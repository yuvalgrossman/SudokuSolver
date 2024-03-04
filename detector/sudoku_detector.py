import os.path
import cv2
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class GridNums(Dataset):
    def __init__(self, source_path):
        self.grids = sorted(glob.glob(source_path + '/grids/*.png'))
        # self.nums = sorted(glob.glob(source_path + '/nums/*.png'))
        self.grid_transform = ToTensor()
        self.nums_transform = ToTensor()

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, index):
        img_filename=self.grids[index]
        img_name = img_filename.split(os.path.sep)[-1]

        grid = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
        nums = cv2.imread(img_filename.replace('/grids/', '/nums/'), cv2.IMREAD_GRAYSCALE)

        grid = self.grid_transform(grid)
        nums = self.nums_transform(nums)

        return (grid, nums, img_name)

from torch import nn
import torch.nn.functional as F

class NaiveGridDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 9, 30)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(9, 16, 5)
        self.conv3 = nn.Conv2d(16, 64, 6)
        self.conv4 = nn.Conv2d(64, 1, 12)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        return x

from torchvision.models.resnet import ResNet, BasicBlock

class ResnetDetector(ResNet):
    def __init__(self):
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=10)
        # adjust in layer for grayscale:
        self.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        # adjust output for 9x9 grid (we'll ignore last FC layer):
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(9, 9))
        self.conv_last = nn.Conv2d(512, 10, (1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.conv_last(x)
        x = nn.functional.softmax(x, 1)
        return x

from digits_classifier import SimpleClassifier
class MultiClassifier(nn.Module):
    def __init__(self, PATH = 'digits_classifier.pth'):
        super().__init__()
        self.classifier = SimpleClassifier()
        self.classifier.load_state_dict(torch.load(PATH))
        # self.conv1 = nn.Conv2d(1, 6, 45)
        # self.conv1.weights = self.classifier.conv1.weight.repeat_interleave(9, dim=2).repeat_interleave(9, dim=3)
        # self.conv2 = nn.Conv2d(6, 16, 45)
        # self.conv2.weights = self.classifier.conv2.weight.repeat_interleave(9, dim=2).repeat_interleave(9, dim=3)
        # # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # # self.fc1 = self.classifier.fc1.weight.unsqueeze(2).unsqueeze(3).repeat_interleave(9, dim=2).repeat_interleave(9, dim=3)
        # # self.fc2 = self.classifier.fc2.weight.unsqueeze(2).unsqueeze(3).repeat_interleave(9, dim=2).repeat_interleave(9, dim=3)
        # # self.fc3 = self.classifier.fc3.weight.unsqueeze(2).unsqueeze(3).repeat_interleave(9, dim=2).repeat_interleave(9, dim=3)
        #
        # self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.pool(F.relu(x))
        # x = self.conv2(x)
        # x = self.pool(F.relu(x))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        # 1 split into 9x9 cells:
        b,c,w,h = x.shape
        output = torch.zeros(b, c, 9, 9)
        for i in range(9):
            ind_x = int(i * w / 9)
            for j in range(9):
                ind_y = int(j * h / 9)
                cell = x[:,:,ind_x:ind_x+int(w/9), ind_y:ind_y+int(h/9)]
                # TODO resize
                output[:,:,i,j] = self.classifier(cell).argmax(1, keepdims=True)
        return output

def train():
    dataset = GridNums('data/GridNums/printed/train')
    trainloader = DataLoader(dataset, batch_size=1)

    model = NaiveGridDetector().to('cuda')
    model = ResnetDetector().to('cuda')
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    # criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MultiLabelSoftMarginLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_logger = []

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            if epoch%50==0:
                plt.hist(model.conv1.weight.flatten().detach().cpu(), 100)
                plt.title(model.conv1.weight.sum().item())
                plt.show()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to('cuda'))

            # build target one hot encoding from labels:
            target = torch.zeros_like(outputs).scatter(1, (labels * 255).to(device='cuda', dtype=int), 1.)

            loss = criterion(outputs, labels.to('cuda'))
            print(outputs.sum(), labels.sum())
            loss.backward()
            optimizer.step()


            # print statistics
            running_loss += loss.item()
            loss_logger.append(loss.item()*255)
            if i % 100 == 99:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/100:.3f}')
                running_loss = 0.0


    plt.plot(loss_logger)
    plt.show()
    print('Finished Training')
    PATH = './naive_grid_detector.pth'
    torch.save(model.state_dict(), PATH)

def test(model_weights='digits_classifier.pth'):
    # dataset = GridNums('data/GridNums/printed/test')
    # dataset = GridNums('data/GridNums/mnist_based/test')
    dataset = GridNums('data/GridNums/combine_mnist/test')
    testloader = DataLoader(dataset, batch_size=1000, shuffle=False)

    model = MultiClassifier(model_weights)
    accuracy = []

    for i, data in enumerate(testloader):
        inputs, labels, names = data
        outputs = model(inputs)

        accuracy.append((labels*255==outputs).sum()*1.0/len(outputs.flatten()))

    print('accuracy {:.3f}'.format(np.mean(accuracy)))


def infer(image):
    model = MultiClassifier()
    from img_proc import user_crop
    from PIL import Image
    croped = user_crop(image)
    flipped = 255 - croped
    image = Image.fromarray(flipped)
    image = image.convert("L").resize((300, 300))
    inputs = ToTensor()(image).unsqueeze(0)
    outputs = model(inputs)
    return outputs


if __name__ == "__main__":
    test(model_weights='digits_classifier_augmentations.pth')
    # train()

    # img = cv2.imread('data/soduko.png')
    # img = cv2.imread('data/ex1.jpeg')
    # print(infer(img))

