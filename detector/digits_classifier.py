import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import cv2

batch_size = 1000
num_epochs = 50

class MNIST19wEmpty(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        # here we replace the zeros with empty image
        if target == 0:
            img = torch.zeros_like(img)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class Digits(MNIST19wEmpty):
    def __len__(self):
        return int(1.9*len(self.data)) # we don't need to produce 0's, just use the regular empty cell from before

    def __getitem__(self, index):
        if index<len(self.data):
            return super().__getitem__(index)
        else:
            # produce a digit:
            img = np.zeros([28, 28], 'int8')
            digit = int((index-len(self.data))/10000)+1
            ind_x = img.shape[1]//3 + np.random.randint(-5, 5)
            ind_y = img.shape[0]*2//3 + np.random.randint(-3, 3)
            text_size = np.random.random()*1.5+0.5
            gray_level = np.random.randint(200, 255)
            thickness = np.random.randint(1,2)
            cv2.putText(img, str(digit), (ind_x, ind_y), 1, text_size, gray_level, thickness);
            # plt.imshow(img);plt.show()

            img = Image.fromarray(img, mode='L')

            if self.transform is not None:
                img = self.transform(img)

            return img, digit

class RandomEdgeLines:
    """randomly add vertical and horizontal lines on edges of the image
    operates on tensors"""
    def __call__(self, x):
        #is to add lines?
        xi, xf, yi, yf = np.random.random(4)>0.5
        #thickness of lines:
        xit, xft, yit, yft = np.random.randint(1, 4, 4)
        gray_level = np.random.random()/4+0.75
        # add first horizontal line:
        if xi:
            x[:, :xit, :] = gray_level
        # add last horizontal line:
        if xf:
            x[:, -xft:, :] = gray_level
        # add first vertical line:
        if yi:
            x[:, :, :yit] = gray_level
        # add last vertical line:
        if yf:
            x[:, :, -yft:] = gray_level

        return x

class RandomBackground:
    """randomly add background to the image
    operates on tensors"""
    def __call__(self, x):
        x[x==0] = torch.rand(1)/3
        return x

class RandomScratch: #TODO
    """randomly add a scratch to the image
    operates on tensors"""
    def __call__(self, x):
        return x


def init_train():
    global trainloader, testloader
    transform = transforms.Compose(
        [transforms.Resize(32),
         transforms.ToTensor(),
         RandomEdgeLines(),
         RandomBackground()
         # RandomAffine(degrees, translate=None, scale=None, shear=None, interpolation=<InterpolationMode.NEAREST: 'nearest'>, fill=0, fillcolor=None, resample=None),
        # RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    ])
    trainset = Digits(root='./data', train=True,
                      download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    testset = Digits(root='./data', train=False,
                     download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)
    classes = list(map(str, range(10)))



# functions to show an image

def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(device='cpu'):
    model = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    test_accuracy = []

    # measure the initial test accuracy - we expect to get around 10% which is random for 10 classes:
    acc = test(model)
    test_accuracy.append(acc)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.to(device)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

        print('test epoch on ', epoch)
        test_accuracy.append(test(model))

    print('Finished Training')
    PATH = 'digits_classifier.pth'
    torch.save(model.state_dict(), PATH)

    plt.plot(test_accuracy);
    plt.xlabel('training epochs')
    plt.ylabel('test accuracy')
    plt.title('Digits Classifier Training')
    plt.grid()
    plt.savefig('outputs/Digits_training_w_edges.png')

def test(model=None, device='cpu', verbose=True):

    if model is None:
        model = SimpleClassifier()
        PATH = 'digits_classifier.pth'
        model.load_state_dict(torch.load(PATH))

    model = model.to(device)
    l = []
    p = []
    for i, (images, labels) in enumerate(testloader):
        # print(i)
        predictions = model(images).argmax(1)
        l += labels
        p += predictions

    l = np.array(l)
    p = np.array(p)
    acc = l==p
    if verbose:
        print('accuracy: ')
        print(sum(acc)*1.0/len(acc))
    return sum(acc)*1.0/len(acc)

    # print(l[l!=p])

if __name__ == "__main__":
    # test()
    init_train()
    train(device='cuda')
