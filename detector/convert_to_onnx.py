import torch
from sudoku_detector import MultiClassifier
from digits_classifier import SimpleClassifier, SimpleNN

def main():
    # model = SimpleClassifier()
    # model.load_state_dict(torch.load('digits_classifier_augmentations.pth'))
    model = SimpleNN()
    model.eval()
    dummy_input = torch.zeros([1,1,32,32])
    # print(model(dummy_input))
    torch.onnx.export(model, dummy_input, 'digits_classifier_augmentations.onnx', verbose=True)

if __name__ == '__main__':
    main()