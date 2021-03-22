import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.mobilenetv2 import MobileNetV2

def load_data(data_dir):
    '''Load dataset.
    
    Args:
        data_dir (str): dataset path to load dataset
    Returns:
        dataloaders (dict): data loaders for training and validation
        dataset_sizes (dict): each dataset size
    '''
    data_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010)),
                    ])

    # Download or load image datasets
    image_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                     transform=data_transform)

    # Create dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=128,
                                             shuffle=False, num_workers=6)
    dataset_size = len(image_dataset)

    return dataloader, dataset_size

def load_model(device, model_path, width_mult=1.0, use_res_connect=True, res_loc=0, use_multi_gpu=False):
    '''Load model, loss function, optimizer and scheduler.

    Args:
        device (obj): device object for cpu or gpu
        model_path (str): pretrained model path for evaluation
        width_mult (float): multilplier value
        use_res_connect (bool): whether to use residual connection or not
        res_loc (int): residual location
                       e.g. 0: between bottlenecks, 1: between expansions,
                            2: between depthwise layers

    Returns:
        model (obj): loaded model
        criterion (obj): loss function
    '''
    model = MobileNetV2(num_classes=10,
                        width_mult=width_mult,
                        use_res_connect=use_res_connect,
                        res_loc=res_loc)

    # Device Settings (Single GPU or Multi-GPU)
    if use_multi_gpu:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()

    return model, criterion

def eval_model(dataloader, dataset_size, device, model, criterion):
    '''Evaluate model.

    Args:
        dataloaders (dict): data loaders for training and validation
        dataset_sizes (dict): each dataset size
        device (obj): device object for cpu or gpu
        model (obj): loaded model
        criterion (obj): loss function

    Returns:
        model (obj): trained model
    '''
    since = time.time()

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)


    total_loss = running_loss / dataset_size
    total_acc = running_corrects.double() / dataset_size

    time_elapsed = time.time() - since
    print('Inference complete in {:.0f}m {:.4f}s'.format(
        time_elapsed // 60, time_elapsed - time_elapsed // 60))
    print('Loss: {:.4f} Acc: {:.4f}'.format(total_loss, total_acc))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate MobileNetV2')
    parser.add_argument('--data_dir', type=str, help='Path of input data', default='./data')
    parser.add_argument('--model_path', type=str, help='Path of model to save', default='')
    parser.add_argument('--width_mult', type=float, help='Width for multiplier', default=1.0)
    parser.add_argument('--use_res_connect', action='store_true', help='Whether to use residual connection or not')
    parser.add_argument('--res_loc', type=int, help='Location of residual connections (e.g. Between bottlenecks -> 0', default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', help='Whether to use multi-gpu or not')
    args = parser.parse_args()

    # Check a model path
    assert args.model_path != '', 'You need to input a trained model path.'

    # Load dataset
    dataloader, dataset_size = load_data(args.data_dir)

    # Load and train MobileNetV2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, criterion = load_model(device, args.model_path, args.width_mult,
                                  args.use_res_connect, args.res_loc, args.use_multi_gpu)
    eval_model(dataloader, dataset_size, device, model, criterion)
