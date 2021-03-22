import argparse
import os
import copy
import time
import random
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from models.mobilenetv2 import MobileNetV2

def set_random_seeds(random_seed, use_multi_gpu=False):
    '''Set random seeds.
    
    Args:
        random_seed (int): random seed to fix randomness
    '''
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if use_multi_gpu:
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def load_data(data_dir, dataset=0):
    '''Load dataset.
    
    Args:
        data_dir (str): dataset path to load dataset
        dataset (int): dataset number (e.g. 0: CIFAR-10, 1: CIFAR-100)
    Returns:
        dataloaders (dict): data loaders for training and validation
        dataset_sizes (dict): each dataset size
    '''
    CIFAR10, CIFAR100 = 0, 1
    DATASET = ['CIFAR-10', 'CIFAR-100']

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    }

    # Download or load image datasets
    if dataset == CIFAR10:
        image_datasets = {x: datasets.CIFAR10(root='./data',
                                              train=True if x == 'train' else False,
                                              download=True, transform=data_transforms[x])
                             for x in ['train', 'val']}
    elif dataset == CIFAR100:
        image_datasets = {x: datasets.CIFAR100(root='./data',
                                               train=True if x == 'train' else False,
                                               download=True, transform=data_transforms[x])
                             for x in ['train', 'val']}
    else:
        assert False, 'Invalid Dataset.'

    # Create dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                                  shuffle=True if x == 'train' else False,
                                                  num_workers=6)
                      for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}    

    return dataloaders, dataset_sizes

def load_model(device, width_mult=1.0, use_res_connect=True, linear_bottleneck=True,
               res_loc=0, num_classes=10, pretrained_path='', use_multi_gpu=False):
    '''Load model, loss function, optimizer and scheduler.

    Args:
        device (obj): device object for cpu or gpu
        width_mult (float): multilplier value
        use_res_connect (bool): whether to use residual connection or not
        linear_bottleneck (bool): whether to use linear bottleneck or not
        res_loc (int): residual location
                       e.g. 0: between bottlenecks, 1: between expansions,
                            2: between depthwise layers
        num_classes (int): the number of classes
        pretrained_path (str): pretrained model path for transfer learning

    Returns:
        model (obj): loaded model
        criterion (obj): loss function
        optimizer (obj): optimizer (e.g. SGD, RMSprop)
        scheduler (obj): learning scheduler (e.g. Step, Cosine Annealing)
    '''
    model = MobileNetV2(num_classes=num_classes,
                        width_mult=width_mult,
                        use_res_connect=use_res_connect,
                        linear_bottleneck=linear_bottleneck,
                        res_loc=res_loc)

    # Device Settings (Single GPU or Multi-GPU)
    if use_multi_gpu:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=0.045, momentum=0.9, weight_decay=0.00004)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    return model, criterion, optimizer, scheduler


def train_model(dataloaders, dataset_sizes, device,
                model, criterion, optimizer, scheduler,
                save_model, model_dir, save_acc, result_dir,
                num_epochs=200):
    '''Train and evaluate model.

    Args:
        dataloaders (dict): data loaders for training and validation
        dataset_sizes (dict): each dataset size
        device (obj): device object for cpu or gpu
        model (obj): loaded model
        criterion (obj): loss function
        optimizer (obj): optimizer (e.g. SGD, RMSprop)
        scheduler (obj): learning scheduler (e.g. Step, Cosine Annealing)
        save_model (bool) whether to save model or not
        model_dir (str): path to save model
        save_acc (bool): whether to save model accuracy or not
        result_dir (str): path to save accruacy
        num_epochs (int): the number of epochs

    Returns:
        model (obj): trained model
    '''
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_opt_wts = copy.deepcopy(optimizer.state_dict())
    best_acc = 0.0
    running_accs = []

    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_acc = torch.sum(preds == labels.data).double() / batch_size
                running_accs.append(running_acc.item())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_opt_wts = copy.deepcopy(optimizer.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save best model weights
    id = round(time.time())
    if save_model:
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        torch.save(model.state_dict(), f'{model_dir}/mobilenetv2_{id}')
        torch.save(best_opt_wts, f'{model_dir}/mobilenetv2_{id}_opt')

    # save accruacies
    # https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/
    if save_acc:
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        steps = range(len(running_accs))
        
        # plotting the points  
        plt.plot(steps, running_accs)
        
        # naming the x and y axis 
        plt.xlabel('Steps') 
        plt.ylabel('Accruacy') 
        
        # giving a title to my graph 
        plt.title('Training Results') 

        plt.savefig(f'{result_dir}/acc_{id}.png')

        with open(f'{result_dir}/acc_{id}.txt', 'w') as f:
            f.write('\n'.join(list(map(str, running_accs))))

    return model


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train MobileNetV2')
    parser.add_argument('--data_dir', type=str, help='Path of input data', default='./data')
    parser.add_argument('--dataset', type=int, help='Dataset: 0 (CIFAR-10), 1 (CIFAR-100)', default=0)
    parser.add_argument('--save_model', action='store_true', help='Whether to save model or not')
    parser.add_argument('--pretrained_path', type=str, help='Path of model to save', default='')
    parser.add_argument('--model_dir', type=str, help='Path of model to save', default='./trained_models')
    parser.add_argument('--save_acc', action='store_true', help='Whether to save accruacies or not')
    parser.add_argument('--result_dir', type=str, help='Path of results to save', default='./results')
    parser.add_argument('--width_mult', type=float, help='Width for multiplier', default=1.0)
    parser.add_argument('--use_res_connect', action='store_true', help='Whether to use residual connection or not')
    parser.add_argument('--res_loc', type=int, help='Location of residual connections (e.g. Between bottlenecks -> 0', default=0)
    parser.add_argument('--linear_bottleneck', action='store_true', help='Whether to use Linear Bottlenck or not')
    parser.add_argument('--epoch', type=int, help='Epoch', default=200)
    parser.add_argument('--random_seed', type=int, help='Random seed for reproducibility', default=0)
    parser.add_argument('--print_to_file', action='store_true', help='Whether to print results or not')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Whether to use multi-gpu or not')

    args = parser.parse_args()

    # Set print function
    from utils import init_file_for_print, set_print_to_file
    id = round(time.time())
    init_file_for_print(id)
    print = set_print_to_file(print, args.print_to_file, id)

    # Set Random Seeds
    set_random_seeds(args.random_seed, args.use_multi_gpu)

    # Load dataset
    dataloaders, dataset_sizes = load_data(args.data_dir, args.dataset)
    NUM_CLS = [10, 100]

    # Load and train MobileNetV2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, criterion, optimizer, scheduler = load_model(device, args.width_mult,
                                                        args.use_res_connect,
                                                        args.linear_bottleneck,
                                                        args.res_loc,
                                                        NUM_CLS[args.dataset],
                                                        args.pretrained_path,
                                                        args.use_multi_gpu)
    model = train_model(dataloaders, dataset_sizes, device,
                        model, criterion, optimizer, scheduler,
                        args.save_model, args.model_dir,
                        args.save_acc, args.result_dir, args.epoch)
