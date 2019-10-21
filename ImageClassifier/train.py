# Imports here
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import copy
from collections import OrderedDict
import argparse
parser = argparse.ArgumentParser(description='Train.py')
parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--dropout', dest = "dropout", action = "store", default=0.5)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=10)
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
parser.add_argument('--hidden_units', dest="hidden_units", action="store", default=1024)

args = parser.parse_args()
where = args.data_dir
path = args.save_dir
lr = args.learning_rate
structure = args.arch
dropout = args.dropout
power = args.gpu
epochs = args.epochs
hidden_units = args.hidden_units

print("ARGS : ", args)

# ## Load the data

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

device = 'cuda' if args.gpu else 'cpu'

# Defining your transforms for the training, validation, and testing sets
data_transforms = {'train':transforms.Compose([transforms.RandomRotation(45),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ]),
                   'valid_test':transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ])}

# TODO: Load the datasets with ImageFolder
image_datasets = {'train':datasets.ImageFolder(train_dir,data_transforms['train']),
                 'valid':datasets.ImageFolder(valid_dir,data_transforms['valid_test']),
                 'test':datasets.ImageFolder(test_dir,data_transforms['valid_test'])}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {i:torch.utils.data.DataLoader(image_datasets[i],batch_size=32,shuffle=True)
              for i in image_datasets}
dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}

# Building and training the classifier


#model = models.vgg16(pretrained=True)

if args.arch=='vgg16':
    print('Choosen Model : vgg16')
    model = models.vgg16(pretrained=True)
    
elif args.arch=='vgg13':
    print('Choosen Model : vgg13')
    model = models.vgg13(pretrained=True)




# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False



#  Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
classifier = nn.Sequential(OrderedDict([
    ('fc1',nn.Linear(in_features=25088, out_features=4096)),
    ('relu1',nn.ReLU()),
    ('dropout1',nn.Dropout(p=0.5)),
    ('fc2',nn.Linear(in_features=4096, out_features=args.hidden_units)),
    ('relu2',nn.ReLU()),
    ('dropout2',nn.Dropout(p=0.5)),
    ('fc3',nn.Linear(in_features=args.hidden_units,out_features=102)),
    ('output',nn.LogSoftmax(dim=1))
]))


# Setting a classifier to model
model.classifier = classifier




def train_model(model, criterion, optimizer, scheduler, num_epochs,device='cuda'):
    torch.cuda.is_available()
    if device=='cuda':
        model.cuda()
    else:
        model.cpu()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                print('Training...')
                model.train()  # Set model to training mode
            else:
                print('Validating...')
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


criterion = nn.NLLLoss()
# OPtimizer
optimizer_ft = optim.Adam(list(model.classifier.parameters()), lr=args.learning_rate)
# Scheduler
exp_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft,step_size=4)
# Training Model
model_trained = train_model(model, criterion, optimizer_ft, exp_scheduler, args.epochs, device)

def get_accuracy(model_trainedA, data_set):
    model_trainedA.eval()
    model_trainedA.to(device='cuda')
    accuracy_rate = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data_set]):
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model_trainedA.forward(inputs)
            max_log, predicted = output.max(dim=1)

            correct_out = predicted == labels.data
            accuracy_rate.append(correct_out.float().mean().cpu().numpy())

    print(f'Total Accuracy on Test Data : {sum(accuracy_rate)*100/len(accuracy_rate)}')


get_accuracy(model_trained, 'test')

model_trained.cpu()
model_trained.class_to_idx = image_datasets['train'].class_to_idx



torch.save({'state_dict':model_trained.state_dict(),
            'optimizer_state':optimizer_ft.state_dict(),
            'scheduler':exp_scheduler.state_dict(),
            'criteria':criterion.state_dict(),
            'arch':args.arch,
            'epochs':args.epochs,
            'hidden_units': args.hidden_units,
            'learning_rate': args.learning_rate,
            'class_to_idx':model_trained.class_to_idx},
           args.save_dir)