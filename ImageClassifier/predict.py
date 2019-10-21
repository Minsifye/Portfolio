
# Imports here
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import copy
from collections import OrderedDict
import argparse
#Reading inputs from cmd prompt
parser = argparse.ArgumentParser(description='predict.py') 
#parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/test/1/image_06743.jpg")
parser.add_argument('--image_path', dest="image_path", action="store", default="./flowers/test/22/image_05366.jpg")
#parser.add_argument('image_path', nargs='*', action="store", default="./flowers/")
parser.add_argument('--top_k', dest="top_k", action="store", default=5, type=int)
parser.add_argument('--checkpoint', dest="checkpoint", action="store", default="./checkpoint.pth") 
parser.add_argument('--category_names', dest="category_names", action="store", default="cat_to_name.json")
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
args = parser.parse_args()
#print("args :", args)
image_path = args.image_path
checkpoint = args.checkpoint
category_names = args.category_names
top_k = args.top_k
use_gpu = args.gpu

# Label mapping
import json
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
    
# Function that loads a checkpoint and rebuilds the model
def load_model(checkpoint_path):
    checkpoint_ld=torch.load(checkpoint_path)
    criteria=checkpoint_ld['criteria']
    optimiz=checkpoint_ld['optimizer_state']
    scheduler = checkpoint_ld['scheduler']

    if checkpoint_ld['arch']=='vgg16':
        model_ld = models.vgg16(pretrained=True)
    else:
        model_ld = models.vgg13(pretrained=True)
        
    model_ld.class_to_idx = checkpoint_ld['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(in_features=25088, out_features=4096)),
            ('relu1',nn.ReLU()),
            ('dropout1',nn.Dropout(p=0.5)),
            ('fc2',nn.Linear(in_features=4096, out_features=checkpoint_ld['hidden_units'])),
            ('relu2',nn.ReLU()),
            ('dropout2',nn.Dropout(p=0.5)),
            ('fc3',nn.Linear(in_features=checkpoint_ld['hidden_units'],out_features=102)),
            ('output',nn.LogSoftmax(dim=1))
        ]))
    
    model_ld.classifier = classifier
    
    for param in model_ld.parameters():
        param.requires_grad = False
    model_ld.load_state_dict(checkpoint_ld['state_dict'])
    return model_ld
    
model_ld = load_model('checkpoint.pth')


#Converting Image(.jpg) to numpy array
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)

    #Resizing using thumbnail
    image.thumbnail((256,256), Image.ANTIALIAS)
    
    width, height = image.size
    size = 256
    shortest_side = min(width, height)
    image = image.resize((int((width/shortest_side)*size), int((height/shortest_side)*size)), resample=0)
    
    crop_size = 224
    
    #Margins for crop
    left_margin = (width - crop_size)/2
    right_margin = left_margin + crop_size
    bottom_margin = (height - crop_size)/2
    top_margin = bottom_margin + crop_size
    
    #Cropping
    image = image.crop((left_margin,bottom_margin,right_margin,top_margin))
    
    image = np.array(image)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    #Normalize
    image = (image/255-mean)/std
    
    #Color Channel adjustment
    image = image.transpose(2,0,1)
    
    return image


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.cpu()
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.unsqueeze_(0)
    
    with torch.no_grad():
        outcome = model.forward(image)
        probabilities = torch.exp(outcome)
        probability_test = F.softmax(outcome.data,dim=1)
        top_probs, top_labels = probabilities.topk(topk)
    
    top_labels = top_labels.numpy()[0]
    
    idx_to_class = {val:key for key,val in model_ld.class_to_idx.items()}
    top_labels = [idx_to_class[i] for i in top_labels]
    top_flowers = [cat_to_name[i] for i in top_labels]
    top_probs = ["%.4f" % e for e in top_probs.tolist()[0]]
    top_probs = [float(i) for i in top_probs]
    return (top_probs, top_labels, top_flowers)

top_probs, top_labels, top_flowers = predict(image_path, model_ld, top_k)
print(f'probs:{top_probs} \n top_labels:{top_labels} \n top_flowers:{top_flowers}')
print("Image Path :", image_path)
df = pd.DataFrame()
df['Probability'] = top_probs
df['Label'] = top_labels
df['Flower'] = top_flowers
print(df)