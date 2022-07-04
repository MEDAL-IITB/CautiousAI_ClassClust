import os
from glob import glob
from tqdm import tqdm
import time
import copy

import cv2
import random
import numpy as np
from numpy import newaxis
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('Agg')

import pydicom as dicom
import pydicom.data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import torchxrayvision as xrv

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from scipy import stats
from scipy import ndimage, misc
from sklearn import metrics
import seaborn as sns

seed = 1997
shuffle_seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

transform = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    }

batch_size = 128
train_data = torchvision.datasets.ImageFolder(root='/home/Drive/abhiraj/data/mias_gen/train', transform=transform['train'])
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_data = torchvision.datasets.ImageFolder(root='/home/Drive/abhiraj/data/mias_gen/test', transform=transform['test'])
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)
ood_data = torchvision.datasets.ImageFolder(root='/home/Drive/abhiraj/data/mias_gen/ood/birads', transform=transform['test'])
ood_loader = torch.utils.data.DataLoader(dataset=ood_data, batch_size=batch_size, shuffle=False, num_workers=0)

dataloaders = {'train': train_loader,
               'test' : test_loader,
               'ood'  : ood_loader}
dataset_sizes = {'train': len(train_data),
                 'test' : len(test_data),
                 'ood'  : len(ood_data)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
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
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

num_classes = 3
#model = models.vgg16(pretrained=True)
#model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.01)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_fit = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=20)

#evaluate performance
def validate_softmax(model):
  model.eval()
  df_scores = pd.DataFrame(columns = ['im_id', 'type','label', 'pred', 'score'])
  im_id = 0
  with torch.set_grad_enabled(False):
    for set in ['test','ood']:
      for inputs, labels in dataloaders[set]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        outputs = F.softmax(outputs)
        scores, preds = torch.max(outputs, 1)
        
        for i in range(len(scores)):
          if set == 'test':
            df_scores = df_scores.append({'im_id':im_id, 'type':'id','label':labels[i].item(), 'pred':preds[i].item(), 'score':scores[i].item()}, ignore_index = True)
          else:
            df_scores = df_scores.append({'im_id':im_id, 'type':'od','label':'OOD', 'pred':preds[i].item(), 'score':scores[i].item()}, ignore_index = True)
          im_id+=1
  df_scores.to_csv('softmax_mias_test_scores.csv',index=False)
           
validate_softmax(model_fit)     

def validate_odin(model):
  model.eval()
  df_scores = pd.DataFrame(columns = ['im_id', 'type','label', 'pred', 'score'])
  im_id = 0
  
  for set in ['test','ood']:
    for inputs, labels in dataloaders[set]:
      #inputs = inputs.to(device)
      inputs = Variable(inputs.to(device), requires_grad = True)
      labels = labels.to(device)
      
      outputs = model(inputs)
      scores, preds = torch.max(outputs, 1)
      
      loss = criterion(outputs, preds)
      loss.backward()
      gradient =  torch.ge(inputs.grad.data, 0)
      gradient = (gradient.float() - 0.5) * 2
      gradient[0][0] = (gradient[0][0] )/(0.5) 
      gradient[0][1] = (gradient[0][1] )/(0.5)
      gradient[0][2] = (gradient[0][2] )/(0.5)
      tempInputs = torch.add(inputs.data,  (-1)*0.002, gradient)
      logits = model(Variable(tempInputs))
      logits = logits / 1000
      softmax = F.softmax(logits, dim=1)
      prediction = logits.max(1, keepdim=True)[1]
      softmax = F.softmax(logits, dim=1)
      max_softmax, _ = torch.max(softmax.data, 1)
      
      for i in range(len(scores)):
        if set == 'test':
          df_scores = df_scores.append({'im_id':im_id, 'type':'id','label':labels[i].item(), 'pred':int(prediction[i]), 'score':max_softmax[i].detach().cpu().item()}, ignore_index = True)
        else:
          df_scores = df_scores.append({'im_id':im_id, 'type':'od','label':labels[i].item(), 'pred':int(prediction[i]), 'score':max_softmax[i].detach().cpu().item()}, ignore_index = True)
        im_id+=1
  df_scores.to_csv('odin_mias_test_scores.csv',index=False)
           
validate_odin(model_fit)  

def generate_gradcam_vis(model, set):
  model.eval()
  target_layer = model.resnet.layer4[1].conv2
  cam = GradCAM(model=model, target_layer=target_layer)
  for i, sample in enumerate(dataloaders[set], 0):
    img, label = sample
    img = img.to(device)
    for j in range(len(label)):
      input_tensor = img[j].unsqueeze(0)
      rgb_img = (img[j]).float().permute(1, 2, 0).cpu().numpy()
      rgb_img = rgb_img/np.max(rgb_img)
      target_category = int(label[j].item())
      if target_category > 1:
        target_category = int(0)
      grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
      grayscale_cam = grayscale_cam[0, :]
      visualization = show_cam_on_image(rgb_img, grayscale_cam)
      
      fig = plt.figure()
      ax1 = fig.add_subplot(1,2,1)
      ax1.imshow(rgb_img)
      ax2 = fig.add_subplot(1,2,2)
      ax2.imshow(visualization)
      bach_class = str(label[j].item())
      fig.suptitle(set+'_'+str(target_category)+'_'+bach_class, fontsize=12)
      fig.savefig('/home/Drive/abhiraj/eval/GRADCAM/SOFTMAX/softmax_gradcam_'+set+'_'+str(target_category)+'_'+bach_class+'_'+'_'+str(j)+'.png')
      
generate_gradcam_vis(model_fit, 'test')  
generate_gradcam_vis(model_fit, 'ood')