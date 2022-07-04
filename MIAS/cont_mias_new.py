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
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Subset
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
from sklearn.decomposition import PCA
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
    
num_classes = 3
batch_size = 128
train_data = torchvision.datasets.ImageFolder(root='/home/Drive/abhiraj/data/mias_gen/train', transform=transform['train'])
#idx = [i for i in range(len(train_data)) if train_data.imgs[i][1] != train_data.class_to_idx['N']]
#train_data = Subset(train_data, idx)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_data = torchvision.datasets.ImageFolder(root='/home/Drive/abhiraj/data/mias_gen/test', transform=transform['test'])
#idx = [i for i in range(len(test_data)) if test_data.imgs[i][1] != test_data.class_to_idx['N']]
#test_data = Subset(test_data, idx)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)
ood_data = torchvision.datasets.ImageFolder(root='/home/Drive/abhiraj/data/mias_gen/ood/birads', transform=transform['test'])
ood_loader = torch.utils.data.DataLoader(dataset=ood_data, batch_size=batch_size, shuffle=False, num_workers=0)
sys.exit()
dataloaders = {'train': train_loader,
               'test' : test_loader,
               'ood'  : ood_loader}
dataset_sizes = {'train': len(train_data),
                 'test' : len(test_data),
                 'ood'  : len(ood_data)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SupConLoss(nn.Module):
    """GITHUB: https://github.com/HobbitLong/SupContrast
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

SupConLoss = SupConLoss(temperature=0.1)
SupConLoss = SupConLoss.to(device)

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
                    feats, outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    feat_cat = torch.cat([feats.unsqueeze(1), feats.unsqueeze(1)], dim=1)
                    
                    loss_cont = SupConLoss(feat_cat, labels) 
                    loss_ce = criterion(outputs, labels)
                    loss = loss_ce + 0.5*loss_cont

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


class CustomModel(nn.Module):
  def __init__(self):
    super(CustomModel, self).__init__()
    self.resnet = models.resnet18(pretrained=True)
    num_ftrs = self.resnet.fc.in_features
    self.resnet.fc1 = nn.Linear(num_ftrs, 512)
    self.resnet.fc3 = nn.Linear(512, 128)
    self.linout = nn.Linear(128, num_classes)

  def forward(self, x):
    out = self.resnet.conv1(x)
    out = self.resnet.bn1(out)
    out = self.resnet.relu(out)
    out = self.resnet.maxpool(out)
    out = self.resnet.layer1(out)
    out = self.resnet.layer2(out)
    out = self.resnet.layer3(out)
    out = self.resnet.layer4(out)
    out = self.resnet.avgpool(out)
    out = out.view(-1,512)
    out = self.resnet.fc1(out)
    out = F.normalize(self.resnet.fc3(out))
    lin_out = self.linout(out)

    return out, lin_out

model = CustomModel()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_fit = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=20)

def estimate_density(model, loader, n_clusters):
  model.eval()
  df = pd.DataFrame(columns = ['label']+list(str(i) for i in range(128)))
  with torch.no_grad():
    for i, sample in enumerate(loader, 0):
      x_i ,label  = sample
      x_i= x_i.to(device)
      label = label.to(device)
      z_i, p_i = model(x_i)
      z_i = np.array(z_i.detach().cpu())*1 ############
      lb = np.array(label.detach().cpu())
      for it in range(z_i.shape[0]):
        temp = {}
        temp['label'] = lb[it]
        for jt in range(z_i.shape[1]):
          temp[str(jt)] = z_i[it,jt]
        df = df.append(temp, ignore_index = True)
  
  pca = PCA(n_components=8)
  og_label = df['label']
  del df['label']
  y = pca.fit_transform(df)
  print('explained var :', pca.explained_variance_ratio_)
  df = pd.DataFrame(y, columns = [str(i) for i in range(8)])
  df['label'] = og_label
  
  mean_vectors = df.groupby('label', as_index=False).mean()
  del mean_vectors['label']
  mean_vectors = mean_vectors
  mean_vectors = mean_vectors.to_numpy()
  std_mats = []
  for i in range(n_clusters):
    sub_df = df[df['label'] == i]
    del sub_df['label']
    temp = sub_df.cov()
    std_mats.append(temp.to_numpy())
    
  return mean_vectors, std_mats, pca
  
mean_vectors_train, std_mats_train, pca_obj = estimate_density(model_fit, dataloaders['train'], num_classes)

def calc_score(z_i, mean_vectors, std_mats,n_clusters):
  max_s = [[float('-inf'),0] for i in range(z_i.shape[0])]
  for i in range(n_clusters):
    mu = mean_vectors[i]
    E = std_mats[i]
    E_inv = np.linalg.inv(E)
    s = -np.diag((z_i-mu)@E_inv@(z_i-mu).T) - np.log((2*np.pi)**(z_i.shape[1])*np.linalg.det(E)) # check #
    for j in range(len(max_s)):
      if (max_s[j][0] < s[j]):
         max_s[j][0] = s[j]
         max_s[j][1] = i
  return max_s

#evaluate performance
def validate(model):
  model.eval()
  df_scores = pd.DataFrame(columns = ['im_id', 'type','label', 'pred', 'score'])
  im_id = 0
  with torch.set_grad_enabled(False):
    for set in ['test','ood']:
      for inputs, labels in dataloaders[set]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        feats, outputs = model(inputs)
        outputs = F.softmax(outputs)
        _, preds = torch.max(outputs, 1)
        
        feats = np.array(feats.detach().cpu())
        feats = pca_obj.transform(feats)
        lb = np.array(labels.detach().cpu())
        scores = calc_score(feats*1, mean_vectors_train, std_mats_train, num_classes)
        
        for i in range(len(scores)):
          if set == 'test':
            df_scores = df_scores.append({'im_id':im_id, 'type':'id','label':labels[i].item(), 'pred':scores[i][1], 'score':scores[i][0]}, ignore_index = True)
          else:
            df_scores = df_scores.append({'im_id':im_id, 'type':'od','label':labels[i].item(), 'pred':scores[i][1], 'score':scores[i][0]}, ignore_index = True)
          im_id+=1
  df_scores.to_csv('cont_mias_test_scores.csv',index=False)
           
validate(model_fit)   

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
      fig.savefig('/home/Drive/abhiraj/eval/GRADCAM/CONT/cont_gradcam_'+set+'_'+str(target_category)+'_'+bach_class+'_'+'_'+str(j)+'.png')
      
generate_gradcam_vis(model_fit, 'test')  
generate_gradcam_vis(model_fit, 'ood')