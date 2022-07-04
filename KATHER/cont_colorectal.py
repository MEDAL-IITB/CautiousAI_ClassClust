seed = 0
import torch
torch.manual_seed(seed)
import torchvision
import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
np.random.seed(seed)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
random.seed(seed)

from random import randrange
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import numbers

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn import metrics
########################################################################################
#path = "/home/abhiraj/DDP/CL/data/COLORECTAL/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/"
path = "/home/abhiraj/DDP/CL/data/full_kather/img/"
classes = os.listdir(path)
classes.sort()
#id_classes = [classes[0],classes[1],classes[3]]                                                        ######
#od_classes = [classes[2]] 
print(classes)                                                                  ######[2,3,5]
id_classes = [classes[0],classes[1],classes[2],classes[3],classes[4],classes[5],classes[6],classes[7],classes[8]]                                                        ######
#od_classes = [classes[3],classes[5]]                                                                 ###### [2,3,5]
od_classes = []
id_classes.sort()
od_classes.sort()

def pretrain_colorectal():
  all_file_paths = []
  train_file_paths = []
  val_file_paths = []
  test_file_paths = []
  od_trainval_count = 0
  od_test_count = 1000

  for i in range(len(id_classes)):
    filenames = os.listdir(os.path.join(path,id_classes[i]))
    print(id_classes[i],len(filenames))
    count = 0
    for f in filenames:
      file_path = os.path.join(os.path.join(path,id_classes[i]),f)
      if count < int(0.9*len(filenames)):
        train_file_paths.append([file_path, i, 'id'])
      elif (count >= int(0.9*len(filenames)) and count < int(0.95*len(filenames))):
        val_file_paths.append([file_path, i, 'id'])
      elif count >= int(0.95*len(filenames)):
        test_file_paths.append([file_path, i, 'id'])
      all_file_paths.append([file_path, i, 'id'])
      count += 1
  for i in range(len(od_classes)):
    filenames = os.listdir(os.path.join(path,od_classes[i]))
    print(od_classes[i],len(filenames))
    count = 0
    for f in filenames:
      file_path = os.path.join(os.path.join(path,od_classes[i]),f)
      if count < od_trainval_count:
        flip = random.uniform(0,1)
        if flip > 0.5:
          idx = random.randint(0, len(id_classes)-1)
          train_file_paths.append([file_path, idx, 'od'])
          all_file_paths.append([file_path, idx, 'od'])
          count += 1
        else:
          idx = random.randint(0, len(id_classes)-1)
          val_file_paths.append([file_path, idx, 'od'])
          all_file_paths.append([file_path, idx, 'od'])
          count += 1
      elif (count < od_trainval_count+od_test_count and count >= od_trainval_count):
        idx = random.randint(0, len(id_classes)-1)
        test_file_paths.append([file_path, idx, 'od'])
        all_file_paths.append([file_path, idx, 'od'])
        count += 1
      else:
        idx = random.randint(0, len(id_classes)-1)
        all_file_paths.append([file_path, idx, 'od'])
        count += 1

  random.shuffle(all_file_paths)
  random.shuffle(train_file_paths)
  random.shuffle(val_file_paths)
  random.shuffle(test_file_paths)
  return all_file_paths, train_file_paths, val_file_paths, test_file_paths

class CRPreTrainDataset(Dataset):
    def __init__(self, transform=None):
      self.transform = transform
      self.all_file_paths, self.train_file_paths, self.val_file_paths, self.test_file_paths =  pretrain_colorectal()

    def __len__(self):
      return len(self.all_file_paths)
    
    def get_train_paths(self):
      return self.train_file_paths
    def get_val_paths(self):
      return self.val_file_paths
    def get_test_paths(self):
      return self.test_file_paths
    
    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      img_path = self.all_file_paths[idx][0]
      image = Image.open(img_path).convert("RGB")
      img_class = self.all_file_paths[idx][1]
      img_type = self.all_file_paths[idx][2]

      if self.transform:
        x_i = self.transform(image)
        x_j = self.transform(image)

      sample = {'x_i':x_i,'x_j':x_j,'label':img_class, 'type': img_type }

      return sample


class TrainDataset(Dataset):
    def __init__(self, train_paths, transform=None):
      self.transform = transform
      self.train_file_paths =  train_paths

    def __len__(self):
      return len(self.train_file_paths)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      img_path = self.train_file_paths[idx][0]
      image = Image.open(img_path).convert("RGB")
      img_class = self.train_file_paths[idx][1]
      img_type = self.train_file_paths[idx][2]

      if self.transform:
        x_i = self.transform(image)
        x_j = self.transform(image)

      sample = {'x_i':x_i,'x_j':x_j,'label':img_class, 'type': img_type }

      return sample

class ValDataset(Dataset):
    def __init__(self, val_paths, transform=None):
      self.transform = transform
      self.val_file_paths =  val_paths

    def __len__(self):
      return len(self.val_file_paths)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      img_path = self.val_file_paths[idx][0]
      image = Image.open(img_path).convert("RGB")
      img_class = self.val_file_paths[idx][1]
      img_type = self.val_file_paths[idx][2]

      if self.transform:
        x_i = self.transform(image)
        x_j = self.transform(image)

      sample = {'x_i':x_i,'x_j':x_j,'label':img_class, 'type': img_type }

      return sample

class TestDataset(Dataset):
    def __init__(self, test_paths, mean, std, transform=None):
      self.transform = transform
      self.test_file_paths =  test_paths
      self.inv_norm = transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],std= [1/s for s in std])

    def __len__(self):
      return len(self.test_file_paths)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      img_path = self.test_file_paths[idx][0]
      image = Image.open(img_path).convert("RGB")
      img_class = self.test_file_paths[idx][1]
      img_type = self.test_file_paths[idx][2]

      if self.transform:
        x_i = self.transform(image)
        x_j = self.transform(image)
        
      im_arr = self.inv_norm(x_i)
        
      im_arr = np.array(im_arr).transpose(1,2,0)
      
      sample = {'x_i':x_i,'x_j':x_j,'label':img_class, 'type': img_type, 'path': img_path, 'og_img': np.asarray(im_arr) }

      return sample
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

composed_tf = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                #transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1),
                                transforms.RandomVerticalFlip(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                #transforms.GaussianBlur(7, sigma=(0.1, 2.0)),
                                transforms.Normalize(mean = mean, std = std)
                              ])
composed_notf = transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean, std = std)
                              ])
pretrain_dat = CRPreTrainDataset(composed_tf)
pretrain_dat_notf = CRPreTrainDataset(composed_notf)
pretraindataloader = DataLoader(pretrain_dat, batch_size=128,shuffle=True, num_workers=1)
pretraindataloader_notf = DataLoader(pretrain_dat_notf, batch_size=128,shuffle=True, num_workers=1)

train_set = pretrain_dat.get_train_paths()
train_dat = TrainDataset(train_set,composed_tf)
trainloader = DataLoader(train_dat, batch_size=128,shuffle=True, num_workers=1)
train_dat = TrainDataset(train_set,composed_notf)
trainloader_notf = DataLoader(train_dat, batch_size=128,shuffle=True, num_workers=1)
val_set = pretrain_dat.get_val_paths()
val_dat = ValDataset(val_set,composed_notf)
valloader = DataLoader(val_dat, batch_size=128,shuffle=False, num_workers=1)
test_set = pretrain_dat.get_test_paths()
test_dat = TestDataset(test_set,mean,std,composed_notf)
testloader = DataLoader(test_dat, batch_size=128,shuffle=False, num_workers=1)
#############################################################################################################
class CustomModel(nn.Module):
  def __init__(self):
    super(CustomModel, self).__init__()
    self.resnet = models.resnet18(pretrained=False)
    num_ftrs = self.resnet.fc.in_features
    self.resnet.fc1 = nn.Linear(num_ftrs, 512)
    self.resnet.fc3 = nn.Linear(512, 128)

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

    return out

pre_model = CustomModel()
trained_model = torch.load("/home/abhiraj/DDP/CL/models/HistoSimCLR/tenpercent_resnet18.ckpt", map_location=torch.device('cpu'))

for name , param in trained_model['state_dict'].items():
  if isinstance(param, nn.parameter.Parameter):
    param = param.data
  if 'fc' not in name: 
    pre_model.state_dict()['resnet.'+name[13:]].copy_(param)
  elif (name == 'model.resnet.fc.1.weight'):
    pre_model.state_dict()['resnet.fc1.weight'].copy_(param)
  elif (name == 'model.resnet.fc.1.bias'):
    pre_model.state_dict()['resnet.fc1.bias'].copy_(param)
  elif (name == 'model.resnet.fc.3.weight'):
    pre_model.state_dict()['resnet.fc3.weight'].copy_(param)
  elif (name == 'model.resnet.fc.3.bias'):
    pre_model.state_dict()['resnet.fc3.bias'].copy_(param)
##############################################################################################################
class contrastive_loss(nn.Module):
    def __init__(self, tau=0.1, normalize=False):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss
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
######################################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

epoch = 20
early_stop_ep = 10
cont_loss_type = 'simclr'

pre_model = pre_model.to(device)
optimizer = torch.optim.Adam(pre_model.parameters(), lr=0.0001, weight_decay = 0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=5, factor=0.01)

NTXloss = contrastive_loss(normalize=True)
NTXloss = NTXloss.to(device)
SupConLoss = SupConLoss(temperature=0.1)
SupConLoss = SupConLoss.to(device)
##########################################################################################################
def tsne_visualize(model,loader, feats,perplexity,learning_rate, n_iter, random_state, file_name):
  model.eval()
  df = pd.DataFrame(columns = list(str(i) for i in range(128)))
  df['label'] = None
  for i, sample in enumerate(loader, 0):
    x_i, x_j ,label, d_type  = sample['x_i'], sample['x_j'], sample['label'], sample['type']
    x_i = x_i.to(device)
    label = label.to(device)
    if (feats == 1):
      z_i = model(x_i)
    else:
      z_i, _ = model(x_i)
    z_i = np.array(z_i.detach().cpu())
    lb = np.array(label.detach().cpu())
    for it in range(z_i.shape[0]):
      temp = {}
      temp['label'] = lb[it]
      if (d_type[it] == 'od'):
        temp['label'] = len(id_classes)
      else:
        temp['label'] = lb[it]
      for jt in range(z_i.shape[1]):
        temp[str(jt)] = z_i[it,jt]
      df = df.append(temp, ignore_index = True)

  m = TSNE(perplexity = perplexity, learning_rate = learning_rate, n_iter = n_iter, random_state = random_state)
  df_feat = df[list(str(i) for i in range(128))]
  tsne_data = m.fit_transform(df_feat)
  tsne_df = pd.DataFrame(tsne_data, columns = ['x', 'y'])
  tsne_df['label'] = df['label'].tolist()
  sns.FacetGrid(tsne_df,hue='label', size = 6).map(plt.scatter, 'x', 'y')
  plt.legend(loc = 'best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/COLORECTAL/'+ file_name)
##############################################################################################################
#tsne_visualize(pre_model,pretraindataloader_notf, 1,30, 200, 1000, 0, 'cont_before_TSNE.png')
###########################################################################################################
class LinearClassifierModel(nn.Module):
  def __init__(self, pre_model, n_classes):
    super(LinearClassifierModel, self).__init__()
    self.pre_model = pre_model
    for param in self.pre_model.parameters():
      param.requires_grad = True  # freeze
    self.lin1 = nn.Linear(128, 256)
    self.bn1 = nn.BatchNorm1d(256)
    self.lin2 = nn.Linear(256, 512)
    self.bn2 = nn.BatchNorm1d(512)
    self.lin3 = nn.Linear(512, 128)
    self.bn3 = nn.BatchNorm1d(128)
    self.relu = nn.ReLU()
    self.linout = nn.Linear(128, n_classes)

  def forward(self, x):
    out1 = self.pre_model(x)
    out = self.relu(self.bn1(self.lin1(out1)))
    out = self.relu(self.bn2(self.lin2(out)))
    out = self.relu(self.bn3(self.lin3(out)))
    out2 = self.linout(out)

    return out1, out2

full_model = LinearClassifierModel(pre_model, len(id_classes)) # number if id classes
full_model = full_model.to(device)

criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
#################################################################################################################
def train(model,loader, cont_loss_type):
  model.train()
  running_loss = 0.0
  for i, sample in enumerate(loader, 0):
    x_i, x_j ,label  = sample['x_i'], sample['x_j'], sample['label']
    x_i = x_i.to(device)
    x_j = x_j.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    z_i, p_i = model(x_i)
    z_j, p_j = model(x_j)
    z_cat = torch.cat([z_i.unsqueeze(1), z_j.unsqueeze(1)], dim=1)

    if cont_loss_type == 'simclr' :
      loss_cl = NTXloss(z_i, z_j)
      #loss_cl = SupConLoss(z_cat)
    if cont_loss_type == 'supcon':
      loss_cl = SupConLoss(z_cat, label)
    loss_ce_i = criterion(p_i, label)
    loss_ce_j = criterion(p_j, label)
    loss = 0.5*(loss_ce_i+loss_ce_j) + loss_cl#*(0.5)**((ep-10)/10)
    loss.backward()

    optimizer.step()
    running_loss += loss.item()
    
  print('loss: %.5f' % (running_loss/len(train_dat)))
  return (running_loss/len(train_dat))

def test(model,loader):
  model.eval()
  correct = 0
  total = 0
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  with torch.no_grad():
    for i, sample in enumerate(loader, 0):
      x_i, x_j ,label, d_type  = sample['x_i'], sample['x_j'], sample['label'], sample['type']
      x_i = x_i.to(device)
      label = label.to(device)
      z_i, p_i = model(x_i)
      _, predicted = torch.max(p_i.data, 1)
      for i in range(label.shape[0]):
        if d_type[i] == 'id':
          total += 1
          if (predicted[i] == label[i]):
            correct += 1 
            class_correct[label[i]] += 1
          class_total[label[i]] += 1
  for i in range(len(id_classes)):
    print('Accuracy of %5s : %.2f %%' % (id_classes[i], 100 * class_correct[i] / class_total[i]))
  print('Accuracy of the network on the val images: %.2f %%' % (100 * correct / total))
##################################################################################################
import time
print('FT-Training')
tic = time.time()
min_val_loss = float('inf')
early_stop_counter = 0
for ep in range(1,epoch+1):
  cont_loss = train(full_model,trainloader,cont_loss_type)
  print("Epoch :",ep)
  test(full_model,valloader)
  scheduler.step(cont_loss)
  if min_val_loss > cont_loss:
    min_val_loss = cont_loss
    early_stop_counter = 0
  else:
    early_stop_counter += 1
  if early_stop_counter == early_stop_ep:
    print("Early Stopping...")
    break
toc = time.time()
print("Time elapsed :", toc - tic)
##############################################################################################################
test(full_model,valloader)
#tsne_visualize(full_model,pretraindataloader_notf, 0,30, 200, 1000, 0, 'cont_after_TSNE.png')
##############################################################################################################
def kmeans_predictor(model, train_loader, test_loader, n_clusters, test_type, thresh_ip = None):
  model.eval()
  df = pd.DataFrame(columns = ['label']+list(str(i) for i in range(128)))
  with torch.no_grad():
    for i, sample in enumerate(train_loader, 0):
      x_i, x_j ,label, d_type  = sample['x_i'], sample['x_j'], sample['label'], sample['type']
      x_i = x_i.to(device)
      label = label.to(device)
      z_i, p_i = model(x_i)
      z_i = np.array(z_i.detach().cpu())
      lb = np.array(label.detach().cpu())
      for it in range(z_i.shape[0]):
        temp = {}
        if d_type[it] == 'id':
          temp['label'] = lb[it]
        else:
          temp['label'] = n_clusters
        for jt in range(z_i.shape[1]):
          temp[str(jt)] = z_i[it,jt]
        df = df.append(temp, ignore_index = True)
  
  labels = df['label']
  del df['label']
  kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=1000).fit(df.to_numpy())
  mapping = [0 for i in range(n_clusters)]
  for i in range(n_clusters):
    mapping[i] = stats.mode(kmeans.labels_[labels==i])[0][0]
  centers = kmeans.cluster_centers_
  
  class_wise_correct = [0 for i in range(n_clusters)]
  class_wise_total = [0 for i in range(n_clusters)]
  od_correct = 0
  od_total = 0
  id_correct = 0
  id_total = 0
  id_hist = []
  od_hist = []
  TPR_arr = []
  FPR_arr = []
  correct_list = []
  ood_list = []
  df_scores = pd.DataFrame(columns = ['type', 'label','pred','score','path'])
  with torch.no_grad():
    for i, sample in enumerate(test_loader, 0):
      x_i, x_j ,label, d_type, im_paths, og_imgs  = sample['x_i'], sample['x_j'], sample['label'], sample['type'], sample['path'], sample['og_img']
      x_i = x_i.to(device)
      label = label.to(device)
      z_i, p_i = model(x_i)
      z_i = np.array(z_i.detach().cpu())
      lb = np.array(label.detach().cpu())
      for it in range(z_i.shape[0]):
        if d_type[it] == 'id':
          id_hist.append(kmeans.score(np.expand_dims(z_i[it], axis=0)))
          if lb[it] == mapping[kmeans.predict(np.expand_dims(z_i[it], axis=0))[0]]:
            class_wise_correct[lb[it]] += 1
            correct_list.append([kmeans.score(np.expand_dims(z_i[it], axis=0)),1, im_paths[it], og_imgs[it], lb[it]])
          else:
            correct_list.append([kmeans.score(np.expand_dims(z_i[it], axis=0)),0, im_paths[it], og_imgs[it], lb[it]])
          class_wise_total[lb[it]] += 1
          df_scores = df_scores.append({'type':'id','label':int(lb[it]),'pred': int(mapping[kmeans.predict(np.expand_dims(z_i[it], axis=0))[0]]),'score':kmeans.score(np.expand_dims(z_i[it], axis=0)),'path':im_paths[it]}, ignore_index = True )
          id_total+=1
        else:
          od_hist.append(kmeans.score(np.expand_dims(z_i[it], axis=0)))
          ood_list.append([kmeans.score(np.expand_dims(z_i[it], axis=0)), im_paths[it], og_imgs[it]])
          df_scores = df_scores.append({'type':'od','label':int(lb[it]),'pred': 'OOD','score':kmeans.score(np.expand_dims(z_i[it], axis=0)),'path':im_paths[it]}, ignore_index = True )
          od_total+=1
          
  df_scores.to_csv('/home/abhiraj/DDP/CL/eval/COLORECTAL/ckmeans_test_scores.csv', index=False)
  max_score = max([max(id_hist),max(od_hist)])
  min_score = min([min(id_hist),min(od_hist)])
  max_sum = 0
  thresh = 0
  th_arr = np.linspace(min_score,max_score+0.001,int((max_score-min_score)/0.0001))
  
  fpr95 = 0
  tpr95 = 2
  odac25 = 0
  odac50 = 0
  odac75 = 0
  odac90 = 0
  idac25 = 0
  idac50 = 0
  idac75 = 0
  idac90 = 0
  
  if thresh_ip is not None:
    id_correct = np.sum(np.array(id_hist) >= thresh_ip)
    od_correct = np.sum(np.array(od_hist) < thresh_ip)
    thresh = thresh_ip
  else:
    for i in range(len(th_arr)):
      id_corr = np.sum(np.array(id_hist) >= th_arr[i])
      id_wrng = np.sum(np.array(id_hist) < th_arr[i])
      od_corr = np.sum(np.array(od_hist) < th_arr[i])
      od_wrng = np.sum(np.array(od_hist) >= th_arr[i])
      tpr = od_corr/(od_corr+od_wrng)
      fpr = id_wrng/(id_corr+id_wrng)
      TPR_arr.append(tpr)
      FPR_arr.append(fpr)
      if abs(tpr - 0.95) < tpr95:
        fpr95 = fpr
        tpr95 = abs(tpr - 0.95)
      if i == int(0.25*(len(th_arr)-10)): ############## depends on 625
        odac25 = od_corr/(od_wrng+od_corr)
        idac25 = id_corr/(id_wrng+id_corr)
      if i == int(0.5*(len(th_arr)-10)): ############## depends on 625
        odac50 = od_corr/(od_wrng+od_corr)
        idac50 = id_corr/(id_wrng+id_corr)
      if i == int(0.75*(len(th_arr)-10)): ############## depends on 625
        odac75 = od_corr/(od_wrng+od_corr)
        idac75 = id_corr/(id_wrng+id_corr)
      if i == int(0.9*(len(th_arr)-10)): ############## depends on 625
        odac90 = od_corr/(od_wrng+od_corr)
        idac90 = id_corr/(id_wrng+id_corr)
      if ((id_corr+od_corr) >= max_sum):
        max_sum = id_corr+od_corr
        id_correct = id_corr
        od_correct = od_corr
        thresh = th_arr[i]

  auc_score = metrics.auc(FPR_arr, TPR_arr)

  for i in range(n_clusters):
    print(id_classes[i],':',class_wise_correct[i]/class_wise_total[i])
    
  print('Threshold :',thresh)
  print('ID correct :',100*(id_correct/id_total))
  print('OD correct :',100*(od_correct/od_total))
  print('AUC :', auc_score)
  
  plt.figure(figsize = (5,5))
  plt.hist(id_hist, bins=100, density=True, cumulative=True, label='id', histtype='step', alpha=0.55, color='g')
  plt.hist(od_hist, bins=100, density=True, cumulative=True, label='id', histtype='step', alpha=0.55, color='r')
  plt.xlabel('score')
  plt.title('Cummulative density of ID and OD')
  plt.legend(loc = 'best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/COLORECTAL/cont_kmeans_OOD_cdf.png')

  plt.figure()
  plt.hist(id_hist, bins=100, fc=(0, 1, 0, 0.5), label='id', density=True)
  plt.hist(od_hist, bins=100, fc=(1, 0, 0, 0.5),label='od', density=True)
  plt.axvline(x=thresh,color='k')
  plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/COLORECTAL/cont_kmeans_OOD_score.png')

  plt.figure()
  plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  plt.title('ROC curve, AUC= '+str(auc_score))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/COLORECTAL/cont_kmeans_AUROC.png')
  
  fp = open('/home/abhiraj/DDP/CL/eval/COLORECTAL/colo_scores_lists.txt','a')
  fp.write(test_type+' CONT_KMEANS \n') #######
  fp.write('id_hist: '+str(id_hist)+'\n')
  fp.write('od_hist: '+str(od_hist)+'\n')
  fp.close()
  
  fp = open('/home/abhiraj/DDP/CL/eval/COLORECTAL/colo_auroc_lists.txt','a')
  fp.write(test_type+' CONT_KMEANS \n') #######
  fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  fp.close()
  
  ood_list.sort()
    
  fp = open('colorectal_od_paths.txt','a')
  fp.write(test_type+' Cont-Kmeans \n') #######
  for dat in ood_list:
   fp.write(str(dat[0])+' : '+str(dat[1])+'\n')
  fp.close()
  
  count = 0
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0])) 
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_od_cont_kmeans_'+str(count)+'.png') #####################
    if count == 5:
      break
  
  ood_list.sort(reverse=True)
  
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_od_cont_kmeans_'+str(count)+'.png') ######################
    if count == 10:
      break
  
  correct_list.sort()
  
  acc_list = []
  fp = open('colorectal_im_paths.txt','a')
  fp.write(test_type+' Cont-Kmeans \n')
  for dat in correct_list:
   acc_list.append([dat[0], dat[1]])
   fp.write(str(dat[0])+' : '+str(dat[1])+' : '+dat[2]+'\n')
  fp.close()
  
  count = 0
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_cont_kmeans_'+str(count)+'.png')
    if count == 5:
      break
  
  correct_list.sort(reverse=True)
  
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_cont_kmeans_'+str(count)+'.png')
    if count == 10:
      break
  
  correct_arr = np.array(acc_list)
  acc_list = []
  data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  for i in range(len(data_left_list)):
    temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
    acc_list.append(np.sum(temp_arr)/len(temp_arr))
    
  fp = open('colo_acc_curves.txt', 'a')
  fp.write('CONT_kmeans: '+str(acc_list)+'\n')
  fp.close()  
    
  plt.figure()
  plt.plot(data_left_list, acc_list, '-', label='accuracy')
  plt.xlabel('% data ignored')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs %data left')
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/eval/COLORECTAL/cont_kmeans_accuracy.png')  
##############################################################################################################
#kmeans_predictor(full_model, trainloader, testloader, len(id_classes),'no_tf')
###############################################################################################################

def lof_predictor(model, train_loader, test_loader, n_clusters, test_type, thresh_ip = None):
  model.eval()
  df = pd.DataFrame(columns = ['label']+list(str(i) for i in range(128)))
  with torch.no_grad():
    for i, sample in enumerate(train_loader, 0):
      x_i, x_j ,label, d_type  = sample['x_i'], sample['x_j'], sample['label'], sample['type']
      x_i = x_i.to(device)
      label = label.to(device)
      z_i, p_i = model(x_i)
      z_i = np.array(z_i.detach().cpu())
      lb = np.array(label.detach().cpu())
      for it in range(z_i.shape[0]):
        temp = {}
        temp['label'] = lb[it]
        #if d_type[it] == 'id':
        #  temp['label'] = lb[it]
        #else:
        #  temp['label'] = n_clusters
        for jt in range(z_i.shape[1]):
          temp[str(jt)] = z_i[it,jt]
        df = df.append(temp, ignore_index = True)
  
  predictors = []
  for i in range(n_clusters):
    sub_df = df[df['label']==i]
    del sub_df['label']
    clf = LocalOutlierFactor(n_neighbors=200, novelty=True)
    clf.fit(sub_df)
    predictors.append(clf)
  
  class_wise_correct = [0 for i in range(n_clusters)]
  class_wise_total = [0 for i in range(n_clusters)]
  od_correct = 0
  od_total = 0
  id_correct = 0
  id_total = 0
  id_hist = []
  od_hist = []
  TPR_arr = []
  FPR_arr = []
  correct_list = []
  ood_list = []
  df_scores = pd.DataFrame(columns = ['type', 'label','pred','score','path'])
  with torch.no_grad():
    for i, sample in enumerate(test_loader, 0):
      x_i , x_j ,label, d_type, im_paths, og_imgs  = sample['x_i'], sample['x_j'], sample['label'], sample['type'], sample['path'], sample['og_img']
      x_i = x_i.to(device)
      label = label.to(device)
      z_i, p_i = model(x_i)
      z_i = np.array(z_i.detach().cpu())
      lb = np.array(label.detach().cpu())
      for it in range(z_i.shape[0]):
        if d_type[it] == 'id':
          max_score = float('-inf')
          p_idx = 0
          for p in range(len(predictors)):
            score = predictors[p].score_samples(np.expand_dims(z_i[it], axis=0))[0]
            if score>max_score:
              max_score = score
              p_idx = p
          id_hist.append(max_score)
          if lb[it] == p_idx:
            class_wise_correct[lb[it]] += 1
            correct_list.append([max_score,1, im_paths[it], og_imgs[it], lb[it]])
          else:
            correct_list.append([max_score,0, im_paths[it], og_imgs[it], lb[it]])
          class_wise_total[lb[it]] += 1
          df_scores = df_scores.append({'type':'id','label':int(lb[it]),'pred': int(p_idx),'score':max_score,'path':im_paths[it]}, ignore_index = True )
          id_total+=1
        else:
          max_score = float('-inf')
          for p in range(len(predictors)):
            score = predictors[p].score_samples(np.expand_dims(z_i[it], axis=0))[0]
            if score>max_score:
              max_score = score
          ood_list.append([max_score, im_paths[it], og_imgs[it]])
          od_hist.append(max_score)
          df_scores = df_scores.append({'type':'od','label':int(lb[it]),'pred': 'OOD','score':max_score,'path':im_paths[it]}, ignore_index = True )
          od_total+=1
  df_scores.to_csv('/home/abhiraj/DDP/CL/eval/COLORECTAL/clof_test_scores.csv', index=False)
  max_score = max([max(id_hist),max(od_hist)])
  min_score = min([min(id_hist),min(od_hist)])
  max_sum = 0
  thresh = 0
  th_arr = np.linspace(min_score,max_score+0.001,int((max_score-min_score)/0.0001)) ###############
  
  fpr95 = 0
  tpr95 = 2
  odac25 = 0
  odac50 = 0
  odac75 = 0
  odac90 = 0
  idac25 = 0
  idac50 = 0
  idac75 = 0
  idac90 = 0

  if thresh_ip is not None:
    id_correct = np.sum(np.array(id_hist) >= thresh_ip)
    od_correct = np.sum(np.array(od_hist) < thresh_ip)
    thresh = thresh_ip
  else:
    for i in range(len(th_arr[::-1])):
      id_corr = np.sum(np.array(id_hist) >= th_arr[i])
      id_wrng = np.sum(np.array(id_hist) < th_arr[i])
      od_corr = np.sum(np.array(od_hist) < th_arr[i])
      od_wrng = np.sum(np.array(od_hist) >= th_arr[i])
      tpr = od_corr/(od_corr+od_wrng)
      fpr = id_wrng/(id_corr+id_wrng)
      TPR_arr.append(tpr)
      FPR_arr.append(fpr)
      if abs(tpr - 0.95) < tpr95:
        fpr95 = fpr
        tpr95 = abs(tpr - 0.95)
      if i == int(0.25*(len(th_arr)-10)): ############## depends on 882
        odac25 = od_corr/(od_wrng+od_corr)
        idac25 = id_corr/(id_wrng+id_corr)
      if i == int(0.5*(len(th_arr)-10)): ############## depends on 882
        odac50 = od_corr/(od_wrng+od_corr)
        idac50 = id_corr/(id_wrng+id_corr)
      if i == int(0.75*(len(th_arr)-10)): ############## depends on 882
        odac75 = od_corr/(od_wrng+od_corr)
        idac75 = id_corr/(id_wrng+id_corr)
      if i == int(0.9*(len(th_arr)-10)): ############## depends on 882
        odac90 = od_corr/(od_wrng+od_corr)
        idac90 = id_corr/(id_wrng+id_corr)
      if ((id_corr+od_corr) >= max_sum):
        max_sum = id_corr+od_corr
        id_correct = id_corr
        od_correct = od_corr
        thresh = th_arr[i]

  auc_score = metrics.auc(FPR_arr, TPR_arr)

  for i in range(n_clusters):
    print(i,':',class_wise_correct[i]/class_wise_total[i])
    
  print('Threshold :',thresh)
  print('ID correct :',100*(id_correct/id_total))
  print('OD correct :',100*(od_correct/od_total))
  print('AUC :', auc_score)
  
  plt.figure(figsize = (5,5))
  plt.hist(id_hist, bins=100, density=True, cumulative=True, label='id', histtype='step', alpha=0.55, color='g')
  plt.hist(od_hist, bins=100, density=True, cumulative=True, label='id', histtype='step', alpha=0.55, color='r')
  plt.xlabel('score')
  plt.title('Cummulative density of ID and OD')
  plt.legend(loc = 'best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/COLORECTAL/cont_lof_OOD_cdf.png')

  plt.figure()
  plt.hist(id_hist, bins=100, fc=(0, 1, 0, 0.5), label='id', density=True)
  plt.hist(od_hist, bins=100, fc=(1, 0, 0, 0.5),label='od', density=True)
  plt.axvline(x=thresh,color='k')
  plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/COLORECTAL/cont_lof_OOD_score.png')

  plt.figure()
  plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  plt.title('AUC : '+ str(auc_score))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/COLORECTAL/cont_lof_AUROC.png')
  
  fp = open('/home/abhiraj/DDP/CL/eval/COLORECTAL/colo_scores_lists.txt','a')
  fp.write(test_type+' CONT_LOF \n') #######
  fp.write('id_hist: '+str(id_hist)+'\n')
  fp.write('od_hist: '+str(od_hist)+'\n')
  fp.close()
  
  fp = open('/home/abhiraj/DDP/CL/eval/COLORECTAL/colo_auroc_lists.txt','a')
  fp.write(test_type+' CONT_LOF \n') #######
  fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  fp.close()
  
  ood_list.sort()
    
  fp = open('colorectal_od_paths.txt','a')
  fp.write(test_type+' Cont-LOF \n') #######
  for dat in ood_list:
   fp.write(str(dat[0])+' : '+str(dat[1])+'\n')
  fp.close()
  
  count = 0
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0])) 
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_od_cont_lof_'+str(count)+'.png') #####################
    if count == 5:
      break
  
  ood_list.sort(reverse=True)
  
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_od_cont_lof_'+str(count)+'.png') ######################
    if count == 10:
      break
  
  correct_list.sort()
  
  acc_list = []
  fp = open('colorectal_im_paths.txt','a')
  fp.write(test_type+' Cont-LOF \n')
  for dat in correct_list:
   acc_list.append([dat[0], dat[1]])
   fp.write(str(dat[0])+' : '+str(dat[1])+' : '+dat[2]+'\n')
  fp.close()
  
  count = 0
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_cont_lof_'+str(count)+'.png')
    if count == 5:
      break
  
  correct_list.sort(reverse=True)
  
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_cont_lof_'+str(count)+'.png')
    if count == 10:
      break
  
  correct_arr = np.array(acc_list)
  acc_list = []
  data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  for i in range(len(data_left_list)):
    temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
    acc_list.append(np.sum(temp_arr)/len(temp_arr))
    
  fp = open('colo_acc_curves.txt', 'a')
  fp.write('CONT_lof: '+str(acc_list)+'\n')
  fp.close()  
  
  plt.figure()
  plt.plot(data_left_list, acc_list, '-', label='accuracy')
  plt.xlabel('% data ignored')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs %data left')
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/eval/COLORECTAL/cont_lof_accuracy.png')

################################################################################################################
#lof_predictor(full_model, trainloader, testloader, len(id_classes),'no_tf')
##########################################################################################################

#def estimate_density(model, loader, n_clusters):
full_model.eval()
df = pd.DataFrame(columns = ['label']+list(str(i) for i in range(128)))
with torch.no_grad():
  for i, sample in enumerate(trainloader, 0):
    x_i, x_j ,label, d_type  = sample['x_i'], sample['x_j'], sample['label'], sample['type']
    x_i = x_i.to(device)
    label = label.to(device)
    z_i, p_i = full_model(x_i)
    z_i = np.array(z_i.detach().cpu())*100
    lb = np.array(label.detach().cpu())
    for it in range(z_i.shape[0]):
      temp = {}
      temp['label'] = lb[it]
      #if d_type[it] == 'id':
      #  temp['label'] = lb[it]
      #else:
      #  temp['label'] = n_clusters
      for jt in range(z_i.shape[1]):
        temp[str(jt)] = z_i[it,jt]
      df = df.append(temp, ignore_index = True)

mean_vectors = df.groupby('label', as_index=False).mean()
del mean_vectors['label']
mean_vectors = mean_vectors
mean_vectors = mean_vectors.to_numpy()
std_mats = []
for i in range(len(id_classes)):
  sub_df = df[df['label'] == i]
  del sub_df['label']
  temp = sub_df.cov()
  std_mats.append(temp.to_numpy())

######################################################################################################

#mean_vectors_train, std_mats_train = estimate_density(full_model, trainloader, len(id_classes))

##########################################################################################################################

def calc_score(z_i, mean_vectors, std_mats):
  max_s = [[float('-inf'),0] for i in range(z_i.shape[0])]
  for i in range(len(id_classes)):
    mu = mean_vectors[i]
    E = std_mats[i]
    E_inv = np.linalg.inv(E)
    s = -np.diag((z_i-mu)@E_inv@(z_i-mu).T) - np.log((2*np.pi)**(z_i.shape[1])*np.linalg.det(E)) # check #
    for j in range(len(max_s)):
      if (max_s[j][0] < s[j]):
         max_s[j][0] = s[j]
         max_s[j][1] = i
  return max_s
############################################################################################################
def detect_ood(model, loader, mean_vectors, std_mats, n_clusters, test_type, thresh_ip=None):
  model.eval()
  
  class_wise_correct = [0 for i in range(n_clusters)]
  class_wise_total = [0 for i in range(n_clusters)]
  od_correct = 0
  od_total = 0
  id_correct = 0
  id_total = 0
  avg_ood = []
  avg_clean = []
  TPR_arr = []
  FPR_arr = []
  correct_list = []
  ood_list = []
  df_scores = pd.DataFrame(columns = ['type', 'label','pred','score','path'])
  with torch.no_grad():
    for i, sample in enumerate(loader, 0):
      x_i, x_j ,label, d_type, im_paths, og_imgs  = sample['x_i'], sample['x_j'], sample['label'], sample['type'], sample['path'], sample['og_img']
      x_i = x_i.to(device)
      label = label.to(device)
      z_i, p_i = model(x_i)
      z_i = np.array(z_i.detach().cpu())
      lb = np.array(label.detach().cpu())
      score = calc_score(z_i*100, mean_vectors, std_mats)
      
      for j in range(len(score)):
        if d_type[j] == 'od':
          avg_ood.append(score[j][0])
          ood_list.append([score[j][0], im_paths[j], og_imgs[j]])
          df_scores = df_scores.append({'type':'od','label':int(lb[j]),'pred': 'OOD','score':score[j][0],'path':im_paths[j]}, ignore_index = True )
          od_total+=1
        else:
          avg_clean.append(score[j][0])
          if lb[j] == score[j][1]:
            class_wise_correct[lb[j]] += 1
            correct_list.append([score[j][0],1, im_paths[j], og_imgs[j], lb[j]])
          else:
            correct_list.append([score[j][0],0, im_paths[j], og_imgs[j], lb[j]])
          class_wise_total[lb[j]] += 1
          df_scores = df_scores.append({'type':'id','label':int(lb[j]),'pred': int(score[j][1]),'score':score[j][0],'path':im_paths[j]}, ignore_index = True )
          id_total+=1
  
  df_scores.to_csv('/home/abhiraj/DDP/CL/eval/COLORECTAL/cgfit_test_scores.csv', index=False)
  '''
  max_score = max([max(avg_clean),max(avg_ood)])
  min_score = min([min(avg_clean),min(avg_ood)])
  max_sum = 0
  thresh = 0
  th_arr = np.linspace(min_score,max_score+1,int((max_score-min_score)/0.1))
  
  fpr95 = 0
  tpr95 = 2
  odac25 = 0
  odac50 = 0
  odac75 = 0
  odac90 = 0
  idac25 = 0
  idac50 = 0
  idac75 = 0
  idac90 = 0
  
  if thresh_ip is not None:
    id_correct = np.sum(np.array(avg_clean) >= thresh_ip)
    od_correct = np.sum(np.array(avg_ood) < thresh_ip)
    thresh = thresh_ip
  else:
    for i in range(len(th_arr)):
      id_corr = np.sum(np.array(avg_clean) >= th_arr[i])
      id_wrng = np.sum(np.array(avg_clean) < th_arr[i])
      od_corr = np.sum(np.array(avg_ood) < th_arr[i])
      od_wrng = np.sum(np.array(avg_ood) >= th_arr[i])
      tpr = od_corr/(od_corr+od_wrng)
      fpr = id_wrng/(id_corr+id_wrng)
      TPR_arr.append(tpr)
      FPR_arr.append(fpr)
      if abs(tpr - 0.95) < tpr95:
        fpr95 = fpr
        tpr95 = abs(tpr - 0.95)
      if i == int(0.25*(len(th_arr)-10)): ############## depends on 1047
        odac25 = od_corr/(od_wrng+od_corr)
        idac25 = id_corr/(id_wrng+id_corr)
      if i == int(0.5*(len(th_arr)-10)): ############## depends on 1047
        odac50 = od_corr/(od_wrng+od_corr)
        idac50 = id_corr/(id_wrng+id_corr)
      if i == int(0.75*(len(th_arr)-10)): ############## depends on 1047
        odac75 = od_corr/(od_wrng+od_corr)
        idac75 = id_corr/(id_wrng+id_corr)
      if i == int(0.9*(len(th_arr)-10)): ############## depends on 1047
        odac90 = od_corr/(od_wrng+od_corr)
        idac90 = id_corr/(id_wrng+id_corr)
      if ((id_corr+od_corr) >= max_sum):
        max_sum = id_corr+od_corr
        id_correct = id_corr
        od_correct = od_corr
        thresh = th_arr[i]

  auc_score = metrics.auc(FPR_arr, TPR_arr)
  
  for i in range(n_clusters):
    print(i,':',class_wise_correct[i]/class_wise_total[i])

#  print("clean avg :",np.mean(np.array(avg_clean)))
#  print("clean std :",np.std(np.array(avg_clean)))
#  print("ood avg :",np.mean(np.array(avg_ood)))
#  print("ood std :",np.std(np.array(avg_ood)))
  print('Threshold :',thresh)
  print('ID correct :',100*(id_correct/id_total))
  print('OD correct :',100*(od_correct/od_total))
  print('AUC :', auc_score)
  
  plt.figure(figsize = (5,5))
  plt.hist(avg_clean, bins=100, density=True, cumulative=True, label='id', histtype='step', alpha=0.55, color='g')
  plt.hist(avg_ood, bins=100, density=True, cumulative=True, label='id', histtype='step', alpha=0.55, color='r')
  plt.xlabel('score')
  plt.title('Cummulative density of ID and OD')
  plt.legend(loc = 'best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/COLORECTAL/cont_gfit_OOD_cdf.png')

  plt.figure()
  plt.hist(avg_clean, bins=100, fc=(0, 1, 0, 0.5), label='clean', density=True)
  plt.hist(avg_ood, bins=100, fc=(1, 0, 0, 0.5),label='ood', density=True)
  plt.axvline(x=thresh,color='k')
  plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/COLORECTAL/cont_gfit_OOD_score.png')

  plt.figure()
  plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  plt.title('ROC curve, AUC= '+str(auc_score))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/COLORECTAL/cont_gfit_AUROC.png')
  
  fp = open('/home/abhiraj/DDP/CL/eval/COLORECTAL/colo_scores_lists.txt','a')
  fp.write(test_type+' CONT_GFIT \n') #######
  fp.write('id_hist: '+str(avg_clean)+'\n')
  fp.write('od_hist: '+str(avg_ood)+'\n')
  fp.close()
  
  fp = open('/home/abhiraj/DDP/CL/eval/COLORECTAL/colo_auroc_lists.txt','a')
  fp.write(test_type+' CONT_GFIT \n') #######
  fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  fp.close()
  
  ood_list.sort()
    
  fp = open('colorectal_od_paths.txt','a')
  fp.write(test_type+' Cont-Gfit \n') #######
  for dat in ood_list:
   fp.write(str(dat[0])+' : '+str(dat[1])+'\n')
  fp.close()
  
  count = 0
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0])) 
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_od_cont_gift_'+str(count)+'.png') #####################
    if count == 5:
      break
  
  ood_list.sort(reverse=True)
  
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_od_cont_gfit_'+str(count)+'.png') ######################
    if count == 10:
      break
  
  correct_list.sort()
  
  acc_list = []
  fp = open('colorectal_im_paths.txt','a')
  fp.write(test_type+' Cont-Gfit \n')
  for dat in correct_list:
   acc_list.append([dat[0], dat[1]])
   fp.write(str(dat[0])+' : '+str(dat[1])+' : '+dat[2]+'\n')
  fp.close()
  
  count = 0
  for dat in correct_list:
   count +=1
   plt.figure()
   plt.imshow(np.asarray(dat[3]))
   plt.title(str(dat[4])+ ' : '+ str(dat[0]))
   plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_cont_gfit_'+str(count)+'.png')
   if count == 5:
     break
  
  correct_list.sort(reverse=True)
  
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/COLORECTAL/'+test_type+'_cont_gfit_'+str(count)+'.png')
    if count == 10:
      break
  
  correct_arr = np.array(acc_list)
  acc_list = []
  data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  for i in range(len(data_left_list)):
    temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
    acc_list.append(np.sum(temp_arr)/len(temp_arr))
  
  fp = open('colo_acc_curves.txt', 'a')
  fp.write('CONT_gfit: '+str(acc_list)+'\n')
  fp.close()  
  
  plt.figure()
  plt.plot(data_left_list, acc_list, '-', label='accuracy')
  plt.xlabel('% data ignored')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs %data left')
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/eval/COLORECTAL/cont_gfit_accuracy.png')
  '''
##########################################################################################################
detect_ood(full_model, testloader, mean_vectors, std_mats,len(id_classes), 'no_tf')
##########################################################################################################

def generate_gradcam_vis(model, loader, mean, std):
  model.eval()
  inv_norm = transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],std= [1/s for s in std])
  target_layer = model.pre_model.resnet.layer4[-1]
  cam = GradCAM(model=model, target_layer=target_layer)
  for i, sample in enumerate(loader, 0):
    img, label, d_type, im_paths, og_imgs  = sample['x_i'], sample['label'], sample['type'], sample['path'], sample['og_img']
    img = img.to(device)
    for j in range(len(label)):
      input_tensor = img[j].unsqueeze(0)
      rgb_img = inv_norm(img[j]).float().permute(1, 2, 0).cpu().numpy()
      target_category = label[j].item()
      grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
      grayscale_cam = grayscale_cam[0, :]
      visualization = show_cam_on_image(rgb_img, grayscale_cam)
      
      fig = plt.figure()
      ax1 = fig.add_subplot(1,2,1)
      ax1.imshow(rgb_img)
      ax2 = fig.add_subplot(1,2,2)
      ax2.imshow(visualization)
      bach_class = im_paths[j].split('/')[-2]
      fig.suptitle(d_type[j]+'_'+id_classes[target_category]+'_'+bach_class, fontsize=12)
      fig.savefig('/home/abhiraj/DDP/CL/eval/COLORECTAL/GRADCAM/CONT/cont_gradcam_'+d_type[j]+'_'+id_classes[target_category]+'_'+bach_class+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'.png')
      
generate_gradcam_vis(full_model, testloader, mean, std)

print("FINISHED")
