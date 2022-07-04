import os
from glob import glob
from tqdm import tqdm

import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# sklearn libraries
from scipy import stats
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
import seaborn as sns

seed = 1
shuffle_seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

base_dir = "/home/abhiraj/DDP/CL/data/HAM"
all_image_path = glob(os.path.join(base_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

df_original = pd.read_csv(os.path.join(base_dir, 'HAM10000_metadata.csv'))
df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs
    

# norm_mean,norm_std = compute_img_mean_std(all_image_path)
normMean = [0.7630358, 0.54564357, 0.5700475]
normStd = [0.14092763, 0.15261263, 0.16997081]

df_undup = df_original.groupby('lesion_id').count()
df_undup = df_undup[df_undup['image_id'] == 1]
df_undup.reset_index(inplace=True)

def get_duplicates(x):
    unique_list = list(df_undup['lesion_id'])
    if x in unique_list:
        return 'unduplicated'
    else:
        return 'duplicated'

# create a new colum that is a copy of the lesion_id column
df_original['duplicates'] = df_original['lesion_id']
# apply the function to this new column
df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)


df_undup = df_original[df_original['duplicates'] == 'unduplicated']
y = df_undup['cell_type_idx']
_, df_val = train_test_split(df_undup, test_size=0.2, random_state=shuffle_seed, stratify=y)
print(df_val['cell_type'].value_counts())

# This set will be df_original excluding all rows that are in the val set
# This function identifies if an image is part of the train or val set.
def get_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val['image_id'])
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'

# identify train and val rows
# create a new colum that is a copy of the image_id column
df_original['train_or_val'] = df_original['image_id']
# apply the function to this new column
df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
# filter out train rows
df_train_nv = df_original[(df_original['train_or_val'] == 'train') & (df_original['dx'] == 'nv')] ##############
df_train_mel = df_original[(df_original['train_or_val'] == 'train') & (df_original['dx'] == 'mel')] ##############
df_train_bkl = df_original[(df_original['train_or_val'] == 'train') & (df_original['dx'] == 'bkl')] ##############
df_train_bcc = df_original[(df_original['train_or_val'] == 'train') & (df_original['dx'] == 'bcc')] ##############
df_train_akiec = df_original[(df_original['train_or_val'] == 'train') & (df_original['dx'] == 'akiec')] ##############
df_train_vasc = df_original[(df_original['train_or_val'] == 'train') & (df_original['dx'] == 'vasc')] ##############
df_train_df = df_original[(df_original['train_or_val'] == 'train') & (df_original['dx'] == 'df')] ##############

print('MEL train samples :',len(df_train_mel))
print('NV train samples :',len(df_train_nv))
print('BKL train samples :',len(df_train_bkl))

#df_train = pd.concat([df_train_mel, df_train_nv, df_train_bkl]) ##############
df_all = pd.concat([df_train_mel, df_train_nv, df_train_bkl, df_train_bcc, df_train_akiec, df_train_vasc, df_train_df]) ##############
df_train = pd.concat([df_train_mel, df_train_nv, df_train_bkl, df_train_bcc, df_train_akiec, df_train_vasc, df_train_df])

data_aug_rate = [15,10,5,50,0,40,5]#[0,0,5,0,0,0,5]  #[15,10,5,50,0,40,5] ##############
for i in range(7):
    if data_aug_rate[i]:
        df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)

data_aug_rate = [15,10,5,50,0,40,5] ##############
for i in range(7):
    if data_aug_rate[i]:
        df_all=df_all.append([df_all.loc[df_all['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)        
        
df_train = df_train.reset_index()
df_val = df_val.reset_index()
df_all = df_all.reset_index()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CustomModel(nn.Module):
  def __init__(self):
    super(CustomModel, self).__init__()
    self.resnet = models.resnet18(pretrained=True)
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

    return out, x

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

input_size = 224
train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(normMean, normStd)])
# define the transformation of the val images.
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(normMean, normStd)])
                                    
                                    
def map_indices_train(cell_type_idx):
      map = [0,1,2,3,4,5,6]#[0,0,0,0,1,0,2] ##############
      d_type = ''
      if cell_type_idx in [0,1,3,5]: ##############
        d_type = 'id' ####
      else:
        d_type = 'id'
      return map[cell_type_idx]

def map_indices_val(cell_type_idx):
      map = [0,1,2,3,4,5,6]#[0,0,0,0,1,0,2] ##############
      d_type = ''
      if cell_type_idx in [0,1,3,5]: ##############
        d_type = 'id' ####
      else:
        d_type = 'id'
      return [map[cell_type_idx], d_type]

def map_indices_all(cell_type_idx):
      map = [0,1,2,3,4,5,6] ##############
      d_type = ''
      if cell_type_idx in [0,1,3,5]: ##############
        d_type = 'id' ####
      else:
        d_type = 'id'
      return [map[cell_type_idx], d_type]

class CustomTrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(map_indices_val(int(self.df['cell_type_idx'][index]))[0])
        d_type = map_indices_val(int(self.df['cell_type_idx'][index]))[1]

        if self.transform:
            X_i = self.transform(X)
            X_j = self.transform(X)
        return X_i,X_j, y, d_type

class CustomValDataset(Dataset):
    def __init__(self, df, mean, std, transform=None):
        self.df = df
        self.transform = transform
        self.inv_norm = transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],std= [1/s for s in std])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        im_path = self.df['path'][index]
        X = Image.open(self.df['path'][index])
        y = torch.tensor(map_indices_val(int(self.df['cell_type_idx'][index]))[0])
        d_type = map_indices_val(int(self.df['cell_type_idx'][index]))[1]

        if self.transform:
            X_i = self.transform(X)
            X_j = self.transform(X)
            
        im_arr = self.inv_norm(X_i)
        im_arr = np.array(im_arr).transpose(1,2,0)
        
        return X_i,X_j, y, d_type, im_path, np.asarray(im_arr)

class CustomAllDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(map_indices_all(int(self.df['cell_type_idx'][index]))[0])
        d_type = map_indices_all(int(self.df['cell_type_idx'][index]))[1]

        if self.transform:
            X_i = self.transform(X)
            X_j = self.transform(X)
        return X_i,X_j, y, d_type

# Define the training set using the table train_df and using our defined transitions (train_transform)
training_set = CustomTrainDataset(df_train, transform=train_transform)
train_loader = DataLoader(training_set, batch_size=64, shuffle=True, num_workers=1)
# Same for the validation set:
validation_set = CustomValDataset(df_val,normMean,normStd,transform=val_transform)
val_loader = DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=1)
# Same for the complete set:
all_set = CustomAllDataset(df_all, transform=train_transform)
all_loader = DataLoader(all_set, batch_size=128, shuffle=True, num_workers=1)

# this function is used during training process, to calculation the loss and accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def tsne_visualize(model,loader, perplexity,learning_rate, n_iter, random_state, file_name):
  model.eval()
  df = pd.DataFrame(columns = list(str(i) for i in range(3)))
  df['label'] = None
  for i, sample in enumerate(loader, 0):
    x_i,_, label, d_type, _, _  = sample
    x_i = x_i.to(device)
    label = label.to(device)
    z_i, _ = model(x_i)
    z_i = np.array(z_i.detach().cpu())
    lb = np.array(label.detach().cpu())
    for it in range(z_i.shape[0]):
      temp = {}
      #temp['label'] = lb[it]
      if (d_type[it] == 'od'):
        temp['label'] = 4
      else:
        temp['label'] = lb[it]
      for jt in range(z_i.shape[1]):
        temp[str(jt)] = z_i[it,jt]
      df = df.append(temp, ignore_index = True)

  m = TSNE(perplexity = perplexity, learning_rate = learning_rate, n_iter = n_iter, random_state = random_state)
  df_feat = df[list(str(i) for i in range(3))]
  tsne_data = m.fit_transform(df_feat)
  tsne_df = pd.DataFrame(tsne_data, columns = ['x', 'y'])
  tsne_df['label'] = df['label'].tolist()
  sns.FacetGrid(tsne_df,hue='label', size = 6).map(plt.scatter, 'x', 'y')
  plt.legend(loc = 'best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/'+file_name+'.png')

def pretrain(model,loader,optimizer, epoch):
  model.train()
  running_loss = 0.0
  for i, data in enumerate(loader):
    x_i, x_j, label, d_type = data
    x_i = x_i.to(device)
    x_j = x_j.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    z_i,_ = model(x_i)
    z_j,_ = model(x_j)
    z_cat = torch.cat([z_i.unsqueeze(1), z_j.unsqueeze(1)], dim=1)
 
    loss = SupConLoss(z_cat)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if i % 100 == 99: 
      print('[%d, %5d / %5d] loss: %.3f' % (epoch, i + 1, len(loader), running_loss / 100))
  return (running_loss/len(all_set))


total_loss_train, total_acc_train = [],[]
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        images_i, images_j, labels, d_type = data
        N = images_i.size(0)
        # print('image shape:',images.shape, 'label shape',labels.shape)
        images_i = Variable(images_i).to(device)
        images_j = Variable(images_j).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        feat_i, outputs_i = model(images_i)
        feat_j, outputs_j = model(images_j)
        feat_cat = torch.cat([feat_i.unsqueeze(1), feat_j.unsqueeze(1)], dim=1)

        loss_cl = SupConLoss(feat_cat, labels)                                                               #########################################################
        #loss_cl = SupConLoss(feat_cat)
        loss_ce_i = criterion(outputs_i, labels)
        loss_ce_j = criterion(outputs_j, labels)
        loss = 0.5*(loss_ce_i+loss_ce_j) + 0.1*loss_cl ##########

        loss.backward()
        optimizer.step()
        prediction = outputs_i.max(1, keepdim=True)[1]
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % 100 == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
    return train_loss.avg, train_acc.avg
    

def validate(val_loader, model, criterion, optimizer, epoch):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, _, labels, d_type, _, _ = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            _, outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            matches = prediction.eq(labels.view_as(prediction))
            match_sum = 0
            for idx in range(len(matches)):
              if d_type[idx] == 'id' and matches[idx][0] == True:
                match_sum += 1 
            val_acc.update(match_sum/N)

            val_loss.update(criterion(outputs, labels).item())

    print('\n ------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
    print(' ------------------------------------------------------------')
    return val_loss.avg, val_acc.avg
    
try:
  pre_model.load_state_dict(torch.load("/home/abhiraj/DDP/CL/models/HistoSimCLR/cont_ham_pretrain100.pt"))
  print('Model Loaded ... ')
except:
  pass

pre_model.to(device)
optimizer = optim.Adam(pre_model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.1)

#tsne_visualize(pre_model,val_loader, 30,200, 1000, 0, 'cont_ham_pretrain_before_tsne')

do_pretrain = False #########
if do_pretrain:
  pretrain_epoch = 100
  print('Pre-Training')
  for ep in range(1,pretrain_epoch+1):
    cont_loss = pretrain(pre_model,all_loader, optimizer, ep)
    scheduler.step(cont_loss)
    print('Epoch:', ep,'/',pretrain_epoch,':',cont_loss)

  tsne_visualize(pre_model,val_loader, 30,200, 1000, 0, 'cont_ham_pretrain_tsne')

  torch.save(pre_model.state_dict(), "/home/abhiraj/DDP/CL/models/HistoSimCLR/cont_ham_pretrain100.pt")

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
    out1, _ = self.pre_model(x)
    out = self.relu(self.bn1(self.lin1(out1)))
    out = self.relu(self.bn2(self.lin2(out)))
    out = self.relu(self.bn3(self.lin3(out)))
    out2 = self.linout(out)

    return out1, out2
    
model = LinearClassifierModel(pre_model, 7)
    
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)
'''
epoch_num = 10
best_val_acc = 0
total_loss_val, total_acc_val = [],[]
for epoch in tqdm(range(1, epoch_num+1)):
    loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch)
    loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch)
    total_loss_val.append(loss_val)
    total_acc_val.append(acc_val)
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        print('*****************************************************')
        print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
        print('*****************************************************')
        torch.save(model.state_dict(), "/home/abhiraj/DDP/CL/models/HistoSimCLR/cont_ham.pt")
'''        
  
model.load_state_dict(torch.load("/home/abhiraj/DDP/CL/models/HistoSimCLR/cont_ham.pt"))
      
#tsne_visualize(model,val_loader, 30,200, 1000, 0, 'cont_ham_pretrain_after_tsne')
'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/cont_pretrain_confusion.png')
    
model.eval()
y_label = []
y_predict = []
with torch.no_grad():
    for i, data in enumerate(val_loader):
        images, _, labels, d_type, _, _ = data
        N = images.size(0)
        images = Variable(images).to(device)
        _, outputs = model(images)
        prediction = outputs.max(1, keepdim=True)[1]
        y_label.extend(labels.cpu().numpy())
        y_predict.extend(np.squeeze(prediction.cpu().numpy().T))
'''
# compute the confusion matrix
#confusion_mtx = confusion_matrix(y_label, y_predict)
# plot the confusion matrix
plot_labels = ['bkl', 'nv','mel']  #['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
# plot_confusion_matrix(confusion_mtx, plot_labels)
    
import itertools
#plot_confusion_matrix(confusion_mtx, plot_labels)

#report = classification_report(y_label, y_predict, target_names=plot_labels)
#print(report)


def kmeans_predictor(model, train_loader, test_loader, n_clusters, test_type, thresh_ip = None):
  model.eval()
  df = pd.DataFrame(columns = ['label']+list(str(i) for i in range(128)))
  with torch.no_grad():
    for i, sample in enumerate(train_loader, 0):
      x_i, x_j ,label, d_type  = sample
      x_i = x_i.to(device)
      label = label.to(device)
      z_i, p_i = model(x_i)
      z_i = np.array(z_i.detach().cpu())
      lb = np.array(label.detach().cpu())
      for it in range(z_i.shape[0]):
        temp = {}
        temp['label'] = lb[it]
        # if d_type[it] == 'id':
        #   temp['label'] = lb[it]
        # else:
        #   temp['label'] = n_clusters
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
      x_i, x_j, label, d_type, im_paths, og_imgs = sample
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
          
  df_scores.to_csv('/home/abhiraj/DDP/CL/eval/HAM/ckmeans_test_scores.csv', index=False)
  max_score = max([max(id_hist),max(od_hist)])
  min_score = min([min(id_hist),min(od_hist)])
  max_sum = 0
  thresh = 0
  
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
  
  th_arr = np.linspace(min_score,max_score+0.01,int((max_score-min_score)/0.001)) ##############
  
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
      if i == int(0.25*(len(th_arr)-10)): ############## depends on 744
        odac25 = od_corr/(od_wrng+od_corr)
        idac25 = id_corr/(id_wrng+id_corr)
      if i == int(0.5*(len(th_arr)-10)): ############## depends on 744
        odac50 = od_corr/(od_wrng+od_corr)
        idac50 = id_corr/(id_wrng+id_corr)
      if i == int(0.75*(len(th_arr)-10)): ############## depends on 744
        odac75 = od_corr/(od_wrng+od_corr)
        idac75 = id_corr/(id_wrng+id_corr)
      if i == int(0.9*(len(th_arr)-10)): ############## depends on 744
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
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/cont_pretrain_kmeans_OOD_cdf.png')

  plt.figure()
  plt.hist(id_hist, bins=100, fc=(0, 1, 0, 0.5), label='id', density=True)
  plt.hist(od_hist, bins=100, fc=(1, 0, 0, 0.5),label='od', density=True)
  plt.axvline(x=thresh,color='k')
  plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/cont_pretrain_kmeans_OOD_score.png')

  plt.figure()
  plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.title('AUC : '+ str(auc_score))
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/cont_pretrain_kmeans_AUROC.png')
  
  fp = open('/home/abhiraj/DDP/CL/eval/HAM/ham_scores_lists.txt','a')
  fp.write(test_type+' CONT_KMEANS \n') #######
  fp.write('id_hist: '+str(id_hist)+'\n')
  fp.write('od_hist: '+str(od_hist)+'\n')
  fp.close()
  
  fp = open('/home/abhiraj/DDP/CL/eval/HAM/ham_auroc_lists.txt','a')
  fp.write(test_type+' CONT_KMEANS \n') #######
  fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  fp.close()
  
  ood_list.sort()
    
  fp = open('ham_od_paths.txt','a')
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
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_od_cont_kmeans_'+str(count)+'.png') #####################
    if count == 5:
      break
  
  ood_list.sort(reverse=True)
  
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_od_cont_kmeans_'+str(count)+'.png') ######################
    if count == 10:
      break
  
  correct_list.sort()
  
  acc_list = []
  fp = open('ham_im_paths.txt','a')
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
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_cont_kmeans_'+str(count)+'.png')
    if count == 5:
      break
  
  correct_list.sort(reverse=True)
  
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_cont_kmeans_'+str(count)+'.png')
    if count == 10:
      break
  
  correct_arr = np.array(acc_list)
  acc_list = []
  data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  for i in range(len(data_left_list)):
    temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
    acc_list.append(np.sum(temp_arr)/len(temp_arr))
    
  fp = open('ham_acc_curves.txt', 'a')
  fp.write('CONT_kmeans: '+str(acc_list)+'\n')
  fp.close()  
    
  plt.figure()
  plt.plot(data_left_list, acc_list, '-', label='accuracy')
  plt.xlabel('% data ignored')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs %data left')
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/eval/HAM/cont_kmeans_accuracy.png') 
  
#kmeans_predictor(model, train_loader, val_loader, 3, 'no_tf')

def lof_predictor(model, train_loader, test_loader, n_clusters, test_type, thresh_ip = None):
  model.eval()
  df = pd.DataFrame(columns = ['label']+list(str(i) for i in range(128)))
  with torch.no_grad():
    for i, sample in enumerate(train_loader, 0):
      x_i, x_j ,label, d_type  = sample
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
  ood_list =[]
  df_scores = pd.DataFrame(columns = ['type', 'label','pred','score','path'])
  with torch.no_grad():
    for i, sample in enumerate(test_loader, 0):
      x_i, x_j, label, d_type, im_paths, og_imgs = sample
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
  df_scores.to_csv('/home/abhiraj/DDP/CL/eval/HAM/clof_test_scores.csv', index=False)
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
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/cont_pretrain_lof_OOD_cdf.png')

  plt.figure()
  plt.hist(id_hist, bins=100, fc=(0, 1, 0, 0.5), label='id', density=True)
  plt.hist(od_hist, bins=100, fc=(1, 0, 0, 0.5),label='od', density=True)
  plt.axvline(x=thresh,color='k')
  plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/cont_pretrain_lof_OOD_score.png')

  plt.figure()
  plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  plt.title('AUC : '+ str(auc_score))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/cont_pretrain_lof_AUROC.png')
  
  fp = open('/home/abhiraj/DDP/CL/eval/HAM/ham_scores_lists.txt','a')
  fp.write(test_type+' CONT_LOF \n') #######
  fp.write('id_hist: '+str(id_hist)+'\n')
  fp.write('od_hist: '+str(od_hist)+'\n')
  fp.close()
  
  fp = open('/home/abhiraj/DDP/CL/eval/HAM/ham_auroc_lists.txt','a')
  fp.write(test_type+' CONT_LOF \n') #######
  fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  fp.close()
  
  ood_list.sort()
    
  fp = open('ham_od_paths.txt','a')
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
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_od_cont_lof_'+str(count)+'.png') #####################
    if count == 5:
      break
  
  ood_list.sort(reverse=True)
  
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_od_cont_lof_'+str(count)+'.png') ######################
    if count == 10:
      break
  
  correct_list.sort()
  
  acc_list = []
  fp = open('ham_im_paths.txt','a')
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
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_cont_lof_'+str(count)+'.png')
    if count == 5:
      break
  
  correct_list.sort(reverse=True)
  
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_cont_lof_'+str(count)+'.png')
    if count == 10:
      break
  
  correct_arr = np.array(acc_list)
  acc_list = []
  data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  for i in range(len(data_left_list)):
    temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
    acc_list.append(np.sum(temp_arr)/len(temp_arr))
    
  fp = open('ham_acc_curves.txt', 'a')
  fp.write('CONT_lof: '+str(acc_list)+'\n')
  fp.close()  
  
  plt.figure()
  plt.plot(data_left_list, acc_list, '-', label='accuracy')
  plt.xlabel('% data ignored')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs %data left')
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/eval/HAM/cont_lof_accuracy.png')
  
#lof_predictor(model, train_loader, val_loader, 3, 'no_tf')

def estimate_density(model, loader, n_clusters):
  model.eval()
  df = pd.DataFrame(columns = ['label']+list(str(i) for i in range(128)))
  with torch.no_grad():
    for i, sample in enumerate(loader, 0):
      x_i, x_j ,label, d_type  = sample
      x_i = x_i.to(device)
      label = label.to(device)
      z_i, p_i = model(x_i)
      z_i = np.array(z_i.detach().cpu())*100 ############
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
  
  #del df['label']
  #kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=1000).fit(df.to_numpy())
  #df['label'] = kmeans.labels_
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
    
  return mean_vectors, std_mats
  
mean_vectors_train, std_mats_train = estimate_density(model, train_loader, 7)

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
      x_i, x_j, label, d_type, im_paths, og_imgs = sample
      x_i = x_i.to(device)
      label = label.to(device)
      z_i, p_i = model(x_i)
      z_i = np.array(z_i.detach().cpu())
      lb = np.array(label.detach().cpu())
      score = calc_score(z_i*100, mean_vectors, std_mats, n_clusters) ############
      
      for j in range(len(score)):
        if d_type[j] == 'od':
          avg_ood.append(score[j][0])
          ood_list.append([score[j][0], im_paths[j], og_imgs[j]])
          df_scores = df_scores.append({'type':'od','label':int(lb[j]),'pred': int(score[j][1]),'score':score[j][0],'path':im_paths[j]}, ignore_index = True )
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
  
  df_scores.to_csv('/home/abhiraj/DDP/CL/eval/HAM/cgfit_test_scores.csv', index=False)
  '''
  max_score = max([max(avg_clean),max(avg_ood)])
  min_score = min([min(avg_clean),min(avg_ood)])
  max_sum = 0
  thresh = 0
  th_arr = np.linspace(min_score,max_score+1,int((max_score-min_score)/0.1)) ###########
  
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
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/cont_pretrain_gfit_OOD_cdf.png')

  plt.figure()
  plt.hist(avg_clean, bins=100, fc=(0, 1, 0, 0.5), label='clean', density=True)
  plt.hist(avg_ood, bins=100, fc=(1, 0, 0, 0.5),label='ood', density=True)
  plt.axvline(x=thresh,color='k')
  plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/cont_pretrain_gfit_OOD_score.png')

  plt.figure()
  plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  plt.title('AUC : '+ str(auc_score))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/cont_pretrain_gfit_AUROC.png')
  
  fp = open('/home/abhiraj/DDP/CL/eval/HAM/ham_scores_lists.txt','a')
  fp.write(test_type+' CONT_GFIT \n') #######
  fp.write('id_hist: '+str(avg_clean)+'\n')
  fp.write('od_hist: '+str(avg_ood)+'\n')
  fp.close()
  
  fp = open('/home/abhiraj/DDP/CL/eval/HAM/ham_auroc_lists.txt','a')
  fp.write(test_type+' CONT_GFIT \n') #######
  fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  fp.close()
  
  ood_list.sort()
    
  fp = open('ham_od_paths.txt','a')
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
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_od_cont_gift_'+str(count)+'.png') #####################
    if count == 5:
      break
  
  ood_list.sort(reverse=True)
  
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_od_cont_gfit_'+str(count)+'.png') ######################
    if count == 10:
      break
  
  correct_list.sort()
  
  acc_list = []
  fp = open('ham_im_paths.txt','a')
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
   plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_cont_gfit_'+str(count)+'.png')
   if count == 5:
     break
  
  correct_list.sort(reverse=True)
  
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_cont_gfit_'+str(count)+'.png')
    if count == 10:
      break
  
  correct_arr = np.array(acc_list)
  acc_list = []
  data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  for i in range(len(data_left_list)):
    temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
    acc_list.append(np.sum(temp_arr)/len(temp_arr))
  
  fp = open('ham_acc_curves.txt', 'a')
  fp.write('CONT_gfit: '+str(acc_list)+'\n')
  fp.close()  
  
  plt.figure()
  plt.plot(data_left_list, acc_list, '-', label='accuracy')
  plt.xlabel('% data ignored')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs %data left')
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/eval/HAM/cont_gfit_accuracy.png')
  '''
detect_ood(model, val_loader, mean_vectors_train, std_mats_train, 7, 'no_tf')

def generate_gradcam_vis(model, loader, mean, std):
  model.eval()
  inv_norm = transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],std= [1/s for s in std])
  target_layer = model.pre_model.resnet.layer4[-1]
  cam = GradCAM(model=model, target_layer=target_layer)
  id_samples = 0
  od_samples = 0
  for i, dat in enumerate(loader, 0):
    img, _, label, d_type, im_paths, og_imgs = dat
    img = img.to(device)
    for j in range(len(label)):
      input_tensor = img[j].unsqueeze(0)
      rgb_img = inv_norm(img[j]).float().permute(1, 2, 0).cpu().numpy()
      rgb_img = rgb_img/np.max(rgb_img)
      target_category = label[j].item()
      grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
      grayscale_cam = grayscale_cam[0, :]
      visualization = show_cam_on_image(rgb_img, grayscale_cam)
      
      fig = plt.figure()
      ax1 = fig.add_subplot(1,2,1)
      ax1.imshow(rgb_img)
      ax2 = fig.add_subplot(1,2,2)
      ax2.imshow(visualization)
      file_name = im_paths[j].split('/')[-1][:-4]
      fig.suptitle(d_type[j]+'_'+str(target_category)+'_'+str(label[j].item()), fontsize=12)
      fig.savefig('/home/abhiraj/DDP/CL/eval/HAM/GRADCAM/CONT/cont_gradcam_'+d_type[j]+'_'+str(target_category)+'_'+str(label[j].item())+'_'+file_name+'_'+str(j)+'.png')
      
#generate_gradcam_vis(model, val_loader, normMean, normStd)