import os
from glob import glob
from tqdm import tqdm

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

seed = 4110
shuffle_seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#####################
base_dir = "/home/Drive/abhiraj/data/mias/"

#info_f = open(os.path.join(base_dir,'info.txt'))
#df_all = pd.DataFrame(columns = ['path', 'bg_tissue','class', 'severity', 'set'])
#for l in info_f.readlines()[1:]:
#  s_l = l.split(' ')
#  if '144' not in s_l[0]:
#    if 'NORM' in s_l[2]:
#      rp = random.random()
#      if rp>0.3:
#        df_all = df_all.append({'path':os.path.join("/home/Drive/abhiraj/data/mias/images/",s_l[0]+".pgm"), 'bg_tissue':s_l[1], 'class':'NORM', 'severity':"N", 'set':'train'}, ignore_index = True)
#      else:
#        df_all = df_all.append({'path':os.path.join("/home/Drive/abhiraj/data/mias/images/",s_l[0]+".pgm"), 'bg_tissue':s_l[1], 'class':'NORM', 'severity':"N", 'set':'test'}, ignore_index = True)
#    elif 'B' in s_l[3]:
#      rp = random.random()
#      if rp>0.3:
#        df_all = df_all.append({'path':os.path.join("/home/Drive/abhiraj/data/mias/images/",s_l[0]+".pgm"), 'bg_tissue':s_l[1], 'class':s_l[2], 'severity':s_l[3][0], 'set':'train'}, ignore_index = True)
#      else:
#        df_all = df_all.append({'path':os.path.join("/home/Drive/abhiraj/data/mias/images/",s_l[0]+".pgm"), 'bg_tissue':s_l[1], 'class':s_l[2], 'severity':s_l[3][0], 'set':'test'}, ignore_index = True)
#    elif 'M' in s_l[3]:
#      rp = random.random()
#      if rp>0.3:
#        df_all = df_all.append({'path':os.path.join("/home/Drive/abhiraj/data/mias/images/",s_l[0]+".pgm"), 'bg_tissue':s_l[1], 'class':s_l[2], 'severity':s_l[3][0], 'set':'train'}, ignore_index = True)
#      else:
#        df_all = df_all.append({'path':os.path.join("/home/Drive/abhiraj/data/mias/images/",s_l[0]+".pgm"), 'bg_tissue':s_l[1], 'class':s_l[2], 'severity':s_l[3][0], 'set':'test'}, ignore_index = True)
#df_all.drop_duplicates('path', keep="last", inplace=True)
#df_all.to_csv('all_data.csv', index=False)

df_all = pd.read_csv('all_data.csv')
print(df_all['severity'].value_counts())


df_inbreast = pd.DataFrame(columns = ['path', 'bg_tissue','class', 'severity', 'set','type', 'class_id'])
inbreast_dir_path = '/home/Drive/INBREAST/ALL-IMGS'
ib_csv_path = '/home/Drive/INBREAST/INbreast.csv'
dcm_files = os.listdir(inbreast_dir_path)
df = pd.read_csv(ib_csv_path)
for i in range(6):
  for j in range(len(df)):
    if (str(i+1) in df['Bi-Rads'][j]):
      file_name = df['File Name'][j]
      for k in range(len(dcm_files)):
        if str(file_name) in dcm_files[k]:
          df_inbreast = df_inbreast.append({'path':os.path.join(inbreast_dir_path,dcm_files[k]), 'bg_tissue':'inbreast', 'class':str(i+1), 'severity':str(i+1), 'set':'test', 'type':'od', 'class_id':10+i+1}, ignore_index = True)
#for i in range(6):
#  sub_dir_path = os.path.join(inbreast_dir_path,str(i+1))
#  file_paths = os.listdir(sub_dir_path)
#  for f in file_paths:
#    df_inbreast = df_inbreast.append({'path':os.path.join(sub_dir_path,f), 'bg_tissue':'inbreast', 'class':str(i+1), 'severity':str(i+1), 'set':'test', 'type':'od', 'class_id':10+i+1}, ignore_index = True)

def compute_img_mean_std_mias(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """
    img_h, img_w = 512, 512
    means, stdevs = [], []
    sum_img = np.zeros((3,img_h,img_w))

    for i in range(len(image_paths)):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        img = img.astype('float64')
        img = np.reshape(img, (3, img_h ,img_w))
        sum_img += (img/255)
      
    sum_img = sum_img/len(image_paths)

    for i in range(3):
        pixels = sum_img[i, :, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs

def compute_img_mean_std_ib(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """
    img_h, img_w = 512, 512
    means, stdevs = [], []
    sum_img = np.zeros((3,img_h,img_w))

    for i in range(len(image_paths)):
        dcmimg = dicom.dcmread(image_paths[i])
        img = dcmimg.pixel_array
        img = cv2.resize(img, (img_h, img_w))
        img = img.astype('float64')
        img = img[newaxis,:, :]
        img = np.repeat(img, 3, axis=0)
        sum_img += (img/4096)
      
    sum_img = sum_img/len(image_paths)

    for i in range(3):
        pixels = sum_img[i, :, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs

#img_means, img_stds =compute_img_mean_std_mias(df_all['path'])   
#img_means, img_stds =compute_img_mean_std_ib(df_inbreast['path'])   

normMean = [0.11845642201924221, 0.39512413161915483, 0.5423630034884959]
normStd = [0.0863434530534716, 0.1895667292410686, 0.09111656652924158]

normMean_ib = [0.11623384301016873, 0.11623384301016873, 0.11623384301016873]
normStd_ib = [0.06726126934186419, 0.06726126934186419, 0.06726126934186419]

num_classes = 3

def get_type(x):
    return 'id'
df_all['type'] = df_all['severity']
df_all['type'] = df_all['type'].apply(get_type)

map_dict = {'N': 0,
            'B': 1,
            'M': 2}
def get_class(x):
  return map_dict[x]
df_all['class_id'] = df_all['severity']
df_all['class_id'] = df_all['class_id'].apply(get_class)

df_train = df_all[df_all['set']=='train']
y = df_train['class_id']
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=shuffle_seed, stratify=y)
df_test = df_all[df_all['set']=='test']  
df_test = pd.concat([df_test, df_inbreast], ignore_index=True)
df_train = df_train.reset_index()
df_val = df_val.reset_index()
df_test = df_test.reset_index()
data_aug_rate = [0,4,5] 
classes = ['N','B','M']
for i in range(3):
    if data_aug_rate[i]:
        df_train=df_train.append([df_train.loc[df_train['severity'] == classes[i],:]]*(data_aug_rate[i]-1), ignore_index=True)
df_train = df_train.reset_index()

print(df_train['class_id'].value_counts())
print(df_val['class_id'].value_counts())
print(df_test['class_id'].value_counts())

class Resize(object):
    def __init__(self,im_shape):
        self.im_shape = im_shape

    def __call__(self, sample):
        img = sample
        img = img.astype('float32')
        img = cv2.resize(img, self.im_shape)
        return img

class HFlip(object):
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self, sample):
        img = sample
        if np.random.uniform() > self.p:
            img = np.flip(img, 1)
        return img

class VFlip(object):
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self, sample):
        img = sample
        if np.random.uniform() > self.p:
            img = np.flip(img, 0)
        return img

class Rotate(object):
    def __init__(self,theta=0):
        self.theta = theta

    def __call__(self, sample):
        img = sample
        angle = random.randint(-1*self.theta,self.theta)
        if np.random.uniform() > 0.5:
            img = ndimage.rotate(img, angle, reshape=False)
        return img

class ColorJitter(object): #https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    def __init__(self,brightness=0, contrast=0):
        self.brightness = brightness
        self.contrast = contrast
        
    def __call__(self, sample):
        img = sample
        alpha = np.random.uniform(1-self.contrast, 1+self.contrast)
        beta = random.randint(int(-1*self.brightness*255), int(self.brightness*255))
        if np.random.uniform() > 0.5:
            img = alpha*img + beta
        return img

class ToTensor(object):
    def __call__(self, sample):
        img = sample
        img = np.transpose(img, (2,0,1))
        return torch.from_numpy(img.copy())

input_size = 512
train_transform = transforms.Compose([Resize((input_size,input_size)),
                                      HFlip(),
                                      VFlip(),
                                      Rotate(20),
                                      #ColorJitter(brightness=0.01, contrast=0.1),
                                      ToTensor(),
                                      transforms.Normalize(normMean, normStd)])
## define the transformation of the val images.
test_transform = transforms.Compose([Resize((input_size,input_size)),
                                     ToTensor(),
                                     transforms.Normalize(normMean, normStd)])
test_transform_ib = transforms.Compose([Resize((input_size,input_size)),
                                        ToTensor(),
                                        transforms.Normalize(normMean_ib, normStd_ib)])                                    
                                    
class CustomTrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = cv2.imread(self.df['path'][index])/255
        y = torch.tensor(self.df['class_id'][index])
        d_type = self.df['type'][index]

        if self.transform:
            X = self.transform(X)
        return X, y, d_type

class CustomValDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = cv2.imread(self.df['path'][index])/255
        y = torch.tensor(self.df['class_id'][index])
        d_type = self.df['type'][index]

        if self.transform:
            X = self.transform(X)
        return X, y, d_type

class CustomTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.norm = transforms.Normalize(normMean, normStd)
        self.inv_norm = transforms.Normalize(mean= [-m/s for m, s in zip(normMean, normStd)],std= [1/s for s in normStd])
        self.norm_ib = transforms.Normalize(normMean_ib, normStd_ib)
        self.inv_norm_ib = transforms.Normalize(mean= [-m/s for m, s in zip(normMean_ib, normStd_ib)],std= [1/s for s in normStd_ib])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        im_path = self.df['path'][index]
        d_type = self.df['type'][index]
        if d_type == 'id':
            X = cv2.imread(im_path)/255
        else:
            dcmimg = dicom.dcmread(im_path)
            X = (dcmimg.pixel_array)/4096
            X = X[:,:,newaxis]
            X = np.repeat(X, 3, axis=2)
        y = torch.tensor(self.df['class_id'][index])
        

        if self.transform:
            if d_type == 'id':
                X = self.transform(X)
            else:
                X = test_transform_ib(X)
        if d_type == 'id':
            im_arr = self.inv_norm(X)
        else:
            im_arr = self.inv_norm_ib(X)

        im_arr = np.array(im_arr).transpose(1,2,0)
        
        return X, y, d_type, im_path, im_arr
      
# Define the training set using the table train_df and using our defined transitions (train_transform)
training_set = CustomTrainDataset(df_train, transform=train_transform)
train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=1)
# Same for the validation set:
validation_set = CustomValDataset(df_val, transform=test_transform)
val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=1)
# Same for the test set:
test_set = CustomTestDataset(df_test, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=1)

#################################################################################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

d_model = xrv.models.DenseNet(weights="densenet121-res224-all")
d_model.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
class custom_model(nn.Module):
  def __init__(self, densnet):
    super(custom_model, self).__init__()
    self.dnet = densnet
    self.linout = nn.Linear(1024,num_classes)
  def forward(self,x):
    out = self.dnet(x)
    out = self.linout(out)
    return out
    
model = custom_model(d_model)
model.to(device)

def tsne_visualize(model,loader, perplexity,learning_rate, n_iter, random_state, file_name):
  model.eval()
  df = pd.DataFrame(columns = list(str(i) for i in range(num_classes)))
  df['label'] = None
  for i, sample in enumerate(loader, 0):
    x_i,label, d_type, _, _  = sample
    x_i = x_i.to(device)
    label = label.to(device)
    z_i = model(x_i)
    z_i = np.array(z_i.detach().cpu())
    lb = np.array(label.detach().cpu())
    for it in range(z_i.shape[0]):
      temp = {}
      #temp['label'] = lb[it]
      if (d_type[it] == 'od'):
        temp['label'] = num_classes
      else:
        temp['label'] = lb[it]
      for jt in range(z_i.shape[1]):
        temp[str(jt)] = z_i[it,jt]
      df = df.append(temp, ignore_index = True)

  m = TSNE(perplexity = perplexity, learning_rate = learning_rate, n_iter = n_iter, random_state = random_state)
  df_feat = df[list(str(i) for i in range(num_classes))]
  tsne_data = m.fit_transform(df_feat)
  tsne_df = pd.DataFrame(tsne_data, columns = ['x', 'y'])
  tsne_df['label'] = df['label'].tolist()
  sns.FacetGrid(tsne_df,hue='label', size = 6).map(plt.scatter, 'x', 'y')
  plt.legend(loc = 'best')
  plt.savefig('/home/Drive/abhiraj/plots/'+file_name+'.png')

tsne_visualize(model,test_loader, 30,200, 1000, 0, 'softmax_mias_before_tsne')

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
        
total_loss_train, total_acc_train = [],[]
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        images, labels, d_type = data
        N = images.size(0)
        # print('image shape:',images.shape, 'label shape',labels.shape)
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
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
            images, labels, d_type = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            matches = prediction.eq(labels.view_as(prediction))
            match_sum = 0
            for idx in range(len(matches)):
              if d_type[idx] == 'id' and matches[idx][0] == True:
                match_sum += 1 
            val_acc.update(match_sum/N)

            val_loss.update(criterion(outputs, labels).item())

    print('------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
    print('------------------------------------------------------------')
    return val_loss.avg, val_acc.avg
    
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,10,15], gamma=0.1)

epoch_num = 20
best_val_acc = 0
best_val_loss = 100000
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
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        torch.save(model.state_dict(), "/home/Drive/abhiraj/models/softmax_mias.pt")
    scheduler.step()

tsne_visualize(model,test_loader, 30,200, 1000, 0, 'softmax_mias_after_tsne')

model.load_state_dict(torch.load("/home/Drive/abhiraj/models/softmax_mias.pt"))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
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
    plt.savefig('/home/Drive/abhiraj/plots/softmax_mias_confusion.png')
    
model.eval()
y_label = []
y_predict = []
with torch.no_grad():
    for i, data in enumerate(val_loader):
        images, labels, d_type = data
        N = images.size(0)
        images = Variable(images).to(device)
        outputs = model(images)
        prediction = outputs.max(1, keepdim=True)[1]
        y_label.extend(labels.cpu().numpy())
        y_predict.extend(np.squeeze(prediction.cpu().numpy().T))
        
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_label, y_predict)
# plot the confusion matrix
plot_labels = classes
# plot_confusion_matrix(confusion_mtx, plot_labels)
    
import itertools
plot_confusion_matrix(confusion_mtx, plot_labels)

report = classification_report(y_label, y_predict, target_names=plot_labels)
print(report)

def detect_ood_soft(model, loader,n_clusters, test_type):
  model.eval()
  
  class_wise_correct = [0 for i in range(n_clusters)]
  class_wise_total = [0 for i in range(n_clusters)]
  id_hist = []
  od_hist = []
  id_total = 0
  od_total = 0
  TPR_arr = []
  FPR_arr = []
  correct_list = []
  ood_list = []
  df_scores = pd.DataFrame(columns = ['type', 'label', 'pred','score','path'])
  with torch.no_grad():
    for dat in loader:
      img, label, d_type, path, og_imgs = dat
      img = img.to(device)
      label = label.to(device)
      logits = model(img)
      prediction = logits.max(1, keepdim=True)[1]
      softmax = F.softmax(logits, dim=1)
      max_softmax, _ = torch.max(softmax.data, 1)
      for i in range(label.shape[0]):
        if d_type[i] == 'id':
          id_hist.append(max_softmax[i].detach().cpu().item())
          if label[i] == prediction[i]:
            class_wise_correct[int(label[i])] += 1
            correct_list.append([max_softmax[i].detach().cpu().item(),1, path[i], og_imgs[i], label[i].detach().cpu().item()])
          else:
            correct_list.append([max_softmax[i].detach().cpu().item(),0, path[i], og_imgs[i], label[i].detach().cpu().item()])
          class_wise_total[int(label[i])] += 1
          df_scores = df_scores.append({'type':'id','label':int(label[i]), 'pred': int(prediction[i]),'score':max_softmax[i].detach().cpu().item(),'path':path[i]}, ignore_index = True )
          id_total += 1
        else:
          od_hist.append(max_softmax[i].detach().cpu().item())
          ood_list.append([max_softmax[i].detach().cpu().item(), path[i], og_imgs[i]])
          df_scores = df_scores.append({'type':'od','label':int(label[i]), 'pred': 'OOD','score':max_softmax[i].detach().cpu().item(),'path':path[i]}, ignore_index = True )
          od_total += 1
          
  df_scores.to_csv('softmax_mias_test_scores.csv', index=False)
  
  print(len(id_hist))
  print(len(od_hist))
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
    if i == int(0.25*(len(th_arr)-10)): ############## depends on 547
      odac25 = od_corr/(od_wrng+od_corr)
      idac25 = id_corr/(id_wrng+id_corr)
    if i == int(0.5*(len(th_arr)-10)): ############## depends on 547
      odac50 = od_corr/(od_wrng+od_corr)
      idac50 = id_corr/(id_wrng+id_corr)
    if i == int(0.75*(len(th_arr)-10)): ############## depends on 547
      odac75 = od_corr/(od_wrng+od_corr)
      idac75 = id_corr/(id_wrng+id_corr)
    if i == int(0.9*(len(th_arr)-10)): ############## depends on 547
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
  plt.savefig('/home/Drive/abhiraj/plots/softmax_mias_OOD_cdf.png')

  plt.figure(figsize = (7,7))
  weights_id = np.ones_like(np.array(id_hist))/len(id_hist)
  weights_od = np.ones_like(np.array(od_hist))/len(od_hist)
  plt.hist(id_hist, weights=weights_id, bins=100, fc=(0, 1, 0, 0.5), label='id')
  plt.hist(od_hist, weights=weights_od, bins=100, fc=(1, 0, 0, 0.5),label='od')
  plt.axvline(x=thresh,color='k')
  plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  plt.legend(loc='best')
  plt.savefig('/home/Drive/abhiraj/plots/softmax_mias_OOD_score.png')

  plt.figure(figsize = (10,10))
  plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.title('AUC : '+ str(auc_score))
  plt.savefig('/home/Drive/abhiraj/plots/softmax_mias_AUROC.png')
  
  fp = open('/home/Drive/abhiraj/eval/ddsm_scores_lists.txt','a')
  fp.write(test_type+' SOFTMAX \n') #######
  fp.write('id_hist: '+str(id_hist)+'\n')
  fp.write('od_hist: '+str(od_hist)+'\n')
  fp.close()
  
  fp = open('/home/Drive/abhiraj/eval/ddsm_auroc_lists.txt','a')
  fp.write(test_type+' SOFTMAX \n') #######
  fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  fp.close()
  
  ood_list.sort()
    
  fp = open('/home/Drive/abhiraj/eval/ddsm_od_paths.txt','a')
  fp.write(test_type+' SOFTMAX \n') #######
  for dat in ood_list:
   fp.write(str(dat[0])+' : '+str(dat[1])+'\n')
  fp.close()
  
  count = 0
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0])) 
    plt.savefig('/home/Drive/abhiraj/eval/sample_images/'+test_type+'_od_softmax_mias_'+str(count)+'.png') #####################
    if count == 5:
      break
  
  ood_list.sort(reverse=True)
  
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0]))
    plt.savefig('/home/Drive/abhiraj/eval/sample_images/'+test_type+'_od_softmax_mias_'+str(count)+'.png') ######################
    if count == 10:
      break
  
  correct_list.sort()
  
  acc_list = []
  fp = open('/home/Drive/abhiraj/eval/ddsm_im_paths.txt','a')
  fp.write(test_type+' SOFTMAX \n')
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
    plt.savefig('/home/Drive/abhiraj/eval/sample_images/'+test_type+'_softmax_mias_'+str(count)+'.png')
    if count == 5:
      break
  
  correct_list.sort(reverse=True)
  
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/Drive/abhiraj/eval/sample_images/'+test_type+'_softmax_mias_'+str(count)+'.png')
    if count == 10:
      break
  
  correct_arr = np.array(acc_list)
  acc_list = []
  data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  for i in range(len(data_left_list)):
    temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
    acc_list.append(np.sum(temp_arr)/len(temp_arr))
    
  fp = open('/home/Drive/abhiraj/eval/ddsm_acc_curves.txt', 'a')
  fp.write('SOFTMAX: '+str(acc_list)+'\n')
  fp.close()
  
  plt.figure()
  plt.plot(data_left_list, acc_list, '-', label='accuracy')
  plt.xlabel('% data ignored')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs %data left')
  plt.legend(loc='best')
  plt.savefig('/home/Drive/abhiraj/eval/softmax_mias_accuracy.png')

detect_ood_soft(model, test_loader,num_classes, 'no_tf')


def detect_ood_odin(model, loader, temparature, epsilon,n_clusters, test_type):
  model.train()
  
  class_wise_correct = [0 for i in range(n_clusters)]
  class_wise_total = [0 for i in range(n_clusters)]
  id_hist = []
  od_hist = []
  id_total = 0
  od_total = 0
  TPR_arr = []
  FPR_arr = []
  correct_list = []
  ood_list = []
  df_scores = pd.DataFrame(columns = ['type', 'label','pred','score','path'])
  for dat in loader:
    img, label, d_type, path, og_imgs = dat
    img = Variable(img.to(device), requires_grad = True)
    img.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    logits = model(img)
    logits = logits / temparature
    _, predicted = torch.max(logits.data, 1)

    loss = criterion(logits, predicted)
    loss.backward()
    gradient =  torch.ge(img.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    gradient[0][0] = (gradient[0][0] )/(0.0321222) 
    gradient[0][1] = (gradient[0][1] )/(0.0321222)
    gradient[0][2] = (gradient[0][2] )/(0.0321222)
    tempInputs = torch.add(img.data,  (-1)*epsilon, gradient)
    logits = model(Variable(tempInputs))
    logits = logits / temparature
    prediction = logits.max(1, keepdim=True)[1]
    softmax = F.softmax(logits, dim=1)
    max_softmax, _ = torch.max(softmax.data, 1)
    for i in range(label.shape[0]):
      if d_type[i] == 'id':
        id_hist.append(max_softmax[i].detach().cpu().item())
        if label[i] == prediction[i]:
          class_wise_correct[int(label[i])] += 1
          correct_list.append([max_softmax[i].detach().cpu().item(),1, path[i], og_imgs[i], label[i].detach().cpu().item()])
        else:
          correct_list.append([max_softmax[i].detach().cpu().item(),0, path[i], og_imgs[i], label[i].detach().cpu().item()])
        class_wise_total[int(label[i])] += 1
        df_scores = df_scores.append({'type':'id','label':int(label[i]),'pred': int(prediction[i]),'score':max_softmax[i].detach().cpu().item(),'path':path[i]}, ignore_index = True )
        id_total += 1
      else:
        od_hist.append(max_softmax[i].detach().cpu().item())
        ood_list.append([max_softmax[i].detach().cpu().item(), path[i], og_imgs[i]])
        df_scores = df_scores.append({'type':'od','label':int(label[i]),'pred': 'OOD','score':max_softmax[i].detach().cpu().item(),'path':path[i]}, ignore_index = True )
        od_total += 1
  print(len(id_hist))
  print(len(od_hist))
  
  df_scores.to_csv('odin_mias_test_scores.csv', index=False)
  
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
    if i == int(0.25*(len(th_arr)-10)): ############## depends on 670 
      odac25 = od_corr/(od_wrng+od_corr)
      idac25 = id_corr/(id_wrng+id_corr)
    if i == int(0.5*(len(th_arr)-10)): ############## depends on 670
      odac50 = od_corr/(od_wrng+od_corr)
      idac50 = id_corr/(id_wrng+id_corr)
    if i == int(0.75*(len(th_arr)-10)): ############## depends on 670
      odac75 = od_corr/(od_wrng+od_corr)
      idac75 = id_corr/(id_wrng+id_corr)
    if i == int(0.9*(len(th_arr)-10)): ############## depends on 670
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
  plt.savefig('/home/Drive/abhiraj/plots/odin_mias_OOD_cdf.png')

  plt.figure(figsize = (7,7))
  weights_id = np.ones_like(np.array(id_hist))/len(id_hist)
  weights_od = np.ones_like(np.array(od_hist))/len(od_hist)
  plt.hist(id_hist, weights=weights_id, bins=100, fc=(0, 1, 0, 0.5), label='id')
  plt.hist(od_hist, weights=weights_od, bins=100, fc=(1, 0, 0, 0.5),label='od')
  plt.axvline(x=thresh,color='k')
  plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  plt.legend(loc='best')
  plt.savefig('/home/Drive/abhiraj/plots/odin_mias_OOD_score.png')

  plt.figure(figsize = (10,10))
  plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.title('AUC : '+ str(auc_score))
  plt.savefig('/home/Drive/abhiraj/plots/odin_AUROC.png')
  
  fp = open('/home/Drive/abhiraj/eval/ddsm_scores_lists.txt','a')
  fp.write(test_type+' ODIN \n') #######
  fp.write('id_hist: '+str(id_hist)+'\n')
  fp.write('od_hist: '+str(od_hist)+'\n')
  fp.close()
  
  fp = open('/home/Drive/abhiraj/eval/ddsm_auroc_lists.txt','a')
  fp.write(test_type+' ODIN \n') #######
  fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  fp.close()
  
  ood_list.sort()
    
  fp = open('/home/Drive/abhiraj/eval/ddsm_od_paths.txt','a')
  fp.write(test_type+' ODIN \n') #######
  for dat in ood_list:
   fp.write(str(dat[0])+' : '+str(dat[1])+'\n')
  fp.close()
  
  count = 0
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0])) 
    plt.savefig('/home/Drive/abhiraj/eval/sample_images/'+test_type+'_od_odin_mias_'+str(count)+'.png') #####################
    if count == 5:
      break
  
  ood_list.sort(reverse=True)
  
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0]))
    plt.savefig('/home/Drive/abhiraj/eval/sample_images/'+test_type+'_od_odin_mias_'+str(count)+'.png') ######################
    if count == 10:
      break
  
  correct_list.sort()
  
  acc_list = []
  fp = open('/home/Drive/abhiraj/eval/ddsm_im_paths.txt','a')
  fp.write(test_type+' ODIN \n')
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
    plt.savefig('/home/Drive/abhiraj/eval/sample_images/'+test_type+'_odin_mias_'+str(count)+'.png')
    if count == 5:
      break
  
  correct_list.sort(reverse=True)
  
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/Drive/abhiraj/eval/sample_images/'+test_type+'_odin_mias_'+str(count)+'.png')
    if count == 10:
      break
  
  correct_arr = np.array(acc_list)
  acc_list = []
  data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  for i in range(len(data_left_list)):
    temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
    acc_list.append(np.sum(temp_arr)/len(temp_arr))
    
  fp = open('/home/Drive/abhiraj/eval/ddsm_acc_curves.txt', 'a')
  fp.write('ODIN: '+str(acc_list)+'\n')
  fp.close()
  
  plt.figure()
  plt.plot(data_left_list, acc_list, '-', label='accuracy')
  plt.xlabel('% data ignored')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs %data left')
  plt.legend(loc='best')
  plt.savefig('/home/Drive/abhiraj/eval/odin_mias_accuracy.png')
  
detect_ood_odin(model, test_loader,1000,0.002,num_classes, 'no_tf')

def generate_gradcam_vis(model, loader, mean, std):
  model.eval()
  inv_norm = transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],std= [1/s for s in std])
  target_layer = model.dnet.features.denseblock4.denselayer16[-1]
  cam = GradCAM(model=model, target_layer=target_layer)
  for i, sample in enumerate(loader, 0):
    img, label, d_type, im_paths, og_imgs  = sample
    img = img.to(device)
    for j in range(len(label)):
      input_tensor = img[j].unsqueeze(0)
      rgb_img = inv_norm(img[j]).float().permute(1, 2, 0).cpu().numpy()
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
      bach_class = im_paths[j].split('/')[-2]
      fig.suptitle(d_type[j]+'_'+str(target_category)+'_'+bach_class+'_'+im_paths[j].split('/')[-1][:-4], fontsize=12)
      fig.savefig('/home/Drive/abhiraj/eval/GRADCAM/SOFTMAX/softmax_gradcam_'+d_type[j]+'_'+str(target_category)+'_'+bach_class+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'.png')
      
generate_gradcam_vis(model, test_loader, normMean, normStd)
