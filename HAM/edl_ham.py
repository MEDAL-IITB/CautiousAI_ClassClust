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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
df_train = pd.concat([df_train_mel, df_train_nv, df_train_bkl]) ##############


data_aug_rate = [0,0,5,0,0,0,5]  #[15,10,5,50,0,40,5] ##############
for i in range(7):
    if data_aug_rate[i]:
        df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)
        
        
df_train = df_train.reset_index()
df_val = df_val.reset_index()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class PretrainedResNext(nn.Module):
    def __init__(self, num_class=3): ##############
        super().__init__()
        resNext = models.resnext50_32x4d(pretrained=True)
        self.channels = resNext.fc.out_features
        for params in resNext.parameters():
            params.requires_grad_(False)
        self.features = nn.Sequential(*list(resNext.children()))
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channels, num_class)
        self.softmax = nn.Softmax()

    def forward(self, x):
        features = self.features(x)
        out = self.relu(features)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.channels)
        out = self.fc1(out)
        out = self.SoftMax(out)
        return out
        

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(in_features=num_ftrs, out_features=3) ##############
model = model_ft


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
      map = [0,0,0,0,1,0,2] ##############
      d_type = ''
      if cell_type_idx in [0,1,3,5]: ##############
        d_type = 'od'
      else:
        d_type = 'id'
      return map[cell_type_idx]

def map_indices_val(cell_type_idx):
      map = [0,0,0,0,1,0,2] ##############
      d_type = ''
      if cell_type_idx in [0,1,3,5]: ##############
        d_type = 'od'
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
            X = self.transform(X)
        return X, y, d_type

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
            X = self.transform(X)
            
        im_arr = self.inv_norm(X)
        im_arr = np.array(im_arr).transpose(1,2,0)
        
        return X, y, d_type, im_path, np.asarray(im_arr)


# Define the training set using the table train_df and using our defined transitions (train_transform)
training_set = CustomTrainDataset(df_train, transform=train_transform)
train_loader = DataLoader(training_set, batch_size=64, shuffle=True, num_workers=1)
# Same for the validation set:
validation_set = CustomValDataset(df_val,normMean,normStd, transform=train_transform)
print(len(validation_set))
val_loader = DataLoader(validation_set, batch_size=64, shuffle=False, num_workers=1)


def tsne_visualize(model,loader, perplexity,learning_rate, n_iter, random_state, file_name):
  model.eval()
  df = pd.DataFrame(columns = list(str(i) for i in range(3)))
  df['label'] = None
  for i, sample in enumerate(loader, 0):
    x_i,label, d_type, _,_  = sample
    x_i = x_i.to(device)
    label = label.to(device)
    z_i = model(x_i)
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
        
        
        
def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

def kl_divergence(alpha, num_classes, device=None):

    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
        torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl

def loglikelihood_loss(y, alpha, device=None):

    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):

    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * \
        kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):

    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * \
        kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):

    evidence = relu_evidence(output) ###
    alpha = evidence + 1
    loss = torch.mean(mse_loss(target, alpha, epoch_num,
                               num_classes, annealing_step, device=device))
    return loss

def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):

    evidence = relu_evidence(output) ###
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.log, target, alpha,
                               epoch_num, num_classes, annealing_step, device))
    return loss

def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):

    evidence = relu_evidence(output) ###
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.digamma, target, alpha,
                               epoch_num, num_classes, annealing_step, device))
    return loss
    

early_stop_ep = 10
loss_type = 'digamma' #[digamma,log,mse]
annealing_step = 10

total_loss_train, total_acc_train = [],[]
def train(train_loader, model, optimizer, epoch):
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

        ###################################################
        # Extra lines for EDL over HAM10000
        ###################################################
        # second argument is len(id_classes) indistibution classes
        # hardcoded as len ([0,1,2,3,4,5,6]-[0, 1, 3, 5]) = 3
        y = one_hot_embedding(labels, 3)

        if loss_type == 'digamma':
          # changed 3rd arg ep to epoch - couldnt find ep in ref code
          loss = edl_digamma_loss(outputs, y.float(), epoch, 3, annealing_step, device)
        elif loss_type == 'log':
          loss = edl_log_loss(outputs, y.float(), epoch, 3, annealing_step, device)
        else:
          loss = edl_mse_loss(outputs, y.float(), epoch, 3, annealing_step, device)
        ################################################### 

        # loss = criterion(outputs, labels)
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
    
def validate(val_loader, model, optimizer, epoch):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels, d_type, _, _ = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            # ##################################################
            # Extra variables for EDL over HAM10000
            # ##################################################
            # evidence = relu_evidence(outputs)
            # alpha = evidence + 1
            # uncertainty = 3 / torch.sum(alpha, dim=1, keepdim=True)
            y = one_hot_embedding(labels, 3)
            if loss_type == 'digamma':
              # changed 3rd arg ep to epoch - couldnt find ep in ref code
              loss = edl_digamma_loss(outputs, y.float(), epoch, 3, annealing_step, device)
            elif loss_type == 'log':
              loss = edl_log_loss(outputs, y.float(), epoch, 3, annealing_step, device)
            else:
              loss = edl_mse_loss(outputs, y.float(), epoch, 3, annealing_step, device)
            ################################################### 

            prediction = outputs.max(1, keepdim=True)[1]

            matches = prediction.eq(labels.view_as(prediction))
            match_sum = 0
            for idx in range(len(matches)):
              if d_type[idx] == 'id' and matches[idx][0] == True:
                match_sum += 1 
            val_acc.update(match_sum/N)

            val_loss.update(loss.item())

    print('------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
    print('------------------------------------------------------------')
    return val_loss.avg, val_acc.avg
    
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)

tsne_visualize(model,val_loader, 30,200, 1000, 0, 'edl_ham_before_tsne')

epoch_num = 50
best_val_acc = 0
total_loss_val, total_acc_val = [],[]
for epoch in tqdm(range(1, epoch_num+1)):
    loss_train, acc_train = train(train_loader, model, optimizer, epoch)
    loss_val, acc_val = validate(val_loader, model, optimizer, epoch)
    total_loss_val.append(loss_val)
    total_acc_val.append(acc_val)
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        print('*****************************************************')
        print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
        print('*****************************************************')
        torch.save(model.state_dict(), "/home/abhiraj/DDP/CL/models/HistoSimCLR/edl_ham.pt")
        
tsne_visualize(model,val_loader, 30,200, 1000, 0, 'edl_ham_after_tsne')     

model.load_state_dict(torch.load("/home/abhiraj/DDP/CL/models/HistoSimCLR/edl_ham.pt"))

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
    plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/edl_confusion.png')
    
model.eval()
y_label = []
y_predict = []
with torch.no_grad():
    for i, data in enumerate(val_loader):
        images, labels, d_type,_,_ = data
        N = images.size(0)
        images = Variable(images).to(device)
        outputs = model(images)
        prediction = outputs.max(1, keepdim=True)[1]
        y_label.extend(labels.cpu().numpy())
        y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

# compute the confusion matrix
confusion_mtx = confusion_matrix(y_label, y_predict)
# plot the confusion matrix
plot_labels = ['bkl', 'nv','mel']  #['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
# plot_confusion_matrix(confusion_mtx, plot_labels)
    
import itertools
plot_confusion_matrix(confusion_mtx, plot_labels)

report = classification_report(y_label, y_predict, target_names=plot_labels)
print(report)


from scipy import stats
from sklearn import metrics

def detect_ood(model, loader, n_clusters, test_type):
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
  df_scores = pd.DataFrame(columns = ['type', 'label','pred','score','path'])
  with torch.no_grad():
    for dat in loader:
      img, label, d_type, im_paths, og_imgs = dat
      img = img.to(device)
      label = label.to(device)
      logits = model(img)
      prediction = logits.max(1, keepdim=True)[1]
      ###################################################
      # Extra lines for EDL over HAM10000
      ###################################################
      evidence = relu_evidence(logits)
      alpha = evidence + 1
      uncertainty = n_clusters/torch.sum(alpha, dim=1, keepdim=True)
      for i in range(label.shape[0]):
        if d_type[i] == 'id':
          id_hist.append(uncertainty[i].detach().cpu().item())
          if label[i] == prediction[i]:
            class_wise_correct[label[i]] += 1
            correct_list.append([uncertainty[i].detach().cpu().item(),1, im_paths[i], og_imgs[i], label[i].detach().cpu().item()])
          else:
            correct_list.append([uncertainty[i].detach().cpu().item(),0, im_paths[i], og_imgs[i], label[i].detach().cpu().item()])
          class_wise_total[label[i]] += 1
          df_scores = df_scores.append({'type':'id','label':int(label[i]),'pred': int(prediction[i]),'score':uncertainty[i].detach().cpu().item(),'path':im_paths[i]}, ignore_index = True )
          id_total += 1
        else:
          od_hist.append(uncertainty[i].detach().cpu().item())
          ood_list.append([uncertainty[i].detach().cpu().item(), im_paths[i], og_imgs[i]])
          df_scores = df_scores.append({'type':'od','label':int(label[i]),'pred': 'OOD','score':uncertainty[i].detach().cpu().item(),'path':im_paths[i]}, ignore_index = True )
          od_total += 1
  df_scores.to_csv('/home/abhiraj/DDP/CL/eval/HAM/edl_test_scores.csv', index=False)
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
    id_corr = np.sum(np.array(id_hist) < th_arr[i])
    id_wrng = np.sum(np.array(id_hist) >= th_arr[i])
    od_corr = np.sum(np.array(od_hist) >= th_arr[i])
    od_wrng = np.sum(np.array(od_hist) < th_arr[i])
    tpr = od_corr/(od_corr+od_wrng)
    fpr = id_wrng/(id_corr+id_wrng)
    TPR_arr.append(tpr)
    FPR_arr.append(fpr)
    if abs(tpr - 0.95) < tpr95:
      fpr95 = fpr
      tpr95 = abs(tpr - 0.95)
    if i == int(0.25*(len(th_arr)-10)): ############## depends on 613
      odac25 = od_corr/(od_wrng+od_corr)
      idac25 = id_corr/(id_wrng+id_corr)
    if i == int(0.5*(len(th_arr)-10)): ############## depends on 613
      odac50 = od_corr/(od_wrng+od_corr)
      idac50 = id_corr/(id_wrng+id_corr)
    if i == int(0.75*(len(th_arr)-10)): ############## depends on 613
      odac75 = od_corr/(od_wrng+od_corr)
      idac75 = id_corr/(id_wrng+id_corr)
    if i == int(0.9*(len(th_arr)-10)): ############## depends on 613
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
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/edl_OOD_cdf.png')

  plt.figure()
  plt.hist(id_hist, bins=100, fc=(0, 1, 0, 0.5), label='id', density=True)
  plt.hist(od_hist, bins=100, fc=(1, 0, 0, 0.5),label='od', density=True)
  plt.axvline(x=thresh,color='k')
  plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/edl_OOD_score.png')

  plt.figure()
  plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.title('AUC : '+ str(auc_score))
  plt.savefig('/home/abhiraj/DDP/CL/plots/HAM/edl_AUROC.png')
  
  fp = open('/home/abhiraj/DDP/CL/eval/HAM/ham_scores_lists.txt','a')
  fp.write(test_type+' EDL \n') #######
  fp.write('id_hist: '+str(id_hist)+'\n')
  fp.write('od_hist: '+str(od_hist)+'\n')
  fp.close()
  
  fp = open('/home/abhiraj/DDP/CL/eval/HAM/ham_auroc_lists.txt','a')
  fp.write(test_type+' EDL \n') #######
  fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  fp.close()
  
  ood_list.sort()
    
  fp = open('ham_od_paths.txt','a')
  fp.write(test_type+' EDL \n') #######
  for dat in ood_list:
   fp.write(str(dat[0])+' : '+str(dat[1])+'\n')
  fp.close()
  
  count = 0
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0])) 
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_od_edl_'+str(count)+'.png') #####################
    if count == 5:
      break
  
  ood_list.sort(reverse=True)
  
  for dat in ood_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[2]))
    plt.title(str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_od_edl_'+str(count)+'.png') ######################
    if count == 10:
      break
  
  correct_list.sort()
  
  acc_list = []
  fp = open('ham_im_paths.txt','a')
  fp.write(test_type+' EDL \n')
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
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_edl_'+str(count)+'.png')
    if count == 5:
      break
  
  correct_list.sort(reverse=True)
  
  for dat in correct_list:
    count +=1
    plt.figure()
    plt.imshow(np.asarray(dat[3]))
    plt.title(str(dat[4])+ ' : '+ str(dat[0]))
    plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/HAM/'+test_type+'_edl_'+str(count)+'.png')
    if count == 10:
      break
  
  correct_arr = np.array(acc_list)
  acc_list = []
  data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  for i in range(len(data_left_list)):
    temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
    acc_list.append(np.sum(temp_arr)/len(temp_arr))
    
  fp = open('ham_acc_curves.txt', 'a')
  fp.write('EDL: '+str(acc_list)+'\n')
  fp.close()
  
  plt.figure()
  plt.plot(data_left_list, acc_list, '-', label='accuracy')
  plt.xlabel('% data ignored')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs %data left')
  plt.legend(loc='best')
  plt.savefig('/home/abhiraj/DDP/CL/eval/HAM/edl_accuracy.png')
  
detect_ood(model, val_loader,3,'no_tf')

def generate_gradcam_vis(model, loader, mean, std):
  model.eval()
  inv_norm = transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],std= [1/s for s in std])
  target_layer = model.layer4[-1]
  cam = GradCAM(model=model, target_layer=target_layer)
  id_samples = 0
  od_samples = 0
  for i, dat in enumerate(loader, 0):
    img, label, d_type, im_paths, og_imgs = dat
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
      fig.savefig('/home/abhiraj/DDP/CL/eval/HAM/GRADCAM/EDL/edl_gradcam_'+d_type[j]+'_'+str(target_category)+'_'+str(label[j].item())+'_'+file_name+'_'+str(j)+'.png')
      
generate_gradcam_vis(model, val_loader, normMean, normStd)
