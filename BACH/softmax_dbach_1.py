seed = 4
import torch
torch.manual_seed(seed)
import torchvision
import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import sys
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
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn import metrics

############################################################################################

path = "/home/abhiraj/DDP/CL/data/OOD_BACH/"
classes = os.listdir(path)
classes.sort()
classes = classes[:-1]
print(classes)
id_classes = [classes[2],classes[3]]                                                        ######
od_classes = []                                                                   ######begning = 0 insitu = 1 classes[0],classes[1],
id_classes.sort()
od_classes.sort()

def pretrain_bach():
  all_file_paths = []
  train_file_paths = []
  val_file_paths = []
  test_file_paths = []
  
  for i in range(len(id_classes)):
    filenames = os.listdir(os.path.join(path,id_classes[i]))
    for f in filenames:
      file_path = os.path.join(os.path.join(path,id_classes[i]),f)
      train_file_paths.append([file_path, i, 'id'])
  
  all_file_paths = train_file_paths # place holder for now
  
  test_path_id = os.path.join(os.path.join(path,'test'),'iid')
  for i in range(len(id_classes)):
    filenames = os.listdir(os.path.join(test_path_id,id_classes[i]))
    for f in filenames:
      file_path = os.path.join(os.path.join(test_path_id,id_classes[i]),f)
      test_file_paths.append([file_path, i, 'id'])
  
  test_path_od_ip = os.path.join(os.path.join(os.path.join(path,'test'),'ood'),'input')
  defects = os.listdir(test_path_od_ip)
  for i in range(len(defects)):
    filenames = os.listdir(os.path.join(test_path_od_ip,defects[i]))
    for f in filenames:
      file_path = os.path.join(os.path.join(test_path_od_ip,defects[i]),f)
      if 'n' in f:
        idx = id_classes.index('Normal')
      if 'b' in f:
        continue
        #idx = random.randint(0, len(id_classes)-1) #id_classes.index('Benign')
      if 'iv' in f:
        idx = id_classes.index('Invasive')
      if 'is' in f:
        continue
        #idx = random.randint(0, len(id_classes)-1) #id_classes.index('InSitu')
      test_file_paths.append([file_path, idx, 'id'])
      
  test_path_od_op = os.path.join(os.path.join(os.path.join(path,'test'),'ood'),'output')
  for i in range(len(defects)):
    filenames = os.listdir(os.path.join(test_path_od_op,defects[i]))
    for f in filenames:
      file_path = os.path.join(os.path.join(test_path_od_op,defects[i]),f)
      if 'n' in f:
        idx = id_classes.index('Normal')
      if 'b' in f:
        continue
        #idx = random.randint(0, len(id_classes)-1) #id_classes.index('Benign')
      if 'iv' in f:
        idx = id_classes.index('Invasive')
      if 'is' in f:
        continue
        #idx = random.randint(0, len(id_classes)-1) #id_classes.index('InSitu')
      test_file_paths.append([file_path, idx, 'od'])
  
  val_file_paths = test_file_paths  # place holder for now

  random.shuffle(all_file_paths)
  random.shuffle(train_file_paths)
  random.shuffle(val_file_paths)
  random.shuffle(test_file_paths)
  return all_file_paths, train_file_paths, val_file_paths, test_file_paths

class BACHPreTrainDataset(Dataset):
    def __init__(self, transform=None):
      self.transform = transform
      self.all_file_paths, self.train_file_paths, self.val_file_paths, self.test_file_paths =  pretrain_bach()

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

      sample = {'img':x_i,'label':img_class, 'type': img_type }

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

      sample = {'img':x_i,'label':img_class, 'type': img_type }

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

      sample = {'img':x_i,'label':img_class, 'type': img_type }

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
        
      im_arr = self.inv_norm(x_i)
      im_arr = np.array(im_arr).transpose(1,2,0)

      sample = {'img':x_i,'label':img_class, 'type': img_type , 'path': img_path, 'og_img': np.asarray(im_arr)}

      return sample
      
################################################################################################

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

composed_tf = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1),
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
pretrain_dat = BACHPreTrainDataset(composed_tf)
pretrain_dat_notf = BACHPreTrainDataset(composed_notf)
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

#######################################################################################################

#custom model 
full_model = models.resnet18(pretrained=False)
num_ftrs = full_model.fc.in_features
full_model.fc = nn.Linear(num_ftrs, len(id_classes)) #id_classes

#############################################################################################
# sys.exit()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

epoch = 50
early_stop_ep = 20

full_model = full_model.to(device)
optimizer = torch.optim.Adam(full_model.parameters(), lr=0.0001, weight_decay = 0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=5, factor=0.01)

criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

###############################################################################################

#plot TSNE change later
def tsne_visualize(model,loader, perplexity,learning_rate, n_iter, random_state):
  model.eval()
  df = pd.DataFrame(columns = list(str(i) for i in range(len(id_classes)))) #feature length
  df['label'] = None
  for i, sample in enumerate(loader, 0):
    img ,label, d_type  = sample['img'], sample['label'], sample['type']
    img = img.to(device)
    label = label.to(device)
    logits = model(img)
    logits = F.softmax(logits, dim=1)
    logits = np.array(logits.detach().cpu())
    lb = np.array(label.detach().cpu())
    for it in range(logits.shape[0]):
      temp = {}
      temp['label'] = lb[it]
      if (d_type[it] == 'od'):
        temp['label'] = len(id_classes)
      else:
        temp['label'] = lb[it]
      for jt in range(logits.shape[1]):
        temp[str(jt)] = logits[it,jt]
      df = df.append(temp, ignore_index = True)

  m = TSNE(perplexity = perplexity, learning_rate = learning_rate, n_iter = n_iter, random_state = random_state)
  df_feat = df[list(str(i) for i in range(len(id_classes)))]
  tsne_data = m.fit_transform(df_feat)
  tsne_df = pd.DataFrame(tsne_data, columns = ['x', 'y'])
  tsne_df['label'] = df['label'].tolist()
  sns.FacetGrid(tsne_df,hue='label', size = 6).map(plt.scatter, 'x', 'y')
  plt.legend(loc = 'best')
  plt.savefig('/home/abhiraj/DDP/CL/plots/DBACH/softmax_TSNE.png')
  
#####################################################################################

# tsne_visualize(full_model,pretraindataloader_notf, 30, 200, 1000, seed)

###################################################################################

def train(model,loader):
  model.train()
  running_loss = 0.0
  for i, sample in enumerate(loader, 0):
    img, label  = sample['img'], sample['label']
    img = img.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    logits = model(img)
    loss = criterion(logits, label)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    
  print('loss: %.5f' % (running_loss/len(train_dat))) ## train_dat
  return (running_loss/len(train_dat))

def test(model,loader):
  model.eval()
  correct = 0
  total = 0
  class_correct = list(0. for i in range(len(id_classes)))
  class_total = list(0. for i in range(len(id_classes)))
  with torch.no_grad():
    for i, sample in enumerate(loader, 0):
      img, label, d_type  = sample['img'], sample['label'], sample['type']
      img = img.to(device)
      label = label.to(device)
      logits = model(img)
      _, predicted = torch.max(logits.data, 1)
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
  
######################################################################################

# import time
# print('FT-Training')
# tic = time.time()
# min_val_loss = float('inf')
# early_stop_counter = 0
# for ep in range(1,epoch+1):
#   loss = train(full_model,trainloader) #trainloader
#   print("Epoch :",ep) 
#   test(full_model,valloader) #valloader
#   scheduler.step(loss)
#   if min_val_loss > loss:
#     min_val_loss = loss
#     early_stop_counter = 0
#     torch.save(full_model.state_dict(), "/home/abhiraj/DDP/CL/models/HistoSimCLR/softmax_dbach.pt")
#   else:
#     early_stop_counter += 1
#   if early_stop_counter == early_stop_ep:
#     print("Early Stopping...")
#     break
# toc = time.time()
# print("Time elapsed :", toc - tic)

full_model.load_state_dict(torch.load("/home/abhiraj/DDP/CL/models/HistoSimCLR/softmax_dbach.pt"))

########################################################################################

# test(full_model,testloader)

#######################################################################################

def detect_ood(model, loader,n_clusters, test_type):
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
    for i, sample in enumerate(loader, 0):
      img, label, d_type, im_paths, og_imgs  = sample['img'], sample['label'], sample['type'], sample['path'], sample['og_img']
      img = img.to(device)
      label = label.to(device)
      logits = model(img)
      prediction = logits.max(1, keepdim=True)[1]
      softmax = F.softmax(logits, dim=1)
      max_softmax, _ = torch.max(softmax.data, 1)
      # pdb.set_trace()
      for i in range(label.shape[0]):
        if d_type[i] == 'id' or d_type[i] == 'od':
          id_hist.append(max_softmax[i].detach().cpu())
          if label[i] == prediction[i]:
            class_wise_correct[label[i]] += 1
            correct_list.append([max_softmax[i].detach().cpu().item(),1, im_paths[i], og_imgs[i], label[i].detach().cpu()])
          else:
            correct_list.append([max_softmax[i].detach().cpu().item(),0, im_paths[i], og_imgs[i], label[i].detach().cpu()])
          class_wise_total[label[i]] += 1
          df_scores = df_scores.append({'type': d_type[i],'label':int(label[i]), 'pred': int(prediction[i]),'score':max_softmax[i].detach().cpu().item(),'path':im_paths[i]}, ignore_index = True )
          id_total += 1
        # else:
        #   od_hist.append(max_softmax[i].detach().cpu().item())
        #   ood_list.append([max_softmax[i].detach().cpu().item(), im_paths[i], og_imgs[i]])
        #   df_scores = df_scores.append({'type':'od','label':int(label[i]), 'pred': 'OOD','score':max_softmax[i].detach().cpu().item(),'path':im_paths[i]}, ignore_index = True )
        #   od_total += 1
          
  df_scores.to_csv('/home/abhiraj/DDP/CL/eval/DBACH/softmax_test_scores1.csv', index=False)
  
  # max_score = max([max(id_hist),max(od_hist)])
  # min_score = min([min(id_hist),min(od_hist)])
  # max_sum = 0
  # thresh = 0
  # th_arr = np.linspace(min_score,max_score+0.001,int((max_score-min_score)/0.0001))
  
  # fpr95 = 0
  # tpr95 = 2
  # odac25 = 0
  # odac50 = 0
  # odac75 = 0
  # odac90 = 0
  # idac25 = 0
  # idac50 = 0
  # idac75 = 0
  # idac90 = 0
  
  # for i in range(len(th_arr)):
  #   id_corr = np.sum(np.array(id_hist) >= th_arr[i])
  #   id_wrng = np.sum(np.array(id_hist) < th_arr[i])
  #   od_corr = np.sum(np.array(od_hist) < th_arr[i])
  #   od_wrng = np.sum(np.array(od_hist) >= th_arr[i])
  #   tpr = od_corr/(od_corr+od_wrng)
  #   fpr = id_wrng/(id_corr+id_wrng)
  #   TPR_arr.append(tpr)
  #   FPR_arr.append(fpr)
  #   if abs(tpr - 0.95) < tpr95:
  #     fpr95 = fpr
  #     tpr95 = abs(tpr - 0.95)
  #   if i == int(0.25*(len(th_arr)-10)): ############## depends on 400
  #     odac25 = od_corr/(od_wrng+od_corr)
  #     idac25 = id_corr/(id_wrng+id_corr)
  #   if i == int(0.5*(len(th_arr)-10)): ############## depends on 400
  #     odac50 = od_corr/(od_wrng+od_corr)
  #     idac50 = id_corr/(id_wrng+id_corr)
  #   if i == int(0.75*(len(th_arr)-10)): ############## depends on 400
  #     odac75 = od_corr/(od_wrng+od_corr)
  #     idac75 = id_corr/(id_wrng+id_corr)
  #   if i == int(0.9*(len(th_arr)-10)): ############## depends on 400
  #     odac90 = od_corr/(od_wrng+od_corr)
  #     idac90 = id_corr/(id_wrng+id_corr)
  #   if ((id_corr+od_corr) >= max_sum):
  #     max_sum = id_corr+od_corr
  #     id_correct = id_corr
  #     od_correct = od_corr
  #     thresh = th_arr[i]

  # auc_score = metrics.auc(FPR_arr, TPR_arr)
  
  # for i in range(n_clusters):
  #   print(i,':',class_wise_correct[i]/class_wise_total[i])

  # print('Threshold :',thresh)
  # print('ID correct :',100*(id_correct/id_total))
  # print('OD correct :',100*(od_correct/od_total))
  # print('AUC :', auc_score)
  
  # plt.figure(figsize = (5,5))
  # plt.hist(id_hist, bins=100, density=True, cumulative=True, label='id', histtype='step', alpha=0.55, color='g')
  # plt.hist(od_hist, bins=100, density=True, cumulative=True, label='id', histtype='step', alpha=0.55, color='r')
  # plt.xlabel('score')
  # plt.title('Cummulative density of ID and OD')
  # plt.legend(loc = 'best')
  # plt.savefig('/home/abhiraj/DDP/CL/plots/DBACH/softmax_OOD_cdf.png')

  # plt.figure()
  # weights_id = np.ones_like(np.array(id_hist))/len(id_hist)
  # weights_od = np.ones_like(np.array(od_hist))/len(od_hist)
  # plt.hist(id_hist, weights=weights_id, bins=100, fc=(0, 1, 0, 0.5), label='id')
  # plt.hist(od_hist, weights=weights_od, bins=100, fc=(1, 0, 0, 0.5),label='od')
  # plt.axvline(x=thresh,color='k')
  # plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  # plt.legend(loc='best')
  # plt.savefig('/home/abhiraj/DDP/CL/plots/DBACH/softmax_OOD_score.png')

  # plt.figure()
  # plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  # plt.title('ROC curve, AUC= '+ str(auc_score))
  # plt.legend(loc='best')
  # plt.savefig('/home/abhiraj/DDP/CL/plots/DBACH/softmax_AUROC.png')
  
  # fp = open('/home/abhiraj/DDP/CL/eval/DBACH/bach_scores_lists.txt','a')
  # fp.write(test_type+' SOFTMAX \n') #######
  # fp.write('id_hist: '+str(id_hist)+'\n')
  # fp.write('od_hist: '+str(od_hist)+'\n')
  # fp.close()
  
  # fp = open('/home/abhiraj/DDP/CL/eval/DBACH/bach_auroc_lists.txt','a')
  # fp.write(test_type+' SOFTMAX \n') #######
  # fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  # fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  # fp.close()
  
  # ood_list.sort()
    
  # fp = open('bach_od_paths.txt','a')
  # fp.write(test_type+' SOFTMAX \n') #######
  # for dat in ood_list:
  #  fp.write(str(dat[0])+' : '+str(dat[1])+'\n')
  # fp.close()
  
  # count = 0
  # for dat in ood_list:
  #   count +=1
  #   plt.figure()
  #   plt.imshow(np.asarray(dat[2]))
  #   plt.title(str(dat[0])) 
  #   plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/DBACH/'+test_type+'_od_softmax_'+str(count)+'.png') #####################
  #   if count == 5:
  #     break
  
  # ood_list.sort(reverse=True)
  
  # for dat in ood_list:
  #   count +=1
  #   plt.figure()
  #   plt.imshow(np.asarray(dat[2]))
  #   plt.title(str(dat[0]))
  #   plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/DBACH/'+test_type+'_od_softmax_'+str(count)+'.png') ######################
  #   if count == 10:
  #     break
  
  # correct_list.sort()
  
  # acc_list = []
  # fp = open('bach_im_paths.txt','a')
  # fp.write(test_type+' SOFTMAX \n')
  # for dat in correct_list:
  #  acc_list.append([dat[0], dat[1]])
  #  fp.write(str(dat[0])+' : '+str(dat[1])+' : '+dat[2]+'\n')
  # fp.close()
  
  # count = 0
  # for dat in correct_list:
  #   count +=1
  #   plt.figure()
  #   plt.imshow(np.asarray(dat[3]))
  #   plt.title(str(dat[4])+ ' : '+ str(dat[0]))
  #   plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/DBACH/'+test_type+'_softmax_'+str(count)+'.png')
  #   if count == 5:
  #     break
  
  # correct_list.sort(reverse=True)
  
  # for dat in correct_list:
  #   count +=1
  #   plt.figure()
  #   plt.imshow(np.asarray(dat[3]))
  #   plt.title(str(dat[4])+ ' : '+ str(dat[0]))
  #   plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/DBACH/'+test_type+'_softmax_'+str(count)+'.png')
  #   if count == 10:
  #     break
  
  # correct_list.sort()
  # correct_arr = np.array(acc_list)
  # acc_list = []
  # data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  # for i in range(len(data_left_list)):
  #   temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
  #   acc_list.append(np.sum(temp_arr)/len(temp_arr))
    
  # fp = open('acc_curves.txt', 'a')
  # fp.write('Softmax: '+str(acc_list)+'\n')
  # fp.close()
  
  # plt.figure()
  # plt.plot(data_left_list, acc_list, '-', label='accuracy')
  # plt.xlabel('% data ignored')
  # plt.ylabel('Accuracy')
  # plt.title('Accuracy vs %data left')
  # plt.legend(loc='best')
  # plt.savefig('/home/abhiraj/DDP/CL/eval/DBACH/softmax_accuracy.png')
############################################################################################

# detect_ood(full_model, testloader,len(id_classes), 'no_tf')

#############################################################################################

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
  for sample in loader:
    img, label, d_type, im_paths, og_imgs  = sample['img'], sample['label'], sample['type'], sample['path'], sample['og_img']
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
    gradient[0][0] = (gradient[0][0] )/(0.14092763) 
    gradient[0][1] = (gradient[0][1] )/(0.15261263)
    gradient[0][2] = (gradient[0][2] )/(0.16997081)
    tempInputs = torch.add(img.data,  (-1)*epsilon, gradient)
    logits = model(Variable(tempInputs))
    logits = logits / temparature
    prediction = logits.max(1, keepdim=True)[1]
    softmax = F.softmax(logits, dim=1)
    max_softmax, _ = torch.max(softmax.data, 1)
    for i in range(label.shape[0]):
      if d_type[i] == 'id' or d_type[i] == 'od':
        id_hist.append(max_softmax[i].detach().cpu().item())
        if label[i] == prediction[i]:
          class_wise_correct[label[i]] += 1
          correct_list.append([max_softmax[i].detach().cpu().item(),1, im_paths[i], og_imgs[i], label[i].detach().cpu()])
        else:
          correct_list.append([max_softmax[i].detach().cpu().item(),0, im_paths[i], og_imgs[i], label[i].detach().cpu()])
        class_wise_total[label[i]] += 1
        df_scores = df_scores.append({'type':d_type[i],'label':int(label[i]),'pred': int(prediction[i]),'score':max_softmax[i].detach().cpu().item(),'path':im_paths[i]}, ignore_index = True )
        id_total += 1
      # else:
      #   od_hist.append(max_softmax[i].detach().cpu().item())
      #   ood_list.append([max_softmax[i].detach().cpu().item(), im_paths[i], og_imgs[i]])
      #   df_scores = df_scores.append({'type':'od','label':int(label[i]),'pred': 'OOD','score':max_softmax[i].detach().cpu().item(),'path':im_paths[i]}, ignore_index = True )
      #   od_total += 1
  
  df_scores.to_csv('/home/abhiraj/DDP/CL/eval/DBACH/odin_test_scores1.csv', index=False)
  
  # print(len(id_hist))
  # print(len(od_hist))
  # max_score = max([max(id_hist),max(od_hist)])
  # min_score = min([min(id_hist),min(od_hist)])
  # max_sum = 0
  # thresh = 0
  # th_arr = np.linspace(min_score,max_score+0.001,int((max_score-min_score)/0.0001))
  
  # fpr95 = 0
  # tpr95 = 2
  # odac25 = 0
  # odac50 = 0
  # odac75 = 0
  # odac90 = 0
  # idac25 = 0
  # idac50 = 0
  # idac75 = 0
  # idac90 = 0
  
  # for i in range(len(th_arr)):
  #   id_corr = np.sum(np.array(id_hist) >= th_arr[i])
  #   id_wrng = np.sum(np.array(id_hist) < th_arr[i])
  #   od_corr = np.sum(np.array(od_hist) < th_arr[i])
  #   od_wrng = np.sum(np.array(od_hist) >= th_arr[i])
  #   tpr = od_corr/(od_corr+od_wrng)
  #   fpr = id_wrng/(id_corr+id_wrng)
  #   TPR_arr.append(tpr)
  #   FPR_arr.append(fpr)
  #   if abs(tpr - 0.95) < tpr95:
  #     fpr95 = fpr
  #     tpr95 = abs(tpr - 0.95)
  #   if i == int(0.25*(len(th_arr)-10)): ############## depends on 591
  #     odac25 = od_corr/(od_wrng+od_corr)
  #     idac25 = id_corr/(id_wrng+id_corr)
  #   if i == int(0.5*(len(th_arr)-10)): ############## depends on 591
  #     odac50 = od_corr/(od_wrng+od_corr)
  #     idac50 = id_corr/(id_wrng+id_corr)
  #   if i == int(0.75*(len(th_arr)-10)): ############## depends on 591
  #     odac75 = od_corr/(od_wrng+od_corr)
  #     idac75 = id_corr/(id_wrng+id_corr)
  #   if i == int(0.9*(len(th_arr)-10)): ############## depends on 591
  #     odac90 = od_corr/(od_wrng+od_corr)
  #     idac90 = id_corr/(id_wrng+id_corr)
  #   if ((id_corr+od_corr) >= max_sum):
  #     max_sum = id_corr+od_corr
  #     id_correct = id_corr
  #     od_correct = od_corr
  #     thresh = th_arr[i]

  # auc_score = metrics.auc(FPR_arr, TPR_arr)
  
  # for i in range(n_clusters):
  #   print(i,':',class_wise_correct[i]/class_wise_total[i])

  # print('Threshold :',thresh)
  # print('ID correct :',100*(id_correct/id_total))
  # print('OD correct :',100*(od_correct/od_total))
  # print('AUC :', auc_score)
  
  # plt.figure(figsize = (5,5))
  # plt.hist(id_hist, bins=100, density=True, cumulative=True, label='id', histtype='step', alpha=0.55, color='g')
  # plt.hist(od_hist, bins=100, density=True, cumulative=True, label='id', histtype='step', alpha=0.55, color='r')
  # plt.xlabel('score')
  # plt.title('Cummulative density of ID and OD')
  # plt.legend(loc = 'best')
  # plt.savefig('/home/abhiraj/DDP/CL/plots/DBACH/odin_OOD_cdf.png')

  # plt.figure()
  # weights_id = np.ones_like(np.array(id_hist))/len(id_hist)
  # weights_od = np.ones_like(np.array(od_hist))/len(od_hist)
  # plt.hist(id_hist, weights=weights_id, bins=100, fc=(0, 1, 0, 0.5), label='id')
  # plt.hist(od_hist, weights=weights_od, bins=100, fc=(1, 0, 0, 0.5),label='od')
  # plt.axvline(x=thresh,color='k')
  # plt.title('CLS ACC : '+ str(sum(class_wise_correct)/sum(class_wise_total)) + ' FPR95: '+str(fpr95)+'\n O25: '+str(round(odac25,4))+' O50: '+str(round(odac50,4))+' O75: '+str(round(odac75,4))+' O90: '+str(round(odac90,4))+'\n I25: '+str(round(idac25,4))+' I50: '+str(round(idac50,4))+' I75: '+str(round(idac75,4))+' I90: '+str(round(idac90,4)))
  # plt.legend(loc='best')
  # plt.savefig('/home/abhiraj/DDP/CL/plots/DBACH/odin_OOD_score.png')

  # plt.figure()
  # plt.plot(FPR_arr, TPR_arr, '-', label='auc')
  # plt.title('ROC curve')
  # plt.legend(loc='best')
  # plt.title('AUC : '+ str(auc_score))
  # plt.savefig('/home/abhiraj/DDP/CL/plots/DBACH/odin_AUROC.png')
  
  # fp = open('/home/abhiraj/DDP/CL/eval/DBACH/bach_scores_lists.txt','a')
  # fp.write(test_type+' ODIN \n') #######
  # fp.write('id_hist: '+str(id_hist)+'\n')
  # fp.write('od_hist: '+str(od_hist)+'\n')
  # fp.close()
  
  # fp = open('/home/abhiraj/DDP/CL/eval/DBACH/bach_auroc_lists.txt','a')
  # fp.write(test_type+' ODIN \n') #######
  # fp.write('FPR_arr: '+str(FPR_arr)+'\n')
  # fp.write('TPR_arr: '+str(TPR_arr)+'\n')
  # fp.close()
  
  # ood_list.sort()
    
  # fp = open('bach_od_paths.txt','a')
  # fp.write(test_type+' ODIN \n') #######
  # for dat in ood_list:
  #  fp.write(str(dat[0])+' : '+str(dat[1])+'\n')
  # fp.close()
  
  # count = 0
  # for dat in ood_list:
  #   count +=1
  #   plt.figure()
  #   plt.imshow(np.asarray(dat[2]))
  #   plt.title(str(dat[0])) 
  #   plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/DBACH/'+test_type+'_od_odin_'+str(count)+'.png') #####################
  #   if count == 5:
  #     break
  
  # ood_list.sort(reverse=True)
  
  # for dat in ood_list:
  #   count +=1
  #   plt.figure()
  #   plt.imshow(np.asarray(dat[2]))
  #   plt.title(str(dat[0]))
  #   plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/DBACH/'+test_type+'_od_odin_'+str(count)+'.png') ######################
  #   if count == 10:
  #     break
  
  # correct_list.sort()
  
  # acc_list = []
  # fp = open('bach_im_paths.txt','a')
  # fp.write(test_type+' ODIN \n')
  # for dat in correct_list:
  #  acc_list.append([dat[0], dat[1]])
  #  fp.write(str(dat[0])+' : '+str(dat[1])+' : '+dat[2]+'\n')
  # fp.close()
  
  # count = 0
  # for dat in correct_list:
  #   count +=1
  #   plt.figure()
  #   plt.imshow(np.asarray(dat[3]))
  #   plt.title(str(dat[4])+ ' : '+ str(dat[0]))
  #   plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/DBACH/'+test_type+'_odin_'+str(count)+'.png')
  #   if count == 5:
  #     break
  
  # correct_list.sort(reverse=True)
  
  # for dat in correct_list:
  #   count +=1
  #   plt.figure()
  #   plt.imshow(np.asarray(dat[3]))
  #   plt.title(str(dat[4])+ ' : '+ str(dat[0]))
  #   plt.savefig('/home/abhiraj/DDP/CL/code/Contrastive/sample_images/DBACH/'+test_type+'_odin_'+str(count)+'.png')
  #   if count == 10:
  #     break
  
  # correct_list.sort()
  # correct_arr = np.array(acc_list)
  # acc_list = []
  # data_left_list = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
  # for i in range(len(data_left_list)):
  #   temp_arr = correct_arr[int((data_left_list[i]/100)*len(correct_arr)):,1]
  #   acc_list.append(np.sum(temp_arr)/len(temp_arr))
  
  # fp = open('acc_curves.txt', 'a')
  # fp.write('ODIN: '+str(acc_list)+'\n')
  # fp.close()  
  
  # plt.figure()
  # plt.plot(data_left_list, acc_list, '-', label='accuracy')
  # plt.xlabel('% data ignored')
  # plt.ylabel('Accuracy')
  # plt.title('Accuracy vs %data left')
  # plt.legend(loc='best')
  # plt.savefig('/home/abhiraj/DDP/CL/eval/DBACH/odin_accuracy.png')
  
detect_ood_odin(full_model, testloader,1000,0.002,len(id_classes), 'no_tf')

#################################################################################################################################
def generate_gradcam_vis_softmax(model, loader, mean, std):
  model.eval()
  inv_norm = transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],std= [1/s for s in std])
  target_layer = model.layer4[-1]
  for i, sample in enumerate(loader, 0):
    img, label, d_type, im_paths, og_imgs  = sample['img'], sample['label'], sample['type'], sample['path'], sample['og_img']
    img = img.to(device)
    label = label.to(device)
    logits = model(img)
    prediction = logits.max(1, keepdim=True)[1]
    for j in range(len(label)):
      save_file_path = ''
      input_tensor = img[j].unsqueeze(0)
      rgb_img = inv_norm(img[j]).float().permute(1, 2, 0).cpu().numpy()
      cam = GradCAM(model=model, target_layer=target_layer)
      target_category = prediction[j].item()
      grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
      grayscale_cam = grayscale_cam[0, :]
      visualization = show_cam_on_image(rgb_img, grayscale_cam)
      
      fig1,ax1 = plt.subplots(1)
      fig1.subplots_adjust(left=0,right=1,bottom=0,top=1)
      ax1.imshow(rgb_img)
      ax1.axis('off')
      ax1.axis('tight')
      fig2,ax2 = plt.subplots(1)
      fig2.subplots_adjust(left=0,right=1,bottom=0,top=1)
      ax2.imshow(visualization)
      ax2.axis('off')
      ax2.axis('tight')
      bach_class = id_classes[label[j].cpu().item()]
      if d_type[j] == 'id':
        if label[j] == prediction[j]:
          save_file_path_he = '/home/abhiraj/DDP/CL/eval/DBACH/GRADCAM/SOFTMAX/softmax_id_correct_'+bach_class+'_'+id_classes[target_category]+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'_HE.png'
          save_file_path_heat = '/home/abhiraj/DDP/CL/eval/DBACH/GRADCAM/SOFTMAX/softmax_id_correct_'+bach_class+'_'+id_classes[target_category]+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'_HEAT.png'
        else:
          save_file_path_he = '/home/abhiraj/DDP/CL/eval/DBACH/GRADCAM/SOFTMAX/softmax_id_incorrect_'+bach_class+'_'+id_classes[target_category]+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'_HE.png'
          save_file_path_heat = '/home/abhiraj/DDP/CL/eval/DBACH/GRADCAM/SOFTMAX/softmax_id_incorrect_'+bach_class+'_'+id_classes[target_category]+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'_HEAT.png'
      if d_type[j] == 'od':
        save_file_path_he = '/home/abhiraj/DDP/CL/eval/DBACH/GRADCAM/SOFTMAX/softmax_od_gradcam_'+bach_class+'_'+id_classes[target_category]+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'_HE.png'
        save_file_path_heat = '/home/abhiraj/DDP/CL/eval/DBACH/GRADCAM/SOFTMAX/softmax_od_gradcam_'+bach_class+'_'+id_classes[target_category]+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'_HEAT.png'
      
      fig1.savefig(save_file_path_he, dpi = 300)
      fig2.savefig(save_file_path_heat, dpi = 300)
      
      
# generate_gradcam_vis_softmax(full_model, testloader, mean, std)


#def generate_gradcam_vis_odin(model, loader, mean, std):
#  model.eval()
#  inv_norm = transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],std= [1/s for s in std])
#  target_layer = model.layer4[-1]
#  for i, sample in enumerate(loader, 0):
#    img, label, d_type, im_paths, og_imgs  = sample['img'], sample['label'], sample['type'], sample['path'], sample['og_img']
#    img = img.to(device)
#    label = label.to(device)
#    logits = model(img)
#    _, prediction = torch.max(logits.data, 1)
#    model.train()
#    loss = criterion(logits, prediction)
#    loss.backward()
#    gradient =  torch.ge(img.grad.data, 0)
#    gradient = (gradient.float() - 0.5) * 2
#    gradient[0][0] = (gradient[0][0] )/(0.14092763) 
#    gradient[0][1] = (gradient[0][1] )/(0.15261263)
#    gradient[0][2] = (gradient[0][2] )/(0.16997081)
#    tempInputs = torch.add(img.data,  (-1)*epsilon, gradient)
#    logits = model(Variable(tempInputs))
#    logits = logits / temparature
#    prediction = logits.max(1, keepdim=True)[1]
#    model.eval()
#    for j in range(len(label)):
#      input_tensor = img[j].unsqueeze(0)
#      rgb_img = inv_norm(img[j]).float().permute(1, 2, 0).cpu().numpy()
#      cam = GradCAM(model=model, target_layer=target_layer)
#      target_category = prediction[j].item()
#      grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
#      grayscale_cam = grayscale_cam[0, :]
#      visualization = show_cam_on_image(rgb_img, grayscale_cam)
#      
#      fig = plt.figure()
#      ax1 = fig.add_subplot(1,2,1)
#      ax1.imshow(rgb_img)
#      ax2 = fig.add_subplot(1,2,2)
#      ax2.imshow(visualization)
#      
#      bach_class = im_paths[j].split('/')[-2]
#      fig.suptitle(d_type[j]+'_'+bach_class+'_'+id_classes[target_category], fontsize=12)
#      if d_type[j] == 'id':
#        if label[j] == prediction[j]:
#          save_file_path = '/home/abhiraj/DDP/CL/eval/BACH/GRADCAM/ODIN/odin_id_correct_'+bach_class+'_'+id_classes[target_category]+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'.png'
#        else:
#          save_file_path = '/home/abhiraj/DDP/CL/eval/BACH/GRADCAM/ODIN/odin_id_incorrect_'+bach_class+'_'+id_classes[target_category]+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'.png'
#      if d_type[j] == 'od':
#        save_file_path = '/home/abhiraj/DDP/CL/eval/BACH/GRADCAM/ODIN/odin_od_gradcam_'+bach_class+'_'+id_classes[target_category]+'_'+im_paths[j].split('/')[-1][:-4]+'_'+str(j)+'.png'
#      
#      fig.savefig(save_file_path)
#      
#      
#generate_gradcam_vis_odin(full_model, testloader, mean, std)