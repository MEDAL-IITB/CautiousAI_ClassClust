import os
import numpy as np
import cv2
import pandas as pd
import pydicom as dicom
import pydicom.data
import random
import shutil

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
          dcmimg = dicom.dcmread(os.path.join(inbreast_dir_path,dcm_files[k]))
          img = dcmimg.pixel_array
          img = cv2.resize(img, (100,100))
          img = (img/4096)*255
          cv2.imwrite(os.path.join("/home/Drive/abhiraj/data/mias_gen/ood/birads/{}/".format(i+1),dcm_files[k][:-4]+".jpg"), img)
          
base_dir = "/home/Drive/abhiraj/data/mias/"
img_dir = "/home/Drive/abhiraj/data/mias_gen/"
info_f = open(os.path.join(base_dir,'info.txt'))

fnames_N = []
fnames_B = []
fnames_M = []
for l in info_f.readlines()[1:]:
  s_l = l.split(' ')
  if 'NORM' in s_l[2]:
    fnames_N.append(s_l[0])
  elif 'B' in s_l[3]:
    fnames_B.append(s_l[0])
  elif 'M' in s_l[3]:
    fnames_M.append(s_l[0])
  else:
    pass
    
  if '144' not in s_l[0]:
    img = cv2.imread(os.path.join("/home/Drive/abhiraj/data/mias/images/",s_l[0]+".pgm"))
    img = cv2.resize(img, (100,100))
    rows, cols,color = img.shape
    for angle in range(0,360,8):
      M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)    #Rotate 0 degree
      img_rotated = cv2.warpAffine(img, M, (cols, rows))
      if 'NORM' in s_l[2]:
        cv2.imwrite(os.path.join("/home/Drive/abhiraj/data/mias_gen/train/N/",s_l[0]+"_"+str(angle)+".jpg"), img_rotated)
      elif 'B' in s_l[3]:
        cv2.imwrite(os.path.join("/home/Drive/abhiraj/data/mias_gen/train/B/",s_l[0]+"_"+str(angle)+".jpg"), img_rotated)
      elif 'M' in s_l[3]:
        cv2.imwrite(os.path.join("/home/Drive/abhiraj/data/mias_gen/train/M/",s_l[0]+"_"+str(angle)+".jpg"), img_rotated)

         
list_B = os.listdir("/home/Drive/abhiraj/data/mias_gen/train/B")
list_M = os.listdir("/home/Drive/abhiraj/data/mias_gen/train/M")
list_N = os.listdir("/home/Drive/abhiraj/data/mias_gen/train/N")

print('B:',len(fnames_B),'/',len(list_B))
print('M:',len(fnames_M),'/',len(list_M))
print('N:',len(fnames_N),'/',len(list_N))

B_test = 450
M_test = 360
N_test = 1395

#B_test_list = random.sample(fnames_B, B_test)
#for f in B_test_list:
#  fnames_B.remove(f)
#M_test_list = random.sample(fnames_M, M_test)
#for f in M_test_list:
#  fnames_M.remove(f)
#N_test_list = random.sample(fnames_N, N_test)
#for f in N_test_list:
#  fnames_N.remove(f)

B_test_list = random.sample(list_B, B_test)
for f in B_test_list:
  list_B.remove(f)
M_test_list = random.sample(list_M, M_test)
for f in M_test_list:
  list_M.remove(f)
N_test_list = random.sample(list_N, N_test)
for f in N_test_list:
  list_N.remove(f)

#for f in B_test_list:
#  for aug_f in list_B:
#    if f in aug_f:
#      shutil.move("/home/Drive/abhiraj/data/mias_gen/train/B/"+aug_f, "/home/Drive/abhiraj/data/mias_gen/test/B/"+aug_f)
#for f in M_test_list:
#  for aug_f in list_M:
#    if f in aug_f:
#      shutil.move("/home/Drive/abhiraj/data/mias_gen/train/M/"+aug_f, "/home/Drive/abhiraj/data/mias_gen/test/M/"+aug_f)
#for f in N_test_list:
#  for aug_f in list_N:
#    if f in aug_f:
#      shutil.move("/home/Drive/abhiraj/data/mias_gen/train/N/"+aug_f, "/home/Drive/abhiraj/data/mias_gen/test/N/"+aug_f) 

for f in B_test_list:
  shutil.move("/home/Drive/abhiraj/data/mias_gen/train/B/"+f, "/home/Drive/abhiraj/data/mias_gen/test/B/"+f)
for f in M_test_list:
  shutil.move("/home/Drive/abhiraj/data/mias_gen/train/M/"+f, "/home/Drive/abhiraj/data/mias_gen/test/M/"+f)
for f in N_test_list:
  shutil.move("/home/Drive/abhiraj/data/mias_gen/train/N/"+f, "/home/Drive/abhiraj/data/mias_gen/test/N/"+f) 

B_test_list = os.listdir("/home/Drive/abhiraj/data/mias_gen/test/B")
M_test_list = os.listdir("/home/Drive/abhiraj/data/mias_gen/test/M")
N_test_list = os.listdir("/home/Drive/abhiraj/data/mias_gen/test/N")
  
print('train B:',len(fnames_B),'/',len(list_B))
print('train M:',len(fnames_M),'/',len(list_M))
print('train N:',len(fnames_N),'/',len(list_N))
print('test B:',len(B_test_list))
print('test M:',len(M_test_list))
print('test N:',len(N_test_list)) 