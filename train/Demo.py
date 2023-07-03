from scipy.io import loadmat
from torch.nn import functional as F
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from libtiff import TIFF
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import os
import torch.optim as optim
from preprocess.preprocess import split
from model.attention_new import ViTLite
import matplotlib.pyplot as plt
import random
from Loss import NCC
import math
from sklearn import metrics
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def kappa(confusion_matrix, k):
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe  = float(ysum*xsum)/np.sum(dataMat)**2
    OA = float(P0/np.sum(dataMat)*1.0)
    cohens_coefficient = float((OA-Pe)/(1-Pe))
    return cohens_coefficient
# 设置随机数种子
# 1.定义网络超参数
setup_seed(3407)
EPOCH = 30  # 训练多少轮次
BATCH_SIZE =128# 每次喂给的数据量
LR = 0.0012 # 学习率
Train_Rate = 0.1  # 将训练集和测试集按比例分开
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 是否用GPU环视cpu训练

if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)
gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)

# 读取图片——ms4
# datapath present the path of the data
ms4_tif = TIFF.open('datapath/ms4.tif', mode='r')
ms4_np = ms4_tif.read_image()
print('原始ms4图的形状：', np.shape(ms4_np))
pan_tif = TIFF.open('datapath/pan.tif', mode='r')
pan_np = pan_tif.read_image()
print('原始pan图的形状：', np.shape(pan_np))
pan_np =split(pan_np,2)
pan_np=np.transpose(pan_np,(1,2,0))
pan_np= cv2.pyrDown(pan_np)
print('2-split后pan图的形状：',np.shape(pan_np))
label_np = loadmat("datapath/label.mat")
label_np=label_np['label']
print('label数组形状：', np.shape(label_np))

# ms4与pan图补零  (给图片加边框）
Ms4_patch_size = 32 # ms4截块的边长
Interpolation = cv2.BORDER_REFLECT_101
# cv2.BORDER_REPLICATE： 进行复制的补零操作;
# cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
# cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
# cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdefgh|abcdefgh|abcdefg;

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的ms4图的形状：', np.shape(ms4_np))

Pan_patch_size = Ms4_patch_size   # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 1), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 1), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的pan图的形状：', np.shape(pan_np))

# 按类别比例拆分数据集
# label_np=label_np.astype(np.uint8)
label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255
label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
print('类标：', label_element)
print('各类样本数：', element_count)
Categories_Number = len(label_element) - 1  # 数据的类别数
print('标注的类别数：', Categories_Number)
label_row, label_column = np.shape(label_np)  # 获取标签图的行、列

'''归一化图片'''
def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

ground_xy = np.array([[]] * Categories_Number).tolist()   # [[],[],[],[],[],[],[]]  7个类别
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)  # [800*830, 2] 二维数组

count = 0
for row in range(label_row):  
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])     
for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    np.random.shuffle(shuffle_array)

    ground_xy[categories] = ground_xy[categories][shuffle_array]
shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    # print('aaa', categories_number)
    for i in range(categories_number):
        if i < int(categories_number * Train_Rate):
            ground_xy_train.append(ground_xy[categories][i])
        else:
            ground_xy_test.append(ground_xy[categories][i])
    label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
    label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)


shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

print('训练样本数：', len(label_train))
print('测试样本数：', len(label_test))

ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
ms4 = np.array(ms4).transpose((2, 0, 1))  
pan= np.array(pan).transpose((2, 0, 1))

ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)
label_fu,label_pan,label_ms=[],[],[]
class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size
    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int( x_ms)      
        y_pan = int( y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]
        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]
        locate_xy = self.gt_xy[index]
        target = self.train_labels[index]
        return image_ms, image_pan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)

class MyData1(Dataset):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(x_ms)  
        y_pan = int(y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]
        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)

train_data = MyData(ms4, pan, label_train,ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test ,ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

model = ViTLite(img_size=Ms4_patch_size,num_heads=2, mlp_ratio=1, embedding_dim=64, positional_embedding='learnable', num_classes=11).cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
def train_model(model, train_loader, optimizer, epoch,epochs):
    model.train()
    correct = 0.0

    train_bar=tqdm(train_loader)
    for step, (ms, pan, label, _) in enumerate(train_bar):
        ms, pan, label= ms.cuda(), pan.cuda(), label.cuda()
        optimizer.zero_grad()
        output,xo,yo,x2,y2= model(ms,pan)
        pred_train = output.max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        cosinloss = nn.CosineEmbeddingLoss(margin=0.2)
        loss2=cosinloss(xo,yo,torch.ones(len(label)).cuda())+cosinloss(x2,xo,torch.zeros(len(label)).cuda())+cosinloss(y2,yo,torch.zeros(len(label)).cuda())
        loss=F.cross_entropy(output, label.long())+loss2
        loss.backward()
        optimizer.step()
        train_bar.desc=f"train epoch [{epoch}/{epochs}] loss={loss:.3f} loss2={loss2:.3f}"
    print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))

test_losses=[]
import time
for epoch in range(1, EPOCH+1):
    train_model(model,  train_loader, optimizer, epoch,EPOCH)
    if epoch==EPOCH:
        start = time.time()
        test_loss=test_model(model,  test_loader)
        end = time.time()
        print('time:{0:f}'.format(end-start))
    scheduler.step()
torch.save(model, 'model.pkl')
