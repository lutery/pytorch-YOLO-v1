#encoding:utf-8
#
#created by xiongzihua
#
'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
'''
import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt

class yoloDataset(data.Dataset):
    image_size = 448
    def __init__(self,root,list_file,train,transform):
        '''
        root: 数据集根目录
        list_file: 数据集列表文件（存储图片文件以及图片目标位置的信息）
        train: 是否为训练数据集
        transform: 图片预处理
        '''

        print('data init')
        self.root=root
        self.train = train
        self.transform=transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123,117,104)#RGB

        if isinstance(list_file, list):
            # 这里将多个数据集的图片对象数据信息合并到一个文件中
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines  = f.readlines()

        # 遍历所有图片路径以及对象信息（lx,ly,rx,ry,类型）
        # self.fnames 存储所有图片路径
        # self.boxes 存储所有图片的目标位置信息
        # self.labels 存储所有图片的目标类别信息
        # self.num_samples 存储总共的目标数量
        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            # 有多少个目标
            num_boxes = (len(splited) - 1) // 5
            box=[]
            label=[]
            for i in range(num_boxes):
                x = float(splited[1+5*i])
                y = float(splited[2+5*i])
                x2 = float(splited[3+5*i])
                y2 = float(splited[4+5*i])
                c = splited[5+5*i]
                box.append([x,y,x2,y2])
                label.append(int(c)+1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root+fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train:
            #img = self.random_bright(img)
            img, boxes = self.random_flip(img, boxes)
            img,boxes = self.randomScale(img,boxes)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img,boxes,labels = self.randomShift(img,boxes,labels)
            img,boxes,labels = self.randomCrop(img,boxes,labels)
        # #debug
        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = self.BGR2RGB(img)
        # pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))
        # cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=1)
        # plt.figure()
        
        # # cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)
        # plt.imshow(img_show)
        # plt.show()
        # #debug
        h,w,_ = img.shape
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img = self.BGR2RGB(img) #because pytorch pretrained model use RGB
        img = self.subMean(img,self.mean) #减去均值
        img = cv2.resize(img,(self.image_size,self.image_size))
        target = self.encoder(boxes,labels)# 7x7x30
        for t in self.transform:
            img = t(img)

        return img,target
    def __len__(self):
        return self.num_samples

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30

        从这里可以看出，yolov1每个grid cell仅预测两个目标
        '''
        # 网格的数量，即图像被划分成的网格数 todo 这里的14是否有问题，不是7x7网格吗
        grid_num = 14
        # 30 = [2 * [中心点，长宽，置信度], 20个分类的one-hot编码] 创建一个全零张量，用于存储编码后的目标信息
        target = torch.zeros((grid_num,grid_num,30))
        # 网格尺寸归一化后，每个小网格的尺寸
        cell_size = 1./grid_num
        # 计算真实目标的归一化长宽尺寸
        wh = boxes[:,2:]-boxes[:,:2]
        # 计算真实目标的归一化中心点坐标
        cxcy = (boxes[:,2:]+boxes[:,:2])/2
        # 遍历当前图片中的所有目标
        for i in range(cxcy.size()[0]):
            # 计算目标框中心点所在的网格索引，也就是第几个网格
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1
            # 在目标框中心所在的网格的第 5 个通道位置标记为目标存在，也就是置信度为1
            target[int(ij[1]),int(ij[0]),4] = 1
            # 在目标框中心所在的网格的第 10 个通道位置标记为目标存在，也就是置信度为1 todo 两个置信度都为1，会有问题吗
            target[int(ij[1]),int(ij[0]),9] = 1
            # 标记目标的类别
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
            # 计算目标框的在归一化后的网格坐标
            xy = ij*cell_size #匹配到的网格的左上角相对坐标
            #  计算目标框中心点相对于所在网格左上角的偏移量。
            delta_xy = (cxcy_sample -xy)/cell_size
            # 存储目标框的宽度和高度
            target[int(ij[1]),int(ij[0]),2:4] = wh[i]
            # 存储目标框中心点相对于所在网格左上角的偏移量
            target[int(ij[1]),int(ij[0]),:2] = delta_xy
            # 存储目标框的宽度和高度
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]
            # 存储目标框中心点相对于所在网格左上角的偏移量
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy
        return target
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def randomScale(self,bgr,boxes):
        #固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels




    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    file_root = '/home/xzh/data/VOCdevkit/VOC2012/allimgs/'
    train_dataset = yoloDataset(root=file_root,list_file='voc12_trainval.txt',train=True,transform = [transforms.ToTensor()] )
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    for i in range(100):
        img,target = next(train_iter)
        print(img,target)


if __name__ == '__main__':
    main()


