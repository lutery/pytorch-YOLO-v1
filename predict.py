#encoding:utf-8
#
#created by xiongzihua
#
import os
import torch
from torch.autograd import Variable
import torch.nn as nn

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np

VOC_CLASSES = ('i1',
'i10',
'i11',
'i12',
'i13',
'i14',
'i15',
'i2',
'i3',
'i4',
'i5',
'il100',
'il110',
'il50',
'il60',
'il70',
'il80',
'il90',
'io',
'ip',
'p1',
'p10',
'p11',
'p12',
'p13',
'p14',
'p15',
'p16',
'p17',
'p18',
'p19',
'p2',
'p20',
'p21',
'p22',
'p23',
'p24',
'p25',
'p26',
'p27',
'p28',
'p3',
'p4',
'p5',
'p6',
'p7',
'p8',
'p9',
'pa10',
'pa12',
'pa13',
'pa14',
'pa8',
'pb',
'pc',
'pg',
'ph1.5',
'ph2',
'ph2.1',
'ph2.2',
'ph2.4',
'ph2.5',
'ph2.8',
'ph2.9',
'ph3',
'ph3.2',
'ph3.5',
'ph3.8',
'ph4',
'ph4.2',
'ph4.3',
'ph4.5',
'ph4.8',
'ph5',
'ph5.3',
'ph5.5',
'pl10',
'pl100',
'pl110',
'pl120',
'pl15',
'pl20',
'pl25',
'pl30',
'pl35',
'pl40',
'pl5',
'pl50',
'pl60',
'pl65',
'pl70',
'pl80',
'pl90',
'pm10',
'pm13',
'pm15',
'pm1.5',
'pm2',
'pm20',
'pm25',
'pm30',
'pm35',
'pm40',
'pm46',
'pm5',
'pm50',
'pm55',
'pm8',
'pn',
'pne',
'po',
'pr10',
'pr100',
'pr20',
'pr30',
'pr40',
'pr45',
'pr50',
'pr60',
'pr70',
'pr80',
'ps',
'pw2',
'pw2.5',
'pw3',
'pw3.2',
'pw3.5',
'pw4',
'pw4.2',
'pw4.5',
'w1',
'w10',
'w12',
'w13',
'w16',
'w18',
'w20',
'w21',
'w22',
'w24',
'w28',
'w3',
'w30',
'w31',
'w32',
'w34',
'w35',
'w37',
'w38',
'w41',
'w42',
'w43',
'w44',
'w45',
'w46',
'w47',
'w48',
'w49',
'w5',
'w50',
'w55',
'w56',
'w57',
'w58',
'w59',
'w60',
'w62',
'w63',
'w66',
'w8',
'wo',
'i6',
'i7',
'i8',
'i9',
'ilx',
'p29',
'w29',
'w33',
'w36',
'w39',
'w4',
'w40',
'w51',
'w52',
'w53',
'w54',
'w6',
'w61',
'w64',
'w65',
'w67',
'w7',
'w9',
'pax',
'pd',
'pe',
'phx',
'plx',
'pmx',
'pnl',
'prx',
'pwx',
'w11',
'w14',
'w15',
'w17',
'w19',
'w2',
'w23',
'w25',
'w26',
'w27',
'pl0',
'pl4',
'pl3',
'pm2.5',
'ph4.4',
'pn40',
'ph3.3',
'ph2.6')

Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]

def decoder(pred):
    '''
    解码网络预测的结果
    pred (tensor) 1x7x7x30 todo 貌似这里的网络推理得到的是1x14x14x(10 + 分类)
    return (tensor) box[[x1,y1,x2,y2]] label[...] 返回预测框的坐标，类别，置信度
    '''
    grid_num = 14
    boxes=[] # 保存检测到的预测框的坐标信息（坐标形式时左上角和右下角的形式）
    cls_indexs=[] # 预测的类别的索引
    probs = [] # 类别的置信度
    cell_size = 1./grid_num # 这里的1应该表示整个grid cell的总边长是1 归一化后的 todo 这里计算的单个grid cell的尺寸
    # pred = pred.data # 
    pred = pred.squeeze(0) #7x7x30 去掉最低的维度
    contain1 = pred[:,:,4].unsqueeze(2) # 提取所有grid cell第一个预测框的置信度
    contain2 = pred[:,:,9].unsqueeze(2) # 提取所有grid cell第二个预测框的置信度
    contain = torch.cat((contain1,contain2),2) # 结合所有grid cell 两个预测框的置信度
    mask1 = contain > 0.1 #大于阈值 拿到所有置信度大于0.1的预测框的位置
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9 todo 这里是在找到置信度最大的概率的那个预测框的位置 todo 为啥要这么做，是因为有可能置信度的大小小于0.1吗
    mask = (mask1+mask2).gt(0)# 合并两个mask所找到的预测框的位置信息 todo 运行代码看下效果
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    # 遍历所有的预测框
    # todo 难道这里的每个cell的两个预测框可以分别预测不同的物体吗？
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    #print(i,j,b)
                    # 拿到置信度符合要求的预测框
                    box = pred[i,j,b*5:b*5+4]
                    # 拿到预测框的置信度
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    # 拿到基于当前cell所在的位置转换为基于整个grid cell所在的坐标位置
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                    # 根据预测框的位置转换为基于整个grid cell所在的坐标位置 之所以要这么转换，是因为预测框的位置是相对于当前cell的位置的
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    # 将预测框的位置坐标（中心点坐标+预测框尺寸）转换为左上角右下角的坐标形式
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    # 拿到预测框的类别以及概率
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    # 仅处理是否有目标的概率乘以预测物品的类别的概率大于0.1的预测框
                    if float((contain_prob*max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob*max_prob)
    if len(boxes) ==0:
        # 为了保持兼容性，所以当预测框的数量为0的时候，返回一个空的tensor
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        # 否则将预测框转换为tensor
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.cat([x.unsqueeze(0) for x in cls_indexs],0) #(n,)
    # 对于预测框进行nms操作
    keep = nms(boxes,probs)
    # 从nms返回的预测框索引中，提取对应的预测框、预测类型、置信度
    return boxes[keep],cls_indexs[keep],probs[keep]

def nms(bboxes,scores,threshold=0.5):
    '''
    对所有的预测框进行nms操作，避免重复检测到同一个物体

    bboxes(tensor) [N,4]
    scores(tensor) [N,]

    reture:返回最后保留预测框的索引，索引的维度是和boxes一致
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    # 找到所有的预测框的面积
    areas = (x2-x1) * (y2-y1)

    #根据置信度的大小降序排序
    # order代表排序后的索引，维度和原先的一样，用来查询原先的数据在排序后的位置
    _,order = scores.sort(0,descending=True)
    keep = []  # 存储最后保留的预测框的索引
    while order.numel() > 0:
        if order.numel() == 1:
            # 如果只有一个预测框那就没有必要进行nms操作了直接返回
            keep.append(order.item())
            break
        
        # 处理现在最高置信度的预测框
        i = order[0].item()
        keep.append(i)

        # 将剩余的预测框和当前最高置信度的预测框进行比较，如果有重叠的部分就去掉
        # 这里的去掉逻辑采用的是使用clamp函数，进行对比，例如：如果剩余的部分的左上角的坐标有小于x1[i]或者小于y1[i]或者大于x2[i]或者大于y2[i]的
        # 部分就去掉，使用x1 y1 x2 y2替换剩余预测框的坐标。
        # 并将替换得到的坐标返回到xx1 yy1 xx2 yy2
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        # 计算剩余预测框中的重叠部分的面积
        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        # 计算剩余预测框与当前预测框的交际和并集之间的重叠比例
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 找到重叠比例小于阈值的预测框的索引
        # 如果没有说明剩余预测框的所有预测框都与当前预测框的重叠比例较大，说明剩余预测框都是预测同一个物体，则不需要继续进行nms操作
        # 直接返回
        # 如果有则继续进行bms操作
        # 并且通过ovr<=threshold得到重叠区域小于阈值的索引
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        # 根据索引提取剩余重叠区域小的预测框继续进行nms操作
        order = order[ids+1]
    # 返回最后保留的预测框的索引
    return torch.LongTensor(keep)
#
#start predict one image
#
def predict_gpu(model,image_name,root_path=''):

    result = []
    image = cv2.imread(os.path.join(root_path, image_name))
    h,w,_ = image.shape
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mean = (123,117,104)#RGB
    img = img - np.array(mean,dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    img = Variable(img[None,:,:,:],volatile=True)
    img = img.cuda()

    pred = model(img) #1x7x7x30
    pred = pred.cpu()
    boxes,cls_indexs,probs =  decoder(pred)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],image_name,prob])
    return result


def predict_gpu_opencvimg(model,cv_image):
    '''
    通过模型预测一张图片的目标

    param model: yolov1模型
    cv_Image: 待预测的图片

    return todo
    '''

    result = []
    image = cv_image
    h,w,_ = image.shape
    # 将原始图片转换为448*448的图片 ，todo 训练的时候图片是否是转换为448*448的图片
    img = cv2.resize(image,(448,448))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mean = (123,117,104)#RGB
    # 归一化图片，但是貌似缺少除法操作，todo 确认训练的时候是否也是这样
    img = img - np.array(mean,dtype=np.float32)

    # todo 合并为transforms里面
    transform = transforms.Compose([transforms.ToTensor(),])
    img = transform(img)
    with torch.no_grad():
        # 将图片扩展为为批次大小为1的数据
        img = img[None,:,:,:]
        img = img.cuda()

        # 进行推理，todo 但是貌似整个网络推理的结果是1x14x14x（10+分类）吧？
        pred = model(img) #1x7x7x30
        pred = pred.cpu()
        boxes,cls_indexs,probs =  decoder(pred)

        # 遍历每个预测框，将预测框的位置（相对于图片的位置）、类别、置信度保存到result中
        for i,box in enumerate(boxes):
            x1 = int(box[0]*w)
            x2 = int(box[2]*w)
            y1 = int(box[1]*h)
            y2 = int(box[3]*h)
            cls_index = cls_indexs[i]
            cls_index = int(cls_index) # convert LongTensor to int
            prob = probs[i]
            prob = float(prob)
            result.append([(x1,y1),(x2,y2),VOC_CLASSES[cls_index],prob])
        return result
        
        



if __name__ == '__main__':
    model = resnet50()
    print('load model...')
    model.load_state_dict(torch.load('best.pth'))
    model.eval()
    model.cuda()
    image_name = r"F:\Projects\datasets\oc\TK100\data\test\204.jpg"
    image = cv2.imread(image_name)
    print('predicting...')
    result = predict_gpu(model,image_name)
    for left_up,right_bottom,class_name,_,prob in result:
        color = Color[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image,left_up,right_bottom,color,2)
        label = class_name+str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite('result.jpg',image)




