#encoding:utf-8
#
#created by xiongzihua 2017.12.26
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class yoloLoss(nn.Module):
    '''
    创建yolo损失
    '''
    def __init__(self,S,B,l_coord,l_noobj, category_count):
        '''
        param S： todo
        param B: todo
        param l_coord： todo
        param l_noobj: todo
        param category_count: 分类数
        '''
        super(yoloLoss,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.category_count = category_count + 10

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    def forward(self,pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+self.category_count) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,self.category_count)
        '''
        N = pred_tensor.size()[0] # 获取批次大小
        # 找到第一个预测框的置信度大于0，并返回一个bool 矩阵，标识为True表示置信度大于0
        coo_mask = target_tensor[:,:,:,4] > 0 # 获取target中所有置信度大于0的boolean矩阵 todo 为什么这里的仅获取第5个维度的置信度大于0
# 找到第一个预测框的置信度等于0，并返回一个bool 矩阵，标识为True表示置信度等于0
        noo_mask = target_tensor[:,:,:,4] == 0 # 获取target中所有置信度等于0的boolean矩阵
        # 将得到的bool矩阵维度扩展为target_tensor的维度，todo 这里应该是将最后一个维度扩展为 [2 * [中心点，长宽，置信度], 分类数的one-hot编码]
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        # 从pred_tensor中取出target中所有置信度大于0的预测值
        # view 展平维度，coo_pred的维度编程(-1, [2 * [中心点，长宽，置信度], 分类数的one-hot编码])
# 根据真实标签的预测框的位置提取预测tensorf中对应的预测框 todo 作用
        coo_pred = pred_tensor[coo_mask].view(-1,self.category_count)
        # 获取所有置信度大于0的预测框（有两个预测框），todo 这里的分类预测是决定两个预测的类型吗？
# 提取根据真实标签的预测框的位置提取预测tensorf中对应的预测框的位置信息
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        # 获取每个置信度大于0的预测分类
# 根据真实标签的预测框的位置提取预测tensorf中对应的预测框的类别信息
        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]
        
        # 按照相同的方法从target中获取 预测框（两个）以及预测分类
        coo_target = target_tensor[coo_mask].view(-1,self.category_count)
        # box_target是目标预测框的中心点、长宽、置信度,并且进行展平，使得第二维度的含义编程每个预测框的信息，第一维度的信息是所有预测框
        # 维度信息[batch, gw, gh, 2 * 5], [batch * gw * gh * 2, 5]
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        # class_target是目标预测框的分类，
        # 维度信息[batch, gw, gh, categroy_count - 10]
        class_target = coo_target[:,10:]

        # compute not contain obj loss
# 从预测的tensor和真实tensor中提取目标的预测框的位置信息和类别信息
        # 获取预测pred中所有置信度应该等于0的预测值
        noo_pred = pred_tensor[noo_mask].view(-1,self.category_count)
        # 获取真实target中所有置信度等于0的预测值
        noo_target = target_tensor[noo_mask].view(-1,self.category_count)
        # 创建一个size和noo_pred一样的boolean矩阵，并将矩阵的默认值设置为False todo 这个是在做什么？
        noo_pred_mask = torch.cuda.BoolTensor(noo_pred.size())
        noo_pred_mask.zero_()
# 将非目标物体的置信度设置为1
        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1 # 这里的将4 和 9的位置设置为1，是为了下一步将对应位置的预测置信度以及目标置信度的数据取出来
# todo 这边实在提取非目标物体的置信度信息（预测tensor和真实tensor的）吗？
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
# 这里出现了第一个损失，也就是非目标物体的置信度损失 todo 确认yolov1中是否存在这个损失
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,reduction='sum') # 计算第一个损失：不是目标位置的预测框置信度设置为0

        #compute contain obj loss 这里开始计算真实目标的置信度
        #todo 这两的两个mask的作用是什么
        coo_response_mask = torch.cuda.BoolTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.BoolTensor(box_target.size())
        coo_not_response_mask.zero_()
        # todo 这里的iou的作用是什么？
        box_target_iou = torch.zeros(box_target.size()).cuda()
        # 这里开始遍历所有grid cell，因为每个cell有两个预测框，所以这里range最后一个参数是2
        for i in range(0,box_target.size()[0],2): #choose the best iou box
            # 获取两个预测框[2, 5]
            box1 = box_pred[i:i+2]
        #创建一个和box1一样大小的tensor
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            # 将box1的中心点坐标转换为所处网格的内部 todo 这里除以14是否有问题？需要和数据集中的dateset中14对比
            # todo 确认target中的目标是否一致
            box1_xyxy[:,:2] = box1[:,:2]/14. -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2]/14. -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()
        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],reduction='sum') + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),reduction='sum')
        #2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0
        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],reduction='sum')
        
        #I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],reduction='sum')

        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,reduction='sum')

        return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N




