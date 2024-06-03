import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50, resnet18
from yoloLoss import yoloLoss
from dataset import yoloDataset

from visualize import Visualizer
import numpy as np

use_gpu = torch.cuda.is_available()

def train():
    # 数据集路径
    file_root = r'F:\Projects\datasets\oc\TK100\data'
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 24
    use_resnet = True

    if use_resnet:
        net = resnet50()
    else:
        net = vgg16_bn()
    # net.classifier = nn.Sequential(
    #             nn.Linear(512 * 7 * 7, 4096),
    #             nn.ReLU(True),
    #             nn.Dropout(),
    #             #nn.Linear(4096, 4096),
    #             #nn.ReLU(True),
    #             #nn.Dropout(),
    #             nn.Linear(4096, 1470),
    #         )
    #net = resnet18(pretrained=True)
    #net.fc = nn.Linear(512,1470)
    # initial Linear
    # for m in net.modules():
    #     if isinstance(m, nn.Linear):
    #         m.weight.data.normal_(0, 0.01)
    #         m.bias.data.zero_()
    print(net)
    #net.load_state_dict(torch.load('yolo.pth'))
    print('load pre-trined model')
    if use_resnet:
        # 这里使用的torch内置的resnet模型
        resnet = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        new_state_dict = resnet.state_dict()
        dd = net.state_dict()
        for k in new_state_dict.keys():
            print(k)
            # 将预训练模型中的非fc层拷贝到yolo 骨干resnet中
            # 这里可以知道如何进行迁移学习
            if k in dd.keys() and not k.startswith('fc'):
                print('yes')
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)
    else:
        # 这里也是加载vgg16预训练模型
        vgg = models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1)
        new_state_dict = vgg.state_dict()
        dd = net.state_dict()
        for k in new_state_dict.keys():
            print(k)
            if k in dd.keys() and k.startswith('features'):
                print('yes')
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)
    if True:
        # 不使用预训练模型
        net.load_state_dict(torch.load('yolo.pth'))
    print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

    criterion = yoloLoss(7,2,5,0.5,221)
    if use_gpu:
        net.cuda()

    # 切换训练模式
    net.train()
    # different learning rate
    params=[]
    # 获取网络所有的参数以及参数名字
    params_dict = dict(net.named_parameters())
    for key,value in params_dict.items():
        # todo 这里为啥要单独设置学习率，且还是*1，感觉没啥用
        if key.startswith('features'):
            params += [{'params':[value],'lr':learning_rate*1}]
        else:
            params += [{'params':[value],'lr':learning_rate}]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

    # train_dataset = yoloDataset(root=file_root,list_file=['voc12_trainval.txt','voc07_trainval.txt'],train=True,transform = [transforms.ToTensor()] )
    # 这边是写死了读取voc2012和voc2007的数据集
    train_dataset = yoloDataset(root=file_root,list_file=['tk100.txt'],train=True,catetory_path='tk100-catetory.txt',transform = [transforms.ToTensor()] )
    train_loader = DataLoader(train_dataset,batch_size=batch_size * 2,shuffle=True,num_workers=4)
    # test_dataset = yoloDataset(root=file_root,list_file='voc07_test.txt',train=False,transform = [transforms.ToTensor()] )
    test_dataset = yoloDataset(root=file_root,list_file='tk100-test.txt',train=False,catetory_path='tk100-catetory.txt',transform = [transforms.ToTensor()] )
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    print('the dataset has %d images' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))
    logfile = open('log.txt', 'w')

    num_iter = 0
    vis = Visualizer(env='xiong')
    best_test_loss = 3

    for epoch in range(38, num_epochs):
        net.train()
        # 这边在手动调整学习率？估计是进行超参数调整
        # if epoch == 1:
        #     learning_rate = 0.0005
        # if epoch == 2:
        #     learning_rate = 0.00075
        # if epoch == 3:
        #     learning_rate = 0.001
        if epoch >= 30:
            learning_rate=0.0001
        if epoch >= 40:
            learning_rate=0.00001
        # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))
        
        total_loss = 0.
        
        for i,(images,target) in enumerate(train_loader):
            # 貌似新版本不需要这里用Variable
            images = Variable(images)
            target = Variable(target)
            if use_gpu:
                images,target = images.cuda(),target.cuda()
            
            pred = net(images)
            loss = criterion(pred,target)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 5 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
                %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
                num_iter += 1
                vis.plot_train_val(loss_train=total_loss/(i+1))

        #validation
        validation_loss = 0.0
        net.eval()
        for i,(images,target) in enumerate(test_loader):
            with torch.no_grad():
                images = Variable(images)
                target = Variable(target)
            if use_gpu:
                images,target = images.cuda(),target.cuda()
            
            pred = net(images)
            loss = criterion(pred,target)
            validation_loss += loss.item()
        validation_loss /= len(test_loader)
        vis.plot_train_val(loss_val=validation_loss)
        
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)

            
            torch.save(net.state_dict(),'best.pth')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
        logfile.flush()      
        torch.save(net.state_dict(),'yolo.pth')
   

if __name__ == "__main__":
    train()
    print("Done!")