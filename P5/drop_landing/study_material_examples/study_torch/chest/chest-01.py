# 1 加载库
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import copy
from torch.optim import lr_scheduler


def tb_writer():
    timestr=time.strftime("%Y%m%d_%H%M%S")
    writer=SummaryWriter("logdir/"+timestr)
    return writer


# 2 定义一个方法：显示图片
def image_show(inp, title=None):
    plt.figure(figsize=(14, 3))
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self,size=None):
        super().__init__()
        size= size or (1,1) # default keral size 
        self.pool_one=nn.AdaptiveAvgPool2d(size)
        self.pool_two=nn.AdaptiveAvgPool2d(size)

    def forward(self, x):
        # cat two pool layers
        return torch.cat([self.pool_one(x), self.pool_two(x)],1)

def visualization_model(model, num=6):
    was_training = model.training
    model.eval()
    images_sof_far =0
    with torch.no_grad():
        for i, (datas, targets) in enumerate(dataloaders['val']):
            datas, targets= datas.to(device), targets.to(device)
            output=model(datas)
            _,preds=troch.max(output,dim=1)
            for j in range(datas.size()[0]):
                images_so_far += 1 # 累计图片数量
                ax = plt.subplot(num_images // 2, 2, images_so_far) # 显示图片
                ax.axis('off') # 关闭坐标轴
                ax.set_title('predicted:{}'.format(class_names[preds[j]]))
                imshow(datas.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)



def get_model():
    # obatin a trained model using torchvision.models
    model_pre=models.resnet50(pretrained=True)

    # frezze parameters in the trained models
    for param in model_pre.parameters():
        param.requires_grad = False

    #fine-tune mode:
    ## modify pooling layers
    model_pre.avgpool = AdaptiveConcatPool2d()
    ## modify full connection layer
    model_pre.fc=nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(4096),
        nn.Dropout(p=0.5),
        nn.Linear(4096,512),
        nn.ReLU(),
        nn.BatchNorm1d(512), # normalized
        nn.Dropout(p=0.5),
        nn.Linear(512,2),
        nn.LogSoftmax(dim=1)
            )
    return model_pre

def train_epochs(model,device,dataloaders,criterion,optimizer,num_epochs,writer):

    start=time.time()

    best_score=np.inf
    

    for epoch in num_epochs:
        train_loss=train(model, device, dataloaders['train'],criterion, optimizer,epoch, writer)
        test_loss, accuracy = test(model, device, dataloaders['test'],criterion,epoch, writer)

    if test_loss < best_score:
        best_score=test_loss
        torch.save(model.state_dict(),model_path)
        
    print("{0:>20}|{1:>20}|{2:>20}|{3:>20.2f}|".format(epoch,train_loss,test_loss,accuracy))

    writer.flush

    time_all=time.tiem()-start

    print("Training complete in {:.2f}m {:.2f}s".format(time_all//60,time_all%60))

# Define a training function
# Inputs: model, deive, train_loader, criterion, optimizer, epoch,
def train(model,device,train_loader, criterion, optimizer, epoch, writer):
    model.train()# train function of the base class

    total_loss=0.0
    # training epoch
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()# 初始化参数
        output=model(data)# forward
        loss=criterion(output,target)# loss
        loss.backward()# backward
        optimizer.step() # update parameters
        total_loss+=loss.item()

    writer.add_scalar("Train Loss", total_loss/len(train_loader), epoch)
    writer.flush()
    return total_loss/len(train_loader)


def test(model, device, test_loader,criterion, epoch, writer):
    model.eval() #evaluate model for test

    total_loss=0.0
    correct=0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output=model(data)
            loss=criterion(output,target)
            total_loss=+loss.item()
            _,preds=torch.max(output,dim=1)

            correct+=torch.sum(preds==target)

        total_loss/=len(test_loader)
        accuracy= correct/len(test_loader)
        writer.add_scalar("Test Loss", total_loss, epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)
        writer.flush()

        print("test loss : {:.4f}, Accuracy : {:4f}".format(total_loss, accuracy))
        return total_loss, accuracy





def main():
    # 3 定义超参数
    model_path='model.pth'
    batch_size = 8 # 每批处理的数据数量
    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

    # 4 图片转换
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize(300),
                transforms.RandomResizedCrop(300),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        'test': transforms.Compose([
                transforms.Resize(size=300),
                transforms.CenterCrop(size=256),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]),
        'val':
            transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
    }

    # 5 操作数据集
    # 5.1 数据集路径
    #data_path = "/Users/tommy/Desktop/9_datasets/chest_xray"
    data_path = "./chest_xray"
    # 5.2 加载数据集train 和 val
    image_datasets = { x : datasets.ImageFolder(os.path.join(data_path, x),
                                                data_transforms[x]) for x in ['train', 'val','test']}
    # 5.3 为数据集创建一个迭代器，读取数据
    dataloaders = {x : DataLoader(image_datasets[x], shuffle=True,
                                  batch_size=batch_size) for x in ['train', 'val','test']}

    # 5.3 训练集和验证集的大小（图片的数量）
    data_sizes = {x : len(image_datasets[x]) for x in ['train', 'val','test']}

    # 5.4 获取标签的类别名称:  NORMAL 正常 --- PNEUMONIA 感染
    target_names = image_datasets['train'].classes

    # 6 显示一个batch_size的图片（8张图片）
    # 6.1 读取8张图片
    datas, targets = next(iter(dataloaders['train']))
    # 6.2 将若干张图片拼成一幅图像
    out = make_grid(datas, nrow=4, padding=10)
    # 6.3 显示图片
    #image_show(out, title=[target_names[x] for x in targets])


    # 将tensor转换为image
    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255]
    )


    writer=tb_writer()
    images, labels = next(iter(dataloaders['train'])) # 获取一批数据
    grid = torchvision.utils.make_grid([inv_normalize(image) for image in images[:32]]) # 读取32张图片
    writer.add_image('X-Ray grid', grid, 0) # 添加到TensorBoard
    writer.flush() # 将数据读取到存储器中
    
    model = get_model().to(device) # 获取模型
    criterion = nn.NLLLoss() # 损失函数
    optimizer = optim.Adam(model.parameters())
    train_epochs(model, device, dataloaders, criterion, optimizer, range(0,10), writer)
    writer.close() 


if __name__ == '__main__':
    main()




