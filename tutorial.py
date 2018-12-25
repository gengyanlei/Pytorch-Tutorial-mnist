'''
author : leilei
A whole Pytorch tutorial : set different layer's lr , update lr (One to one correspondence)
                           output middle layer's feature and fine-tune
'''
import torch
import torchvision
import numpy as np
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.utils import data

EPOCH=20
BATCH_SIZE=64
LR=1e-4

# mnist download,transform NHWC=>NCHW and 0,255=>0,1
train_data=torchvision.datasets.MNIST(root='./mnist',train=True,
                                      transform=torchvision.transforms.ToTensor(),download=False)
# pytorch's dataset loader
train_loader=data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
# test data
test_data=torchvision.datasets.MNIST(root='./mnist',train=False)
# test Variable  need transform  gpu
test_x=Variable(torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)).cuda()/255
test_y=test_data.test_labels.cuda()
# create model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
                nn.Conv2d(in_channels=1,out_channels=16,kernel_size=4,stride=1,padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,4,1,2),nn.ReLU(),nn.MaxPool2d(2,2))
        self.out=nn.Linear(32*7*7,10)
        
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        per_out=[]
        x=self.conv1(x)
        per_out.append(x)
        x=self.conv2(x)
        per_out.append(x)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        # can output middle layer's features
        return output,per_out

cnn=CNN().cuda()# gpu

# fine-tune
new_params=cnn.state_dict()
pretrain_dict=torch.load('./model/model.pth')
pretrain_dict={k:v for k,v in pretrain_dict.items() if k in new_params and v.size()==new_params[k].size()}#dict gennerator
new_params.update(pretrain_dict)
cnn.load_state_dict(new_params)

cnn.train()# if you want test ; just modify to cnn.eval()
# set different layer's learning rate: [conv1 conv2] lr*10 ; [out]  lr
def get_10x_lr_params(net):
    b=[net.conv1,net.conv2]
    for i in b:
        for j in i.modules():
            for k in j.parameters():
                yield k
                # generator

# set weight bias different lr; bias_lr = 2 * weight_lr
def get_weight_params(net):
    for i in net.modules(): # if error just add some 'for' 
        if isinstance(i,nn.Conv2d):
            yield i.weight
        if isinstance(i,nn.BatchNorm2d):
            yield i.weight
        if isinstance(i,nn.Linear):
            yield i.weight

# update lr
def lr_poly(base_lr,iters,max_iter,power):
    return base_lr*((1-float(iters)/max_iter)**power)
def adjust_lr(optimizer,base_lr,iters,max_iter,power):
    lr=lr_poly(base_lr,iters,max_iter,power)
    optimizer.param_groups[0]['lr']=lr # first param iterator 
    if len(optimizer.param_groups)>1:
        optimizer.param_groups[1]['lr']=lr*10

Params=get_10x_lr_params(cnn)
# optimizer             first params  lr=LR Internal overlapping external lr; second params
optimizer=torch.optim.Adam([{'params':cnn.out.parameters()},{'params':Params,'lr':LR*10}],lr=LR)
# loss function
loss_func=nn.CrossEntropyLoss().cuda()

iters=0
for epoch in range(EPOCH):
    i_iter=train_data.train_data.shape[0]//BATCH_SIZE
    for step,(x,y) in enumerate(train_loader):
        optimizer.zero_grad()# clear gradient
        adjust_lr(optimizer,LR,iters,EPOCH*i_iter,0.9)# update lr
        iters+=1
        b_x=Variable(x).cuda()# if channel==1 auto add c=1
        b_y=Variable(y).cuda()
#        print(cnn.state_dict()['conv1.0.weight'])
        output=cnn(b_x)[0]
        loss=loss_func(output,b_y)# Variable need to get .data
        loss.backward() # backward loss
        optimizer.step() # compute per gradient
        
        if step%50==0:
            test_output=cnn(test_x)[0]
            pred_y=torch.max(test_output,1)[1].cuda().data.squeeze()
            '''
            why data ,because Variable .data to Tensor;and cuda() not to numpy() ,must to cpu 
            and to numpy and .float compute decimal
            '''
            accuracy=torch.sum(pred_y==test_y).data.float()/test_y.size(0)
            print('EPOCH: ',epoch,'| train_loss:%.4f'%loss.data[0],'| test accuracy:%.2f'%accuracy)
        #                                           loss.data.cpu().numpy().item() get one value
    torch.save(cnn.state_dict(),'./model/model.pth')
# test phase
test_output=cnn(test_x[:13])[0]
pred_y=torch.max(test_output,1)[1].cuda().data.squeeze()
print(pred_y)
print(test_y[:13])
