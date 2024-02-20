import numpy as np
np.random.seed(123)

import torch
from torch import nn
torch.manual_seed(123)

from torch.nn import functional as F
import math
import random
random.seed(123)
from pickle import dump,load

###Save Rndm States
#torchRndmState=torch.get_rng_state()# torch.set_rng_state
#numPyRndmState=np.random.get_state()#numpy.random.set_state

#torch.save(torchRndmState,'torchRndmStateModels.pth')
#with open('numPyRndmStateModels.obj', 'wb') as f:
#    dump(numPyRndmState, f)

###load Rndm States
#revis_data=torch.load('torchRndmStateModels.pth')
#torch.set_rng_state(revis_data)
#with open('numPyRndmStateModels.obj', 'rb') as f:
#   np.random.set_state(load(f))


class Discriminator(nn.Module):
    def __init__(self, channel=64*2,out_class=1):
        super(Discriminator, self).__init__()
        self.channel = channel
        n_class = out_class 
  
        self.conv1 = nn.Conv3d(1,channel//8, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=0)
        self.in2 = nn.InstanceNorm3d(channel//4,affine=True)
        self.conv3 = nn.Conv3d(channel//4, n_class, kernel_size=4, stride=2, padding=1)
    def forward(self, x):
    
        h1 = F.leaky_relu(self.conv1(torch.nn.functional.pad(x,(2,1,2,1,2,1))), negative_slope=0.2)
        h2 = F.leaky_relu(self.in2(self.conv2(torch.nn.functional.pad(h1,(1,2,1,2,1,2)))), negative_slope=0.2)
        h3 = self.conv3(h2)
        output = h3.mean(axis=(2,3,4))
        return output

class multConvFModel(nn.Module): #Forward Model

    def __init__(self, nDepths=53,s=3,V=51,NxN=361,haarL=8,l=3,c=400):
    
        self.s=s
        self.V=V
        self.NxN=NxN
        self.nDepths=nDepths
        self.haarL=haarL      
        self.cnst=1e3
        super(multConvFModel, self).__init__()

        self.convBias=nn.Sequential(
             nn.Conv2d(nDepths*s*s,c,kernel_size=l, bias=False,padding=0,padding_mode='reflect'),
             nn.Conv2d(c,c,kernel_size=l, bias=False,padding=0,padding_mode='reflect'),
             nn.Conv2d(c,c,kernel_size=l, bias=False,padding=0,padding_mode='reflect'),
             nn.Conv2d(c,c,kernel_size=l, bias=False,padding=0,padding_mode='reflect'),
             nn.Conv2d(c,c,kernel_size=l, bias=False,padding=0,padding_mode='reflect'),
             nn.Conv2d(c,c,kernel_size=l, bias=False,padding=0,padding_mode='reflect'),
             nn.Conv2d(c,c,kernel_size=l, bias=False,padding=0,padding_mode='reflect'),
             nn.Conv2d(c,NxN,kernel_size=l, bias=False,padding=0,padding_mode='reflect')
        )

        
    def _initialize_weights(self):

        for m in self.convBias:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=1.0e-5, std=math.sqrt(2.0e-10/(m.out_channels*m.weight.data[0][0].numel())))
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)                        
        
    def forward(self, x):

        xShap=x.shape
        x=x.view(xShap[0],xShap[1],xShap[2]//self.s,self.s,xShap[3]//self.s,self.s) 
        x=x.permute(0,1,3,5,2,4)
        x=x.reshape(xShap[0],xShap[1]*(self.s*self.s),xShap[2]//self.s,xShap[3]//self.s)

        output=self.cnst*self.convBias(x)

        return output


class oneCnvFModel(nn.Module):#Forward Model
    def __init__(self, nDepths=53,s=3,V=51,NxN=361,haarL=8,L2=17):
    
        self.s=s
        self.V=V
        self.NxN=NxN
        self.nDepths=nDepths
        self.haarL=haarL
        
        super(oneCnvFModel, self).__init__()
        self.frwdApPadd=nn.Sequential(
             nn.Conv2d(nDepths*s*s,NxN, kernel_size=L2, bias=False,padding=L2//2)#,padding=10
        )

        for param in self.frwdApPadd.parameters():
            param.requires_grad = False

        self.trnspApPadd=nn.Sequential(
             nn.Conv2d(NxN,nDepths*s*s, kernel_size=L2, bias=False,padding=L2//2),#,padding=10
        )
        for param in self.trnspApPadd.parameters():
            param.requires_grad = False


        self.haarPLF=nn.Sequential(
             nn.Conv2d(NxN,NxN, kernel_size=haarL, bias=False,stride=haarL,padding=0,groups=NxN),#,padding=10
             nn.ConvTranspose2d(NxN,NxN, kernel_size=haarL,stride=haarL, bias=False,padding=0,groups=NxN)#,padding=10
        )
        for param in self.haarPLF.parameters():
            param.requires_grad = False
        for m in self.haarPLF:
            nn.init.constant_(m.weight.data,1/haarL)

    def isra(self, x,iterIs=1):

        xTrnsp=self.trnspApPadd(x)
        output=(xTrnsp)
        for i in range (iterIs):
            output=output*(xTrnsp/self.trnspApPadd(self.frwdApPadd(output)))

        output=output.view(output.shape[0],self.nDepths,self.s,self.s,output.shape[2],output.shape[3])
        output=output.permute(0,1,4,2,5,3)
        output=output.reshape(output.shape[0],output.shape[1],output.shape[2]*output.shape[3],output.shape[4]*output.shape[5])

        return output

        
    def forward(self, x):

        
        output=self.frwdApPadd(x)
        return output



class InvrsModel(nn.Module):#Reconstruction Network
    def __init__(self,nIter=6, nDepths=53,s=3,V=51,NxN=361):#56,12
        super(InvrsModel, self).__init__()
        self.nIter=nIter
        self.s=s
        self.V=V
        self.NxN=NxN
        self.nDepths=nDepths

        for i in range(nIter):

            setattr(self,"nonLin%d" % i, nn.Sequential(nn.ReLU()))
            if i<nIter:
                for param in getattr(self,"nonLin%d" % i).parameters():
                    param.requires_grad = True

        self.padLista=1

        for i in range(nIter):
            setattr(self,"W0Lis%d" % i, nn.Sequential(
            nn.Conv2d(V,V, kernel_size=3, bias=False,padding=0)#True
            ))
            if i>1:
                for param in getattr(self,"W0Lis%d" % i).parameters():
                    param.requires_grad = False

        for i in range(nIter):
            setattr(self,"W1Lis%d" % i, nn.Sequential(
            nn.Conv2d(V,nDepths*s*s, kernel_size=3, bias=True,padding=0,padding_mode='reflect')#True
            ))
            if i>1:
                for param in getattr(self,"W1Lis%d" % i).parameters():
                    param.requires_grad = False

        for i in range(1,nIter):
            setattr(self,"W2Lis%d" % i, nn.Sequential(
            nn.Conv2d(nDepths*s*s,V, kernel_size=3, bias=False,padding=0,padding_mode='reflect'),
            nn.Conv2d(V,nDepths*s*s, kernel_size=3, bias=True,padding=0,padding_mode='reflect')#True
            ))
            if i<nIter:
                for param in getattr(self,"W2Lis%d" % i).parameters():
                    param.requires_grad = False
        
        self.cmprs=nn.Conv2d(NxN,V, kernel_size=15, bias=False)
        self.padCmprs=7

        self._initialize_weights()
        
    def _initialize_weights(self):
        m=self.cmprs
        nn.init.normal_(m.weight.data, mean=1.0e-5, std=math.sqrt(2.0e-10/(m.out_channels*m.weight.data[0][0].numel())))
        for i in range(self.nIter):
            for m in getattr(self,"W1Lis%d" % i):
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, mean=1.0e-5, std=math.sqrt(2.0e-10/(m.out_channels*m.weight.data[0][0].numel())))
                    #nn.init.constant_(m.weight.data,1.0e-3)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)
        for i in range(1,self.nIter):
            for m in getattr(self,"W2Lis%d" % i):
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, mean=1.0e-5, std=math.sqrt(2.0e-10/(m.out_channels*m.weight.data[0][0].numel())))
                    #nn.init.constant_(m.weight.data,1.0e-3)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)

        for i in range(1,self.nIter):
            for m in getattr(self,"W0Lis%d" % i):
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, mean=1.0e-5, std=math.sqrt(2.0e-10/(m.out_channels*m.weight.data[0][0].numel())))
                    #nn.init.constant_(m.weight.data,1.0e-3)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)
    def randPadd(self):
        paddRnd=np.random.randint(4)
        if paddRnd==0:
            paddM='replicate'
        if paddRnd==1:
            paddM='circular'
        if paddRnd==2:
            paddM='reflect'
        if paddRnd==3:
            paddM='constant'
        return paddM
                        
    def forward(self, x, test=0):

        paddM=self.randPadd()
        if test==1:
            paddM='reflect'
        padL=self.padCmprs
        x=torch.nn.functional.pad(x,(padL,padL,padL,padL),paddM)
        x=self.cmprs(x)

        paddM=self.randPadd()
        if test==1:
            paddM='reflect'  
            
        padL=self.padLista           
        x=torch.nn.functional.pad(x,(padL,padL,padL,padL),paddM)

        tmp=self.W1Lis0(x)
        out=self.nonLin0(tmp)

        tmp1Prev=x
        for i in range(1,self.nIter):

            tmp1=getattr(self,"W1Lis%d" % i)(tmp1Prev)
            tmp1Prev=getattr(self,"W0Lis%d" % i)(tmp1Prev)
            padL=self.padLista                       
            tmp1Prev=torch.nn.functional.pad(tmp1Prev,(padL,padL,padL,padL),paddM)

            paddM=self.randPadd()
            if test==1:
                paddM='reflect'
                
            padL=self.padLista                                   
            tmp2=getattr(self,"W2Lis%d" % i)[0](torch.nn.functional.pad(out,(padL,padL,padL,padL),paddM))
            
            paddM=self.randPadd()
            if test==1:
                paddM='reflect'    
                
            padL=self.padLista                                                   
            tmp2=getattr(self,"W2Lis%d" % i)[1](torch.nn.functional.pad(tmp2,(padL,padL,padL,padL),paddM))   
            
            tmp=tmp1+out-tmp2#grad step
            out=getattr(self,"nonLin%d" % (i))(tmp)#non-linearity
            
        #Reshaping

        out=out.view(out.shape[0],self.nDepths,self.s,self.s,out.shape[2],out.shape[3])
        out=out.permute(0,1,4,2,5,3)
        out=out.reshape(out.shape[0],out.shape[1],out.shape[2]*out.shape[3],out.shape[4]*out.shape[5])

        return out

