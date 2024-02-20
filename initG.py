import argparse
import os
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
np.random.seed(123)

import hdf5storage
from pickle import dump,load

import torch
torch.manual_seed(123)

from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


from tqdm import tqdm

import random
random.seed(123)
from random import randint

from utils import AverageMeter, calc_psnr
from modelsAdv import *

 
 

#python initG.py --trnVol-file "../vol_2P.mat"   --trnLF-file "../LF1_StackAndTmp.mat"  --outputs-dir "./outputs"           --weights-fileG "./outputs/weights/epochG_40.pth"     --batch-size 2 --num-epochs 400
 




parser = argparse.ArgumentParser()
parser.add_argument('--trnVol-file', type=str, required=True)
parser.add_argument('--trnLF-file', type=str, required=True)
parser.add_argument('--outputs-dir', type=str, required=True)
parser.add_argument('--weights-fileG', type=str)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--num-epochs', type=int, default=400)




args = parser.parse_args()

args.outputs_dir = os.path.join(args.outputs_dir,'weights')

if not os.path.exists(args.outputs_dir):
    os.makedirs(args.outputs_dir)

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#torch.manual_seed(args.seed)

###Save Rndm States

###if saveRnd==1:
#torchRndmState=torch.get_rng_state()# torch.set_rng_state
#numPyRndmState=np.random.get_state()#numpy.random.set_state

#torch.save(torchRndmState,'torchRndmState.pth')
#with open('numPyRndmState.obj', 'wb') as f:
#    dump(numPyRndmState, f)

####load Rndm States

#revis_data=torch.load('torchRndmState.pth')
#torch.set_rng_state(revis_data)
#with open('numPyRndmState.obj', 'rb') as f:
#   np.random.set_state(load(f))

_eps = 1e-15
s=3
N_=19
nDepths=53
centDpt=nDepths//2
L2=17
V=51
haarL=8
nLFs=28
nLayers=6
nLF_Seq=80

G=InvrsModel(nIter=nLayers,nDepths=nDepths,s=s,V=V,NxN=N_*N_).to(device)


#state_dict=G.state_dict()
#for n, p in torch.load(args.weights_fileG, map_location=lambda storage, loc: storage).items():
#    if n in state_dict.keys():
#        #print(n)
#        #print('G')
#        state_dict[n].copy_(p)




G_optimizer = optim.Adam(G.parameters(), lr=1e-6)#5e-7


size_input2=47#Size of the patches for training
size_input=47#Size of the patches for training

size_label2=(size_input2)*s
size_label=(size_input)*s



file = hdf5storage.loadmat(args.trnVol_file)
hrT=file['vol']
hrT=np.array(hrT).astype(np.float32)
hrT=hrT[None,:,:,:]
hrT=np.transpose(hrT,(0,3,1,2))
hrT=np.maximum(hrT,0)
maxhRT=np.amax(hrT,axis=(1,2,3))
hrT=255*hrT/maxhRT[:,None,None,None]
hrT=np.array(hrT).astype(np.uint8)
hrT=torch.from_numpy(hrT).to(device)
hrT=torch.nn.functional.relu(hrT)



file = hdf5storage.loadmat(args.trnLF_file)
lfTrainF=file['lfTrain']
lfTrainF=np.array(lfTrainF).astype(np.uint8)
lfTrainF=torch.from_numpy(lfTrainF).to(device)
lfTrainF=torch.nn.functional.relu(lfTrainF)       
    

#remove empty spaces
lfTrainF=lfTrainF[:,:,13:-13,13:-13]#remove empty spaces
hrT=hrT[:,:,13:-13,13:-13]#remove empty spaces


numBatch=36*nLF_Seq# manually chosen numBatch to simplify training
lenDatSet=args.batch_size*numBatch


volBatch=torch.zeros((args.batch_size,nDepths,size_label2,size_label)).to(device)
lfBatch=torch.zeros((args.batch_size,N_*N_,size_input2,size_input)).to(device)



lossTot1=np.zeros((args.num_epochs))



###############################################
# Training
###############################################

    
                
for epoch in range(args.num_epochs):


    epoch_losses1 = AverageMeter()


    
    with tqdm(total=(lenDatSet - lenDatSet % args.batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for dummIn in range(numBatch):


            for p in G.parameters():  
                p.requires_grad = True


            with torch.no_grad():

                for j in range(args.batch_size):

    
                    indx0=np.random.randint(0,hrT.shape[1]-nDepths+1)
                    indx1=np.random.randint(lfTrainF.shape[2]-size_input2+1)
                    indx2=np.random.randint(lfTrainF.shape[3]-size_input+1)

                    #crop patch in the 2D LF image and corresponding patch in the 3D image
                    tmpLf_real=lfTrainF[indx0+centDpt,:,indx1: size_input2 +indx1, indx2: size_input + indx2].float()
                    locL1 =int(s*(indx1)+(s*size_input2-size_label2)/2)#-5*s
                    locL2 =int(s*(indx2)+(s*size_input-size_label)/2)#-5*s
                    tmpVol= hrT[0,indx0:indx0+nDepths,locL1: size_label2 +locL1, locL2: size_label + locL2].float()

                    #Data augmentation 3 axes reflection and permutation in xy
                    augmLF= tmpLf_real
                    augmLF=augmLF[None,:,:,:]
                    augmLFShap=augmLF.shape
                    augmLF=augmLF.view(augmLFShap[0],N_,N_,augmLFShap[2],augmLFShap[3]) 
                    augmLF=augmLF.permute(0,2,1,3,4)#to match matlab reshape


                    swapAx=np.random.randint(2)
                    reflec0=np.random.randint(2)
                    reflec1=np.random.randint(2)
                    reflec2=np.random.randint(2)#Depths
                    idxP=[i for i in range(augmLF.shape[3]-1, -1, -1)]
                    idx19=[i for i in range(augmLF.shape[1]-1, -1, -1)]

                    idxD1=[i for i in range(tmpVol.shape[0]-1, -1, -1)]
                    idxD2=[i for i in range(tmpVol.shape[1]-1, -1, -1)]
                    idxD3=[i for i in range(tmpVol.shape[2]-1, -1, -1)]


                    if reflec0==1:
                        augmLF=augmLF[:,idx19,:,:,:]
                        augmLF=augmLF[:,:,:,idxP,:]
                        tmpVol=tmpVol[:,idxD2,:]

                    if reflec1==1:
                        augmLF=augmLF[:,:,idx19,:,:]
                        augmLF=augmLF[:,:,:,:,idxP]
                        tmpVol=tmpVol[:,:,idxD3]

                    if reflec2==1:
                        augmLF=augmLF[:,idx19,:,:,:]
                        augmLF=augmLF[:,:,idx19,:,:]
                        tmpVol=tmpVol[idxD1,:,:]

                    if swapAx==1:
                        augmLF=augmLF.permute(0,2,1,4,3)  
                        tmpVol=tmpVol.permute(0,2,1)

                    augmLF=augmLF.permute(0,2,1,3,4)#because reshape matlab is different
                    augmLF=augmLF.reshape(augmLFShap[0],N_*N_,augmLFShap[2],augmLFShap[3])


                    lfBatch[j]=augmLF[0]
                    volBatch[j]=tmpVol

                stdLF=torch.sqrt((torch.sum((volBatch)**2,dim=(1,2,3))).view(volBatch.shape[0],1,1,1))
                volBatch=(volBatch)/(stdLF+_eps)

 



            volHat =G(lfBatch)

            stdLF=torch.sqrt((torch.sum((volHat)**2,dim=(1,2,3))).view(volHat.shape[0],1,1,1))
            volHat_n=(volHat.clone())/(stdLF+_eps)

            dc_loss=(((volHat_n-volBatch))**2).mean()
            loss1=1e5*dc_loss
            
            G_optimizer.zero_grad()
            loss1.backward()
            G_optimizer.step()

            epoch_losses1.update(loss1.item(), len(lfBatch))
            
            t.set_postfix(loss='{:.6f}'.format(epoch_losses1.avg))
            t.update(len(lfBatch))


    lossTot1[epoch]=epoch_losses1.avg



    if (epoch)%10 ==0: 
        torch.save(G.state_dict(),os.path.join(args.outputs_dir, 'epochG_{}.pth'.format(epoch//10)))
    scio.savemat('lossesFile.mat', mdict={'lossTot1': lossTot1})
