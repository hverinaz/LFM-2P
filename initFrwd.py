import argparse
import os
import copy
import h5py
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
np.random.seed(123)

import hdf5storage
import cv2
from pickle import dump,load

import torch
torch.manual_seed(123)

from torch import nn
from torch.autograd import Variable
import nibabel as nib
from nilearn import plotting
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as TF


from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import random
random.seed(123)
from random import randint

from utils import AverageMeter, calc_psnr
from modelsAdv import *



parser = argparse.ArgumentParser()
parser.add_argument('--weights-fileF', type=str)
parser.add_argument('--weights-fileFl', type=str)
parser.add_argument('--outputs-dir', type=str, required=True)
parser.add_argument('--num-epochs', type=int, default=3000)

# python initFrwd.py --outputs-dir "./outputs" --weights-fileF "../lfMatrix.mat"   --weights-fileFl "./outputs/weights/epochFl.pth"



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


N_=19
nDepths=53
L2=17
V=51
s=3
haarL=8
l=3
c=400



Fl=multConvFModel(nDepths=nDepths,s=s,V=V,NxN=N_*N_,haarL=haarL,l=l,c=c).to(device)

#state_dict=Fl.state_dict()
#for n, p in torch.load(args.weights_fileFl, map_location=lambda storage, loc: storage).items():
#    if n in state_dict.keys():
#        print(n)
#        #print('G')
#        state_dict[n].copy_(p)


file = hdf5storage.loadmat(args.weights_fileF)
wg4=file['H']

#fixed preprocessing because of computation of H
wg4=wg4/np.max(wg4)
wg4 = np.array(wg4).astype(np.float32)
filtSiz=19
wg4=np.reshape(wg4,(filtSiz*N_,filtSiz*N_,nDepths,s,s)) 
wg4=np.transpose(wg4,(0,1,3,4,2))
wg4=wg4[:,:,::-1,::-1,::-1]#last dimension because it is opposite according to the ISRA reconstruction

#reshaping of H
wg4=np.reshape(wg4,(filtSiz,N_,filtSiz,N_,s,s,nDepths)) 
wg4=np.transpose(wg4,(3,1,0,2,4,5,6))
wg4=np.reshape(wg4,(N_*N_,filtSiz,filtSiz,s,s,nDepths)) 
wg4=wg4[:,::-1,::-1,:,:,:]
wg4=np.transpose(wg4,(0,5,3,4,1,2))
wg4=np.reshape(wg4,(N_*N_,nDepths*s*s,filtSiz,filtSiz)) 
wg4=wg4[:,:,1:-1,1:-1]#removing elements since they contain little information

knownW = torch.from_numpy(wg4).to(device)
idxD1=[i for i in range(knownW.shape[2]-1, -1, -1)]#due to Conv/correlation mismatch in definition
knownW=knownW[:,:,idxD1,:]
knownW=knownW[:,:,:,idxD1]
knownW=knownW.to(device)


Fl_optimizer = optim.Adam(Fl.parameters(), lr=1e-4)#5e-7


batch_size=3
lenDatSet=nDepths*s*s
numBatch=lenDatSet//batch_size


lfBatchImp=torch.zeros((batch_size,N_*N_,L2,L2)).to(device)
lossF=np.zeros((args.num_epochs))


for epoch in range(args.num_epochs):
    #cnt1Cont=cnt1Cont+5e-8


    epoch_lossesF=AverageMeter()



    with tqdm(total=(lenDatSet - lenDatSet % batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for dummIn in range(numBatch):

            for p in Fl.parameters():  
                p.requires_grad = True


            Fl.train()


            with torch.no_grad():
                volImp=torch.zeros((batch_size,nDepths,s,s),device=device)

                for j in range(batch_size):

                    indx0Inp=np.random.randint(volImp.shape[1])
                    indx1Inp=np.random.randint(volImp.shape[2])
                    indx2Inp=np.random.randint(volImp.shape[3])   

                    volImp[j,indx0Inp,indx1Inp,indx2Inp]=1                            
                    equivInd=indx0Inp*s*s+indx1Inp*s+indx2Inp

                    lfBatchImp[j]=knownW[:,equivInd,:,:]



            lfSyntImp=Fl(torch.nn.functional.pad(volImp,(s*(L2-1),s*(L2-1),s*(L2-1),s*(L2-1))))
            dc_loss2Imp=((lfSyntImp-lfBatchImp)**2).mean()
         

            loss1=1e-2*dc_loss2Imp

            Fl_optimizer.zero_grad()
            loss1.backward(retain_graph=False)
            Fl_optimizer.step()

            epoch_lossesF.update(dc_loss2Imp.item(), len(lfBatchImp))

            t.set_postfix(loss='{:.6f}'.format(epoch_lossesF.avg))
            t.update(len(lfBatchImp))




    lossF[epoch]=epoch_lossesF.avg




    if (epoch)%10 ==0: 

        torch.save(Fl.state_dict(),os.path.join(args.outputs_dir, 'epochFl_{}.pth'.format(epoch//10)))
    torch.save(Fl.state_dict(),os.path.join(args.outputs_dir, 'epochFLast.pth'))
    scio.savemat('lossesSyn.mat', mdict={'lossF':lossF})
