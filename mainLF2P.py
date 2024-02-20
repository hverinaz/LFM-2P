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

#python mainLF2P.py --trnVol-file "../../advContZReal/forward2P/vol_newD2PwithTmp.mat"   --trnLF-file "../../advContZReal/forward2P/newD2PwithTmpPad.mat"  --trnLF2-file "../../advContZReal/forward2P/newD2PwithTmpS1A2Pad.mat"  --trnLF3-file "../../advContZReal/forward2P/newD2PwithTmpS2A3Pad.mat"  --outputs-dir "./outputs"           --weights-fileFl "./outputs/weights/epochFl_24.pth"    --weights-fileG "./outputs/weights/epochG_40.pth"   --weights-fileD "./outputs/weights/epochD_40.pth"     --batch-size 2 --num-epochs 3000     




parser = argparse.ArgumentParser()
parser.add_argument('--trnVol-file', type=str, required=True)
parser.add_argument('--trnLF-file', type=str, required=True)
parser.add_argument('--trnLF2-file', type=str, required=True)
parser.add_argument('--trnLF3-file', type=str, required=True)
parser.add_argument('--outputs-dir', type=str, required=True)
parser.add_argument('--weights-fileFl', type=str)
parser.add_argument('--weights-fileG', type=str)
parser.add_argument('--weights-fileD', type=str)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--num-epochs', type=int, default=3000)
parser.add_argument('--noisy-GT', type=bool, default=True)




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


#print(err)

_eps = 1e-15
s=3
N_=19
nDepths=53
centDpt=nDepths//2
L2=17
V=51
   
haarL=8
nLFs=28

F=oneCnvFModel(nDepths=nDepths,s=s,V=V,NxN=N_*N_,haarL=haarL,L2=L2).to(device)
l=3
c=400
Fl=multConvFModel(nDepths=nDepths,s=s,V=V,NxN=N_*N_,haarL=haarL,l=l,c=c).to(device)
nLayers=6
G=InvrsModel(nIter=nLayers,nDepths=nDepths,s=s,V=V,NxN=N_*N_).to(device)
D=Discriminator().to(device)



#state_dict=D.state_dict()
#for n, p in torch.load(args.weights_fileD, map_location=lambda storage, loc: storage).items():
#    if n in state_dict.keys():
        #print(n)
        #print('D')
#        state_dict[n].copy_(p)


#Load Trained Forward Model Weights
state_dict=Fl.state_dict()
for n, p in torch.load(args.weights_fileFl, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        #print(n)
        #print('G')
        state_dict[n].copy_(p)

#Load Pre-trained reconstruction Network
state_dict=G.state_dict()
for n, p in torch.load(args.weights_fileG, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        #print(n)
        #print('G')
        state_dict[n].copy_(p)




Fl_optimizer = optim.Adam(Fl.parameters(), lr=2e-5)#5e-7
G_optimizer = optim.Adam(G.parameters(), lr=2e-7)#5e-7
d_optimizer = optim.Adam(D.parameters(), lr=4e-7)#5e-6



size_input2=47#2*np.random.randint(22,25)+1#47
size_input=47#2*np.random.randint(22,25)+1#47

size_label2=(size_input2)*s
size_label=(size_input)*s


nLF_Seq=80

file = hdf5storage.loadmat(args.trnLF_file)
lfTrainFTmp=file['lfTrainTmp']
lfTrainFTmp=np.array(lfTrainFTmp).astype(np.uint8)
lfTrainFTmp=torch.from_numpy(lfTrainFTmp).to(device)
lfTrainFTmp=torch.nn.functional.relu(lfTrainFTmp)       
lfTrainFTmp=lfTrainFTmp[0:nLF_Seq]

file = hdf5storage.loadmat(args.trnLF2_file)#Slc1A2
lfTrainFTmp2=file['lfTrainTmpd1']
lfTrainFTmp2=np.array(lfTrainFTmp2).astype(np.uint8)
lfTrainFTmp2=torch.from_numpy(lfTrainFTmp2).to(device)
lfTrainFTmp2=torch.nn.functional.relu(lfTrainFTmp2)       
lfTrainFTmp2=lfTrainFTmp2[0:nLF_Seq]

file = hdf5storage.loadmat(args.trnLF3_file)#Slc2A3
lfTrainFTmp3=file['lfTrainTmp']
lfTrainFTmp3=np.array(lfTrainFTmp3).astype(np.uint8)
lfTrainFTmp3=torch.from_numpy(lfTrainFTmp3).to(device)
lfTrainFTmp3=torch.nn.functional.relu(lfTrainFTmp3)       
lfTrainFTmp3=lfTrainFTmp3[0:nLF_Seq]


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

if args.noisy_GT:
    #Use GT from pre-trained G for noisy 2P
    with torch.no_grad():
        totVol=torch.zeros_like(hrT).float()
        contSum=torch.zeros((1,totVol.shape[1])).to(device)
        for j in range(hrT.shape[1]-nDepths+1):
            tmpLf_real=1*lfTrainF[j+nDepths//2,:,:,:].float()
            tmpVol_Hat=G(tmpLf_real[None,:,:,:])
            totVol[0,j:j+nDepths,:,:]=totVol[0,j:j+nDepths,:,:]+tmpVol_Hat
            contSum[0,j:j+nDepths]=contSum[0,j:j+nDepths]+1

        totVol=totVol/contSum.view(1,hrT.shape[1],1,1)
        totVol=torch.nn.functional.relu(totVol)
        totVol=totVol/totVol.max()
        hrT=255*totVol    
        hrT=hrT.type(torch.uint8)
    



#remove empty spaces
lfTrainF=lfTrainF[:,:,13:-13,13:-13]#remove empty spaces
hrT=hrT[:,:,13:-13,13:-13]#remove empty spaces


numBatch=36*nLF_Seq#manually chosen numBatch
lenDatSet=args.batch_size*numBatch


volBatch=torch.zeros((args.batch_size,nDepths,size_label2,size_label)).to(device)
lfBatch=torch.zeros((args.batch_size,N_*N_,size_input2,size_input)).to(device)
lfBatchSeq=torch.zeros((args.batch_size,N_*N_,size_input2,size_input)).to(device)



lossTot1=np.zeros((args.num_epochs))
lossAdvEpc=np.zeros((args.num_epochs))
lossC1Epc=np.zeros((args.num_epochs))
lossTot2=np.zeros((args.num_epochs))
                                     
lossC2Epc=np.zeros((args.num_epochs))
lossDR=np.zeros((args.num_epochs))
lossDH=np.zeros((args.num_epochs))



###############################################
# Adversarial Training
###############################################

for p in Fl.parameters():  
    p.requires_grad = False
    
                
for epoch in range(args.num_epochs):


    epoch_losses1 = AverageMeter()
    epoch_losses2 = AverageMeter()
    epoch_losses1A = AverageMeter()
    epoch_losses1C=AverageMeter()
    epoch_losses1C2=AverageMeter()
    epoch_lossesGr1=AverageMeter()
    epoch_lossesDR = AverageMeter()
    epoch_lossesDH = AverageMeter()

    
        
    with tqdm(total=(lenDatSet - lenDatSet % args.batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for dummIn in range(numBatch):

###############################################
# Update Generator 
###############################################

            for p in D.parameters():  
                p.requires_grad = False
            for p in G.parameters():  
                p.requires_grad = True


            with torch.no_grad():

                for j in range(args.batch_size):

    
                    indx0=np.random.randint(0,hrT.shape[1]-nDepths+1)
                    indx1=np.random.randint(lfTrainF.shape[2]-size_input2+1)
                    indx2=np.random.randint(lfTrainF.shape[3]-size_input+1)

                    tmpLf_real=lfTrainF[indx0+centDpt,:,indx1: size_input2 +indx1, indx2: size_input + indx2].float()
                    locL1 =int(s*(indx1)+(s*size_input2-size_label2)/2)#-5*s
                    locL2 =int(s*(indx2)+(s*size_input-size_label)/2)#-5*s
                    tmpVol= hrT[0,indx0:indx0+nDepths,locL1: size_label2 +locL1, locL2: size_label + locL2].float()


                    indx0=np.random.randint(lfTrainFTmp.shape[0])
                    indx1T=np.random.randint(lfTrainFTmp.shape[2]-size_input2+1)
                    indx2T=np.random.randint(lfTrainFTmp.shape[3]-size_input+1)
                    fctLF=0.5*torch.rand(1,device=device)+0.75
                    indxFnl=np.random.randint(3)

                    if indxFnl<1:
                        augmLF2=fctLF*lfTrainFTmp[indx0,:,indx1T: size_input2 +indx1T, indx2T: size_input + indx2T].float()
                    elif indxFnl==1:
                        augmLF2=fctLF*lfTrainFTmp2[indx0,:,indx1T: size_input2 +indx1T, indx2T: size_input + indx2T].float()
                    else:
                        augmLF2=fctLF*lfTrainFTmp3[indx0,:,indx1T: size_input2 +indx1T, indx2T: size_input + indx2T].float()


                    locL1 =int(s*(indx1T)+(s*size_input2-size_label2)/2)
                    locL2 =int(s*(indx2T)+(s*size_input-size_label)/2)



                    augmLF= tmpLf_real
                    augmLF=augmLF[None,:,:,:]
                    augmLFShap=augmLF.shape
                    augmLF=augmLF.view(augmLFShap[0],N_,N_,augmLFShap[2],augmLFShap[3]) 
                    augmLF=augmLF.permute(0,2,1,3,4)#to match matlab reshape


                    augmLF2=augmLF2[None,:,:,:]
                    augmLF2Shap=augmLF2.shape
                    augmLF2=augmLF2.view(augmLF2Shap[0],N_,N_,augmLF2Shap[2],augmLF2Shap[3]) 
                    augmLF2=augmLF2.permute(0,2,1,3,4)#to match matlab reshape

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
                        augmLF2=augmLF2[:,idx19,:,:,:]
                        augmLF2=augmLF2[:,:,:,idxP,:]
                        tmpVol=tmpVol[:,idxD2,:]

                    if reflec1==1:
                        augmLF=augmLF[:,:,idx19,:,:]
                        augmLF=augmLF[:,:,:,:,idxP]
                        augmLF2=augmLF2[:,:,idx19,:,:]
                        augmLF2=augmLF2[:,:,:,:,idxP]
                        tmpVol=tmpVol[:,:,idxD3]

                    if reflec2==1:
                        augmLF=augmLF[:,idx19,:,:,:]
                        augmLF=augmLF[:,:,idx19,:,:]
                        augmLF2=augmLF2[:,idx19,:,:,:]
                        augmLF2=augmLF2[:,:,idx19,:,:]
                        tmpVol=tmpVol[idxD1,:,:]

                    if swapAx==1:
                        augmLF=augmLF.permute(0,2,1,4,3)  
                        augmLF2=augmLF2.permute(0,2,1,4,3)  
                        tmpVol=tmpVol.permute(0,2,1)

                    augmLF=augmLF.permute(0,2,1,3,4)#because reshape matlab is different
                    augmLF=augmLF.reshape(augmLFShap[0],N_*N_,augmLFShap[2],augmLFShap[3])

                    augmLF2=augmLF2.permute(0,2,1,3,4)#because reshape matlab is different
                    augmLF2=augmLF2.reshape(augmLF2Shap[0],N_*N_,augmLF2Shap[2],augmLF2Shap[3])



                    lfBatch[j]=augmLF[0]
                    lfBatchSeq[j]=augmLF2[0]
                    volBatch[j]=tmpVol

                stdLF=torch.sqrt((torch.sum((volBatch)**2,dim=(1,2,3))).view(volBatch.shape[0],1,1,1))
                volBatch=(volBatch)/(stdLF+_eps)

 



            volHat =G(lfBatch)

            stdLF=torch.sqrt((torch.sum((volHat)**2,dim=(1,2,3))).view(volHat.shape[0],1,1,1))
            volHat_n=(volHat.clone())/(stdLF+_eps)

            dc_loss=(((volHat_n-volBatch))**2).mean()

            fakeVSeq=G(lfBatchSeq)
            lfSyntSeq=Fl(torch.nn.functional.pad(fakeVSeq,((L2//2)*s,(L2//2)*s,(L2//2)*s,(L2//2)*s),'reflect'))


            tmpPd=torch.nn.functional.pad(lfSyntSeq-lfBatchSeq,(1,0,1, 0),'reflect')
            tmpPd=F.haarPLF(tmpPd)[:,:,1::,1::]
            tmpPd=lfSyntSeq-lfBatchSeq-tmpPd
            dc_loss2Seq=((tmpPd)**2).mean()


            fakeVSeq = fakeVSeq[None,:,:,:,:]
            fakeVSeq=fakeVSeq.permute(1,0,3,4,2)
            fakeVSeq=fakeVSeq/torch.max(torch.max(torch.max(fakeVSeq,dim=4)[0],dim=3)[0],dim=2)[0].view(fakeVSeq.shape[0],1,1,1,1)
            fakeVSeq=2*fakeVSeq-1

                
            d_val=D(fakeVSeq)
            d_loss = ((d_val-1)**2).mean() 

            

            loss1=2.5e3*dc_loss+15e-4*dc_loss2Seq+0.001*d_loss
            
            G_optimizer.zero_grad()
            loss1.backward()
            G_optimizer.step()

            epoch_losses1C.update(dc_loss.item(), len(lfBatch))
            epoch_losses1.update(loss1.item(), len(lfBatch))
            epoch_lossesGr1.update(dc_loss2Seq.item(), len(lfBatch))
            epoch_losses1A.update(d_val.mean().item(), len(lfBatch))
            
###############################################
# Update Discriminator
###############################################     
            
            for p in D.parameters():
                p.requires_grad = True
            for p in G.parameters():  
                p.requires_grad = False


            D.train()

            with torch.no_grad(): 
                                     

                realInpBatch = volBatch[None,:,:,:,:].clone()
                realInpBatch=realInpBatch.permute(1,0,3,4,2)
                realInpBatch=realInpBatch/torch.max(torch.max(torch.max(realInpBatch,dim=4)[0],dim=3)[0],dim=2)[0].view(realInpBatch.shape[0],1,1,1,1)
                realInpBatch=2*realInpBatch-1
                fakeInpBatch=fakeVSeq.clone()


            tmpDR=D(realInpBatch)#
            tmpDH=D(fakeInpBatch)#
        
            x_lossR=((tmpDR-1)**2).mean()
            x_lossH=((tmpDH+1)**2).mean()

            loss2=x_lossR+x_lossH#


            d_optimizer.zero_grad()
            loss2.backward()
            d_optimizer.step()


            epoch_losses2.update(loss2.item(), len(fakeInpBatch))
            epoch_lossesDR.update(tmpDR.mean().item(), len(fakeInpBatch))
            epoch_lossesDH.update(tmpDH.mean().item(), len(fakeInpBatch))

            t.set_postfix(loss1='{:.2f}'.format(epoch_losses1.avg)+' loss2={:.2f}'.format(epoch_losses2.avg))
            t.update(len(lfBatch))


    lossTot1[epoch]=epoch_losses1.avg
    lossC1Epc[epoch]=epoch_losses1C.avg
    lossC2Epc[epoch]=epoch_lossesGr1.avg
    lossAdvEpc[epoch]=epoch_losses1A.avg
    lossTot2[epoch]=epoch_losses2.avg                                     
    lossDR[epoch]=epoch_lossesDR.avg
    lossDH[epoch]=epoch_lossesDH.avg




    if (epoch)%10 ==0: 
        torch.save(G.state_dict(),os.path.join(args.outputs_dir, 'epochG_{}.pth'.format(epoch//10)))
        torch.save(D.state_dict(),os.path.join(args.outputs_dir, 'epochD_{}.pth'.format(epoch//10)))
    scio.savemat('lossesSyn.mat', mdict={'lossTot1': lossTot1,'lossC1Epc': lossC1Epc,'lossAdvEpc': lossAdvEpc,'lossTot2': lossTot2,'lossDR': lossDR,'lossDH': lossDH,'lossC2Epc':lossC2Epc})
