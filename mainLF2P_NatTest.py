import argparse
import os
import copy
import h5py
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
import hdf5storage
import cv2

import torch
from torch import nn
from torch.autograd import Variable
import nibabel as nib
from nilearn import plotting
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as TF


from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from random import randint
import random
from utils import AverageMeter, calc_psnr
#from datasetWGanNoUp import *
from modelsAdv import *
#python mainLF2P_NatTest.py --trnVol-file "./vol_newD2PwithTmpS2A3.mat"   --trnLF-file "./newD2PwithTmpS2A3Pad.mat"  --tstVol-file "./newDCalciumVol_2.mat"   --tstLF-file "./newDCalcium2P_2.mat"  --outputs-dir "./outputs"                 --weights-fileF "./HTomato3x3Smpl_514_1919.mat"   --weights-fileFl "./outputs/weights/epochFLast.pth"    --weights-fileG "./outputs/weights/epochG_40.pth"        --lr 1e-6                --batch-size 2 --num-epochs 3000        --num-workers 8



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trnVol-file', type=str, required=True)
    parser.add_argument('--trnLF-file', type=str, required=True)
    parser.add_argument('--tstVol-file', type=str, required=False)
    parser.add_argument('--tstLF-file', type=str, required=False)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--weights-fileG', type=str)
    parser.add_argument('--weights-fileF', type=str)
    parser.add_argument('--weights-fileFl', type=str)
    parser.add_argument('--weights-fileD', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)


 #python main2PForward.py --trnVol "./newDCalcium2P.mat"                --trnLF-file "./newDCalcium2PVol.mat"    --outputs-dir "./outputs"                 --weights-fileF "./weightsInitReal29VP_NoUp.mat"   -weights-fileFl "./weightsInitReal29VP_NoUp.mat"        --lr 1e-6                --batch-size 2 --num-epochs 3000        --num-workers 8

#python mainLF2P_PlusFrwd.py --trnVol-file "./newDCalcium2PVol.mat"   --trnLF-file "./newDCalcium2P.mat"  --tstVol-file "./newDCalcium2PVol_2.mat"   --tstLF-file "./newDCalcium2P_2.mat"  --outputs-dir "./outputs"                 --weights-fileF "./HTomato3x3Smpl.mat"   --weights-fileFl "./outputs/weights/epochFl_1.pth"    --weights-fileD "./outputs/weights/epochD_68.pth"        --lr 1e-6                --batch-size 2 --num-epochs 3000        --num-workers 8

#python mainLF2P_NatTest.py --trnVol-file "./vol_newD2PwithTmp.mat"   --trnLF-file "./newD2PwithTmp.mat"  --tstVol-file "./newDCalcium2PVol_2.mat"   --tstLF-file "./newDCalcium2P_2.mat"  --outputs-dir "./outputs"                 --weights-fileF "./HTomato3x3Smpl_514_1919.mat"   --weights-fileFl "./outputs/weights/epochFLast.pth"    --weights-fileG "./outputs/weights/epochLast.pth"        --lr 1e-6                --batch-size 2 --num-epochs 3000        --num-workers 8

    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir,'weights')

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)


    _eps = 1e-15



    F=knownModel().to(device)
    Fl=unknownModel().to(device)
    D=Discriminator().to(device)
    #GFilt=GaussFilt().to(device)
    G=InvrsModel().to(device)
    #GFilt=GaussFilt3D().to(device)
    #Drv=Derivative().to(device)


    #state_dict=D.state_dict()
    #for n, p in torch.load(args.weights_fileD, map_location=lambda storage, loc: storage).items():
    #    if n in state_dict.keys():
            #print(n)
            #print('D')
    #        state_dict[n].copy_(p)


    state_dict=Fl.state_dict()
    for n, p in torch.load(args.weights_fileFl, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            #print(n)
            #print('G')
            state_dict[n].copy_(p)
        #if n=='kernelConv':
        #    print(n)
        #    knownW=p
        #    idxD1=[i for i in range(knownW.shape[2]-1, -1, -1)]#due to Conv/correlation
        #    knownW=knownW[:,:,idxD1,:]
        #    knownW=knownW[:,:,:,idxD1]**2
        #    scio.savemat('volSeq42_.mat', mdict={'knownW':knownW.cpu().numpy()})
        #    print(err)

    state_dict=G.state_dict()
    for n, p in torch.load(args.weights_fileG, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            #print(n)
            #print('G')
            state_dict[n].copy_(p)
    #print(err)

    ##state_dict=F.state_dict()
    #for n, p in torch.load(args.weights_fileCmp, map_location=lambda storage, loc: storage).items():
    #    if n in state_dict.keys():
            #print(n)
            #print('G')
    #        state_dict[n].copy_(p)
    #for n, p in torch.load(args.weights_fileDmp, map_location=lambda storage, loc: storage).items():
    #    if n in state_dict.keys():
            #print(n)
            #print('G')
    #        state_dict[n].copy_(p)



    file = hdf5storage.loadmat(args.weights_fileF)
    #state_dict=F.state_dict()
    wg4=file['H']
    print(wg4.shape)
    print(errr)
    #fix because of computation of H
    wg4=wg4/np.max(wg4)
    wg4 = np.array(wg4).astype(np.float32)
    #print(wg4.shape)
    #print(errrrr)
    filtSiz=19
    wg4=np.reshape(wg4,(filtSiz*19,filtSiz*19,53,3,3)) 
    wg4=np.transpose(wg4,(0,1,3,4,2))
    wg4=wg4[:,:,::-1,::-1,::-1]#last dimension because it is opposite according to the ISRA reconstruction
    #scio.savemat('Htest.mat', mdict={'wg4':wg4})
    #print(errrrr)
    wg4=np.reshape(wg4,(filtSiz,19,filtSiz,19,3,3,53)) 
    wg4=np.transpose(wg4,(3,1,0,2,4,5,6))
    wg4=np.reshape(wg4,(19*19,filtSiz,filtSiz,3,3,53)) 
    wg4=wg4[:,::-1,::-1,:,:,:]
    ##wg4=np.transpose(wg4,(5,3,4,0,1,2))
    #wg4=np.reshape(wg4,(75*9,19,19,13,13)) 
    #wg4=np.transpose(wg4,(0,3,2,4,1))
    #wg4=np.reshape(wg4,(75*9,19*13,19*13)) 
    #wg4=wg4[None,:,:,:]
    wg4=np.transpose(wg4,(0,5,3,4,1,2))

    #wg4=np.transpose(wg4,(5,3,4,1,2,0))
    wg4=np.reshape(wg4,(19*19,53*9,filtSiz,filtSiz)) 
    #wg4=np.transpose(wg4,(3,0,1,2))
    wg4=wg4[:,:,1:-1,1:-1]#remove elements
    #print(wg4.shape)

    #lfEval=wg4[:,:,:,1,1,11].squeeze()
    #feat=lfEval.permute(1,2,0)
    #feat=np.squeeze(feat.data.cpu().numpy())
    #plt.imshow(feat[:,:,100])
    #plt.show()
    #print(wg4.shape)
    #print(errr)
    #wg5=file['filtNet']
    ##wg6=file['weightCompr']
    ##wg6=wg6[:,:,None,None]
    ##wg7=file['weightDcompr']
    ##wg7=wg7[:,:,None,None]


    #wg4 = np.array(wg4).astype(np.float32)
    #wg4=np.transpose(wg4,(2,3,0,1))

    wgTmp = torch.from_numpy(wg4).to(device)
    #knownW=torch.sqrt(wgTmp)
    #wgTmp=knownW
    #wgTmp=1.2*torch.sqrt(wgTmp)+0.07*torch.rand_like(wgTmp)
    #wgTmp=0.7*torch.sqrt(wgTmp)
    #wgTmp[wgTmp<0.1]=0.1*torch.rand_like(wgTmp[wgTmp<0.1])
    #knownW=wgTmp
    #stdW=torch.sqrt(torch.sum((knownW)**2,dim=(0,2,3)).view(1,knownW.shape[1],1,1))
    #knownW=knownW/(stdW+_eps)
    F.state_dict()['frwdApPadd.0.weight'].copy_(wgTmp)
    #state_dict['frwdAp.0.weight'].copy_(wgTmp)
    #state_dict['frwdApPadd.0.weight'].copy_(wg
    #Fl.state_dict()['frwdAp.0.weight'].copy_(wgTmp)

    #Fl.state_dict()['kernelConv'].copy_(wgTmp)

    #Fl.kernelConv=wgTmp
    #for i in range(75*9):
    #    Fl.state_dict()['oneDFr{}'.format(i)+'.0.weight'].copy_(wgTmp[:,i,:,:][:,None,:,:])

    #Fl.state_dict()['frwdAp.0.weight'].copy_(wgTmp)
    wg4=np.transpose(wg4,(1,0,2,3)) #for transpose
    wg4=np.array(wg4[:,:,::-1,::-1]).astype(np.float32)
    wgTmp = torch.from_numpy(wg4).to(device)
    F.state_dict()['trnspApPadd.0.weight'].copy_(wgTmp)
    #state_dict['trnspAp.0.weight'].copy_(wgTmp)
    #state_dict['trnspApPadd.0.weight'].copy_(wgTmp)
    ##Fl.state_dict()['trnspAp.0.weight'].copy_(wgTmp)


    #wg4=np.zeros((1,1,1,1,3)).astype(np.float32)
    #wg4[0,0,0,0,0]=0.5;
    #wg4[0,0,0,0,2]=0.5;
    #wgTmp = torch.from_numpy(wg4).to(device)
    #state_dict['lowPassZ.weight'].copy_(wgTmp)

    #wg4=np.zeros((2,1,2,2,1)).astype(np.float32)
    #wg4[0,0,0,0,0]=-1;
    #wg4[0,0,1,0,0]=1;
    #wg4[1,0,0,0,0]=-1;
    #wg4[1,0,0,1,0]=1;

    #wgTmp = torch.from_numpy(wg4).to(device)
    #Drv.state_dict()['convXY.weight'].copy_(wgTmp)

    #wg4=np.zeros((1,1,5,5)).astype(np.float32)    
    #wg4[0,0,:,:]=[[0.0030,0.0133,0.0219,0.0133,0.0030],[0.0133, 0.0596, 0.0983,0.0596,0.0133],[0.0219, 0.0983, 0.1621,0.0983,0.0219], [0.0133, 0.0596, 0.0983,0.0596,0.0133],[0.0030, 0.0133, 0.0219,0.0133,0.0030]]

    #wgTmp = torch.from_numpy(wg4).to(device)
    #GFilt.state_dict()['convGauss.weight'].copy_(wgTmp)
    wg4=np.zeros((1,1,5,5,5)).astype(np.float32)  
    wg4[0,0,:,:,:]=[[[0.0002,    0.0007,    0.0012,    0.0007,    0.0002],
    [0.0007,    0.0032,    0.0054,    0.0032,    0.0007],
    [0.0012,    0.0054,    0.0088,    0.0054,    0.0012],
    [0.0007,    0.0032,    0.0054,    0.0032,    0.0007],
    [0.0002,    0.0007,    0.0012,    0.0007,    0.0002]],

    [[0.0007,    0.0032,    0.0054,    0.0032,    0.0007],
    [0.0032,    0.0146,    0.0240,    0.0146,    0.0032],
    [0.0054,    0.0240,    0.0396,    0.0240,    0.0054],
    [0.0032,    0.0146,    0.0240,    0.0146,    0.0032],
    [0.0007,    0.0032,    0.0054,    0.0032,    0.0007]],

    [[0.0012,    0.0054,    0.0088,    0.0054,    0.0012],
    [0.0054,    0.0240,    0.0396,    0.0240,    0.0054],
    [0.0088,    0.0396,    0.0653,    0.0396,    0.0088],
    [0.0054,    0.0240,    0.0396,    0.0240,    0.0054],
    [0.0012,    0.0054,    0.0088,    0.0054,    0.0012]],

    [[0.0007,    0.0032,    0.0054,    0.0032,    0.0007],
    [0.0032,    0.0146,    0.0240,    0.0146,    0.0032],
    [0.0054,    0.0240,    0.0396,    0.0240,    0.0054],
    [0.0032,    0.0146,    0.0240,    0.0146,    0.0032],
    [0.0007,    0.0032,    0.0054,    0.0032,    0.0007]],

    [[0.0002,    0.0007,    0.0012,    0.0007,    0.0002],
    [0.0007,    0.0032,    0.0054,    0.0032,    0.0007],
    [0.0012,    0.0054,    0.0088,    0.0054,    0.0012],
    [0.0007,    0.0032,    0.0054,    0.0032,    0.0007],
    [0.0002,    0.0007,    0.0012,    0.0007,    0.0002]]]
    wgTmp = torch.from_numpy(wg4).to(device)
    #GFilt.state_dict()['convGauss.weight'].copy_(wgTmp)
    #criterion = nn.MSELoss()#MSE
    Fl_optimizer = optim.Adam(Fl.parameters(), lr=1e-5)#5e-7
    G_optimizer = optim.Adam(G.parameters(), lr=1e-5)#5e-7
    d_optimizer = optim.Adam(D.parameters(), lr=5e-5)#5e-6




    size_input2=47#2*np.random.randint(22,25)+1#47
    size_input=47#2*np.random.randint(22,25)+1#47
    upFa=3
    size_label2=(size_input2-0)*upFa
    size_label=(size_input-0)*upFa

    size_input2F=size_label2/upFa-16#2*np.random.randint(22,25)+1#47
    size_inputF=size_label/upFa-16#2*np.random.randint(22,25)+1#47



    #file = hdf5storage.loadmat(args.tstVol_file)
    #hrTTst=file['vol']
    #hrTTst=np.array(hrTTst).astype(np.float32)
    #hrTTst=hrTTst[None,:,:,:]
    #hrTTst=np.transpose(hrTTst,(0,3,1,2))
    #hrTTst=torch.from_numpy(hrTTst).to(device)
    #hrTTst=torch.nn.functional.relu(hrTTst)
    #hrTTst=hrTTst[:,:,3*13:-3*13,3*13:-3*13]
    #print(hrTTst.dtype)
    #print(hrTTst.shape)


    file = hdf5storage.loadmat(args.trnLF_file)
    lfTrainFTmp=file['lfTrainTmp']
    lfTrainFTmp=np.array(lfTrainFTmp).astype(np.uint8)
    lfTrainFTmp=torch.from_numpy(lfTrainFTmp).to(device)
    #lfTrainFTmp=torch.nn.functional.relu(lfTrainFTmp[0,:,:,:][None,:,:,:])      
    #lfTrainFTmp=lfTrainFTmp[:,:,19:-19,19:-19] 


    file = hdf5storage.loadmat(args.trnVol_file)
    hrT=file['vol']
    #hrT=file['totVol']
    ##hrT=file['tmpVol1']
    #hrT=np.squeeze(hrT)
    #hrT=np.transpose(hrT,(1,2,0))
    hrT=np.array(hrT).astype(np.float32)
    #hrT=hrT[3*19:-3*19,3*19:-3*19,:]
    hrT=hrT[None,:,:,:]
    hrT=np.transpose(hrT,(0,3,1,2))
    hrT=np.maximum(hrT,0)
    maxhRT=np.amax(hrT,axis=(1,2,3))
    hrT=255*hrT/maxhRT[:,None,None,None]
    hrT=np.array(hrT).astype(np.uint8)
    hrT=torch.from_numpy(hrT).to(device)
    hrT=torch.nn.functional.relu(hrT)

    print(hrT.dtype)
    print(hrT.shape)


    file = hdf5storage.loadmat(args.trnLF_file)
    lfTrainF=file['lfTrain']
    lfTrainF=np.array(lfTrainF).astype(np.uint8)


    lfTrainF=torch.from_numpy(lfTrainF).to(device)
    lfTrainF=torch.nn.functional.relu(lfTrainF)       
    #lfTrainF=lfTrainF[:,:,19:-19,19:-19]   



    print(lfTrainF.shape)   
    print(lfTrainF.max())   

    nBtchRealLF=80
    numBatch=3*12*nBtchRealLF#843
    #numBatch=1
    lenDatSet=args.batch_size*numBatch
    size_W=13
    volBatch=torch.zeros((args.batch_size,53,size_label2,size_label)).to(device)
    lfBatch=torch.zeros((args.batch_size,361,size_input2,size_input)).to(device)
    #volBatchSeq=torch.zeros((args.batch_size,75,size_label2,size_label)).to(device)
    lfBatchSeq=torch.zeros((args.batch_size,361,size_input2,size_input)).to(device)

    #realInpBatch=torch.zeros((args.batch_size,1,size_W*19,size_W*19)).to(device)
    #fakeInpBatch=torch.zeros((args.batch_size,1,size_W*19,size_W*19)).to(device)
    #fakeInpBatchG=torch.zeros((args.batch_size,1,size_W*19,size_W*19)).to(device)


    wGrad=1.00
    wGrad2=10.00


    fctL=1.0/(60.0+1.0)
    fctL=1e-4
    fctL=1e-9
    fctL=0.99999#0.84 1e-3
    fctL=0.6#0.84 1e-3
    fctL=1/191#0.84 1e-3
    fctL=0.05
    lossGCEpc=np.zeros((args.num_epochs))
    lossGAEpc=np.zeros((args.num_epochs))
    lossGCEpcDiv=np.zeros((args.num_epochs))
    lossGCSpec=np.zeros((args.num_epochs))
    lossDEpc=np.zeros((args.num_epochs))
    lossDcEpc=np.zeros((args.num_epochs))
    lossDPnlEpc=np.zeros((args.num_epochs))
    lossDPnlEpc2=np.zeros((args.num_epochs))
    lossDPnlEpc3=np.zeros((args.num_epochs))
    lossGr1=np.zeros((args.num_epochs))
    lossGr2=np.zeros((args.num_epochs))
    lossGr3=np.zeros((args.num_epochs))
    fctEpc=np.zeros((args.num_epochs))
    psnrEpoch=np.zeros((args.num_epochs))
    psnrEpochTst=np.zeros((args.num_epochs))
    wEpc=np.zeros((args.num_epochs))
    w2Epc=np.zeros((args.num_epochs))
    constG=np.zeros((args.num_epochs))
    constG2=np.zeros((args.num_epochs))
    lossGCEpc2Div=np.zeros((args.num_epochs))
    GATrain=np.zeros((args.num_epochs))
    GATest=np.zeros((args.num_epochs))
    GCTrain=np.zeros((args.num_epochs))
    GCTest=np.zeros((args.num_epochs))
    GC3Train=np.zeros((args.num_epochs))
    GC3Test=np.zeros((args.num_epochs))
    GC2Train=np.zeros((args.num_epochs))
    GC2Test=np.zeros((args.num_epochs))
    inpLFEpoch=np.zeros((args.num_epochs))
    inpTrEpoch=np.zeros((args.num_epochs))
    psnrTrn=np.zeros((args.num_epochs))
    psnrTst=np.zeros((args.num_epochs))

    tmpLoss=nn.MSELoss()
    prevx_loss2=0.0
    meanX_loss2=0.0
    x_lossR=torch.zeros((args.batch_size)).to(device)
    #indx0I=np.random.permutation(knownW.shape[1])[0:args.batch_size]
    #indx1I=np.random.randint(knownW.shape[2]-size_W+1)
    #indx2I=np.random.randint(knownW.shape[3]-size_W+1)

    #for p in G.parameters():  
    #    p.requires_grad = False
    #G._initialize_weights()
    cnt1Cont=4e-6
    gradM=torch.ones((1),device=device)
    #print(err)
    for epoch in range(args.num_epochs):
        #cnt1Cont=cnt1Cont+5e-8

        epoch_losses1 = AverageMeter()
        epoch_losses2 = AverageMeter()
        epoch_losses1A = AverageMeter()
        epoch_losses1C=AverageMeter()
        epoch_losses1C2=AverageMeter()
        epoch_losses1C3=AverageMeter()
        epoch_lossesGr1=AverageMeter()
        epoch_lossesGr2=AverageMeter()
        epoch_lossesGr3=AverageMeter()
        epoch_lossesInpLF=AverageMeter()
        epoch_lossesInpLTr=AverageMeter()
        epoch_losses2C = AverageMeter()
        epoch_losses2Pnl = AverageMeter()
        epoch_losses2Pnl2 = AverageMeter()
        epoch_losses2Pnl3 = AverageMeter()
        #if epoch%30==0:
        #    GPrev = copy.deepcopy(G)

        with tqdm(total=(lenDatSet - lenDatSet % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for dummIn in range(numBatch):

    ###############################################
    # Train  Generator 
    ###############################################
                for dummInG in range(0):

                    for p in D.parameters():  
                        p.requires_grad = False
                    for p in G.parameters():  
                        p.requires_grad = True
                    #G.trnspApPadd.requires_grad=False
                    #G.convZDir.requires_grad=False
                    #for p in Fl.parameters():  
                    #    p.requires_grad = True



                    G.train()
                    #D.train()
                    Fl.train()
#                    G2.train()


                    with torch.no_grad():
                        #volImp=torch.zeros((args.batch_size,75*9,19,19),device=device)
                        #outLFImp=torch.rand((args.batch_size,361,13,13),device=device)
                        #trnspInpl=torch.rand((args.batch_size,361,size_input2,size_input),device=device)

                        for j in range(args.batch_size):


                            
                            indx0=np.random.randint(0,28)
                            indx1=np.random.randint(lfTrainF.shape[2]-size_input2+1)
                            indx2=np.random.randint(lfTrainF.shape[3]-size_input+1)
                            tmpLf_real=lfTrainF[indx0+26,:,indx1: size_input2 +indx1, indx2: size_input + indx2].float()
                            locL1 =int(upFa*(indx1)+(upFa*size_input2-size_label2)/2)#-5*upFa
                            locL2 =int(upFa*(indx2)+(upFa*size_input-size_label)/2)#-5*upFa
                            tmpVol= hrT[0,indx0:indx0+53,locL1: size_label2 +locL1, locL2: size_label + locL2].float()

                            volBatch[j]=tmpVol  
                            lfBatch[j]=tmpLf_real

                            indx0=np.random.randint(0,28)

                            lfBatchSeq[j]=lfTrainFTmp[indx0,:,indx1: size_input2 +indx1, indx2: size_input + indx2].float()


                            #shfZ=np.random.randint(-20,20)  
                            #swapAx=np.random.randint(2)
                            #reflec1=np.random.randint(2)#Depths
                            #reflec2=np.random.randint(2)
                            #reflec3=np.random.randint(2)
        
                            #idxD1=[i for i in range(tmpVol.shape[0]-1, -1, -1)]
                            #idxD2=[i for i in range(tmpVol.shape[1]-1, -1, -1)]
                            #idxD3=[i for i in range(tmpVol.shape[2]-1, -1, -1)]


                            #if reflec1==1:
                            #    tmpVol=tmpVol[idxD1,:,:]
                            #if reflec2==1:
                            #    tmpVol=tmpVol[:,idxD2,:]
                            #if reflec3==1:
                            #    tmpVol=tmpVol[:,:,idxD3]
                            #if swapAx==1:
                            #    tmpVol=tmpVol.permute(0,2,1)


                            #tmpVol=torch.roll(tmpVol, shifts=shfZ, dims=0)
                            #tmpVol=tmpVol.permute(1,2,0)

                            #tmpVolShap=tmpVol.shape
                            #tmpVol=tmpVol.view(tmpVolShap[0]//3,3,tmpVolShap[1]//3,3,tmpVolShap[2]) 
                            #tmpVol=tmpVol.permute(4,1,3,0,2)
                            #tmpVol=tmpVol.reshape(tmpVolShap[2]*9,tmpVolShap[0]//3,tmpVolShap[1]//3)
                            #volAug[j,:,:,:]=tmpVol


                       
                        #outLFImp=F(torch.nn.functional.pad(volImp,(12,12,12,12)),0)#[:,:,2:-2,2:-2]
                        #outLFImpMx1=outLFImp/torch.max(torch.max(torch.max(outLFImp,dim=3)[0],dim=2)[0],dim=1)[0].view(outLFImp.shape[0],1,1,1)
                        #outLFImp=outLFImp.reshape(outLFImp.shape[0],19,19,13,13)
                        #outLFImp=outLFImp.permute(0,3,2,4,1)
                        #outLFImp=outLFImp.reshape(outLFImp.shape[0],1,13*19,13*19)
                        #outLFImp=GFilt(outLFImp,3)
                        #stdLF=torch.sqrt((torch.sum((outLFImp)**2,dim=(1,2,3))).view(outLFImp.shape[0],1,1,1))
                        #outLFImp=(outLFImp)/(stdLF+_eps)
                        #outLFImp=(outLFImp)/100000.0
                        

                        stdLF=torch.sqrt((torch.sum((volBatch)**2,dim=(1,2,3))).view(volBatch.shape[0],1,1,1))
                        volBatch=(volBatch)/(stdLF+_eps)

                        #lfAug=Fl(volAug)
                        #volAugrshp=volAug[:,:,8:-8,8:-8]
                        #tmpVolShap=volAugrshp.shape
                        #volAugrshp=volAugrshp.view(tmpVolShap[0],tmpVolShap[1]//9,3,3,tmpVolShap[2],tmpVolShap[3]) 
                        #volAugrshp=volAugrshp.permute(0,1,4,2,5,3)
                        #volAugrshp=volAugrshp.reshape(tmpVolShap[0],tmpVolShap[1]//9,tmpVolShap[2]*3,tmpVolShap[3]*3)

                        #stdLF=torch.sqrt((torch.sum((volAugrshp)**2,dim=(1,2,3))).view(volAugrshp.shape[0],1,1,1))
                        #volAugrshp=(volAugrshp)/(stdLF+_eps)
                        
                        #outLFImp=outLFImp/torch.max(torch.max(torch.max(outLFImp,dim=3)[0],dim=2)[0],dim=1)[0].view(outLFImp.shape[0],1,1,1)
                        #volImp=volImp/torch.max(torch.max(torch.max(outLFImp,dim=3)[0],dim=2)[0],dim=1)[0].view(outLFImp.shape[0],1,1,1)
  
                        #mx1V=volBatch[0].max()
                        #mx2V=volBatch[1].max()                    
                        #maskoutLFImp=outLFImp
                        #maskoutLFImp[maskoutLFImp>=0.05]=1
                        #maskoutLFImp[maskoutLFImp<0.05]=0
                        #stdLF=torch.sqrt((torch.sum((outLFImp)**2,dim=(1,2,3))).view(outLFImp.shape[0],1,1,1))
                        #outLFImp=(outLFImp)/(stdLF+_eps)
                        #print(outLFImp.shape)

                        #lfEval=maskoutLFImp.reshape(2,19,19,13,13)
                        #lfEval=lfEval.permute(0,3,2,4,1)
                        #lfEval=lfEval.reshape(2,19*13,19*13)



                    volHat =G(lfBatch)
                    stdLF=torch.sqrt((torch.sum((volHat)**2,dim=(1,2,3))).view(volHat.shape[0],1,1,1))
                    volHat_n=(volHat)/(stdLF+_eps)

                    dc_loss=((volHat_n-volBatch)**2).mean()
                    #print(lfBatch.shape)
                    #print(volHat.shape)
                    #print(errrr)
                    #feat=volBatch[0].permute(1,2,0)
                    #featInp=feat
                    #feat=np.squeeze(feat.data.cpu().numpy())
                    #plt.imshow(feat[:,:,37])
                    #plt.show()
                    #feat=volHat[0].permute(1,2,0)
                    #featInp=feat
                    #feat=np.squeeze(feat.data.cpu().numpy())
                    #plt.imshow(feat[:,:,37])
                    #plt.show()


                    #volHat2 =G(lfAug)
                    #stdLF=torch.sqrt((torch.sum((volHat2)**2,dim=(1,2,3))).view(volHat2.shape[0],1,1,1))
                    #volHat2=(volHat2)/(stdLF+_eps)
                    #lfSynt=Fl(volHat)
                    #dc_loss2=((lfSynt-lfBatch[:,:,8:-8,8:-8])**2).mean()

                  

                    #fakeVSeq=G(lfBatchSeq)
                    #lfSyntSeq=Fl(fakeVSeq)
                    #dc_loss2Seq=((lfSyntSeq-lfBatchSeq[:,:,8:-8,8:-8])**2).mean()


                    #fakeVSeq = fakeVSeq[None,:,:,:,:]
                    #fakeVSeq=fakeVSeq.permute(1,0,3,4,2)
                    #fakeVSeq=fakeVSeq/torch.max(torch.max(torch.max(fakeVSeq,dim=4)[0],dim=3)[0],dim=2)[0].view(fakeVSeq.shape[0],1,1,1,1)
                    #fakeVSeq=2*fakeVSeq-1
                    #d_loss = ((D(fakeVSeq)-1)**2).mean() 

                    
                    #lf_hatLin =Fl(volBatchLin)
                    #stdLF=torch.sqrt((torch.sum((lf_hatLin)**2,dim=(1,2,3))).view(lf_hatLin.shape[0],1,1,1))
                    #lf_hatLin=(lf_hatLin)/(stdLF+_eps)


                    #lf_hatI =Fl(torch.nn.functional.pad(volImp,(14,14,14,14)))
                    #lf_hatI=lf_hatI.reshape(lf_hatI.shape[0],19,19,13,13)
                    #lf_hatI=lf_hatI.permute(0,3,2,4,1)
                    #lf_hatI=lf_hatI.reshape(lf_hatI.shape[0],1,13*19,13*19)
                    #lf_hatI=GFilt(lf_hatI,3)

                    #stdLF=torch.sqrt((torch.sum((lf_hatI)**2,dim=(1,2,3))).view(lf_hatI.shape[0],1,1,1))
                    #lf_hatI=(lf_hatI)/(stdLF+_eps)
                    #lf_hatI=lf_hatI
                    #lf_hatI=lf_hatI/torch.max(torch.max(torch.max(lf_hatI,dim=3)[0],dim=2)[0],dim=1)[0].view(lf_hatI.shape[0],1,1,1)
                    #dc2_loss=-(maskoutLFImp*lf_hatI*outLFImp).mean()
                    #dc2_loss=((lf_hatI-outLFImp)**2).mean()
                    #dc2_loss=((lf_hatLin-lfBatchLin)**2).mean()

                    #loss1=1e4*dc_loss+0.01*d_loss# 1e3*dc_loss+ +1e-7*dc_lossInpSparse+7*dc_lossInp#+1e3*dc_lossPosi
                    #loss1=5e4*dc_loss+1*dc2_loss#+0.01*d_loss# 1e3*dc_loss+ +1e-7*dc_lossInpSparse+7*dc_lossInp#+1e3*dc_lossPosi
                    #loss1=5e4*dc_loss+1*dc_loss2#+0.01*d_loss# 1e3*dc_loss+ +1e-7*dc_lossInpSparse+7*dc_lossInp#+1e3*dc_lossPosi
                    loss1=5e4*dc_loss#+1*dc_loss2+dc_loss2Seq#+d_loss
                    G_optimizer.zero_grad()
                    Fl_optimizer.zero_grad()
                    loss1.backward(retain_graph=False)
                    Fl_optimizer.step()
                    G_optimizer.step()

                    #g_optimizer.step()

                    #epoch_lossesInpLF.update(regContinuityW.item(), len(x_hat))
                    #epoch_lossesInpLTr.update(inpTrp_loss.item(), len(x_hat))
                    #epoch_losses1C.update(dc_loss.item()/Dc.constantG.mean().data, len(x_hat))
                    #epoch_losses1C2.update(dc2_loss.item()/Dc2.constantG.mean().data, len(x_hat))
                    epoch_losses1C.update(dc_loss.item(), len(lfBatch))
                    #epoch_losses1C2.update(dc_loss2.item(), len(lfBatch))
                    #epoch_losses1C3.update(dc_loss2Seq.item(), len(lfBatch))
                    epoch_losses1.update(loss1.item(), len(lfBatch))
                    #epoch_lossesGr1.update(dc_lossInpSparse.mean().item(), len(lf_hat))
                    #epoch_lossesGr2.update(dc_lossPosi.item(), len(lf_hat))
                    #epoch_lossesGr3.update(dc_lossSymm.item(), len(lf_hat))
                    #epoch_losses2C.update(lossTrp, len(x_hat))
                    #epoch_losses1A.update(d_loss.item(), len(fakeInpBatchG))
                    del loss1#,dc_loss#,d_loss#dc2_loss,regContinuity,                
    ###############################################
    # Train D
    ###############################################     
                nItD=0
                for dummInD in range(nItD):
   
                    for p in D.parameters():
                        p.requires_grad = True

                    for p in G.parameters():  
                        p.requires_grad = False

#                    for p in G2.parameters():  
#                        p.requires_grad = False
                    D.train()
                    #Dc.train()
                    #Dc2.train()

                    with torch.no_grad(): 
                                                 
                        indx1=np.random.randint(fakeVSeq.shape[2]-64+1)
                        indx2=np.random.randint(fakeVSeq.shape[3]-64+1)
                        indx3=np.random.randint(fakeVSeq.shape[4]-64+1)
                        fakeInpBatch=fakeVSeq[:,:,indx1:64+indx1,indx2:64+indx2,indx3:64+indx3].clone()

                        realInpBatch = volBatch[None,:,:,:,:].clone()
                        realInpBatch=realInpBatch.permute(1,0,3,4,2)
                        realInpBatch=realInpBatch/torch.max(torch.max(torch.max(realInpBatch,dim=4)[0],dim=3)[0],dim=2)[0].view(realInpBatch.shape[0],1,1,1,1)
                        realInpBatch=2*realInpBatch-1

                        realInpBatch=realInpBatch[:,:,indx1:64+indx1,indx2:64+indx2,indx3:64+indx3]



                    #x_lossR=D(realInpBatch)#.mean()
                    #x_lossH=D(fakeInpBatch)#.mean()
                    #loss2 =((torch.nn.functional.leaky_relu(-x_lossR+x_lossH+2,negative_slope=0.01))**2).mean()




                    tmpDR=D(realInpBatch)#.mean()
                    tmpDH=D(fakeInpBatch)#.mean()
                    
                    x_lossR=((tmpDR-1)**2).mean()
                    x_lossH=((tmpDH+1)**2).mean()

                    loss2=x_lossR+x_lossH#+0.1*gradient_penalty_h#+0.1*l1_reg#



                    d_optimizer.zero_grad()
                    loss2.backward(retain_graph=False)
                    d_optimizer.step()


                    epoch_losses2.update(loss2.item(), len(fakeInpBatch))
                    #epoch_losses2Pnl.update(gradient_penalty_h.item()/wGrad, len(x_hat))
                    epoch_losses2Pnl.update(x_lossR.mean().item(), len(fakeInpBatch))
                    #epoch_losses2Pnl.update((gradient_penalty_h.item()/wGrad), len(x_hat))
                    epoch_losses2Pnl2.update(x_lossH.mean().item(), len(fakeInpBatch))
                    #epoch_losses2Pnl2.update(l1_reg, len(x_hat))
                    #epoch_losses2Pnl3.update(gradient_penalty_hDc2.item()/wGrad2, len(x_hat))
                    del loss2,x_lossH#x_lossR,tmpDR,tmpDH

                #t.set_postfix(loss1='{:.2f}'.format(epoch_losses1.avg)+' loss2={:.2f}'.format(epoch_losses2.avg))
                t.set_postfix(loss='{:.6f}'.format(epoch_losses1.avg))

                t.update(len(lfBatch))


        #print('D: {:<8.3}'.format(loss2.data.cpu().numpy()), 
        #      'G: {:<8.3}'.format(loss1.data.cpu().numpy()),
        #      'gDc: {:<8.3}'.format(dc_loss.data.cpu().numpy()),
        #      'gD: {:<8.3}'.format(d_loss.data.cpu().numpy()),
#       #       'DLRc: {:<8.3}'.format(tmpDRc.data.cpu().numpy()),
#       #       'DLHc: {:<8.3}'.format(tmpDHc.data.cpu().numpy()),
        #      'penaltyDc: {:<8.3}'.format(gradient_penalty_hDc.data.cpu().numpy()/wGrad2),
        #      'w2: {:<8.3}'.format(wGrad2),
        #      'DLR: {:<8.3}'.format(tmpDR.data.cpu().numpy()),
        #      'DLH: {:<8.3}'.format(tmpDH.data.cpu().numpy()),
        #      'penalty: {:<8.3}'.format(gradient_penalty_h.data.cpu().numpy()/wGrad),
        #      'w1: {:<8.3}'.format(wGrad),
        #      'cG: {:<8.3}'.format(Dc.constantG.mean().data.cpu().numpy()),
        #      )
#        print(gradient_penalty_h.data.cpu()/wGrad)
#        print(gradient_penalty_hDc.data.cpu()/wGrad2)
        #meanX_loss2=meanX_loss2/numBatch
        # wGrad=wGrad+1e-1*numBatch*np.maximum(prevx_loss2-meanX_loss2,0.0)#1e-2 without patch, 1e-4 works withouth patch
        #wGrad=wGrad+1e-1*numBatch*np.abs(prevx_loss2-meanX_loss2)#1e-2 without patch, 1e-4
        #prevx_loss2=meanX_loss2
        #meanX_loss2=0.0


        lossGr1[epoch]=epoch_lossesGr1.avg
        lossGr2[epoch]=epoch_lossesGr2.avg
        lossGr3[epoch]=epoch_lossesGr3.avg
        lossGCEpc[epoch]=(epoch_losses1.avg-epoch_losses1A.avg)
        lossGAEpc[epoch]=epoch_losses1A.avg
        lossGCEpcDiv[epoch]=epoch_losses1C.avg
        lossGCEpc2Div[epoch]=epoch_losses1C2.avg
        lossGCSpec[epoch]=epoch_losses1C3.avg
        lossDEpc[epoch]=epoch_losses2.avg
        lossDcEpc[epoch]=epoch_losses2C.avg
        lossDPnlEpc[epoch]=epoch_losses2Pnl.avg
        lossDPnlEpc2[epoch]=epoch_losses2Pnl2.avg
        lossDPnlEpc3[epoch]=epoch_losses2Pnl3.avg
        wEpc[epoch]=wGrad
        w2Epc[epoch]=wGrad2
        inpLFEpoch[epoch]=epoch_lossesInpLF.avg
        inpTrEpoch[epoch]=epoch_lossesInpLTr.avg


        #constG[epoch]=Dc.constantG.mean()
        #constG2[epoch]=Dc2.constantG.mean()
        with torch.no_grad():

            size_input2Tmp=107#size_input2#69,81-12
            size_inputTmp=107#size_input#69,81-12
            size_label2Tmp=(size_input2Tmp-0)*upFa
            size_labelTmp=(size_inputTmp-0)*upFa
            indx1=np.random.randint(lfTrainF.shape[2]-size_input2Tmp+1)
            indx2=np.random.randint(lfTrainF.shape[3]-size_inputTmp+1)
            indx0=np.random.randint(0,28)#28,65,42
            locL1 =int(upFa*(indx1)+(upFa*size_input2Tmp-size_label2Tmp)/2)#-5*upFa
            locL2 =int(upFa*(indx2)+(upFa*size_inputTmp-size_labelTmp)/2)#-5*upFa
            indx0=14#3,14,10, with structural +4 in LF
            tmpVol= hrT[0,indx0:indx0+53,locL1: size_label2Tmp +locL1, locL2: size_labelTmp + locL2].float()
            tmpVol1= tmpVol
            mxtmpVol=torch.max(tmpVol1)
            #print(mxtmpVol)
            #print(errr)
            tmpVol1=tmpVol1/mxtmpVol
            #tmpVol=tmpVol/255.0
            #stdLF=torch.sqrt(torch.sum((tmpVol)**2))
            #tmpVol=(tmpVol)/(stdLF+_eps)

            tmpLf_real=lfTrainF[indx0+26+4,:,indx1: size_input2Tmp +indx1, indx2: size_inputTmp + indx2].float()
            #tmpLf_real[tmpLf_real>255]=255
            indx0=np.random.randint(lfTrainFTmp.shape[0])
            tmpLf_real=lfTrainFTmp[299,:,indx1: size_input2Tmp +indx1, indx2: size_inputTmp + indx2].float()

            #totLF=torch.zeros((28,361,91,91)).to(device)#,dtype=torch.uint8
            #for indx0 in range(28):
            #    #tmpVol1= hrT[0,indx0:indx0+53,locL1: size_label2Tmp +locL1, locL2: size_labelTmp + locL2].float()
            #    tmpVol1= hrT[0,indx0:indx0+53,locL1: size_label2Tmp +locL1, locL2: size_labelTmp + locL2].float()
            #    #tmpLf_real=lfTrainF[indx0+26,:,indx1: size_input2Tmp +indx1, indx2: size_inputTmp + indx2].float()
            #    #tmpVol_Hat=G(tmpLf_real[None,:,:,:])
            #    tmpVol1=torch.zeros_like(tmpVol1)
            #    tmpVol1[indx0+13,160,160]=1
            #    #tmpVol1[indx0+13-1:indx0+13+1+1,160-1:160+1+1,160-1:160+1+1]=1
            #    frwdHt=Fl(4785.2002*tmpVol1[None,:,:,:])
            #    #frwdHt=Fl(tmpVol_Hat)
            #    #mxtmpLF=torch.max(frwdHt)
            #    #frwdHt=255*frwdHt/mxtmpLF
            #    totLF[indx0]=frwdHt

            #scio.savemat('lfHat.mat', mdict={'totLF':totLF.cpu().numpy()})
            #print(errr)


            #quantiles=np.zeros((100))
            #for i in range(900,1000):
            #    totMax=0*tmpVol.max()
            #    for j in range(len(lfTrainFTmp)):#range(len(lfTrainFTmp)):
            #        tmpLf_real=lfTrainFTmp[j,:,:,:].float()
            #        tmpVol_Hat=G(tmpLf_real[None,:,:,:])
            #        tmpMax=torch.quantile(tmpVol_Hat,i/1000);
            #        #print(tmpMax)
            #        #print(err)
            #        if tmpMax>totMax:
            #            totMax=tmpMax
            #    print(i)
            #    quantiles[i-900]=totMax;
            #scio.savemat('quantiles.mat', mdict={'quantiles':quantiles})
            #print(err)
            #totMax=0*tmpVol.max()
            #for j in range(len(lfTrainFTmp)):#range(len(lfTrainFTmp)):
            #    tmpLf_real=lfTrainFTmp[j,:,:,:].float()
            #    tmpVol_Hat=G(tmpLf_real[None,:,:,:])
            #    tmpMax=tmpVol_Hat.max()
            #    #print(tmpMax)
            #    #tmpMax=torch.quantile(tmpVol_Hat,0.99995);#0.99999
            #    #print(tmpMax)
            #    #print(err)
            #    if tmpMax>totMax:
            #        totMax=tmpMax
            #print(totMax)
            ##print(err)
            #totVol=torch.zeros((500,53,321,321),dtype=torch.uint8).to(device)
            ##totVol=torch.zeros((125,53,321,321)).to(device)
                
            #for j in range(len(lfTrainFTmp)):#range(len(lfTrainFTmp)):
            ##for j in range(125):#range(len(lfTrainFTmp)):
            #    tmpLf_real=lfTrainFTmp[j,:,:,:].float()
            #    tmpVol_Hat=torch.nn.functional.relu(G(tmpLf_real[None,:,:,:]))
            #    tmpVol_Hat[tmpVol_Hat>totMax]=totMax
            #    tmpVol_Hat=255*tmpVol_Hat/totMax
            #    totVol[j]=tmpVol_Hat
                    
                
            ######print(totVol.dtype)
            #scio.savemat('volSeq42_.mat', mdict={'totVol':totVol.cpu().numpy()})
            #print(totVol.shape)
            #print(errr)

            #totVol=torch.zeros((len(lfTrainFTmp),53,9*3,9*3)).to(device)
            #for j in range(len(lfTrainFTmp)):#range(len(lfTrainFTmp)):

            #    tmpLf_real=lfTrainFTmp[j,:,42:51,39:48].float()
            #    tmpVol_Hat=torch.nn.functional.relu(G(tmpLf_real[None,:,:,:]))
            #    totVol[j]=tmpVol_Hat
                    
                
            ####print(totVol.dtype)
            #scio.savemat('volSeq42_.mat', mdict={'totVol':totVol.cpu().numpy()})
            #print(totVol.shape)
            #print(errr)




            augmLF=tmpLf_real
            augmLF=augmLF[None,:,:,:]
            augmLFShap=augmLF.shape
            augmLF=augmLF.view(augmLFShap[0],19,19,augmLFShap[2],augmLFShap[3]) 
            augmLF=augmLF.permute(0,2,1,3,4)#because reshape matlab is different


            strBn=format(epoch, '#06b')[2::]

            #swapAx=np.random.randint(2)
            swapAx=int(strBn[0])
            #reflec0=np.random.randint(2)
            reflec0=int(strBn[1])
            #reflec1=np.random.randint(2)
            reflec1=int(strBn[2])
            #reflec2=np.random.randint(2)
            reflec2=int(strBn[3])

            idxP=[i for i in range(augmLF.shape[3]-1, -1, -1)]
            idx19=[i for i in range(augmLF.shape[1]-1, -1, -1)]

            if reflec0==1:
                augmLF=augmLF[:,idx19,:,:,:]
                augmLF=augmLF[:,:,:,idxP,:]
            if reflec1==1:
                augmLF=augmLF[:,:,idx19,:,:]
                augmLF=augmLF[:,:,:,:,idxP]
            if reflec2==1:
                augmLF=augmLF[:,idx19,:,:,:]
                augmLF=augmLF[:,:,idx19,:,:]
            if swapAx==1:
                augmLF=augmLF.permute(0,2,1,4,3)  

            augmLF=augmLF.permute(0,2,1,3,4)#because reshape matlab is different
            augmLF=augmLF.reshape(augmLFShap[0],361,augmLFShap[2],augmLFShap[3])
            #tmpLf_real=augmLF[0]

            ##New Ground-Truth After First Training
            #totVol=torch.zeros((1,80,321,321)).to(device)
            #contSum=torch.zeros((1,80)).to(device)
            #for j in range(28):
            #    tmpLf_real=1*lfTrainF[j+26,:,:,:].float()
            #    tmpVol_Hat=G(tmpLf_real[None,:,:,:])
            #  #print(tmpVol_Hat.shape)
            #    #print(tmpLf_real.shape)
            #    totVol[0,j:j+53,:,:]=totVol[0,j:j+53,:,:]+tmpVol_Hat
            #    contSum[0,j:j+53]=contSum[0,j:j+53]+1

            totVol=torch.zeros((28,53,321,321)).to(device)

            for j in range(28):
                tmpLf_real=1*lfTrainF[j+30,:,:,:].float()
                tmpVol_Hat=G(tmpLf_real[None,:,:,:])
              #print(tmpVol_Hat.shape)
                #print(tmpLf_real.shape)
                totVol[j,:,:,:]=tmpVol_Hat
            scio.savemat('volTotForPSNR_.mat', mdict={'totVol':totVol.cpu().numpy()})
            print(errr)


            #totMax=0*tmpVol.max()
            #for j in range(len(lfTrainFTmp)):
            #    tmpLf_real=lfTrainFTmp[j,:,:,:].float()
            #    tmpVol_Hat=G(tmpLf_real[None,:,:,:])
            #    tmpVol_Hat=torch.nn.functional.pad(tmpVol_Hat,(8*3,8*3,8*3,8*3),'reflect')
            #    frwdHt=Fl(tmpVol_Hat)
            #    tmpMax=frwdHt.max()
            #    if tmpMax>totMax:
            #        totMax=tmpMax

            #totLF=torch.zeros((500,361,107,107),dtype=torch.uint8).to(device)
                
            #for j in range(len(lfTrainFTmp)):
            #    tmpLf_real=lfTrainFTmp[j,:,:,:].float()
            #    tmpVol_Hat=G(tmpLf_real[None,:,:,:])
            #    tmpVol_Hat=torch.nn.functional.pad(tmpVol_Hat,(8*3,8*3,8*3,8*3),'reflect')
            #    frwdHt=Fl(tmpVol_Hat)
            #    frwdHt=255*frwdHt/totMax
            #    totLF[j]=frwdHt                    
                
            #print(totLF.dtype)
            #scio.savemat('LFSeq.mat', mdict={'totLF':totLF.cpu().numpy()})
            #print(errr)

            #qntile20=torch.quantile(tmpLf_real,0.2);
            #print(qntile20)
            #tmpLf_real[tmpLf_real==0]=qntile20

            #x=G(tmpLf_real[None,:,:,:])
            #x=torch.nn.functional.pad(x,(4,0,4,0),'reflect')
            #xShap=x.shape
            #x=x.view(xShap[0]*xShap[1],1,xShap[2],xShap[3])
            #x=Fl.smooth3DInp(x)
            #x=x.view(xShap[0],xShap[1],x.shape[2],x.shape[3])

            tmpVol_Hat=G(tmpLf_real[None,:,:,:])
            #tmpVol_Hat=G(torch.nn.functional.pad(tmpLf_real[None,:,13:-13,13:-13],(13,13,13,13),'reflect'))

            #print(tmpLf_real.shape)
            #tmpVol_Hat=F.isra(tmpLf_real[None,:,:,:],8)
            #print(tmpVol_Hat.shape)
            #scio.savemat('volTot42_.mat', mdict={'tmpVol_Hat':tmpVol_Hat.cpu().numpy()})
            #scio.savemat('volTot.mat', mdict={'grt':tmpVol1.cpu().numpy(),'tmpVol_Hat':tmpVol_Hat.cpu().numpy()})
            #print(errr)
            #totVol=totVol/contSum.view(1,80,1,1)
            #mxtmpVol_Hat=torch.max(tmpVol_Hat)
            #tmpVol_Hat=tmpVol_Hat/mxtmpVol_Hat     
            #scio.savemat('totVolCyc.mat', mdict={'totVol':totVol.cpu().numpy()})
            #print(totVol.shape)
            #print(errr)
            #psnrTrn[epoch]=calc_psnr(tmpVol1,tmpVol_Hat)
            #tmpVol_Hat=torch.zeros_like(tmpVol_Hat)
            #tmpVol_Hat[0,19,121,121]=1 tmpLf_real
            #scio.savemat('volTot45.mat', mdict={'tmpVol_Hat':tmpVol_Hat.cpu().numpy(),'tmpVol1':tmpVol1.cpu().numpy(),'tmpLf_real':tmpLf_real.cpu().numpy()})
            #print(errr) 

            #tmpVolShap=tmpVol_Hat.shape
            #tmpVol_Hat=tmpVol_Hat.view(tmpVolShap[0],tmpVolShap[1],tmpVolShap[2]//3,3,tmpVolShap[3]//3,3) 
            #tmpVol_Hat=tmpVol_Hat.permute(0,1,3,5,2,4)
            #tmpVol_Hat=tmpVol_Hat.reshape(tmpVolShap[0],tmpVolShap[1]*9,tmpVolShap[2]//3,tmpVolShap[3]//3)
            #print(tmpVol_Hat.shape) 
            #print(errr) 

    
            #frwdHt=F.frwdApPadd(tmpVol_Hat)
            #tmpVol_Hat=torch.zeros_like(tmpVol_Hat)
            #tmpVol_Hat[0,epoch,103,103]=1e4
            #tmpVol_Hat[0,0:14,:,:]=0
            #tmpVol_Hat[tmpVol_Hat>=12]=0
            #print(Fl.kernelConv.shape)
            #print(errrr)
            #tmpVol_Hat=torch.roll(tmpVol_Hat, shifts=-10, dims=1)
            #tmpVol_Hat=torch.zeros_like(tmpVol_Hat)
            #tmpVol_Hat[0,12,160,160]=1
            #tmpVol_Hat[:,43,:,:]=0
            print(tmpVol_Hat.shape)
            frwdHt=(Fl(tmpVol_Hat))
            #frwdHt=Fl(torch.nn.functional.pad(tmpVol_Hat,(8*3,8*3,8*3,8*3),'reflect'))
            print(frwdHt.min())

            #tmpPd=torch.nn.functional.pad(frwdHt,(3,2,3, 2),'reflect')
            #tmpPd=F.haarPLF(tmpPd)[:,:,3:-2,3:-2]

            #frwdHt=frwdHt-tmpPd



            #frwdHt=Fl(tmpVol[None,:,:,:])

            #frwdHtPd=tmpLf_real[None,:,:,:].clone()
            #frwdHtPd[:,:,8:-8,8:-8]=frwdHt
            #frwdHtPd=Fl(torch.nn.functional.pad(tmpVol_Hat,(8*3,8*3,8*3,8*3),'reflect'))
            #tmpVol_Hat=G(frwdHtPd)

            #xShap=tmpVol_Hat.shape
            #x=tmpVol_Hat.view(xShap[0],xShap[1],xShap[2]//3,3,xShap[3]//3,3).clone()
            #x=x.permute(0,1,3,5,2,4)
            #x=x.reshape(xShap[0],xShap[1]*9,xShap[2]//3,xShap[3]//3)

            #frwdHt=Fl.convBias[2](Fl.convBias[1](Fl.convBias[0](x)))
            #frwdHt=Fl.convBias[6](Fl.convBias[5](Fl.convBias[4](Fl.convBias[3](frwdHt))))


            frwdGT=tmpLf_real[:,8:-8,8:-8]
            #tmpVol1=G(frwdGT[None,:,:,:])[0]

            print(frwdGT.shape)

            #tmpPd=torch.nn.functional.pad(frwdGT[None,:,:,:],(3,2,3, 2),'reflect')
            #tmpPd=F.haarPLF(tmpPd)[:,:,3:-2,3:-2]

            #frwdGT=frwdGT[None,:,:,:]-tmpPd
            #frwdGT=frwdGT[0]
            print(frwdGT.shape)

            #stdLF=torch.sqrt(torch.sum((frwdHt)**2,dim=(2,3)).view(frwdHt.shape[0],frwdHt.shape[1],1,1))  
            #frwdHt=(frwdHt)/(stdLF+_eps)

            #stdLF=torch.sqrt(torch.sum((frwdGT[None,:,:,:])**2,dim=(2,3)).view(1,frwdGT.shape[0],1,1))  
            #frwdGT=(frwdGT)/(stdLF+_eps)


            #tmpPd=torch.nn.functional.pad(frwdHt,(3,2,3, 2),'reflect')
            #tmpPd=F.haarPLF(tmpPd)[:,:,3:-2,3:-2]
            ##tmpPd=frwdHt-frwdGT-tmpPd
            #frwdHt=frwdHt-tmpPd

            #tmpPd=torch.nn.functional.pad(frwdGT[None,:,:,:],(3,2,3, 2),'reflect')
            #tmpPd=F.haarPLF(tmpPd)[:,:,3:-2,3:-2]
            ##tmpPd=frwdHt-frwdGT-tmpPd
            #frwdGT=frwdGT[None,:,:,:]-tmpPd
            #frwdGT=frwdGT[0,:,:,:]
            ##print(1e4*((tmpPd)**2).mean())
            ##print(errr)

            #tmpVol1=G(frwdGT[None,:,:,:])[0]
            #tmpVol_Hat=G(frwdHt)
            #scio.savemat('lfHat.mat', mdict={'tmpVol_Hat':frwdHt.cpu().numpy()})
            #print(errr)
            scio.savemat('volTot42_.mat', mdict={'grt':tmpVol1.cpu().numpy(),'tmpVol_Hat':tmpVol_Hat.cpu().numpy(),'frwdHt':frwdHt.cpu().numpy(),'frwdGT':frwdGT.cpu().numpy()})
            #print(errr)
            #size_input2Tmp=81#size_input2#81-12
            #size_inputTmp=81#size_input#81-12
            #size_label2Tmp=(size_input2Tmp-0)*upFa
            #size_labelTmp=(size_inputTmp-0)*upFa
            #indx1=np.random.randint(lfTrainFTmp.shape[2]-size_input2Tmp+1)
            #indx2=np.random.randint(lfTrainFTmp.shape[3]-size_inputTmp+1)
            #indx0=np.random.randint(0,28)

            #locL1 =int(upFa*(indx1)+(upFa*size_input2-size_label2)/2)#-5*upFa
            #locL2 =int(upFa*(indx2)+(upFa*size_input-size_label)/2)#-5*upFa

            #tmpVol= hrTTst[0,indx0:indx0+53,locL1: size_label2Tmp +locL1, locL2: size_labelTmp + locL2]
            #tmpVol1= tmpVol
            #mxtmpVol=torch.max(tmpVol1)
            #tmpVol1=tmpVol1/mxtmpVol

            #tmpLf_real=lfTrainFTmp[indx0+26,:,indx1: size_input2Tmp +indx1, indx2: size_inputTmp + indx2]#indx0+37
            #tmpVol_Hat=G(tmpLf_real[None,:,:,:])
            #mxtmpVol_Hat=torch.max(tmpVol_Hat)
            #tmpVol_Hat=tmpVol_Hat/mxtmpVol_Hat    

            #psnrTst[epoch]=calc_psnr(tmpVol1,tmpVol_Hat)


        #if (epoch)%10 ==0: 
        #torch.save(G.state_dict(), 'epochTmpOldV.pth.tar', _use_new_zipfile_serialization=False)
        #torch.save(Fl.state_dict(), 'epochFTmpOldV.pth.tar', _use_new_zipfile_serialization=False)
        #    torch.save(G.state_dict(),os.path.join(args.outputs_dir, 'epochG_{}.pth'.format(epoch//10)))
        #    #torch.save(D.state_dict(),os.path.join(args.outputs_dir, 'epochD_{}.pth'.format(epoch//10)))
        #    torch.save(Fl.state_dict(),os.path.join(args.outputs_dir, 'epochFl_{}.pth'.format(epoch//10)))
        #torch.save(G.state_dict(),os.path.join(args.outputs_dir, 'epochLast.pth'))
        #torch.save(Fl.state_dict(),os.path.join(args.outputs_dir, 'epochFLast.pth'))
        #scio.savemat('lossesSyn.mat', mdict={'psnrTrn':psnrTrn,'psnrTst':psnrTst,'GC': lossGCEpc, 'GCDiv': lossGCEpcDiv,'GC3Div':lossGCSpec,'GA': lossGAEpc,'D': lossDEpc,'GC2Div': lossGCEpc2Div,'w': wEpc,'DPnl': lossDPnlEpc,'w2': w2Epc,'DPnl2': lossDPnlEpc2,'DPnl3':lossDPnlEpc3,'constG':constG,'constG2':constG2,'lossGr1':lossGr1,'lossGr2':lossGr2,'lossGr3':lossGr3,'GATrain':GATrain,'GATest':GATest,'GCTrain':GCTrain,'GCTest':GCTest,'GC3Train':GC3Train,'GC3Test':GC3Test,'GC2Train':GC2Train,'GC2Test':GC2Test,'inpLFEpoch':inpLFEpoch,'inpTrEpoch':inpTrEpoch})

        #rl_hat=rl_hat.permute(0,2,3,1)[:,None,:,:,:]
        #realVolP=x_hat22.detach()
        #realsVolP=real_labl2LF[:,:,6:-6,6:-6].detach()
        #realVolP=hrLF[:,:,6:-6,6:-6].detach()
        #realVol=tmpX.permute(0,2,3,1).detach()
        #print(torch.min(realVol))
        print(tmpVol1.shape)
        print(tmpVol_Hat.shape)
        #print(errrr)
        feat = np.squeeze(tmpVol1.permute(1,2,0).data.cpu().numpy())
        feat = nib.Nifti1Image(feat,affine = np.eye(4))
        plotting.plot_img(feat,title="X_Real",cut_coords=[102,102, 35])
        plotting.show()

        #x_hat=G(torch.div(lfTrainCmpF,torch.norm(lfTrainCmpF,p=2,dim=(1,2,3)).view(lfTrainCmpF.shape[0],1,1,1))).detach()        
        #x_hat=trnspRef.detach()
        feat = np.squeeze(tmpVol_Hat[0].permute(1,2,0).data.cpu().numpy())
        feat=(feat-np.amin(feat))/(np.amax(feat)-np.amin(feat))
        feat = nib.Nifti1Image(feat,affine = np.eye(4))
        plotting.plot_img(feat,title="X_DEC",cut_coords=[102,102, 35])
        plotting.show()

        feat=tmpVol_Hat[0,40,:,:].squeeze()
        feat=np.squeeze(feat.data.cpu().numpy())
        plt.imshow(feat)
        plt.show()


        #print(G2.convFxM.weight.data[14,0,:,:])
        
        
        feat=frwdHt[0].permute(1,2,0)
        feat=np.squeeze(feat.data.cpu().numpy())
        plt.imshow(feat[:,:,240])
        plt.show()
        feat=frwdGT.permute(1,2,0)
        feat=np.squeeze(feat.data.cpu().numpy())
        plt.imshow(feat[:,:,240])
        plt.show()

        feat=frwdHt[0]
        print(frwdHt.shape)
        sizFt=91#15,17,31
        feat=feat.reshape(19,19,sizFt,sizFt)
        feat=feat.permute(2,1,3,0)
        feat=feat.reshape(19*sizFt,19*sizFt)
        feat=np.squeeze(feat.data.cpu().numpy())
        plt.imshow(feat)
        plt.show()

        feat=tmpLf_real
        print(tmpLf_real.shape)
        sizFt=107#15,17,31
        feat=feat.reshape(19,19,sizFt,sizFt)
        feat=feat.permute(2,1,3,0)
        feat=feat.reshape(19*sizFt,19*sizFt)
        feat=np.squeeze(feat.data.cpu().numpy())
        plt.imshow(feat)
        plt.show()

         #x_hat=torch.zeros(1,1,65,65,29).to(device)
         #x_hat[0,0,33,33,15]=1
         #x_hat[0,0,10,10,5]=1
         #x_hat[0,0,56,56,25]=1
         #scio.savemat('predGan.mat', mdict={'vol': x_hat[0].cpu().numpy(),'pred': (F.frwdAp(x_hat[0].permute(0,3,1,2))).cpu().numpy()})
#        print(torch.norm(lf_real[0]))
#        print(torch.norm(lf_hat[0]))
#        print(torch.norm(lf_real[0]-lf_hat[0]))


        #feat=lfTrpEvalHat[0].permute(1,2,0)
        #feat=np.squeeze(feat.data.cpu().numpy())
        #plt.imshow(feat[:,:,180])
        #plt.show()

        #israV=F.isra(F.decmprs(real_22))
        #if (epoch)%1 ==0: 
        #    lfEval=tmpLf_real
        #    lfEval=torch.sum(tmpLf_real,dim=0)[None,:,:]
        #    feat=lfEval.permute(1,2,0)
        #    feat=np.squeeze(feat.data.cpu().numpy())
        #    plt.imshow(feat[:,:])
        #    plt.show()

        #    lfEval=tmpLf_Hat
        #    lfEval=tmpLf_Hat.reshape(1,19,19,69,69)
        #    lfEval=lfEval.permute(0,3,2,4,1)
        #    lfEval=lfEval.reshape(1,19*69,19*69)
        #    feat=lfEval[0]#.permute(1,2,0)
        #    feat=np.squeeze(feat.data.cpu().numpy())
        #    plt.imshow(feat[:,:])
        #    plt.show()
        #    scio.savemat('predGan.mat', mdict={'lfInp': lfEval[0].cpu().numpy()})
            #print(tmpLf_real.shape)
            #print(tmpVol1.shape)
            #lfEval=torch.sum(tmpVol1,dim=0)[None,6*3:-6*3,6*3:-6*3]
        #    lfEval=tmpVol1[37,:,:][None,6*3:-6*3,6*3:-6*3]
        #    #print(lfEval.shape)
        #    feat=lfEval.permute(1,2,0)
        #    feat=np.squeeze(feat.data.cpu().numpy())
        #    plt.imshow(feat[:,:])
        #    plt.show()
#        tmpD1=hrLF[4:8,:,6:-6,6:-6]
#        tmpD1Mx=torch.max(torch.max(torch.max(tmpD1,dim=3)[0],dim=2)[0],dim=1)[0].view(tmpD1.shape[0],1,1,1)
        #tmpD1Min=torch.min(torch.min(torch.min(tmpD1,dim=3)[0],dim=2)[0],dim=1)[0].view(tmpD1.shape[0],1,1,1)
        #tmpD1=(tmpD1-tmpD1Min)/(tmpD1Mx-tmpD1Min)
#        tmpD1=torch.nn.functional.relu(tmpD1)
#        tmpD1=tmpD1/tmpD1Mx

#        tmpD2=x_hat22
#        tmpD2Mx=torch.max(torch.max(torch.max(tmpD2,dim=3)[0],dim=2)[0],dim=1)[0].view(tmpD2.shape[0],1,1,1)
#        tmpD2Min=torch.min(torch.min(torch.min(tmpD2,dim=3)[0],dim=2)[0],dim=1)[0].view(tmpD2.shape[0],1,1,1)
        #tmpD2=(tmpD2-tmpD2Min)/(tmpD2Mx-tmpD2Min)
#        tmpD2=tmpD2/tmpD2Mx
#        tmpD2=torch.nn.functional.relu(tmpD2)
        #print(tmpD2Mx)
        #print(tmpD2Min)
        #print(torch.norm(tmpD1[0]-tmpD2[0]))
        #print(torch.norm(tmpD1[1]-tmpD2[1]))
        #print(torch.norm(tmpD1[2]-tmpD2[2]))
        #print(torch.norm(tmpD1[3]-tmpD2[3]))
       
#        print(calc_psnr(tmpD2[0], tmpD1[0]))
#        print(calc_psnr(tmpD2[1], tmpD1[1]))
#        print(calc_psnr(tmpD2[2], tmpD1[2]))
#        print(calc_psnr(tmpD2[3], tmpD1[3]))
        #del tmtrpEval,tmpGCBoth,stdVol,lfEvalHat,lfEval,stdLfreal,tmpPd,indxGATrn,indxGATst
        torch.cuda.empty_cache() 
#        scio.savemat('predGan.mat', mdict={'volReal': hrLF[3,:,6:-6,6:-6].cpu().numpy(),'vol': x_hat22[0].detach().cpu().numpy(),'volISRA': israV[0].cpu().numpy()})
        #scio.savemat('predGan2.mat', mdict={'predLF': lf_hat.data[0].cpu().numpy(),'realLF': lf_real.data[0].cpu().numpy()})
        #scio.savemat('predGan.mat', mdict={'vol': tmpD2[7].cpu().numpy(),'volISRA': trpEval[7].cpu().numpy()})
        #scio.savemat('predGan.mat', mdict={'grt': lfEval[7].cpu().numpy(),'israLF': lfTrpEvalHat[7].cpu().numpy(),'volLF': lfEvalHat[7].cpu().numpy()})
