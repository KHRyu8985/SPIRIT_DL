""" Training E2E VarNet (Initial Simple Version) """
import matplotlib, pathlib, torch, argparse, os, sys, logging
matplotlib.use('Agg')
sys.path.append(os.getcwd())
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

from logging import FileHandler
from vlogging import VisualRecord
import logging

from sigpy import from_pytorch, to_pytorch_function
import fastmri
from torch.autograd import Variable

import fastmri
from fastmri.data import subsample
from fastmri.data import transforms, mri_data
from fastmri.data import SliceDataset

from torch.utils.data import DataLoader, Dataset
from VN_SPIRIT.utils.network import VarNet2
from VN_SPIRIT.utils.dataset import Data2D

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import json
import robust_loss_pytorch.general


def create_argparser():

    parser = argparse.ArgumentParser(description='Training Original VarNet Reconstruction')

    parser.add_argument('--num-epochs', type=int,
                        default=100, help='Number of Epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')    
    parser.add_argument(
            '--verbose',
            action='store_true',
            help='verbose printing (default: False)')
    parser.add_argument('--loss', type=str,
                    default='l1', help='l1 or adaptive')
    parser.add_argument('--name', type=str, default='E2EVN_brain2D',
                        help='Weight_name') 
    
    return parser


def train_epoch(epoch, data_loader, net, optimizer, criterion, args, adaptive=None):

    avg_loss = 0.
    for iter, (ksp, mask, sens, labels) in enumerate(data_loader):
        ksp, mask, sens, labels = Variable(ksp, requires_grad=True).cuda(), Variable(mask).cuda(), Variable(sens, requires_grad=True).cuda(), Variable(labels, requires_grad=True).cuda()
        
        optimizer.zero_grad()
        out, ksp_out = net(ksp, mask, sens)

        if args.loss == 'adaptive':
            loss = torch.mean(adaptive.lossfun(torch.flatten(out - labels)[:,None]))
        else:
            loss = criterion(out.unsqueeze(1), labels.unsqueeze(1))            

        loss.backward()
        optimizer.step()
        avg_loss = 0.95 * avg_loss + 0.05 * loss.item() if iter > 0 else loss.item()

        if iter % 1000 == 0:
            if epoch == 0:
                logger.info('iter: [{}/{}], Train Loss: {}'.format(iter, len(data_loader), avg_loss))

    return avg_loss
   
    
def validate_epoch(data_loader, net, criterion, args, adaptive=None):

    avg_loss = 0.

    with torch.no_grad():
        for ksp, mask, sens, labels in data_loader:
            ksp, mask, sens, labels = Variable(ksp, requires_grad=True).cuda(), Variable(
                mask).cuda(), Variable(sens, requires_grad=True).cuda(), Variable(labels, requires_grad=True).cuda()

            out, _ = net(ksp, mask, sens)

            if args.loss == 'adaptive':
                loss = torch.mean(adaptive.lossfun(torch.flatten(out - labels)[:,None]))
            else:
                loss = criterion(out.unsqueeze(1), labels.unsqueeze(1))            

            avg_loss += loss.item()

    avg_loss = avg_loss / len(data_loader)

    return avg_loss


def ifftrecon(ksp):
    ksp_np = from_pytorch(ksp[0].cpu(), iscomplex=True)
    im = sp.ifft(ksp_np,axes=(1,2))
    im_comb = sp.rss(im,axes=0)
    im_comb = sp.resize(im_comb,[320,320])
    return im_comb


def test_result(idx, testset, net):
    ksp, mask, sens, labels = testset[idx]
    ksp = ksp.unsqueeze(0).cuda()
    mask = mask.unsqueeze(0).cuda()
    sens = sens.unsqueeze(0).cuda()
    
    with torch.no_grad():
        out, ksp_out = net(ksp, mask, sens)
    out_np = from_pytorch(out[0].cpu().detach(),iscomplex=True)
#    zf = ifftrecon(ksp)
    labels = from_pytorch(labels,iscomplex=True)
    
    out_cat = np.concatenate((np.abs(out_np).squeeze(), np.abs(labels).squeeze()),1)
    error_cat = np.concatenate((np.abs(labels).squeeze(), np.abs(labels).squeeze()),1)
    error_cat = np.abs(error_cat - out_cat) * 5
    out_cat = np.concatenate((out_cat,error_cat),axis=0)
    out_cat = out_cat * 600
    return out_cat



def main(args):
    """ Creating a masking function """

    save_name = '../logs/train_' + args.name + '.html'
    fh = FileHandler(save_name, mode="w")
    logger.addHandler(fh)    
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    net = VarNet2()
    net = net.cuda()

    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
        criterion = criterion.cuda()
        optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
        adaptive = None
    elif args.loss == 'adaptive':
        adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
    num_dims = 1, float_dtype=np.float32, device='cuda:0')
        params = list(net.parameters()) + list(adaptive.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)
        criterion = None
        
    trainset = Data2D('../data/div_brain2d/Train', dset_type='knee2d')

    validset = Data2D('../data/div_brain2d/Val', dset_type='knee2d')
    
    testset = Data2D('../data/div_brain2d/Test', dset_type='knee2d')
    
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=8)

    validloader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=8)
    
    logger.info('Training .... ')

    for epoch in range(args.num_epochs):
        avg_loss = train_epoch(
            epoch, trainloader, net, optimizer, criterion, args, adaptive)
        logger.info('Epoch: {}, Train Loss: {}'.format(epoch, avg_loss))
        
        val_loss = validate_epoch(
            validloader, net, criterion, args, adaptive)
        logger.info('Epoch: {}, Val Loss: {}'.format(epoch, val_loss))


        if epoch % 5 == 0:
            for idx in [10,20,30,40,50]:
                out_cat = test_result(idx,testset,net)
                logger.debug(VisualRecord(
                    "epoch: {}, slice:{}".format(epoch, idx), out_cat, fmt="png"))
            
            # Save network to weight
            weight_name = '../exp/' + args.name + '.pt'
            torch.save(net, weight_name)

if __name__ == "__main__":
    args = create_argparser().parse_args()
    logger = logging.getLogger('Training-VN-SPIRIT-Brain2D')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

    main(args)
