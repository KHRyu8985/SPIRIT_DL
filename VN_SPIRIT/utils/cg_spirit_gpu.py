import numpy as np
import sigpy as sp
from sigpy import from_pytorch, to_pytorch, to_pytorch_function
import scipy.io as io
import numpy as np
from skimage.util import pad, view_as_windows
import cupy
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import fastmri.data.transforms as T
from functools import partial

def torch_VCC(ksp, caxis=1):
    vcc_ksp = ksp * 1
    vcc_ksp[...,1] = -1 * vcc_ksp[...,1]
    vcc_ksp = vcc_ksp.flip([2,3])
    vcc_ksp = vcc_ksp.roll([1,1],[2,3])
    out = torch.cat([ksp,vcc_ksp],caxis)
    return out

def VCC(kspace, caxis=2):
    vcc_ksp = kspace.copy()
    vcc_ksp = np.conj(vcc_ksp[::-1,::-1])
    vcc_ksp = np.roll(vcc_ksp,1,0)
    vcc_ksp = np.roll(vcc_ksp,1,1)
    out = np.concatenate((kspace,vcc_ksp), axis=caxis)
    return out


def dat2AtA(data, kernel_size):
    '''Computes the calibration matrix from calibration data.
    '''

    tmp = im2row(data, kernel_size)
    tsx, tsy, tsz = tmp.shape[:]
    A = np.reshape(tmp, (tsx, tsy*tsz), order='F')
    return np.dot(A.T.conj(), A)


def im2row(im, win_shape):
    '''res = im2row(im, winSize)'''
    sx, sy, sz = im.shape[:]
    wx, wy = win_shape[:]
    sh = (sx-wx+1)*(sy-wy+1)
    res = np.zeros((sh, wx*wy, sz), dtype=im.dtype)

    count = 0
    for y in range(wy):
        for x in range(wx):
            # res[:, count, :] = np.reshape(
            #     im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz), order='F')
            res[:, count, :] = np.reshape(
                im[x:sx-wx+x+1, y:sy-wy+y+1, :], (sh, sz))
            count += 1
    return res

def calibrate_single_coil(AtA, kernel_size, ncoils, coil, lamda, sampling=None):

    kx, ky = kernel_size[:]
    if sampling is None:
        sampling = np.ones((*kernel_size, ncoils))
    dummyK = np.zeros((kx, ky, ncoils))
    dummyK[int(kx/2), int(ky/2), coil] = 1

    idxY = np.where(dummyK)
    idxY_flat = np.sort(
        np.ravel_multi_index(idxY, dummyK.shape, order='F'))
    sampling[idxY] = 0
    idxA = np.where(sampling)
    idxA_flat = np.sort(
        np.ravel_multi_index(idxA, sampling.shape, order='F'))

    Aty = AtA[:, idxY_flat]
    Aty = Aty[idxA_flat]

    AtA0 = AtA[idxA_flat, :]
    AtA0 = AtA0[:, idxA_flat]

    kernel = np.zeros(sampling.size, dtype=AtA0.dtype)
    lamda = np.linalg.norm(AtA0)/AtA0.shape[0]*lamda
    rawkernel = np.linalg.solve(AtA0 + np.eye(AtA0.shape[0])*lamda, Aty) # fast 1s

    kernel[idxA_flat] = rawkernel.squeeze()
    kernel = np.reshape(kernel, sampling.shape, order='F')

    return(kernel, rawkernel)

def spirit_calibrate(acs,kSize, lamda=0.01):
    nCoil = acs.shape[-1]
    AtA = dat2AtA(acs,kSize)
    spirit_kernel = np.zeros((nCoil,nCoil,*kSize),dtype='complex128')
    for c in range(nCoil):
        tmp, _ = calibrate_single_coil(AtA,kernel_size=kSize,ncoils=nCoil,coil=c,lamda=lamda)
        spirit_kernel[c] = np.transpose(tmp,[2,0,1])
    spirit_kernel = np.transpose(spirit_kernel,[2,3,1,0]) # Now same as matlab!
    GOP = np.transpose(spirit_kernel[::-1,::-1],[3,2,0,1])
#    GOP = GOP.copy()
#    for n in range(ncoil):
#        GOP[n,n,kSize[0]//2,kSize[1]//2] = -1    
    return GOP

def Dc(x, m):
    return (1-m) * x 

def D(x, m):
    return m * x 

def GOPHGOP(x: torch.Tensor, GOP: torch.Tensor) -> torch.Tensor:
    GOP = from_pytorch(GOP, iscomplex=True)
    
    GOP_op = sp.linop.ConvolveData(data_shape = x.shape[:-1], filt = GOP, multi_channel=True, mode='full')
    GOP_opH = sp.linop.ConvolveDataAdjoint(data_shape = x.shape[:-1], filt = GOP, multi_channel=True, mode='full')
    
    torch_GOP_op = to_pytorch_function(GOP_op, input_iscomplex=True, output_iscomplex=True).apply
    torch_GOP_opH = to_pytorch_function(GOP_opH, input_iscomplex=True, output_iscomplex=True).apply
    
    out = torch_GOP_opH(torch_GOP_op(x))
    
    return out  

def AHA(x,gop,m):
    x_mask = Dc(x,m) # fill unacquired
    x_normal = GOPHGOP(x_mask,gop)
    return Dc(x_normal,m)

def AHb(x,m,gop):
    out = GOPHGOP(x, gop)
    return - Dc(out,m)

# conjugate gradient algorithm
def dot(x1, x2):
    return torch.sum(x1*x2)


def ip(x):
    return dot(x, x)


def dot_batch(x1, x2):
    return torch.sum(x1*x2, dim=list(range(0, len(x1.shape))))


def ip_batch(x):
    return dot_batch(x, x)


def conjgrad(niter, AHA, b, x, l2lam=0., device='cuda:0'):

    if l2lam > 0:
        r = b - (AHA(x) + l2lam * x)
    else:
        r = b - AHA(x)

    p = r

    rsnot = ip_batch(r)
    error = ip_batch(b - AHA(x) - l2lam * x) / ip_batch(b)

    rsold = rsnot
    rsnew = rsnot
    del rsnot

    reshape = (-1,) + (1,) * (len(x.shape) - 1)
    for i in range(niter):

        if l2lam > 0:
            Ap = AHA(p) + l2lam * p
        else:
            Ap = AHA(p)

        alpha = (rsold / dot_batch(p, Ap)).reshape(reshape)

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = ip_batch(r)

        beta = (rsnew / rsold).reshape(reshape)

        rsold = rsnew

        p = beta * p + r
        error = ip_batch(b - AHA(x) - l2lam * x) / ip_batch(b)

    return x


def SS_refine(ksp, ksp_out, sens, center_fraction = 0.08, algorithm='spirit', l2lam=0.05, kernel_size=[5,5]):
    
    with torch.no_grad():
        ksp_us_vcc = torch_VCC(ksp)
        ksp_out_vcc = torch_VCC(ksp_out)
    ksp_us_vcc_np = from_pytorch(ksp_us_vcc.cpu().detach(), iscomplex=True)[0]
    mask = np.abs(ksp_us_vcc_np) > 0
    
    # Performing calibration
    ksp_vn_np = np.moveaxis(from_pytorch(ksp_out_vcc.detach().cpu(),iscomplex=True)[0],0,-1)
    nro, npe, ncoil = ksp_vn_np.shape
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    calib = sp.resize(ksp_vn_np,[nro,round(npe*center_fraction),ncoil]) # This is the calibration region

    # Preparing calibration kernel

    if algorithm == 'spirit':
        GOP = spirit_calibrate(calib, kernel_size, lamda=0.01)
        GOP = GOP.copy()
        for n in range(ncoil):
            GOP[n,n,kx2,kx2] = -1
#        print('SPIRIT kernel: {}'.format(GOP.shape))
    else:
        C = view_as_windows(
            calib, (kx, ky, ncoil)).reshape((-1, kx*ky*ncoil))
        n = null_space(C, rcond=0)
        n = n[:,-100:] # choosing only 200
        n = np.reshape(n, (kx, ky, ncoil, n.shape[-1]))
        GOP = np.transpose(n[::-1,::-1],[3,2,0,1])
#s        print('PRUNO kernel: {}'.format(GOP.shape))    
        
        
    GOP_t = T.to_tensor(GOP)
    mask_t = T.to_tensor(mask)
    mask_t = mask_t.unsqueeze(0).unsqueeze(-1)

    # To GPU
    GOP_t = GOP_t.type(torch.FloatTensor).to('cuda:0')
    mask_t = mask_t.type(torch.FloatTensor).to('cuda:0')


    # This is solving Ax = b by conjugate gradient algorithm
    with torch.no_grad():
        ATA = partial(AHA, gop=GOP_t, m=mask_t)
        b = AHb(ksp_us_vcc, mask_t, GOP_t) + l2lam * ksp_out_vcc
        res = conjgrad(30, ATA, b, ksp_us_vcc, l2lam=l2lam, device='cuda:0')
    
    # Combining sensitivity
    res_np = from_pytorch(res.detach().cpu(),iscomplex=True)[0][:res.shape[1]//2]
    sens_np = from_pytorch(sens.detach().cpu(), iscomplex=True)[0]
    im_coil = sp.ifft(res_np, axes=(1,2))
    res_vn_spirit = np.sum(im_coil * np.conj(sens_np), axis=0)
    
    return res_vn_spirit, res_np
