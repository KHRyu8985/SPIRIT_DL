import numpy as np
import os 
import h5py, pathlib
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import glob, json

import fastmri
from fastmri.data import subsample
from fastmri.data import transforms, mri_data
from fastmri.data import SliceDataset

from torch.utils.data import DataLoader, Dataset
import torch
import sigpy as sp

import xml.etree.ElementTree as etree
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from VN_SPIRIT.utils.subsample import PoissonDiskMaskFunc

class Data2D(Dataset):
    def __init__(self, root, mask_func = None, dset_type='knee2d'):
        super().__init__()
        self.examples = []
        Files = list(pathlib.Path(root).glob('*.h5'))
        for fname in sorted(Files):
            data = h5py.File(fname, 'r')
            ksp = data['kspace']

            num_slices = ksp.shape[0]
            self.examples += [(fname, slice_num)
                              for slice_num in range(num_slices)]
        if mask_func == None:
            self.mask_func = subsample.EquispacedMaskFunc(
                            center_fractions=[0.08,0.06,0.08],
                            accelerations=[4,6,8])
        else:
            self.mask_func = mask_func
        self.dset_type = dset_type

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname, sl = self.examples[idx]
        with h5py.File(fname, 'r') as hr:
            kspace, sens = hr['kspace'][sl], hr['sens'][sl]
#            if self.dset_type == 'brain2d':
#                kspace = kspace[:,::2]
            et_root = etree.fromstring(hr["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max
            attrs = dict(hr.attrs)
            max_value = attrs['max']
            
        kspace = kspace / max_value
        
        im_coil = sp.ifft(kspace, axes=[1, 2])
        im_comb = np.sum(im_coil * np.conj(sens), axis=0)
        
        mask = self.mask_func(list(im_comb.shape) + [1])[...,0] # weird trick
        mask[:,:padding_left] = 0
        mask[:,padding_right:] = 0
                              
        mask = np.expand_dims(mask, axis=0)
        masked_kspace = kspace * mask
        mask = np.expand_dims(mask, axis=-1)
        
        masked_kspace = transforms.to_tensor(masked_kspace)
        mask = transforms.to_tensor(mask)
        sens = transforms.to_tensor(sens)
        im_comb = np.expand_dims(im_comb, axis=0)
        im_comb = transforms.to_tensor(im_comb)
                        
        return masked_kspace, mask.byte(), sens, im_comb

class Subject3D(Dataset):
    def __init__(self, fname, acc=[9, 15]):
        super().__init__()
        self.examples = []

        data = h5py.File(fname, 'r')
        ksp = data['kspace']

        num_slices = ksp.shape[0]
        self.examples += [(fname, slice_num)
                          for slice_num in range(num_slices)]
        self.mask_func = PoissonDiskMaskFunc(accelerations=acc,
                                             calib_size=20)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname, sl = self.examples[idx]
        with h5py.File(fname, 'r') as hr:
            kspace, sens = hr['kspace'][sl], hr['sens'][sl]
        kspace = kspace * 1e-3
        
        im_coil = sp.ifft(kspace, axes=[1, 2])
        im_comb = np.sum(im_coil * np.conj(sens), axis=0)
        
        mask = self.mask_func(im_comb.shape)
        mask = np.expand_dims(mask, axis=0)
        masked_kspace = kspace * mask
        mask = np.expand_dims(mask, axis=-1)
        
        masked_kspace = transforms.to_tensor(masked_kspace)
        mask = transforms.to_tensor(mask)
        sens = transforms.to_tensor(sens)
        im_comb = np.expand_dims(im_comb, axis=0)
        im_comb = transforms.to_tensor(im_comb)
                        
        return masked_kspace, mask.byte(), sens, im_comb    
    
class Subject3DWhole(Dataset):
    def __init__(self, fname, acc=[9, 15]):
        super().__init__()
        self.examples = []

        data = h5py.File(fname, 'r')
        ksp = data['kspace']

        num_slices = ksp.shape[0]
        self.examples = fname
        self.mask_func = PoissonDiskMaskFunc(accelerations=acc,
                                             calib_size=20)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname = self.examples
        with h5py.File(fname, 'r') as hr:
            kspace, sens = hr['kspace'], hr['sens']
        kspace = kspace * 1e-3 # Too high value
        
        im_coil = sp.ifft(kspace, axes=[1, 2])
        im_comb = np.sum(im_coil * np.conj(sens), axis=0)
        
        mask = self.mask_func(im_comb.shape)
        mask = np.expand_dims(mask, axis=0)
        masked_kspace = kspace * mask
        mask = np.expand_dims(mask, axis=-1)
        
        masked_kspace = transforms.to_tensor(masked_kspace)
        mask = transforms.to_tensor(mask)
        sens = transforms.to_tensor(sens)
        im_comb = np.expand_dims(im_comb, axis=0)
        im_comb = transforms.to_tensor(im_comb)
                        
        return masked_kspace, mask.byte(), sens, im_comb    
        
    
    
def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)