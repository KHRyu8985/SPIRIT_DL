"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import sigpy.mri
import numpy as np
import torch
from math import floor, ceil

class MaskFunc:
    """
    Abstract MaskFunc class for creating undersampling masks of a specified shape.
    """

    def __init__(self, accelerations):
        """
        Args:
            accelerations (List[int]): Range of acceleration rates to simulate.
        """
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def choose_acceleration(self):
        """
        Chooses a random acceleration rate given a range.
        """
        if type(self.accelerations) == int:
            acceleration = self.accelerations
        else:
            accel_range = self.accelerations[1] - self.accelerations[0]
            acceleration = self.accelerations[0] + accel_range*self.rng.rand()
        return acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a 2D uniformly random undersampling mask.
    """
    def __init__(self, accelerations, calib_size):
        super().__init__(accelerations)
        self.calib_size = calib_size

    def __call__(self, out_shape, seed=None):
        #self.rng.seed(seed)

        # Design parameters for mask
        nky = out_shape[1]
        nkz = out_shape[2]
        acceleration = self.choose_acceleration()
        prob = 1.0 / acceleration

        # Generate undersampling mask
        mask = torch.rand([nky, nkz], dtype=torch.float32)
        mask = torch.where(mask < prob, torch.Tensor([1]), torch.Tensor([0]))

        # Add calibration region
        calib = [self.calib_size, self.calib_size]
        mask[int(nky / 2 - calib[-2] / 2):int(nky / 2 + calib[-2] / 2),
             int(nkz / 2 - calib[-1] / 2):int(nkz / 2 + calib[-1] / 2)] = torch.Tensor([1])


        return mask.reshape(out_shape)


class PoissonDiskMaskFunc(MaskFunc):
    """
    PoissonDiskMaskFunc creates a 2D Poisson disk undersampling mask.
    """
    def __init__(self, accelerations, calib_size):
        super().__init__(accelerations)
        self.calib_size = [calib_size, calib_size]

    def __call__(self, out_shape, seed=None):
        #self.rng.seed(seed)

        # Design parameters for mask
        nky = out_shape[0]
        nkz = out_shape[1]
        acceleration = self.choose_acceleration()
        # Generate undersampling mask
        mask = sigpy.mri.poisson([nky, nkz], acceleration,
                calib=self.calib_size,
                dtype=np.float32)

        return mask    
    
