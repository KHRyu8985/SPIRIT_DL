"""Data preparation for training."""
import os
import h5py
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import logging
from sklearn.model_selection import train_test_split
from natsort import natsorted
import glob, json, argparse
import numpy as np
import shutil

logger = logging.getLogger('Data-Preparation-VN-SPIRIT')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
file_handler = logging.FileHandler('../logs/data-prep.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preparation')
    parser.add_argument(
        '--dset-type',
        default='knee2d',
        help='knee2d or brain2d or knee3d')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of test size')
    parser.add_argument('--random-seed', default=1000, help='Random seed')
    parser.add_argument('--copy-data', action='store_true', help='Copy data to train, val, test')    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='verbose printing (default: False)')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.random_seed >= 0:
        np.random.seed(args.random_seed)

    folder_name = 'raw_' + args.dset_type
    folder_dir = os.path.join('../data', folder_name)
    rawfiles = natsorted(glob.glob(os.path.join(folder_dir, '*.h5')))
    logger.info('Number of rawfiles found {}...'.format(len(rawfiles)))

    train_val_rawfile, test_rawfile = train_test_split(
        rawfiles, test_size=args.test_size, random_state=args.random_seed)

    train_rawfile, val_rawfile = train_test_split(
        train_val_rawfile, test_size=0.1, random_state=args.random_seed)

    logger.info('<Dataset Size> Train:{}, Val:{}, Test:{}'.format(len(train_rawfile), len(val_rawfile), len(test_rawfile)))
    
    cfg_name = args.dset_type + '.json'
    cfg_path = os.path.join('../config', cfg_name)
    
    logger.debug('dumping the rawfiles into {}...'.format(cfg_path))
    
    config = {"train": train_rawfile, "val": val_rawfile, "test":test_rawfile}

    with open(cfg_path, 'w') as f:
        json.dump(config, f)
        
    if args.copy_data:
        base_foldername = 'div_' + args.dset_type
        base_folderpath = os.path.join('../data', base_foldername)
        
        if os.path.exists(base_folderpath):
            shutil.rmtree(base_folderpath) # delete previous folder and all subdirectories

        trainpath = os.path.join(base_folderpath, 'Train')
        validpath = os.path.join(base_folderpath, 'Val')
        testpath = os.path.join(base_folderpath, 'Test')
        
        os.mkdir(base_folderpath)
        os.mkdir(trainpath)
        os.mkdir(validpath)
        os.mkdir(testpath)
                                 
        for f in train_rawfile:
            filename = f.split('/')[-1]
            logger.info('Transferring file: {} to Train'.format(filename))
            shutil.copy(f, os.path.join(trainpath, filename))
        
        for f in val_rawfile:
            filename = f.split('/')[-1]
            logger.info('Transferring file: {} to Valid'.format(filename))
            shutil.copy(f, os.path.join(validpath, filename))

        for f in test_rawfile:
            filename = f.split('/')[-1]
            logger.info('Transferring file: {} to Test'.format(filename))
            shutil.copy(f, os.path.join(testpath, filename))
            
        
        
        
        