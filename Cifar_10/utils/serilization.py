import errno
import os
import shutil
import torch


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    fpath = '_'.join(str(state['epoch']), filename)
    fpath = os.path.join(save_dir, fpath)
    mkdir_if_missing(save_dir)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, os.path.join(save_dir, 'model_best.pth.tar'))
