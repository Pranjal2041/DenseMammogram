import shutil
import os
from os.path import join


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_backup(folders = None, files = None, backup_dir = 'experiments'):
    if folders is None:
        folders = ['.', 'corr_lists','detection']
    if files is None:
        files = ['.py', '.txt', '.json','.cfg']

    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(tuple(files)):
                if folder != '.':
                    src = join(folder, file)
                    dest = join(backup_dir, folder, file)
                else:
                    src = file
                    dest = join(backup_dir, file)
                os.makedirs(os.path.split(dest)[0], exist_ok=True)
                shutil.copy(src, dest)