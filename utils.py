import logging
import shutil
import torch
import os
def setup_logging(log_file='log.txt'):
    """
    Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",   
                        datefmt="%Y-%m-%d %H:%M:%S",  
                        filename=log_file,
                        filemode='w')  

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
def save_checkpoint(state, is_best, save_path):
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'), os.path.join(save_path, 'best.pth.tar'))
        
# Dacay learning_rate
def lr_scheduler(optimizer, epoch, lr_decay_epoch=50, decay_factor=0.1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_factor
    return optimizer