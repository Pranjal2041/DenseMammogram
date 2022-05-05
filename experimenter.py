# Experimenter Class is responsible for mainly four things:
# 1. Configuration - Done
# 2. Logging using the AdvancedLogger class - Almost Done
# 3. Model Handling, including loading and saving models - Done(Upgrades Left)
# 4. Running Different Variants Paralelly/Sequentially of experiments
# 5. Combining frcnn training followed by bilateral training and final froc calculation - Done
# 6. Version Control

from advanced_config import AdvancedConfig
from advanced_logger import AdvancedLogger, LogPriority
import os
from os.path import join
from plot_froc import plot_froc
from train_frcnn import main as TRAIN_FRCNN
from train_bilateral import main as TRAIN_BILATERAL
import torch
from model_utils import generate_predictions, generate_predictions_bilateral
import argparse
from dataloaders import get_dict
from utils import create_backup
from torch.utils.tensorboard import SummaryWriter

class Experimenter:

    def __init__(self, cfg_file, BASE_DIR = 'experiments'):
        self.cfg_file = cfg_file        
        
        self.con = AdvancedConfig(cfg_file)
        self.config = self.con.config
        self.exp_dir = join(BASE_DIR,self.config['EXP_NAME'])
        os.makedirs(self.exp_dir, exist_ok=True)
        self.con.save(join(self.exp_dir,'config.cfg'))
    
        self.logger = AdvancedLogger(self.exp_dir)
        self.logger.log('Experiment:',self.config['EXP_NAME'],priority = LogPriority.STATS)
        self.logger.log('Experiment Description:', self.config['EXP_DESC'], priority = LogPriority.STATS)
        self.logger.log('Config File:',self.cfg_file, priority = LogPriority.STATS)
        self.logger.log('Experiment started', priority = LogPriority.LOW)
        self.losses = dict()
        self.frocs = dict()

        self.writer = SummaryWriter(join(self.exp_dir,'tensor_logs'))

        create_backup(backup_dir=join(self.exp_dir,'scripts'))

    def log(self, *args, **kwargs):
        self.logger.log(*args, **kwargs)


    def init_losses(self,mode):
        if mode == 'FRCNN' or mode == 'FRCNN_BILATERAL':
            self.losses['frcnn_loss'] = []
            self.frocs['frcnn_froc'] = []
        elif mode == 'BILATERAL' or mode == 'FRCNN_BILATERAL':
            self.losses['bilateral_loss'] = []
            self.frocs['bilateral_froc'] = []

    def start_epoch(self):
        self.curr_epoch += 1
        self.logger.log('Epoch:',self.curr_epoch, priority = LogPriority.MEDIUM)

    def end_epoch(self, loss, model = None, device = None):
        if self.curr_mode == 'FRCNN':
            self.losses['frcnn_loss'].append(loss)
            self.best_loss = min(self.losses['frcnn_loss'])
            if self.config['EVAL_METHOD'] == 'FROC':
                exp_name = self.config['EXP_NAME']
                _, val_path, _ = self.init_paths()
                generate_predictions(model,device,val_path,f'preds_frcnn_{exp_name}')
                from froc_by_pranjal import get_froc_points
                senses, _ = get_froc_points(f'preds_frcnn_{exp_name}', root_fol= join(self.config['DATA_DIR'],self.config['AIIMS_DATA'], self.config['AIIMS_VAL_SPLIT']), fps_req = [0.2])
                self.frocs['frcnn_froc'].append(senses[0])
                self.best_froc = max(self.frocs['frcnn_froc'])
                self.logger.log(f'Val FROC: {senses[0]}', LogPriority.MEDIUM)
                self.logger.log(f'Best FROC: {self.best_froc}')
        elif self.curr_mode == 'BILATERAL':
            self.losses['bilateral_loss'].append(loss)
            self.best_loss = min(self.losses['bilateral_loss'])    
            if self.config['EVAL_METHOD'] == 'FROC':
                exp_name = self.config['EXP_NAME']
                _, val_path, _ = self.init_paths()
                data_dir = self.config['DATA_DIR']
                print('Generating')
                generate_predictions_bilateral(model,device,val_path,get_dict(data_dir,self.abs_path(self.config['AIIMS_CORRS_LIST'])),preds_folder = f'preds_bilateral_{exp_name}')
                print('Generation Done')
                from froc_by_pranjal import get_froc_points
                senses, _ = get_froc_points(f'preds_bilateral_{exp_name}', root_fol= join(self.config['DATA_DIR'],self.config['AIIMS_DATA'], self.config['AIIMS_VAL_SPLIT']), fps_req = [0.1])
                print('Reading Sens from',f'preds_bilateral_{exp_name}', join(self.config['DATA_DIR'],self.config['AIIMS_DATA'], self.config['AIIMS_VAL_SPLIT']),)
                
                self.frocs['bilateral_froc'].append(senses[0])
                self.best_froc = max(self.frocs['bilateral_froc'])
                self.logger.log(f'Val FROC: {senses[0]}', priority = LogPriority.MEDIUM)
                self.logger.log(f'Best FROC: {self.best_froc}')
  
        self.writer.add_scalar(f"{self.curr_mode}/Loss/Valid", loss, self.curr_epoch)
    
    
    
    def save_model(self, model):
        if self.curr_mode == 'FRCNN':
            self.logger.log('Saving FRCNN Model', priority = LogPriority.LOW)
            model_file = join(self.exp_dir,'frcnn_models',f'frcnn_model.pth')
            if self.config['EVAL_METHOD']:
                SAVE = self.best_froc == self.frocs['frcnn_froc'][-1]
            else:
                SAVE = self.best_loss == self.losses['frcnn_loss'][-1]
        elif self.curr_mode == 'BILATERAL':
            self.logger.log('Saving Bilateral Model', priority = LogPriority.LOW)
            model_file = join(self.exp_dir,'bilateral_models',f'bilateral_model.pth')
            if self.config['EVAL_METHOD'] == 'FROC':
                SAVE = self.best_froc == self.frocs['bilateral_froc'][-1]
            else:
                SAVE = self.best_loss == self.losses['bilateral_loss'][-1]
        os.makedirs(os.path.split(model_file)[0], exist_ok=True)
        if SAVE:
            torch.save(model.state_dict(), model_file)
        
        torch.save(model.state_dict(), f'{model_file[:-4]}_{self.curr_epoch}.pth')

    def init_paths(self,):
        train_path = join(self.config['DATA_DIR'], self.config['AIIMS_DATA'], self.config['AIIMS_TRAIN_SPLIT'])
        val_path = join(self.config['DATA_DIR'],   self.config['AIIMS_DATA'], self.config['AIIMS_VAL_SPLIT'])
        test_path = join(self.config['DATA_DIR'],  self.config['AIIMS_DATA'], self.config['AIIMS_TEST_SPLIT'])
        return train_path, val_path, test_path

    def abs_path(self, path):
        return join(self.config['DATA_DIR'], path)

    # Impure Function, upadtes the model with best state dicts 
    def generate_predictions(self,model, device):
        self.logger.log('Generating Predictions')
        self.logger.flush()
        exp_name = self.config['EXP_NAME']
        train_path, val_path, test_path = self.init_paths()

        # Load the best val_loss model's state dicts
        if self.curr_mode == 'FRCNN':
            model_file = join(self.exp_dir,'frcnn_models','frcnn_model.pth')
        elif self.curr_mode == 'BILATERAL':
            model_file = join(self.exp_dir,'bilateral_models','bilateral_model.pth')
        model.load_state_dict(torch.load(model_file))

        if self.curr_mode == 'FRCNN':
            generate_predictions(model,device,train_path,f'preds_frcnn_{exp_name}')
            generate_predictions(model,device,val_path,f'preds_frcnn_{exp_name}')
            generate_predictions(model,device,test_path,f'preds_frcnn_{exp_name}')
        elif self.curr_mode == 'BILATERAL':
            data_dir = self.config['DATA_DIR']
            generate_predictions_bilateral(model,device,train_path,get_dict(data_dir,self.abs_path(self.config['AIIMS_CORRS_LIST'])),'aiims',f'preds_bilateral_{exp_name}')
            generate_predictions_bilateral(model,device,val_path,get_dict(data_dir,self.abs_path(self.config['AIIMS_CORRS_LIST'])),'aiims',f'preds_bilateral_{exp_name}')
            generate_predictions_bilateral(model,device,test_path,get_dict(data_dir,self.abs_path(self.config['AIIMS_CORRS_LIST'])),'aiims',f'preds_bilateral_{exp_name}')
            test_path = join(self.config['DATA_DIR'], self.config['AIIMS_DATA'], self.config['AIIMS_TEST_SPLIT'])

    def run_experiment(self):

        # First Determine the mode of running the experiment
        mode = self.config['MODE']
        self.init_losses(mode)
        self.curr_mode = 'FRCNN'
        self.curr_epoch = -1
        self.best_loss = 999999
        self.best_froc = 0
        if mode == 'FRCNN':
            TRAIN_FRCNN(self.config['FRCNN'], self)
        elif mode == 'BILATERAL':
            self.curr_mode = 'BILATERAL'
            TRAIN_BILATERAL(self.config['BILATERAL'], self)
        elif mode == 'FRCNN_BILATERAL':
            TRAIN_FRCNN(self.config['FRCNN'], self)
            self.curr_mode = 'BILATERAL'
            self.curr_epoch = -1
            self.best_loss = 999999
            # Note the path to frcnn model must be the same as that dictated by experiment
            self.config['BILATERAL']['FRCNN_MODEL_PATH'] = join(self.exp_dir,'frcnn_models','frcnn_model.pth')
            TRAIN_BILATERAL(self.config['BILATERAL'], self)
        
        self.logger.log(f'Best Loss: {self.best_loss}', priority= LogPriority.STATS)
        self.logger.log('Experiment Training and Generation Ended', priority = LogPriority.MEDIUM)

        # Now evaluate the results

        frcnn_file = join(self.exp_dir, 'senses_fps_frcnn.txt')
        bilateral_file = join(self.exp_dir, 'senses_fps_bilateral.txt')
        from froc_by_pranjal import get_froc_points
        exp_name = self.config['EXP_NAME']
        if mode == 'FRCNN' or mode == 'FRCNN_BILATERAL':
            senses, fps = get_froc_points(f'preds_frcnn_{exp_name}', root_fol= join(self.config['DATA_DIR'],self.config['AIIMS_DATA'], self.config['AIIMS_TEST_SPLIT']), save_to = frcnn_file)
            self.logger.log('FRCNN RESULTS', priority = LogPriority.STATS)
            for s,f in zip(senses, fps):
                self.logger.log(f'Sensitivty at {f}: {s}', priority = LogPriority.STATS)
        if mode == 'BILATERAL' or mode == 'FRCNN_BILATERAL':
            senses, fps = get_froc_points(f'preds_bilateral_{exp_name}', root_fol= join(self.config['DATA_DIR'],self.config['AIIMS_DATA'], self.config['AIIMS_TEST_SPLIT']), save_to = bilateral_file)
            self.logger.log('BILATERAL RESULTS', priority = LogPriority.STATS)
            for s,f in zip(senses, fps):
                self.logger.log(f'Sensitivty at {f}: {s}', priority = LogPriority.STATS)


        # Now draw the graphs.... If FRCNN and BILATERAL both done, draw them on one graph
        # Else draw single graphs only
        if mode == 'FRCNN':
            plot_froc({frcnn_file : 'FRCNN'}, join(self.exp_dir,'plot.png'), TITLE = 'FRCNN FROC')
        elif mode == 'BILATERAL':
            plot_froc({bilateral_file : 'BILATERAL'}, join(self.exp_dir,'plot.png'), TITLE = 'BILATERAL FROC')
        elif mode == 'FRCNN_BILATERAL':
            plot_froc({frcnn_file : 'FRCNN', bilateral_file : 'BILATERAL'}, join(self.exp_dir,'plot.png'), TITLE = 'FRCNN vs BILATERAL FROC')
        self.logger.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='configs/AIIMS_C1.cfg')
    args = parser.parse_args()
    exp = Experimenter(args.cfg_file)
    exp.run_experiment()