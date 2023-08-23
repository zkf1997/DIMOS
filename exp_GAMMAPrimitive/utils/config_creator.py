import yaml
import os
import sys
import pdb

class ConfigCreator(object):
    def __init__(self, cfg_name):
        self.cfg_name = cfg_name
        exppath = os.path.join( *(__file__.split('/')[:-2]) )
        expname = os.path.basename(exppath)
        cfg_file = '/'+exppath+'/cfg/{:s}.yml'.format(cfg_name)
        try:
            cfg = yaml.safe_load(open(cfg_file, 'r'))
        except FileNotFoundError as e:
            print(e)
            sys.exit()

        # create dirs
        self.cfg_exp_dir = os.path.join('results', expname, cfg_name)
        self.cfg_result_dir = os.path.join(self.cfg_exp_dir, 'results')
        self.cfg_ckpt_dir = os.path.join(self.cfg_exp_dir, 'checkpoints')
        self.cfg_log_dir = os.path.join(self.cfg_exp_dir, 'logs')
        os.makedirs(self.cfg_result_dir, exist_ok=True)
        os.makedirs(self.cfg_ckpt_dir, exist_ok=True)
        os.makedirs(self.cfg_log_dir, exist_ok=True)

        # specify missed experiment settings
        cfg['trainconfig']['save_dir'] = self.cfg_ckpt_dir
        cfg['trainconfig']['log_dir'] = self.cfg_log_dir

        # set subconfigs
        self.modelconfig = cfg['modelconfig']
        self.lossconfig = cfg['lossconfig']
        self.trainconfig = cfg['trainconfig']





