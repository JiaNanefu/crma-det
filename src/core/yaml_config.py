"""by lyuwenyu
"""

import torch 
import torch.nn as nn

import re 
import copy

from .config import BaseConfig
from .yaml_utils import load_config, merge_config, create, merge_dict, reset_global_config


class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        merge_dict(cfg, kwargs)

        # pprint(cfg)

        self.yaml_cfg = cfg 

        self.log_step = cfg.get('log_step', 100)
        self.checkpoint_step = cfg.get('checkpoint_step', 1)
        self.epoches = cfg.get('epoches', -1)
        self.resume = cfg.get('resume', '')
        self.tuning = cfg.get('tuning', '')
        self.sync_bn = cfg.get('sync_bn', False)
        self.output_dir = cfg.get('output_dir', None)
        
        self.use_ema = cfg.get('use_ema', False)
        self.use_amp = cfg.get('use_amp', False)
        self.autocast = cfg.get('autocast', dict())
        self.find_unused_parameters = cfg.get('find_unused_parameters', None)
        self.clip_max_norm = cfg.get('clip_max_norm', 0.)

    def _refresh_global_config(self):
        reset_global_config()
        merge_config(self.yaml_cfg)

    @property
    def model(self, ) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._refresh_global_config()
            self._model = create(self.yaml_cfg['model'])
        return self._model 

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._refresh_global_config()
            self._postprocessor = create(self.yaml_cfg['postprocessor'])
        return self._postprocessor

    @property
    def criterion(self, ):
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._refresh_global_config()
            self._criterion = create(self.yaml_cfg['criterion'])
        return self._criterion

    
    @property
    def optimizer(self, ):
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            self._refresh_global_config()
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            self._optimizer = create('optimizer', params=params)

        return self._optimizer
    
    @property
    def lr_scheduler(self, ):
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._refresh_global_config()
            self._lr_scheduler = create('lr_scheduler', optimizer=self.optimizer)
            print('Initial lr: ', self._lr_scheduler.get_last_lr())

        return self._lr_scheduler
    
    @property
    def train_dataloader(self, ):
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._refresh_global_config()
            self._train_dataloader = create('train_dataloader')
            self._train_dataloader.shuffle = self.yaml_cfg['train_dataloader'].get('shuffle', False)

        return self._train_dataloader
    
    @property
    def val_dataloader(self, ):
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            self._refresh_global_config()
            self._val_dataloader = create('val_dataloader')
            self._val_dataloader.shuffle = self.yaml_cfg['val_dataloader'].get('shuffle', False)

        return self._val_dataloader

    @property
    def test_dataloader(self, ):
        if self._test_dataloader is None and 'test_dataloader' in self.yaml_cfg:
            self._refresh_global_config()
            self._test_dataloader = create('test_dataloader')
            self._test_dataloader.shuffle = self.yaml_cfg['test_dataloader'].get('shuffle', False)

        return self._test_dataloader
    
    
    @property
    def ema(self, ):
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            self._refresh_global_config()
            self._ema = create('ema', model=self.model)
            
        return self._ema
    

    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):
            self._refresh_global_config()
            self._scaler = create('scaler')

        return self._scaler

 
    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        '''
        E.g.:
            ^(?=.*a)(?=.*b).*$         means including a and b
            ^((?!b.)*a((?!b).)*$       means including a but not b
            ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
        '''
        assert 'type' in cfg, ''
        cfg = copy.deepcopy(cfg)

        if 'params' not in cfg:
            return model.parameters() 

        assert isinstance(cfg['params'], list), ''

        # 获取默认的 lr 和 weight_decay
        # Bug 3 修复：强制将默认 weight_decay 设为 0.05（而不是 0.0）
        default_lr = cfg.get('lr', 0.0001)
        default_weight_decay = cfg.get('weight_decay', 0.05)

        param_groups = []
        visited = []
        for pg in cfg['params']:
            pattern = pg['params']
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            
            # 创建新的参数组字典，保留所有配置
            new_pg = copy.deepcopy(pg)
            new_pg['params'] = list(params.values())
            
            # 如果没有指定 lr 或 weight_decay，使用默认值
            if 'lr' not in new_pg:
                new_pg['lr'] = default_lr
            if 'weight_decay' not in new_pg:
                new_pg['weight_decay'] = default_weight_decay
            
            param_groups.append(new_pg)
            visited.extend(list(params.keys()))

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            # 未匹配的参数使用默认配置
            param_groups.append({
                'params': list(params.values()),
                'lr': default_lr,
                'weight_decay': default_weight_decay
            })
            visited.extend(list(params.keys()))

        assert len(visited) == len(names), ''

        return param_groups
