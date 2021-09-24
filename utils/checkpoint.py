import os

import torch


# Modified from https://github.com/clovaai/stargan-v2/blob/master/core/checkpoint.py

class CheckpointIO:
    def __init__(self, fname_template, multi_gpu=False, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs
        self.multi_gpu = multi_gpu

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        out_dict = {}
        for name, module in self.module_dict.items():
            if self.multi_gpu:
                out_dict[name] = module.module.state_dict()
            else:
                out_dict[name] = module.state_dict()
        torch.save(out_dict, fname)

    def load(self, step):
        fname = self.fname_template.format(step)
        self.load_from_path(fname)

    def load_from_path(self, fname):
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
        for name, module in self.module_dict.items():
            if self.multi_gpu:
                module.module.load_state_dict(module_dict[name])
            else:
                module.load_state_dict(module_dict[name])
