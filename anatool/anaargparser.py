import argparse
import os

import yaml

from data.join_dataset_path import join_dataset_path


class AnaArgParser:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True, help='train/test configs file')
        parser.add_argument('--dataset_config', type=str, required=True, help='dataset configs file')
        parser.add_argument('--fusion_config', type=str, required=True, help='fusion configs file')
        parser.add_argument('--local_rank', type=int, default=-1, help='distributed training')
        parser.add_argument('--debug', type=bool, default=False, help='debug or not')
        parser.add_argument('--datetime', type=str, default='', help='datetime')
        args = parser.parse_args()
        assert args.config is not None
        self.cfg = self.load_cfg_from_cfg_file([
            args.config,
            args.dataset_config,
            args.fusion_config
        ])
        self.cfg.__setattr__('local_rank', args.local_rank)
        self.cfg.__setattr__('debug', args.debug)
        self.cfg.__setattr__('datetime', args.datetime)
        self.cfg = join_dataset_path(opt=self.cfg, val_set=self.cfg.get('val_set', 'visdial'))

    @staticmethod
    def load_cfg_from_cfg_file(files):
        cfg = {}
        for file in files:
            assert os.path.isfile(file) and file.endswith('.yaml'), \
                '{} is not a yaml file'.format(file)

            with open(file, 'r') as f:
                cfg_from_file = yaml.safe_load(f)

            for key in cfg_from_file:
                for k, v in cfg_from_file[key].items():
                    cfg[k] = v

        cfg = CfgNode(cfg)
        return cfg


class CfgNode(dict):
    def __init__(self, init_dict=None, key_list=None):
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())
