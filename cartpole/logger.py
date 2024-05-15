import os.path as osp
import csv
from collections import defaultdict

import torch
from termcolor import colored


EVAL_FORMAT = [('episode', 'int'), ('step', 'int'),
               ('episode_reward', 'float'), ('norm', 'float')]

TRAIN_FORMAT = [('episode', 'int'), ('step', 'int'),
                ('episode_reward', 'float'), ('batch_reward', 'float'),
                ('actor_loss', 'float'), ('critic_loss', 'float'), ('alpha_loss', 'float'),
                ('ae_loss', 'float'), ('transition_loss', 'float')]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0
    
    def update(self, value, n=1):
        self._sum += value
        self._count += n
    
    def value(self):
        return self._sum / max(1, self._count)
    

class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._csv_file_name = file_name
        self._writeheader = not osp.exists(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, 'a')
        self._csv_writer = None
    
    def log(self, key, value, n=1):
        self._meters[key].update(value, n)
    
    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data
    
    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.)
            if self._writeheader:
                self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)
    
    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, ty in self._formating:
            if key in data:
                pieces.append(self._format(key, data[key], ty))
        print(' | %s' % (' | '.join(pieces)))

    def dump(self, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self._train_mg = MetersGroup(
            osp.join(log_dir, 'train.csv'), formating=TRAIN_FORMAT)
        self._eval_mg = MetersGroup(
            osp.join(log_dir, 'eval.csv'), formating=EVAL_FORMAT)
        
    def log(self, key, value, n=1):
        assert key.startswith('train') or key.startswith('eval')
        if isinstance(value, torch.Tensor):
            value = value.item()
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def dump(self, ty=None):
        if ty is None:
            self._train_mg.dump('train')
            self._eval_mg.dump('eval')
        elif ty == 'eval':
            self._eval_mg.dump('eval')
    