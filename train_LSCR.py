# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Training Loop script"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import glob

# import torch
# from torch.utils.data import DataLoader

import mindspore as ms
from mindspore.dataset import GeneratorDataset

from irgan.data.datasets import DATASETS
from irgan.evaluation.evaluate import Evaluator
from irgan.utils.config import keys, parse_config
from irgan.models.models import MODELS
from irgan.data import codraw_dataset
from irgan.data import clevr_dataset
import time


class Trainer():
    def __init__(self, cfg):
        # the path to save generated images during training, default in logs/exp_name
        img_path = os.path.join(cfg.log_path, cfg.exp_name, 'train_images_*')
        if glob.glob(img_path):
            raise Exception('all directories with name train_images_* under '
                            'the experiment directory need to be removed')
        path = os.path.join(cfg.log_path, cfg.exp_name)

        self.model = MODELS[cfg.gan_type](cfg)
        if cfg.load_snapshot is not None:
            print('-----load pretrained model------')
            self.model.load_model(cfg.load_snapshot)

        self.model.save_model(path, epoch=0, iteration=0)

        self.dataset = DATASETS[cfg.dataset](path=keys[cfg.dataset],
                                             cfg=cfg,
                                             img_size=cfg.img_size)
        self.dataloader = GeneratorDataset(self.dataset,
                                     shuffle=False,
                                     num_parallel_workers=cfg.num_workers,
                                    column_names=["image"])
        self.dataloader = self.dataloader.batch(batch_size=cfg.batch_size, drop_remainder=True)

        if cfg.dataset == 'codraw':
            self.dataloader.collate_fn = codraw_dataset.collate_data
        elif cfg.dataset == 'iclevr':
            self.dataloader.collate_fn = clevr_dataset.collate_data

        self.cfg = cfg

    def train(self):
        iteration_counter = 0
        for epoch in range(self.cfg.epochs):
            if cfg.dataset == 'codraw':
                self.dataset.shuffle()
            epoch_start_time = time.time()
            for batch in self.dataloader:
                if iteration_counter >= 0 and iteration_counter % self.cfg.save_rate == 0:
                    evaluator = Evaluator.factory(self.cfg)
                    evaluator.evaluate(iteration_counter)
                    del evaluator

                iteration_counter += 1

                self.model.train_batch(batch, epoch, iteration_counter)
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, self.cfg.epochs, time.time() - epoch_start_time))


if __name__ == '__main__':
    # print('E4.3')
    print('VISDOM:4003')  #
    # os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6'
    cfg = parse_config()

    trainer = Trainer(cfg)
    trainer.train()
