from torch.utils.tensorboard import SummaryWriter
from utils.events import write_tbPR, write_tbloss
from tqdm import tqdm
import time, os
import numpy as np
import torch

from dataset import create_dataloader
from torch.utils.data.dataloader import DataLoader

class Trainer():
    def __init__(self, cfg, device=torch.device('cpu')):
        self.cfg = cfg
        self.device = device
        self.save_path = self.make_save_path()

        # ===== Tensorboard =====
        self.tblogger = SummaryWriter(self.save_path)

        # =========== DataLoader ===========
        self.train_loader, self.val_loader = self.get_dataloader()

        # =========== Model ===========
        # Lenet, Alexnet, Vggnet, Resnet
        self.model = self.build_model()

        # =========== Optimizer ===========
        self.optimizer = self.build_optimizer()

        # =========== Scheduler ===========
        self.scheduler = self.build_scheduler()

        # =========== Loss ===========
        self.compute_loss = self.set_criterion()

        # =========== Parameters ===========
        self.max_epoch = self.cfg['solver']['max_epoch']
        self.max_stepnum = len(self.train_loader)

    def get_dataloader(self):
        if self.cfg['dataset']['name'] == 'wdm':
            from dataset.wdm import wdm
        else:
            raise ValueError('Invalid dataset name,' 'currently supported [wdm]...')

        train_path = self.cfg['dataset']['train_path']
        val_path = self.cfg['dataset']['val_path']
        batch_size = self.cfg['dataset']['batch_size']
        num_workers = self.cfg['dataset']['num_workers']
        height, width = self.cfg['dataset']['height'], self.cfg['dataset']['width']

        train_object = wdm(path = train_path,
                           height = height,
                           width = width,
                           augmentation = False,
                           task = 'train')

        val_object = wdm(path = val_path,
                           height = height,
                           width = width,
                           augmentation = False,
                           task = 'val')

        train_loader = DataLoader(train_object,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = num_workers,
                                  collate_fn = wdm.collate_fn)

        val_loader = DataLoader(val_object,
                                batch_size = batch_size,
                                shuffle = True,
                                num_workers = num_workers,
                                collate_fn = wdm.collate_fn)

        return train_loader, val_loader

    def make_save_path(self):
        save_path = os.path.join(self.cfg['path']['save_base_path'],
                                 self.cfg['model']['name'])
        os.makedirs(save_path, exist_ok=True)

        return  save_path

    def build_model(self):
        model_name = self.cfg['model']['name']
        if model_name =='lenet':
            from model.lenet import lenet # model 폴더 안에 lenet 스크립트에서 lenet이라는 class를 사용하겠다.
            model = lenet().to(self.device)
        elif model_name =='alexnet':
            from model.alexnet import alexnet
            model = alexnet().to(self.device)
        elif model_name =='resnet':
            from model.resnet import resnet
            model = resnet().to(self.device)
        else:
            raise NotImplementedError

        return model

    def set_criterion(self):
        return torch.nn.BCEWithLogitsLoss(reduction='sum').to(self.device)

    def build_optimizer(self):
        from solver.fn_optimizer import build_optimizer
        return build_optimizer(self.cfg, self.model)

    def build_scheduler(self):
        if self.cfg['scheduler']['name'] == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.9, step_size=5)

        else:
            raise NotImplementedError
        return scheduler

    def start_train(self):
        try:
            for epoch in range(self.max_epoch):
                self.train_one_epoch(epoch)
        except:
            print("Error in trainind loop.....")
            raise

    def train_one_epoch(self, epoch):
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        TP = np.zeros(8)
        FP = np.zeros(8)
        FN = np.zeros(8)

        for step, batch_data in pbar:
            imgs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            out_net = self.model(imgs)

            loss = self.compute_loss(out_net, labels.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Get statistics
            TP, FP, FN = self.get_statistics(self.model.predict(out_net.detach()), labels, TP, FP, FN)

            if step % 2 == 0:
                write_tbloss(self.tblogger, loss.detach().cpu(), (epoch * self.max_epoch + step))

        write_tbPR(self.tblogger, TP, FP, FN, epoch, 'train')
        self.scheduler.step()

    @staticmethod
    def get_statistics(pred, true, TP, FP, FN):
        for defect_idx in range(pred.shape[1]):
            pred_per_defect = pred[:, defect_idx].cpu().detach().numpy()
            true_per_defect = true[:, defect_idx].cpu().detach().numpy()

            TP[defect_idx] += np.sum(pred_per_defect * true_per_defect)
            FP[defect_idx] += np.sum(pred_per_defect * (1 - true_per_defect))
            FN[defect_idx] += np.sum((1 - pred_per_defect) * true_per_defect)

        return TP, FP, FN