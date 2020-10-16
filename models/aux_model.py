import os
import time
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# custom modules
from schedulers import get_scheduler
from optimizers import get_optimizer
from networks import get_network
from utils.metrics import AverageMeter
from utils.utils import to_device

# summary
from tensorboardX import SummaryWriter

def inv_lr_scheduler(optimizer, iter_num, lr, wd, gamma=0.001, power=0.75):
    lr = lr * (1 + gamma * iter_num) ** (-power)
    for pg in optimizer.param_groups:
        pg['lr'] = lr * pg['lr_mult']
        pg['weight_decay'] = wd * pg['decay_mult']
    return optimizer

class AuxModel:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.writer = SummaryWriter(args.log_dir)
        cudnn.enabled = True

        # set up model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_network(args.network.arch)(aux_classes=args.aux_classes + 1, classes=args.n_classes)
        self.model = self.model.to(self.device)

        if args.mode == 'train':
            # set up optimizer, lr scheduler and loss functions
            optimizer = get_optimizer(self.args.training.optimizer)
            optimizer_params = {k: v for k, v in self.args.training.optimizer.items() if k != "name"}
            self.optimizer = optimizer(self.model.get_parameters(), **optimizer_params)
            self.scheduler = get_scheduler(self.optimizer, self.args.training.lr_scheduler)

            self.class_loss_func = nn.CrossEntropyLoss()
            # KL divergence loss
            self.KLD_loss_func = nn.KLDivLoss(reduction='batchmean')

            self.start_iter = 0

            # resume
            if args.training.resume:
                self.load(args.model_dir + '/' + args.training.resume)

            cudnn.benchmark = True

        elif args.mode == 'val':
            self.load(os.path.join(args.model_dir, args.validation.model))
        else:
            self.load(os.path.join(args.model_dir, args.testing.model))

    def entropy_loss(self, x):
        return torch.sum(-F.softmax(x, 1) * F.log_softmax(x, 1), 1).mean()

    def train(self, src_loader, tar_loader, val_loader, test_loader):

        num_batches = len(src_loader)
        print('Number of batches: %d' % num_batches)
        print_freq = max(num_batches // self.args.training.num_print_epoch, 1)
        i_iter = self.start_iter
        start_epoch = i_iter // num_batches
        num_epochs = self.args.training.num_epochs
        best_acc = 0

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            batch_time = AverageMeter()
            losses = AverageMeter()

            for it, (src_batch, tar_batch) in enumerate(zip(src_loader, itertools.cycle(tar_loader))):
                t = time.time()

                if isinstance(src_batch, list):
                    src = src_batch[0] # data, dataset_idx
                else:
                    src = src_batch
                src = to_device(src, self.device)
                src_imgs = src['images']
                src_imgs_ori = src['images_ori']
                src_cls_lbls = src['class_labels']
                src_aux_lbls = src['aux_labels']

                #self.optimizer = inv_lr_scheduler(self.optimizer, i_iter, 
                #        lr=self.args.training.optimizer.lr, wd=self.args.training.optimizer.weight_decay)

                self.optimizer.zero_grad()

                src_aux_logits, src_class_logits = self.model(src_imgs)
                src_aux_loss = self.class_loss_func(src_aux_logits, src_aux_lbls)

                # If true, the network will only try to classify the non scrambled images
                if self.args.training.get('only_non_scrambled'):
                    src_class_loss = self.class_loss_func(
                            src_class_logits[src_aux_lbls == 0], src_cls_lbls[src_aux_lbls == 0])
                else:
                    src_class_loss = self.class_loss_func(src_class_logits, src_cls_lbls)

                src_kld_loss = torch.tensor(0.0)
                if self.args.training.get('src_kld_weight'):
                    _, ori_logits = self.model(src_imgs_ori)
                    aug_logits = src_class_logits
                    src_kld_loss = self.KLD_loss_func(F.log_softmax(aug_logits, 1), 
                            F.softmax(ori_logits.detach(), 1))

                tar = to_device(tar_batch, self.device)
                tar_imgs = tar['images']
                tar_imgs_ori = tar['images_ori']
                tar_aux_lbls = tar['aux_labels']
                tar_aux_logits, tar_class_logits = self.model(tar_imgs)
                tar_aux_loss = self.class_loss_func(tar_aux_logits, tar_aux_lbls)
                tar_entropy_loss = self.entropy_loss(tar_class_logits[tar_aux_lbls==0])
                
                tar_kld_loss = torch.tensor(0.0)
                if self.args.training.get('tar_kld_weight'):
                    _, ori_logits = self.model(tar_imgs_ori)
                    aug_logits = tar_class_logits
                    tar_kld_loss = self.KLD_loss_func(F.log_softmax(aug_logits, 1), 
                            F.softmax(ori_logits.detach(), 1))

                loss = src_class_loss + src_aux_loss * self.args.training.src_aux_weight

                if self.args.training.get('src_kld_weight'):
                    loss += self.args.training.src_kld_weight * src_kld_loss

                loss += tar_aux_loss * self.args.training.tar_aux_weight
                loss += tar_entropy_loss * self.args.training.tar_entropy_weight

                if self.args.training.get('tar_kld_weight'):
                    loss += self.args.training.tar_kld_weight * tar_kld_loss

                loss.backward()
                self.optimizer.step()

                losses.update(loss.item(), src_imgs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - t)

                i_iter += 1

                if i_iter % print_freq == 0:
                    print_string = 'Epoch {:>2} | iter {:>4} | src_class: {:.3f} | src_aux: {:.3f} | tar_aux: {:.3f} | tar_entropy: {:.3f} | src_kld:{:.3f} | tar_kld:{:.3f} | {:4.2f} s/it'
                    self.logger.info(print_string.format(epoch, i_iter,
                        src_class_loss.item(),
                        src_aux_loss.item(),
                        tar_aux_loss.item(),
                        tar_entropy_loss.item(),
                        src_kld_loss.item(),
                        tar_kld_loss.item(),
                        batch_time.avg))
                    self.writer.add_scalar('losses/src_class_loss', src_class_loss, i_iter)
                    self.writer.add_scalar('losses/src_aux_loss', src_aux_loss, i_iter)
                    self.writer.add_scalar('losses/tar_aux_loss', tar_aux_loss, i_iter)
                    self.writer.add_scalar('losses/tar_entropy_loss', tar_entropy_loss, i_iter)

            # adjust learning rate
            self.scheduler.step()

            del loss, src_class_loss, src_aux_loss, tar_aux_loss, tar_entropy_loss
            del src_aux_logits, src_class_logits
            del tar_aux_logits, tar_class_logits

            # validation
            # if val_loader:
                # self.logger.info('validating...')
                # aux_acc, class_acc = self.test(val_loader)
                # self.writer.add_scalar('val/aux_acc', aux_acc, i_iter)
                # self.writer.add_scalar('val/class_acc', class_acc, i_iter)

            if test_loader:
                self.logger.info('testing...')
                aux_acc, class_acc = self.test(test_loader)
                self.writer.add_scalar('test/aux_acc', aux_acc, i_iter)
                self.writer.add_scalar('test/class_acc', class_acc, i_iter)
                if class_acc > best_acc:
                    best_acc = class_acc
                    self.save(self.args.model_dir, i_iter, is_best='True')

                self.logger.info('Best testing accuracy: {:.2f} %'.format(best_acc))

        self.logger.info('Best testing accuracy: {:.2f} %'.format(best_acc))
        self.logger.info('Finished Training.')

    def save(self, path, i_iter, is_best):
        state = {"iter": i_iter + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                }
        save_path = os.path.join(path, 'model_{:06d}.pth'.format(i_iter))
        if is_best:
            save_path = os.path.join(path, 'model_best.pth')
        self.logger.info('Saving model to %s' % save_path)
        torch.save(state, save_path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.logger.info('Loaded model from: ' + path)

        if self.args.mode == 'train':
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.start_iter = checkpoint['iter']
            self.logger.info('Start iter: %d ' % self.start_iter)

    def test(self, test_loader):
        if isinstance(test_loader, list):
            return self.test_multi_crop(test_loader)
        else:
            return self.test_single_crop(test_loader)

    def test_multi_crop(self, test_loaders):
        num_crops = len(test_loaders)
        test_loaders_iterator = [iter(test_loaders[i]) for i in range(num_crops)]

        num_test_iters = len(test_loaders[0])
        tt = tqdm(range(num_test_iters), total=num_test_iters, desc="Multi-crop test")

        aux_correct = 0
        class_correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for cur_it in tt:
                aux_logits_list = []
                class_logits_list = []
                for i in range(num_crops):
                    data = next(test_loaders_iterator[i])
                    if isinstance(data, list):
                        data = data[0]

                    # Get the inputs
                    data = to_device(data, self.device)
                    imgs = data['images']
                    if i == 0:
                        imgs_size = imgs.size(0)
                        cls_lbls = data['class_labels']
                        aux_lbls = data['aux_labels']

                    aux_logits_i, class_logits_i = self.model(imgs)
                    aux_logits_list.append(aux_logits_i)
                    class_logits_list.append(class_logits_i)

                aux_logits = sum(aux_logits_list)
                class_logits = sum(class_logits_list)
                _, cls_pred = class_logits.max(dim=1)
                _, aux_pred = aux_logits.max(dim=1)

                class_correct += torch.sum(cls_pred == cls_lbls.data)
                aux_correct += torch.sum(aux_pred == aux_lbls.data)
                total += imgs_size
        
        tt.close()

        aux_acc = 100 * float(aux_correct) / total
        class_acc = 100 * float(class_correct) / total
        self.logger.info('{} aux_acc: {:.2f} %, class_acc: {:.2f} %'.format(self.args.exp_name, aux_acc, class_acc))
        return aux_acc, class_acc

    def test_single_crop(self, test_loader):
        test_loader_iterator = iter(test_loader)
        num_test_iters = len(test_loader)
        tt = tqdm(range(num_test_iters), total=num_test_iters, desc="Testing")

        aux_correct = 0
        class_correct = 0
        total = 0
        features = []

        self.model.eval()
        with torch.no_grad():
            for cur_it in tt:
                data = next(test_loader_iterator)
                if isinstance(data, list):
                    data = data[0]
                # Get the inputs
                data = to_device(data, self.device)
                imgs = data['images']
                cls_lbls = data['class_labels']
                aux_lbls = data['aux_labels']

                aux_logits, class_logits = self.model(imgs)

                _, cls_pred = class_logits.max(dim=1)
                _, aux_pred = aux_logits.max(dim=1)

                class_correct += torch.sum(cls_pred == cls_lbls.data)
                aux_correct += torch.sum(aux_pred == aux_lbls.data)
                total += imgs.size(0)
                
                if self.args.testing.get('save_features', False):
                    feats = class_logits.cpu().data.numpy()
                    lbls_np = cls_lbls.cpu().data.numpy()
                    lbls_np = lbls_np[:, np.newaxis]
                    features.append(np.hstack((feats, lbls_np)))

            tt.close()

        aux_acc = 100 * float(aux_correct) / total
        class_acc = 100 * float(class_correct) / total
        self.logger.info('{} aux_acc: {:.2f} %, class_acc: {:.2f} %'.format(self.args.exp_name, aux_acc, class_acc))
        
        if self.args.testing.get('save_features', False):
            feature_path = os.path.join(self.args.cache_dir,
                    self.args.datasets.test.name+'_features.npy')
            features = np.asarray(features)
            np.save(feature_path, features)
            self.logger.info('Features are saved at: {:s} '.format(feature_path))

        return aux_acc, class_acc
