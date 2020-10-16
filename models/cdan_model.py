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

def compute_entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def cdan_loss(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def dann_loss(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

class CDANModel:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.writer = SummaryWriter(args.log_dir)
        cudnn.enabled = True

        # set up model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_network(args.network.arch)(output='feature+class_logits', classes=args.n_classes)
        self.model = self.model.to(self.device)
        
        if args.method == 'dann':
            random_dim = self.model.feat_dim
        else:
            random_dim = self.model.feat_dim * args.n_classes
            if random_dim > 4096:
                random_dim = 1024
                self.random_layer = get_network('random_layer')(input_dim_list=[self.model.feat_dim, args.n_classes], output_dim=random_dim)
                self.random_layer.cuda()
                self.logger.info('Use randomized multilinear, random_dim={:d}'.format(random_dim))
            else:
                self.random_layer = None
                self.logger.info('Use multilinear, random_dim={:d}'.format(random_dim))

        self.adv_model = get_network('advnet')(input_dim=random_dim, hidden_dim=1024)
        self.adv_model = self.adv_model.to(self.device)

        if args.mode == 'train':
            # set up optimizer, lr scheduler and loss functions
            optimizer = get_optimizer(self.args.training.optimizer)
            optimizer_params = {k: v for k, v in self.args.training.optimizer.items() if k != "name"}
            parameter_list = self.model.get_parameters() + self.adv_model.get_parameters()
            self.optimizer = optimizer(parameter_list, **optimizer_params)
            # self.scheduler = get_scheduler(self.optimizer, self.args.training.lr_scheduler)

            self.class_loss_func = nn.CrossEntropyLoss()

            self.start_iter = 0

            # resume
            if args.training.resume:
                self.load(args.model_dir + '/' + args.training.resume)

            cudnn.benchmark = True

        elif args.mode == 'val':
            self.load(os.path.join(args.model_dir, args.validation.model))
        else:
            self.load(os.path.join(args.model_dir, args.testing.model))

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
                src_cls_lbls = src['class_labels']

                self.optimizer = inv_lr_scheduler(self.optimizer, i_iter, 
                        lr=self.args.training.optimizer.lr, wd=self.args.training.optimizer.weight_decay)

                self.optimizer.zero_grad()

                src_feats, src_class_logits = self.model(src_imgs)
                src_class_loss = self.class_loss_func(src_class_logits, src_cls_lbls)

                tar = to_device(tar_batch, self.device)
                tar_imgs = tar['images']
                tar_feats, tar_class_logits = self.model(tar_imgs)

                features = torch.cat((src_feats, tar_feats), dim=0)
                outputs = torch.cat((src_class_logits, tar_class_logits), dim=0)
                softmax_out = nn.Softmax(dim=1)(outputs)

                if self.args.method == 'cdan':
                    transfer_loss = cdan_loss([features, softmax_out], 
                            self.adv_model, None, None, self.random_layer)
                elif self.args.method == 'cdan+e':
                    entropy = compute_entropy(softmax_out)
                    transfer_loss = cdan_loss([features, softmax_out], self.adv_model, 
                            entropy, calc_coeff(i_iter), self.random_layer)
                elif self.args.method == 'dann':
                    transfer_loss = dann_loss(features, self.adv_model)
                else:
                    raise ValueError('Method cannot be recognized.')
                
                loss = src_class_loss + transfer_loss * self.args.training.transfer_loss_weight

                loss.backward()
                self.optimizer.step()

                losses.update(loss.item(), src_imgs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - t)

                i_iter += 1

                # adjust learning rate
                # self.scheduler.step()
                
                #for param_group in self.optimizer.param_groups:
                #    print(i_iter, param_group['lr'], param_group['weight_decay'])

                if i_iter % print_freq == 0:
                    print_string = 'Epoch {:>2} | iter {:>4} | class_loss: {:.3f} | transfer_loss: {:.3f} | {:4.2f} s/it'
                    self.logger.info(print_string.format(epoch, i_iter,
                        src_class_loss.item(),
                        transfer_loss.item(),
                        batch_time.avg))
                    self.writer.add_scalar('losses/src_class_loss', src_class_loss, i_iter)
                    self.writer.add_scalar('losses/transfer_loss', transfer_loss, i_iter)


            del loss, src_class_loss, transfer_loss
            del src_class_logits
            del tar_class_logits

            if test_loader:
                self.logger.info('testing...')
                class_acc = self.test(test_loader)
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
                "optimizer_state": self.optimizer.state_dict()
                #"scheduler_state": self.scheduler.state_dict(),
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
            #self.scheduler.load_state_dict(checkpoint['scheduler_state'])
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

        class_correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for cur_it in tt:
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

                    _, class_logits_i = self.model(imgs)
                    class_logits_list.append(class_logits_i)

                class_logits = sum(class_logits_list)
                _, cls_pred = class_logits.max(dim=1)

                class_correct += torch.sum(cls_pred == cls_lbls.data)
                total += imgs_size
        
        tt.close()

        class_acc = 100 * float(class_correct) / total
        self.logger.info('{} class_acc: {:.2f} %'.format(self.args.exp_name, class_acc))
        return class_acc

    def test_single_crop(self, test_loader):
        test_loader_iterator = iter(test_loader)
        num_test_iters = len(test_loader)
        tt = tqdm(range(num_test_iters), total=num_test_iters, desc="Testing")

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

                _, class_logits = self.model(imgs)

                _, cls_pred = class_logits.max(dim=1)

                class_correct += torch.sum(cls_pred == cls_lbls.data)
                total += imgs.size(0)
                
                if self.args.testing.get('save_features', False):
                    feats = class_logits.cpu().data.numpy()
                    lbls_np = cls_lbls.cpu().data.numpy()
                    lbls_np = lbls_np[:, np.newaxis]
                    features.append(np.hstack((feats, lbls_np)))

            tt.close()

        class_acc = 100 * float(class_correct) / total
        self.logger.info('{} class_acc: {:.2f} %'.format(self.args.exp_name, class_acc))

        if self.args.testing.get('save_features', False):
            feature_path = os.path.join(self.args.cache_dir,
                    self.args.datasets.test.name+'_features.npy')
            features = np.asarray(features)
            np.save(feature_path, features)
            self.logger.info('Features are saved at: {:s} '.format(feature_path))
        return class_acc
