import os
import sys
import datetime
import time
import numpy as np

import sklearn.metrics as metrics

import torch
import torchvision as tv
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim

from datasets import SegmentationImageDataset
from datasets import ImageDataset
from networks.unet import UNet_4mp
from networks.unet import Reconstruction_4mp
from networks.discriminator import Discriminator
from utils.gan_loss import AdversarialLoss
from utils.visualizer import Visualizer
from utils.utils import adjust_lr, cuda_visible, print_args, save_ckpt, AverageMeter, LastAvgMeter
from utils.parser import ParserArgs


class PNetModel(nn.Module):
    def __init__(self, args, ablation_mode=4):
        super(PNetModel, self).__init__()
        self.args = args
        if args.gpu >= 0:
            device = torch.device('cuda', args.gpu)
        else:
            device = torch.device('cpu')

        """
        ablation study mode
        """
        # 0: output_structure                       (1 feature)
        # 2: image (1 feature), i.e. auto-encoder
        # 4: output_structure + image               (2 features)

        # model on gpu
        if self.args.data_modality == 'fundus':
            model_G1 = UNet_4mp(n_channels=3, n_classes=1)
            model_G2 = Reconstruction_4mp(image_channels=3, structure_channels=1)
        else:
            model_G1 = UNet_4mp(n_channels=3, n_classes=12)
            model_G2 = Reconstruction_4mp(image_channels=3, structure_channels=12)
        model_D = Discriminator(in_channels=3)

        model_G1 = nn.DataParallel(model_G1).to(device)
        model_G2 = nn.DataParallel(model_G2).to(device)
        model_D = nn.DataParallel(model_D).to(device)

        l1_loss = nn.L1Loss().to(device)
        adversarial_loss = AdversarialLoss().to(device)

        self.add_module('model_G1', model_G1)
        self.add_module('model_G2', model_G2)
        self.add_module('model_D', model_D)

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        # optimizer
        self.optimizer_G = torch.optim.Adam(params=self.model_G2.parameters(),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(params=self.model_D.parameters(),
                                            lr=args.lr * args.d2g_lr,
                                            weight_decay=args.weight_decay,
                                            betas=(args.b1, args.b2))

        # load structure extraction network parameters
        seg_ckpt_path = args.structure_model
        if os.path.isfile(seg_ckpt_path):
            print("=> loading G1 checkpoint")
            checkpoint = torch.load(seg_ckpt_path)
            self.model_G1.load_state_dict(checkpoint['state_dict_G'])
            print("=> loaded G1 checkpoint (epoch {}) \n from {}"
                .format(checkpoint['epoch'], seg_ckpt_path))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(seg_ckpt_path))

        # Optionally resume from a checkpoint
        if self.args.resume:
            ckpt_path = args.resume
            if os.path.isfile(ckpt_path):
                print("=> loading G2 checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(ckpt_path)
                args.start_epoch = checkpoint['epoch']
                self.model_G2.load_state_dict(checkpoint['state_dict_G'])
                print("=> loaded G2 checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def process(self, image):
        # process_outputs
        seg_mask, image_rec, seg_mask_rec = self(image)

        """
        G and D process, this package is reusable
        """
        # zero optimizers
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        gen_loss = 0
        dis_loss = 0

        real_B = image
        fake_B = image_rec

        # discriminator loss
        dis_input_real = real_B
        dis_input_fake = fake_B.detach()
        dis_real, dis_real_feat = self.model_D(dis_input_real)
        dis_fake, dis_fake_feat = self.model_D(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss = dis_loss + (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = fake_B
        gen_fake, gen_fake_feat = self.model_D(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.args.lamd_gen
        gen_loss = gen_loss + gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss = gen_fm_loss + self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.args.lamd_fm
        gen_loss = gen_loss + gen_fm_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(fake_B, real_B) * self.args.lamd_p
        gen_loss = gen_loss + gen_l1_loss
        seg_l1_loss = self.l1_loss(seg_mask_rec, seg_mask) * self.args.lamd_p * 0.5
        gen_loss = gen_loss + seg_l1_loss

        # create logs
        logs = dict(
            gen_gan_loss=gen_gan_loss,
            gen_fm_loss=gen_fm_loss,
            gen_l1_loss=gen_l1_loss,
            seg_l1_loss=seg_l1_loss,
            # gen_content_loss=gen_content_loss,
            # gen_style_loss=gen_style_loss,
        )

        return seg_mask, fake_B, gen_loss, dis_loss, logs

    def forward(self, image):
        with torch.no_grad():
            seg_mask = self.model_G1(image)
        image_rec = self.model_G2(image, seg_mask)
        seg_mask_rec = self.model_G1(image_rec)

        return seg_mask, image_rec, seg_mask_rec

    def backward(self, gen_loss=None, dis_loss=None):
        if dis_loss is not None:
            dis_loss.backward()
        self.optimizer_D.step()
        if gen_loss is not None:
            gen_loss.backward()
        self.optimizer_G.step()


class RunMyModel(object):
    def __init__(self):
        args = ParserArgs().get_args()
        if args.gpu >= 0:
            self.device = torch.device('cuda', args.gpu)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.vis = Visualizer(env='{}'.format(args.version), port=args.port, server=args.vis_server)

        # TODO pass resize and crop size
        train_dataset = ImageDataset(args.label_path, args.image_root, augment=True, resize_size=(256, 256),
                                     crop_size=(224, 224))
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True,
                                                        num_workers=1, pin_memory=True)

        test_dataset = SegmentationImageDataset(args.test_label, args.image_root, augment=False,
                                                resize_size=(256, 256), crop_size=None)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False,
                                                       num_workers=1, pin_memory=True)

        print_args(args)
        self.args = args
        self.new_lr = self.args.lr
        self.model = PNetModel(args)

        if args.predict:
            self.test_acc()
        else:
            self.train_val()

    def train_val(self):
        # general metrics
        self.best_auc = 0
        self.is_best = False
        # self.total_auc_top10 = AverageMeter()
        self.total_auc_last10 = LastAvgMeter(length=10)
        self.acc_last10 = LastAvgMeter(length=10)

        # metrics for iSee
        self.myopia_auc_last10 = LastAvgMeter(length=10)
        self.amd_auc_last10 = LastAvgMeter(length=10)
        self.glaucoma_auc_last10 = LastAvgMeter(length=10)
        self.dr_auc_last10 = LastAvgMeter(length=10)

        for epoch in range(self.args.start_epoch, self.args.n_epochs):
            if self.args.data_modality == 'fundus':
                # total: 1000
                adjust_lr_epoch_list = [40, 80, 160, 240]
            else:
                # total: 180
                adjust_lr_epoch_list = [20, 40, 80, 120]
            _ = adjust_lr(self.args.lr, self.model.optimizer_G, epoch, adjust_lr_epoch_list)
            new_lr = adjust_lr(self.args.lr, self.model.optimizer_D, epoch, adjust_lr_epoch_list)
            self.new_lr = min(new_lr, self.new_lr)

            self.epoch = epoch
            self.train()
            # last 80 epoch, validate with freq
            if epoch > self.args.validate_start_epoch and (epoch % self.args.validate_freq == 0
               or epoch > (self.args.n_epochs - self.args.validate_each_epoch)):
                self.validate_cls()

            print('\n', '*' * 10, 'Program Information', '*' * 10)
            print('GPU: {}'.format(self.args.gpu))
            print('Version: {}\n'.format(self.args.version))

    def train(self):
        self.model.train()
        prev_time = time.time()
        train_loader = self.train_loader

        for i, (image, _,) in enumerate(train_loader):
            image = image.to(self.device)

            # train
            seg_mask, image_rec, gen_loss, dis_loss, logs = \
                self.model.process(image)

            # backward
            self.model.backward(gen_loss, dis_loss)

            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = self.epoch * train_loader.__len__() + i
            batches_left = self.args.n_epochs * train_loader.__len__() - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s" %
                             (self.epoch, self.args.n_epochs,
                              i, train_loader.__len__(),
                              dis_loss.item(),
                              gen_loss.item(),
                              time_left))

            # --------------
            #  Visdom
            # --------------
            if i % self.args.vis_freq == 0:
                image = image[:self.args.vis_batch]

                if self.args.data_modality == 'oct':
                    # BCWH -> BWH, torch.max in Channel dimension
                    seg_mask = torch.argmax(seg_mask[:self.args.vis_batch], dim=1).float()
                    # BWH -> B1WH, 11 -> 1
                    seg_mask = (seg_mask.unsqueeze(dim=1)/11).clamp(0, 1)

                else:
                    seg_mask = seg_mask[:self.args.vis_batch].clamp(0, 1)
                image_rec = image_rec[:self.args.vis_batch].clamp(0, 1)
                image_diff = torch.abs(image-image_rec)

                vim_images = torch.cat([image, seg_mask.expand([-1, 3, -1, -1]), image_rec, image_diff], dim=0)
                self.vis.images(vim_images, win_name='train', nrow=self.args.vis_batch)

                output_save = os.path.join(self.args.output_root,
                                           self.args.project,
                                           'output_v1_0812',
                                           self.args.version,
                                           'train')
                if not os.path.exists(output_save):
                    os.makedirs(output_save)
                tv.utils.save_image(vim_images, os.path.join(output_save, '{}.png'.format(i)), nrow=4)

            if i+1 == train_loader.__len__():
                self.vis.plot_multi_win(dict(dis_loss=dis_loss.item(),
                                             lr=self.new_lr))
                self.vis.plot_single_win(dict(gen_loss=gen_loss.item(),
                                              gen_l1_loss=logs['gen_l1_loss'].item(),
                                              gen_fm_loss=logs['gen_fm_loss'].item(),
                                              gen_gan_loss=logs['gen_gan_loss'].item(),
                                              seg_l1_loss=logs['seg_l1_loss'].item(),
                                              # gen_content_loss=logs['gen_content_loss'].item(),
                                              # gen_style_loss=logs['gen_style_loss'].item()
                                             ),
                                         win='gen_loss')

    def validate_cls(self):
        save_ckpt(version=self.args.version,
                  state={
                      'epoch': self.epoch,
                      'state_dict_G': self.model.model_G2.state_dict(),
                      'state_dict_D': self.model.model_D.state_dict(),
                  },
                  epoch=self.epoch,
                  args=self.args)
        print('\n Save ckpt successfully!')

    def test_acc(self):
        self.model.train()

        with torch.no_grad():
            _, train_predictions = self.forward_cls_dataloader(loader=self.train_loader)
            ground_truths, predictions = self.forward_cls_dataloader(loader=self.test_loader)

            """
            compute metrics
            """
            # get roc curve and compute the auc
            fpr, tpr, thresholds = metrics.roc_curve(np.array(ground_truths), np.array(predictions))
            total_auc = metrics.auc(fpr, tpr)

            """
            compute thereshold, and then compute the accuracy of AMD and Myopia
            """
            percentage = 0.75
            accuracy_threshold = sorted(train_predictions)[int(len(train_predictions) * percentage)]
            ##
            print('threshold', accuracy_threshold)
            print(predictions)
            ##
            class_predictions = [0 if prediction < accuracy_threshold else 1 for prediction in predictions]

            # acc, sensitivity and specifity
            def calculate_class_accuracy(predictions, ground_truths):
                ##
                print(ground_truths)
                print(predictions)
                ##
                accuracy = metrics.accuracy_score(y_true=ground_truths, y_pred=predictions)
                tn, fp, fn, tp = metrics.confusion_matrix(y_true=ground_truths, y_pred=predictions).ravel()
                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                return accuracy, precision, recall

            accuracy, precision, recall = calculate_class_accuracy(class_predictions, ground_truths)

            """
            plot metrics curve
            """
            # ROC curve
            self.vis.draw_roc(fpr, tpr)

            metrics_str = 'AUC = {:.4f}, Accuracy = {:.4f}, Precision = {:.4f}, Recall = {:.4f}'.format(total_auc, accuracy, precision, recall)
            self.vis.text(metrics_str)
            print(metrics_str)

    def forward_cls_dataloader(self, loader):
        if self.args.gpu >= 0:
            device = torch.device('cuda', self.args.gpu)
        else:
            device = torch.device('cpu')
        gt_list = []
        pred_list = []
        for i, items in enumerate(loader):
            if len(items) == 2:
                image, image_name = items
                annotation = None
            else:
                image, annotation, image_name = items
            image = image.to(device)
            # val, forward
            seg_mask, image_rec, seg_mask_rec = self.model(image)

            """
            preditction
            """
            # BCWH -> B, anomaly score
            image_residual = torch.abs(image_rec - image)
            image_diff_mae = image_residual.mean(dim=3).mean(dim=2).mean(dim=1)
            seg_residual = torch.abs(seg_mask_rec - seg_mask)
            seg_mask_mae = seg_residual.mean(dim=3).mean(dim=2).mean(dim=1)

            # image: tensor
            if annotation is None:
                gt_list += [0] * len(image)
            else:
                annotation = annotation.detach().to('cpu').numpy()
                annotation = annotation.reshape((annotation.shape[0], -1))
                gt_list += np.any(annotation >= 0.5, axis=1).astype(np.int32).tolist()
            image_weight = 0.8
            pred_list += (image_diff_mae * image_weight + seg_mask_mae * (1 - image_weight)).tolist()

            """
            visdom
            """
            if i % self.args.vis_freq_inval == 0:
                image = image[:self.args.vis_batch]
                image_rec = image_rec[:self.args.vis_batch].clamp(0, 1)
                image_diff = torch.abs(image - image_rec) * 10

                """
                Difference: seg_mask is different between fundus and oct images
                """
                if self.args.data_modality == 'fundus':
                    seg_mask = seg_mask[:self.args.vis_batch].clamp(0, 1)
                else:
                    seg_mask = torch.argmax(seg_mask[:self.args.vis_batch], dim=1).float()
                    seg_mask = (seg_mask.unsqueeze(dim=1) / 11).clamp(0, 1)

                vim_images = torch.cat([image, seg_mask.expand([-1, 3, -1, -1]), image_rec, seg_mask_rec.expand([-1, 3, -1, -1]), image_diff], dim=0)

                self.vis.images(vim_images, win_name='val', nrow=self.args.vis_batch)

                """
                save images
                """
                save_name = os.path.splitext(image_name[0])[0]
                save_path = os.path.join(self.args.output_root, self.args.project, '{}'.format(self.args.version),
                                         'val', '{}.png'.format(save_name))
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                tv.utils.save_image(vim_images, save_path, nrow=self.args.vis_batch)

        return gt_list, pred_list


class MultiTestForFigures(object):
    def __init__(self):
        args = ParserArgs().args
        if args.gpu >= 0:
            self.device = torch.device('cuda', args.gpu)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        if args.data_modality == 'fundus':
            # IDRiD dataset for segmentation
            # image, mask, image_name_item

            # iSee dataset for classification
            # image, image_name
            self.train_loader, self.normal_test_loader, \
            self.amd_fundus_loader, self.myopia_fundus_loader = \
                ClassificationFundusDataloader(data_root=args.isee_fundus_root,
                                               batch=args.batch,
                                               scale=args.scale).data_load()

        else:
            # Challenge OCT dataset for classification
            self.train_loader, self.normal_test_loader, self.oct_abnormal_loader = OCT_ClsDataloader(
                                                    data_root=args.challenge_oct,
                                                       batch=args.batch,
                                                       scale=args.scale).data_load()

        print_args(args)
        self.args = args

        for ablation_mode in range(6):
            args.resume = 'v22_ablation_{}@fundus@woVGG/latest_ckpt.pth.tar'.format(ablation_mode)
            self.model = PNetModel(args)
            self.test_cls(ablation_mode)


    def test_cls(self, ablation_mode, original_flag=False):
        # self.model.eval()
        self.model.train()

        with torch.no_grad():
            """
            Difference: abnormal dataloader and abnormal_list
            """
            if self.args.data_modality == 'fundus':
                self.forward_cls_dataloader(
                    loader=self.myopia_fundus_loader,
                    ablation_mode=ablation_mode,
                    original_flag=original_flag)

                self.forward_cls_dataloader(
                    loader=self.amd_fundus_loader,
                    ablation_mode=ablation_mode,
                    original_flag=original_flag)
            else:
                raise NotImplementedError('error')

            self.forward_cls_dataloader(
                loader=self.normal_test_loader,
                ablation_mode=ablation_mode,
                original_flag=original_flag)

    def forward_cls_dataloader(self, loader, ablation_mode, original_flag):
        for i, (image, image_name_item) in enumerate(loader):
            image = image.to(self.device)
            # val, forward
            seg_mask, image_rec, seg_mask_rec = self.model(image)

            image_name = image_name_item

            """
            save images
            """
            output_save = os.path.join('/home/imed/new_disk/workspace/',
                                       self.args.project,
                                       'output_v1_0812',
                                       'ablation_study_fundus_ML_feature')

            if not os.path.exists(output_save):
                os.makedirs(output_save)
            if original_flag:
                tv.utils.save_image(image, os.path.join(
                    output_save, '{}_a_input.png'.format(image_name[0])))
                tv.utils.save_image(seg_mask, os.path.join(
                    output_save, '{}_b_mask.png'.format(image_name[0])))
            order = ['c', 'd', 'e', 'f', 'g', 'h', 'i'][ablation_mode]
            tv.utils.save_image(image_rec, os.path.join(
                output_save, '{}_{}_{}.png'.format(image_name[0], order, ablation_mode)))


if __name__ == '__main__':
    import pdb
    RunMyModel()
    # MultiTestForFigures()
