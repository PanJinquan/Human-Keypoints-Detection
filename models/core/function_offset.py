# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

from models.core.config import get_model_name
from models.core.evaluate import accuracy
from models.core.inference import get_final_preds, get_final_preds_offset
from models.tools.transforms import flip_back, flip_back_offset
from models.tools.vis import save_debug_images

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    if config.MODEL.EXTRA.TARGET_TYPE == 'offset':
        losses_hm = AverageMeter()
        losses_os = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.to(device)  # torch.Size([16, 3, 256, 192])
        target = target.to(device)  # torch.Size([16, 10, 64, 48])
        target_weight = target_weight.to(device)  # torch.Size([16, 10, 1])
        # compute output
        output = model(input)  # torch.Size([16, 10, 64, 48])
        # target = target.cuda(non_blocking=True)
        # target_weight = target_weight.cuda(non_blocking=True)

        # loss = criterion(output, target, target_weight)
        if config.MODEL.EXTRA.TARGET_TYPE == 'gaussian':
            loss = criterion(output, target, target_weight)
        elif config.MODEL.EXTRA.TARGET_TYPE == 'offset':
            loss_hm, loss_os = criterion(output, target, target_weight)
            loss = loss_hm + loss_os
        else:
            loss = criterion(output, target, target_weight)
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if config.MODEL.EXTRA.TARGET_TYPE == 'offset':
            losses_hm.update(loss_hm.item(), input.size(0))
            losses_os.update(loss_os.item(), input.size(0))

        if config.MODEL.EXTRA.TARGET_TYPE == 'gaussian':
            _, avg_acc, cnt, pred = accuracy(output.cpu().detach().numpy(),
                                             target.cpu().detach().numpy())
        elif config.MODEL.EXTRA.TARGET_TYPE == 'offset':
            _, avg_acc, cnt, pred = accuracy(output.cpu().detach().numpy()[:, ::3, :, :],
                                             target.cpu().detach().numpy()[:, ::3, :, :])

        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                  'lr:{lr:.7f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc, lr=lr)
            if config.MODEL.EXTRA.TARGET_TYPE == 'offset':
                msg = msg + '\tLs_hm {loss_hm_val:.1f}({loss_hm_avg:.1f})  Ls_os {loss_os_val:.1f}({loss_os_avg:.1f})'.format(
                    loss_hm_val=losses_hm.val * 10 ** 5, loss_hm_avg=losses_hm.avg * 10 ** 5,
                    loss_os_val=losses_os.val * 10 ** 5, loss_os_avg=losses_os.avg * 10 ** 5)
            logger.info(msg)

            # writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            # writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            # writer.add_scalar('lr_global_steps', lr, global_steps)

            writer_dict['train_global_steps'] = global_steps + 1
            prefix = os.path.join(output_dir, "images", 'train', str(i))
            save_debug_images(config, input, meta, target, pred * 4, output, prefix)
    writer = writer_dict['writer']
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_acc', acc.avg, epoch)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict, device, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    if config.MODEL.EXTRA.TARGET_TYPE == 'offset':
        losses_hm = AverageMeter()
        losses_os = AverageMeter()
    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    image_idx = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            input = input.to(device)
            target = target.to(device)
            target_weight = target_weight.to(device)
            output = model(input)  # torch.Size([32, 3, 256, 192])->torch.Size([32, 17, 64, 48])
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                # input_flipped = torch.from_numpy(input_flipped).cuda()
                input_flipped = torch.from_numpy(input_flipped).to(device)
                output_flipped = model(input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(), val_dataset.flip_pairs)
                # output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
                output_flipped = torch.from_numpy(output_flipped.copy()).to(device)

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0
                if config.MODEL.TARGET_TYPE == 'gaussian':
                    output_flipped = flip_back(output_flipped.cpu().numpy(), val_dataset.flip_pairs)
                elif config.MODEL.EXTRA.TARGET_TYPE == 'offset':
                    output_flipped = flip_back_offset(output_flipped.cpu().numpy(), val_dataset.flip_pairs)
                output = (output + output_flipped) * 0.5
            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.cuda(non_blocking=True)

            # loss = criterion(output, target, target_weight)
            if config.MODEL.EXTRA.TARGET_TYPE == 'gaussian':
                loss = criterion(output, target, target_weight)
            elif config.MODEL.EXTRA.TARGET_TYPE == 'offset':
                loss_hm, loss_os = criterion(output, target, target_weight)
                loss = loss_hm + loss_os

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            if config.MODEL.EXTRA.TARGET_TYPE == 'offset':
                losses_hm.update(loss_hm.item(), num_images)
                losses_os.update(loss_os.item(), num_images)

            if config.MODEL.EXTRA.TARGET_TYPE == 'gaussian':
                _, avg_acc, cnt, pred = accuracy(output.cpu().detach().numpy(),
                                                 target.cpu().detach().numpy())
            elif config.MODEL.EXTRA.TARGET_TYPE == 'offset':
                _, avg_acc, cnt, pred = accuracy(output.cpu().detach().numpy()[:, ::3, :, :],
                                                 target.cpu().detach().numpy()[:, ::3, :, :])

            # _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
            #                                  target.cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            image_id = meta['image_id'].numpy().tolist()

            if config.MODEL.EXTRA.TARGET_TYPE == 'gaussian':
                preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)
            elif config.MODEL.EXTRA.TARGET_TYPE == 'offset':
                preds, maxvals, _ = get_final_preds_offset(config, output.clone().cpu().numpy(), c, s)
            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            image_idx.extend(image_id)
            if config.DATASET.DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logger.info(msg)

                # prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                prefix = os.path.join(output_dir, "images", 'val', str(i))
                save_debug_images(config, input, meta, target, pred * 4, output,
                                  prefix)

        name_values, ap = val_dataset.evaluate(config,
                                               all_preds,
                                               output_dir,
                                               all_boxes,
                                               image_path,
                                               filenames,
                                               imgnums,
                                               image_idx=image_idx)

        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            # global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, epoch)
            writer.add_scalar('valid_acc', acc.avg, epoch)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), epoch)
            else:
                name_values = {n: v for n, v in dict(name_values).items() if v >= 0}
                # name_value = dict(name_value)
                writer.add_scalars('valid', name_values, epoch)
            writer_dict['valid_global_steps'] = epoch
    return ap


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
