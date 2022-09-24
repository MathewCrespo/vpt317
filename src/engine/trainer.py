#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage
from ..models.mlp import MLP

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        if 'gan' in self.cfg.MODEL.TRANSFER_TYPE:
            #print(self.model.feat_dim)
            self.discriminator = MLP(
                input_dim=self.model.feat_dim,
                nonlinearity= nn.LeakyReLU,
                mlp_dims=[self.model.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                    [1], # noqa
                special_bias=True,
                dis = True
            )
            self.discriminator = self.discriminator.cuda()
            self.gan_criterion = nn.BCELoss()

        self.device = device
        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model.enc, self.model.head], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)

        if 'gan' in self.cfg.MODEL.TRANSFER_TYPE:
            self.optimizerD = make_optimizer([self.discriminator],cfg.SOLVER)
            self.schedulerD = make_scheduler(self.optimizerD, cfg.SOLVER)
            self.optimizerG = make_optimizer([self.model.enc],cfg.SOLVER)
            self.schedulerG = make_scheduler(self.optimizerG, cfg.SOLVER) # may be a bug here

        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        if 'gan' in self.cfg.MODEL.TRANSFER_TYPE and is_train:
            x = inputs['target'].to(self.device, non_blocking=True)
            source = inputs['source'].to(self.device, non_blocking=True)
            inputs = x
        else:
            inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        
        
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")
        # forward
        with torch.set_grad_enabled(is_train):
            # As an attempt, we first train GAN loss and then the classification loss:
            # only prompt+gan mode and training requires this process
            if 'gan' in self.cfg.MODEL.TRANSFER_TYPE and is_train:
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # optimizer zero_grad                
                real_label = 1
                fake_label = 0

                # get four kinds of outputs
                x_vpt, source_vpt, x_freeze, source_freeze = self.model.forward_encoder(x,source)
                batch_size = x_vpt.shape[0]

                # train with real: source --> frozen encoder/ vpt encoder
                self.optimizerD.zero_grad()
                #source_vpt_D = self.discriminator(source_vpt.detach()) #  source_embed or detach ?
                source_freeze_D = self.discriminator(source_freeze.detach())
                label = torch.full((batch_size,1), real_label, dtype = torch.float,  device='cuda') # check type here
                #lossD_real_vpt = self.gan_criterion(source_vpt_D,label)
                lossD_real_freeze = self.gan_criterion(source_freeze_D,label)
                #lossD_real = lossD_real_vpt + lossD_real_freeze
                lossD_real = lossD_real_freeze
                lossD_real.backward()

                # train with fake: downstream --> frozen encoder/ vpt encoder
                x_vpt_D = self.discriminator(x_vpt.detach())
                #x_freeze_D = self.discriminator(x_freeze.detach())             
                label.fill_(fake_label)
                lossD_fake_vpt = self.gan_criterion(x_vpt_D,label)
                #lossD_fake_freeze = self.gan_criterion(x_freeze_D,label)
                lossD_fake = lossD_fake_vpt# + lossD_fake_freeze
                lossD_fake.backward()
                self.optimizerD.step()
                
                ## optimizer update
                #(2) Update G network (encoder): maximize log(D(G(z)))
                # optimizer.zero_grad()
                self.optimizer.zero_grad()
                label.fill_(real_label)
                x_out = self.discriminator(x_vpt)
                lossG = self.gan_criterion(x_out,label)
                lossG.backward()
                #self.optimizerG.step()
                self.optimizer.step()
        
            # classification loss
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #print('cls loss backward')
        
        if 'gan' in self.cfg.MODEL.TRANSFER_TYPE and is_train:
            return lossD_fake, lossD_real, lossG, loss, outputs

        return loss, outputs

    def get_input(self, data,is_train=True):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)
        if 'gan' in self.cfg.MODEL.TRANSFER_TYPE and is_train:
            inputs = {
                'source': data['source'].float(),
                'target': data['image'].float()
            }
            labels = data["label"]
        
        else:
            inputs = data["image"].float()
            labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        Main training with several epoches
        """
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        if 'gan' in self.cfg.MODEL.TRANSFER_TYPE:
            lossesD_real = AverageMeter('Loss', ':.4e')
            lossesD_fake = AverageMeter('Loss', ':.4e')
            lossesG = AverageMeter('Loss', ':.4e')


        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            if 'gan' in self.cfg.MODEL.TRANSFER_TYPE:
                lossesD_real.reset()
                lossesD_fake.reset()
                lossesG.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break
                
                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                if 'gan' in self.cfg.MODEL.TRANSFER_TYPE:
                    train_lossD_fake, train_lossD_real, train_lossG, train_loss, _ = self.forward_one_batch(X, targets, True)
                else:
                    train_loss, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None
                
                if 'gan' in self.cfg.MODEL.TRANSFER_TYPE:
                    losses.update(train_loss.item(), X['source'].shape[0])
                    lossesD_fake.update(train_lossD_fake.item(), X['source'].shape[0])
                    lossesD_real.update(train_lossD_real.item(), X['source'].shape[0])
                    lossesG.update(train_lossG.item(), X['source'].shape[0])
                else:
                    losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )

            if 'gan' in self.cfg.MODEL.TRANSFER_TYPE:
                logger.info(
                    "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                    + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                        data_time.avg, batch_time.avg)
                    + "average train loss: {:.4f}".format(losses.avg)
                    + "average G loss: {:.4f}, ".format(lossesG.avg)
                    + "average realD loss: {:.4f}, ".format(lossesD_real.avg)
                    + "average fakeD loss: {:.4f}, ".format(lossesD_fake.avg))

            else:
                logger.info(
                    "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                    + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                        data_time.avg, batch_time.avg)
                    + "average train loss: {:.4f}".format(losses.avg))
            # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()
            if 'gan' in self.cfg.MODEL.TRANSFER_TYPE:
                self.schedulerD.step()

            # Enable eval mode
            self.model.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)

            # check the patience
            t_name = "val_" + val_loader.dataset.name
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

        # save the last checkpoints
        # if self.cfg.MODEL.SAVE_CKPT:
        #     Checkpointer(
        #         self.model,
        #         save_dir=self.cfg.OUTPUT_DIR,
        #         save_to_disk=True
        #     ).save("last_model")

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data,is_train=False)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
            joint_logits, total_targets,
            test_name, self.cfg.DATA.MULTILABEL,
        )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")
