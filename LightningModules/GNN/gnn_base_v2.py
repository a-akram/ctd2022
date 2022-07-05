#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
# from .utils import purity_sample
from .utils import load_dataset, LargeDataset

# FIXME::ADAK: I have removed .bool() from 'y_pid', 'y' and 'truth' varialbe, it gave an error.
# TODO::ADAK: Put back the <var>.bool() as 'y_pid.bool()', 'y.bool()' and 'truth.bool()'


class GNNBase(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(hparams)
        
        # Instance Variables
        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage: str = None) -> None:
        
        # Handle any subset of [train, val, test] data split (mind the ordering)
        if self.trainset is None:
            print("Setting up dataset")
            
            input_subdirs = [None, None, None]
            input_subdirs[: len(self.hparams["datatype_names"])] = [
                os.path.join(self.hparams["input_dir"], datatype)
                for datatype in self.hparams["datatype_names"]
            ]
            
            self.trainset, self.valset, self.testset = [
                load_dataset(
                    input_subdir=input_subdir,
                    num_events=self.hparams["datatype_split"][i],
                    **self.hparams
                )
                for i, input_subdir in enumerate(input_subdirs)
            ]
        
    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(
                self.trainset, batch_size=1, num_workers=8
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(
                self.valset, batch_size=1, num_workers=8
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(
                self.testset, batch_size=1, num_workers=8
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None
    
    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    # 1 - Helper Function
    def get_input_data(self, batch):

        if self.hparams["cell_channels"] > 0:
            input_data = torch.cat(
                [batch.cell_data[:, : self.hparams["cell_channels"]], batch.x], dim=-1)
            input_data[input_data != input_data] = 0
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0

        return input_data
    
    # 2 - Helper Function
    def handle_directed(self, batch, edge_sample, truth_sample, sample_indices):

        edge_sample = torch.cat([edge_sample, edge_sample.flip(0)], dim=-1)
        truth_sample = truth_sample.repeat(2)
        sample_indices = sample_indices.repeat(2)

        if ("directed" in self.hparams.keys()) and self.hparams["directed"]:
            direction_mask = batch.x[edge_sample[0], 0] < batch.x[edge_sample[1], 0]
            edge_sample = edge_sample[:, direction_mask]
            truth_sample = truth_sample[direction_mask]

        return edge_sample, truth_sample, sample_indices
    
    # 3 - Helper Function
    def log_metrics(self, output, truth_sample, batch, loss, log):
        """Logging metrics"""
        
        # Edge filter performance
        score = torch.sigmoid(output)
        preds = torch.sigmoid(output) > self.hparams["edge_cut"]
        
        # Positives
        edge_positive = preds.sum().float()
        
        # Truth
        edge_true = truth_sample.sum().float()
        
        # True Positives
        edge_true_positive = ((truth_sample.bool() & preds).sum().float())        
        
        # Efficiency, Purity, AUC, F1-score
        eff = edge_true_positive.clone().detach() / max(1, edge_true)
        pur = edge_true_positive.clone().detach() / max(1, edge_positive)
        auc = roc_auc_score(truth_sample.bool().cpu().detach(), score.cpu().detach())
        
        # Logging Metrics
        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {
                    "val_loss": loss,
                    "current_lr": current_lr,
                    "eff": eff,
                    "pur": pur,
                    "auc": auc,
                }, on_epoch=True, on_step=False, batch_size=10000
            )
        
        return score, preds

    def training_step(self, batch, batch_idx):
        """Train Step Hook."""
        
        # Get Weight
        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )
        
        # Get Truth
        truth = (
            batch.y_pid.bool() if "pid" in self.hparams["regime"] else batch.y.bool()
        )
        
        # True & Input Samples
        edge_sample, truth_sample, sample_indices = batch.edge_index, truth, torch.arange(batch.edge_index.shape[1])
        edge_sample, truth_sample, _ = self.handle_directed(batch, edge_sample, truth_sample, sample_indices)
        
        # Network I/O
        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample).squeeze()
        
        # Manual Weighting
        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights
        else:
            manual_weights = None
        
        # BCE Loss
        loss = F.binary_cross_entropy_with_logits(
            output, truth_sample.float(), weight=manual_weights, pos_weight=weight
        )

        self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=10000)

        return loss
        
    def shared_evaluation(self, batch, batch_idx, log=False):
        """The shared_evaluation() for validation_step() and test_step() hooks."""
        
        # Get Weight
        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )
        
        # Get Truth
        truth = (
            batch.y_pid.bool() if "pid" in self.hparams["regime"] else batch.y.bool()
        )
        
        # True & Input Samples
        edge_sample, truth_sample, sample_indices = batch.edge_index, truth, torch.arange(batch.edge_index.shape[1])
        edge_sample, truth_sample, _ = self.handle_directed(batch, edge_sample, truth_sample, sample_indices)
        
        # Network I/O
        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample).squeeze()
        
        # Manual Weighting
        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights
        else:
            manual_weights = None
        
        # BCE Loss
        loss = F.binary_cross_entropy_with_logits(
            output, truth_sample.float(), weight=manual_weights, pos_weight=weight
        )
                
        # Calculate Metrics
        score, preds = self.log_metrics(output, truth_sample, batch, loss, log)
        
        return {"loss": loss, "preds": preds, "score": score, "truth": truth_sample}

    def validation_step(self, batch, batch_idx):
        """Validation Step Hook."""
        outputs = self.shared_evaluation(batch, batch_idx, log=True)
        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """Test Step Hook"""
        outputs = self.shared_evaluation(batch, batch_idx, log=False)
        return outputs

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx=0,  # ADAK: optimizer_idx to optimizer_idx=0
                       optimizer_closure=None,
                       on_tpu=False,
                       using_native_amp=False,
                       using_lbfgs=False):

        """Optimizer Step Hook"""
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()


class LargeGNNBase(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)

    def setup(self, stage: str = None) -> None:
        # Handle any subset of [train, val, test] data split, assuming that ordering

        self.trainset, self.valset, self.testset = [
            LargeDataset(
                self.hparams["input_dir"],
                subdir,
                split,
                self.hparams
            )
            for subdir, split in zip(self.hparams["datatype_names"], self.hparams["datatype_split"])
        ]

        if (
            self.trainer
            and ("logger" in self.trainer.__dict__.keys())
            and ("_experiment" in self.logger.__dict__.keys())
        ):
            self.logger.experiment.define_metric("val_loss", summary="min")
            self.logger.experiment.define_metric("sig_auc", summary="max")
            self.logger.experiment.define_metric("tot_auc", summary="max")
            self.logger.experiment.define_metric("sig_fake_ratio", summary="max")
            self.logger.experiment.define_metric("custom_f1", summary="max")
            self.logger.experiment.log({"sig_auc": 0})
            self.logger.experiment.log({"sig_fake_ratio": 0})
            self.logger.experiment.log({"custom_f1": 0})
