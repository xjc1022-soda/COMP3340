from argparse import ArgumentParser
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet18, ResNet18_Weights
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datamodule import SADataModule
from dataset import SADataset
import numpy as np
import wandb 
import datetime 
import utils
import time 

class SurvivalModel(LightningModule):
    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 layer_num: int,
                 *args,
                 **kwargs) -> None:

        super().__init__()

        self.save_hyperparameters()
        
        self.layer_num = layer_num
        self.img_encoder = resnet18(pretrained=True)
        in_features = self.img_encoder.fc.in_features
        self.img_encoder.fc = nn.Linear(in_features, 17)
        for index, (name,value) in enumerate(self.img_encoder.named_parameters()):
            if index < self.layer_num:
                value.requires_grad= False
            if value.requires_grad == True:
                print("\t", index, name)

    def shared_step(self, batch, batch_idx, split):
        img, label = batch
        scores = self.img_encoder(img.float())

        # soft max/cross entropy loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "train")
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        wandb.log({"train loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "val")
        self.log("val_loss", loss, prog_bar=True)
        wandb.log({"val_loss": loss})
        return loss


    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()),  
                          lr=self.hparams.learning_rate
                          )
        lr_scheduler = StepLR(optimizer, step_size=20, gamma=1.)
        # lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            # "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--layer_num", type=int, default=58)
        return parser


def cli_main():
    seed_everything(42)



    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = SurvivalModel.add_model_specific_args(parser)
    args = parser.parse_args()
    args.max_epochs = 100
    args.deterministic = True
    args.log_every_n_steps = 25
    args.accelerator = 'gpu'
    args.devices = [0]    
    


    wandb.init(project="flower"+str(datetime.date.today()),
               name=str(args.layer_num)+": "+str(args.learning_rate)+"_"+str(args.max_epochs)
              )    

    model = SurvivalModel(**args.__dict__)
      

    dm = SADataModule(args.batch_size, args.num_workers)

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(dirpath="ckpts", monitor="val_loss", save_last=True, save_top_k=1)
        ])
    trainer.fit(model, dm)
    

if __name__ == "__main__":
    cli_main()
