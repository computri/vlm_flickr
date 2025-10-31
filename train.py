import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from data.data_utils import FlickrDataModule
from models.vlm_module import VLMModel
from torchvision.models import resnet18
from torch import nn
import wandb    

from lightning.pytorch.loggers import WandbLogger

from omegaconf import OmegaConf
from utils.config import load_cfg

class ResNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # remove FC
        self.output_dim = base_model.fc.in_features  # typically 512

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # flatten to [B, output_dim]

def main():
    cfg = load_cfg()
    
    print(OmegaConf.to_yaml(cfg, resolve=True))

    accelerator = cfg.project.device

    # Data
    datamodule = FlickrDataModule(
        data_root="./data/flickr30k",
        tokenizer_name="distilbert-base-uncased",
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        image_size=224,
        max_caption_length=77,
        max_samples=None  # Use full dataset
    )

    # Model
    image_encoder = ResNetWrapper()
    model = VLMModel(
        image_encoder=image_encoder,
        text_encoder_name="distilbert-base-uncased",
        embed_dim=cfg.model.embed_dim,
        lr=cfg.train.lr
    )

    # Checkpointing
    checkpoint_cb = ModelCheckpoint(
        monitor="train/loss",
        save_top_k=1,
        mode="min",
        filename="vlm-{epoch:02d}-{train_loss:.2f}"
    )

    # WandB Logger
    wandb_run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        mode="offline" if not cfg.wandb.log else "online"
        
    )
    wandb_logger = WandbLogger(
        project="vlm", 
        log_model=False,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=accelerator,  # uses GPU if available
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=10
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()