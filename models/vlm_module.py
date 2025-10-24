import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from transformers import AutoModel

class VLMModel(pl.LightningModule):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder_name: str = "distilbert-base-uncased",
        embed_dim: int = 512,
        lr: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        # Image encoder (e.g. wrapped ResNet)
        self.image_encoder = image_encoder
        self.image_proj = nn.Linear(self.image_encoder.output_dim, embed_dim)

        # Text encoder (e.g. DistilBERT)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        # Training config
        self.lr = lr

    def forward(self, images, input_ids, attention_mask):
        # Encode image
        img_feats = self.image_encoder(images)
        img_feats = self.image_proj(img_feats)  # [B, embed_dim]
        img_feats = F.normalize(img_feats, dim=-1)

        # Encode text
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = self.text_proj(text_output.last_hidden_state[:, 0])  # CLS token
        txt_feats = F.normalize(txt_feats, dim=-1)

        return img_feats, txt_feats

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        img_emb, txt_emb = self(images, input_ids, attention_mask)

        # Contrastive loss (InfoNCE / CLIP loss)
        logits = img_emb @ txt_emb.T  # [B, B]
        labels = torch.arange(len(logits)).to(self.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
        
    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        img_emb, txt_emb = self(images, input_ids, attention_mask)

        return {"img_emb": img_emb, "txt_emb": txt_emb}

    def on_validation_epoch_end(self, outputs):
        # Collect all embeddings
        image_embeds = torch.cat([x["img_emb"] for x in outputs], dim=0)
        text_embeds = torch.cat([x["txt_emb"] for x in outputs], dim=0)

        # Cosine similarity matrix
        sim = image_embeds @ text_embeds.T  # [N, N]
        N = sim.size(0)
        labels = torch.arange(N, device=self.device)

        # Image -> Text
        i2t_ranks = sim.argsort(dim=1, descending=True)
        i2t_top1 = (i2t_ranks[:, 0] == labels).float().mean()
        i2t_top5 = (labels.unsqueeze(1) == i2t_ranks[:, :5]).any(dim=1).float().mean()

        # Text -> Image
        t2i_ranks = sim.argsort(dim=0, descending=True)
        t2i_top1 = (t2i_ranks[0, :] == labels).float().mean()
        t2i_top5 = (labels.unsqueeze(0) == t2i_ranks[:5, :]).any(dim=0).float().mean()

        # Log to Lightning + WandB
        self.log_dict({
            "val/i2t_top1": i2t_top1,
            "val/i2t_top5": i2t_top5,
            "val/t2i_top1": t2i_top1,
            "val/t2i_top5": t2i_top5,
        }, prog_bar=True, sync_dist=True)
