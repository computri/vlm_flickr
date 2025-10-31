from matplotlib import transforms
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
import wandb
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModel
from torchvision.transforms import Resize
from torchvision.utils import make_grid

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

        # Image encoder
        self.image_encoder = image_encoder
        self.image_proj = nn.Linear(self.image_encoder.output_dim, embed_dim)

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

        # val logs
        self.val_image_embeds = []
        self.val_text_embeds = []

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

        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
        
    
    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        img_emb, txt_emb = self(images, input_ids, attention_mask)

        self.val_image_embeds.append(img_emb.detach().cpu())
        self.val_text_embeds.append(txt_emb.detach().cpu())

        if batch_idx == 1 and "captions" in batch and "images_raw" in batch:
            
            num_samples = min(4, len(images))
            resize = Resize((224, 224))


            for i in range(num_samples):
                caption = batch["captions"][i]
                # raw_img = batch["images_raw"][i]
                query_img = resize(batch["images_raw"][i])

                # Compute top-3 image matches (caption -> image)
                sim = txt_emb[i] @ img_emb.T
                top_indices = sim.topk(3).indices.tolist()

                # Get retrieved images (raw, resized)
                retrieved_imgs = [resize(batch["images_raw"][j]) for j in top_indices]
                # panel = [wandb.Image(to_pil_image(raw_img.cpu()), caption=captions_for_log[0])]
                 # Combine into grid: [Query | Top 1 | Top 2 | Top 3]
                all_imgs = [query_img] + retrieved_imgs  # each [3, H, W]
                grid = make_grid(all_imgs, nrow=len(all_imgs))  # shape: [3, H, W*4]

                # Compose caption
                caption_text = f"GT Caption: {caption}\n Top 1, Top 2, Top 3"

                # Log to WandB as single image
                self.logger.experiment.log({
                    f"val/retrieval_panel_{i}": wandb.Image(grid, caption=caption_text)
                }, step=self.global_step)
            

    def on_validation_epoch_end(self):
        # Gather all embeddings
        img_embs = torch.cat(self.val_image_embeds, dim=0).to(self.device)
        txt_embs = torch.cat(self.val_text_embeds, dim=0).to(self.device)

        sim = img_embs @ txt_embs.T  # cosine similarity
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

        self.log("val/i2t_top1", i2t_top1)
        self.log("val/i2t_top5", i2t_top5)
        self.log("val/t2i_top1", t2i_top1)
        self.log("val/t2i_top5", t2i_top5)

        # Clear buffers for next epoch
        self.val_image_embeds.clear()
        self.val_text_embeds.clear()
