import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from data.flickr import Flickr30kDataset

class FlickrDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str = "./data/flickr30k",
        tokenizer_name: str = "distilbert-base-uncased",
        batch_size: int = 32,
        num_workers: int = 0,
        image_size: int = 224,
        max_caption_length: int = 77,
    ):
        super().__init__()
        self.data_root = data_root
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_caption_length = max_caption_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def setup(self, stage=None):
        self.train_dataset = Flickr30kDataset(
            data_root=self.data_root,
            split='train',
            image_size=self.image_size,
            max_caption_length=self.max_caption_length,
            auto_download=True,
        )

        self.test_dataset = Flickr30kDataset(
            data_root=self.data_root,
            split='test',
            image_size=self.image_size,
            max_caption_length=self.max_caption_length,
            auto_download=False,
        )
        

    def collate_fn(self, batch):
        images = torch.stack([item['image'] for item in batch])
        images_raw = [item['image_raw'] for item in batch] 
        captions = [item['caption'] for item in batch]
        image_ids = [item['image_id'] for item in batch]

        encoded = self.tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=self.max_caption_length,
            return_tensors='pt'
        )

        return {
            'images': images,
            'images_raw': images_raw,  
            'captions': captions,
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'image_ids': image_ids,
            'batch_size': len(batch)
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )


    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )
