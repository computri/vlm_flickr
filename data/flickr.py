import os
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms
import nltk
from nltk.tokenize import word_tokenize

class Flickr30kDataset(Dataset):
    def __init__(self, data_root="./data/flickr30k", split='train', 
                 image_size=224, max_caption_length=77, 
                 auto_download=True, max_samples=None):
        """
        Complete Flickr30k dataset that downloads and creates dataloaders
        
        Args:
            data_root: Directory for dataset
            split: 'train', 'val', or 'test'
            image_size: Size to resize images to
            max_caption_length: Maximum caption length
            auto_download: Whether to download if not found
            max_samples: Limit samples (None for all)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.max_caption_length = max_caption_length
        self.max_samples = max_samples
        
        # Setup dataset
        if auto_download:
            self._setup_dataset()
        
        # Load data
        self.data = self._load_data()
        
        # Image preprocessing (CLIP-style)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        # Setup NLTK
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    def _setup_dataset(self):
        """Download and setup dataset using clip-benchmark/wds_flickr30k"""
        print(f"Setting up Flickr30k dataset...")
        
        try:
            from datasets import load_dataset
        except ImportError:
            print("Please install: pip install datasets")
            raise
        
        self.data_root.mkdir(parents=True, exist_ok=True)
        images_dir = self.data_root / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        csv_files = list(self.data_root.glob("*_captions.csv"))
        if len(csv_files) >= 3 and images_dir.exists():
            image_count = len(list(images_dir.glob("*.jpg")))
            if image_count > 1000:  # Reasonable threshold
                print(f"Dataset already exists ({image_count} images)")
                return
        
        print("Downloading clip-benchmark/wds_flickr30k...")
        ds = load_dataset("clip-benchmark/wds_flickr30k")
        
        # Debug: check structure
        print("📋 Dataset structure:")
        for split_name in ds.keys():
            print(f"   Split: {split_name}")
            if len(ds[split_name]) > 0:
                sample = ds[split_name][0]
                print(f"   Sample keys: {list(sample.keys())}")
                break
        
        # Process webdataset format
        all_data = []
        
        for split_name in ds.keys():
            print(f"📊 Processing {split_name}...")
            split_data = []
            
            for i, item in enumerate(ds[split_name]):
                if self.max_samples and i >= self.max_samples:
                    break
                
                # Debug first few items
                if i < 3:
                    print(f"  Item {i} keys: {list(item.keys())}")
                
                # Extract image from webdataset format
                image = None
                for img_key in ['jpg', 'png', 'jpeg', 'image', 'img']:
                    if img_key in item:
                        try:
                            image_data = item[img_key]
                            if isinstance(image_data, bytes):
                                # Convert bytes to PIL Image
                                from io import BytesIO
                                image = Image.open(BytesIO(image_data))
                            elif hasattr(image_data, 'save'):  # Already PIL Image
                                image = image_data
                            break
                        except Exception as e:
                            print(f"  Error processing image {i}: {e}")
                            continue
                
                # Extract caption from webdataset format
                caption = None
                for txt_key in ['txt', 'text', 'caption', 'json']:
                    if txt_key in item:
                        try:
                            text_data = item[txt_key]
                            if isinstance(text_data, bytes):
                                text_data = text_data.decode('utf-8')
                            
                            if txt_key == 'json':
                                # Parse JSON for caption
                                if isinstance(text_data, str):
                                    parsed = json.loads(text_data)
                                else:
                                    parsed = text_data
                                caption = parsed.get('caption', parsed.get('text', str(parsed)))
                            else:
                                caption = str(text_data)
                            break
                        except Exception as e:
                            print(f"  Error processing text {i}: {e}")
                            continue
                
                if image is None or caption is None:
                    if i < 5:  # Only show warnings for first few items
                        print(f"  Skipping item {i}: image={image is not None}, caption={caption is not None}")
                    continue
                
                # Save image
                image_filename = f"{split_name}_{i:06d}.jpg"
                image_path = images_dir / image_filename
                
                if not image_path.exists():
                    try:
                        image.save(image_path, 'JPEG', quality=95)
                    except Exception as e:
                        print(f"  Failed to save image {i}: {e}")
                        continue
                
                split_data.append({
                    'image': image_filename,
                    'caption': caption,
                    'image_id': f"{split_name}_{i}",
                    'caption_id': f"{split_name}_{i}_0",
                    'split': split_name
                })
                
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1} items...")
            
            # Save CSV
            if split_data:
                df = pd.DataFrame(split_data)
                csv_file = self.data_root / f"{split_name}_captions.csv"
                df.to_csv(csv_file, index=False)
                print(f"  Saved {len(split_data)} pairs to {csv_file}")
                all_data.extend(split_data)
        
        # Save combined file
        if all_data:
            pd.DataFrame(all_data).to_csv(self.data_root / "all_captions.csv", index=False)
            print(f"Setup complete! {len(all_data)} total pairs")
        
        return True
    
    def _load_data(self):
        """Load caption data for the split"""
        csv_file = self.data_root / f"{self.split}_captions.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Caption file not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        data = [(row['image'], row['caption']) for _, row in df.iterrows()]
        
        if self.max_samples:
            data = data[:self.max_samples]
        
        print(f"Loaded {len(data)} image-caption pairs for {self.split}")
        return data
    
    def _preprocess_text(self, text):
        """Preprocess text for VLM + OT"""
        text = text.lower().strip()
        
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Fixed length for OT alignment
        if len(tokens) > self.max_caption_length:
            tokens = tokens[:self.max_caption_length]
        else:
            tokens.extend(['<pad>'] * (self.max_caption_length - len(tokens)))
        
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_name, caption = self.data[idx]
        
        # Load image
        image_path = self.data_root / "images" / image_name
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Process caption
        caption = self._preprocess_text(caption)
        
        return {
            'image': image,
            'caption': caption,
            'image_id': image_name,
            'index': idx
        }

def create_flickr30k_dataloader(data_root="./data/flickr30k", split='train', 
                               batch_size=32, image_size=224, 
                               num_workers=4, shuffle=None, 
                               auto_download=True, max_samples=None):
    """
    Create Flickr30k DataLoader (downloads if needed)
    
    Args:
        data_root: Dataset directory
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of workers
        shuffle: Whether to shuffle (auto if None)
        auto_download: Download if not found
        max_samples: Limit samples for testing
    """
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Create dataset (downloads if needed)
    dataset = Flickr30kDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        auto_download=auto_download,
        max_samples=max_samples
    )
    
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        captions = [item['caption'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        indices = torch.tensor([item['index'] for item in batch])
        
        return {
            'images': images,
            'captions': captions,
            'image_ids': image_ids,
            'indices': indices,
            'batch_size': len(batch)
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'train'),
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader

def get_flickr30k_info(dataloader):
    """Get dataset statistics"""
    dataset = dataloader.dataset
    
    stats = {
        'split': dataset.split,
        'total_samples': len(dataset),
        'num_batches': len(dataloader),
        'batch_size': dataloader.batch_size,
        'data_root': str(dataset.data_root)
    }
    
    print(f"Flickr30k Dataset Info:")
    print(f"   Split: {stats['split']}")
    print(f"   Samples: {stats['total_samples']}")
    print(f"   Batches: {stats['num_batches']}")
    print(f"   Batch size: {stats['batch_size']}")
    print(f"   Location: {stats['data_root']}")
    
    # Test a batch
    try:
        batch = next(iter(dataloader))
        print(f"   Image shape: {batch['images'].shape}")
        print(f"   Sample caption: {batch['captions'][0][:60]}...")
        stats['image_shape'] = batch['images'].shape[1:]
    except Exception as e:
        print(f"   Error loading batch: {e}")
    
    return stats

def quick_test():
    """Quick test function"""
    print("Quick Flickr30k Test")
    
    # Small test dataset
    train_loader = create_flickr30k_dataloader(
        data_root="./data/flickr30k_test",
        split='train',
        batch_size=4,
        max_samples=100,  # Small for testing
        num_workers=2
    )
    
    # Get info
    stats = get_flickr30k_info(train_loader)
    
    # Test iteration
    print("\nTesting iteration...")
    for i, batch in enumerate(train_loader):
        print(f"   Batch {i+1}: {batch['images'].shape}, {len(batch['captions'])} captions")
        if i >= 2:  # Just test a few batches
            break
    
    print("Test complete")
    return train_loader

# Convenience functions for different use cases
def get_small_flickr30k(batch_size=16):
    """Get small dataset for quick prototyping"""
    return create_flickr30k_dataloader(
        data_root="./data/flickr30k_small",
        split='train',
        batch_size=batch_size,
        max_samples=1000,
        num_workers=2
    )

def get_full_flickr30k(split='train', batch_size=32):
    """Get full dataset for training"""
    return create_flickr30k_dataloader(
        data_root="./data/flickr30k",
        split=split,
        batch_size=batch_size,
        num_workers=4
    )

def get_all_splits(data_root="./data/flickr30k", batch_size=32):
    """Get all splits (train, val, test)"""
    splits = {}
    for split in ['train', 'val', 'test']:
        try:
            splits[split] = create_flickr30k_dataloader(
                data_root=data_root,
                split=split,
                batch_size=batch_size,
                auto_download=(split == 'train')  # Only download once
            )
            print(f"{split} split loaded")
        except Exception as e:
            print(f"Failed to load {split}: {e}")
    
    return splits

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flickr30k Dataset - All-in-One")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")  
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--data_root", default="./data/flickr30k", help="Data directory")
    
    args = parser.parse_args()
    

    # Create dataloader
    loader = create_flickr30k_dataloader(
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Show info
    get_flickr30k_info(loader)
    
