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
                 auto_download=True):
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
        print("Dataset structure:")
        for split_name in ds.keys():
            print(f"   Split: {split_name}")
            if len(ds[split_name]) > 0:
                sample = ds[split_name][0]
                print(f"   Sample keys: {list(sample.keys())}")
                break
        
        # Process webdataset format
        all_data = []
        
        for split_name in ds.keys():
            print(f"Processing {split_name}...")
            split_data = []
            
            for i, item in enumerate(ds[split_name]):
                
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
        

        
        print(f"Loaded {len(data)} image-caption pairs for {self.split}")
        return data
    
    def _preprocess_text(self, text):
        """Preprocess text for VLM"""
        text = text.lower().strip()
        
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Fixed length for alignment
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
            pil_image = Image.open(image_path).convert('RGB')

            # Keep unnormalized version for visualization
            image_raw = transforms.ToTensor()(pil_image)  # [0, 1] float tensor

            # Normalized version for model input
            image = self.image_transform(pil_image)       # normalized for CLIP
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
            image_raw = torch.zeros(3, 224, 224)

    
        
        # Process caption
        caption = self._preprocess_text(caption)
        
        return {
            'image': image,
            'image_raw': image_raw,
            'caption': caption,
            'image_id': image_name,
            'index': idx
        }
