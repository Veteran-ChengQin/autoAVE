"""
Data loader for VideoAVE attribute-conditioned dataset
"""
import os
import csv
import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class VideoAVEAttrDataset:
    """
    Dataset for attribute-conditioned video attribute extraction.
    Flattens VideoAVE data into (video_path, category, title, attr_name, attr_value) tuples.
    """
    
    def __init__(self, data_root: str, domains: Optional[List[str]] = None, 
                 split: str = "train", max_samples: Optional[int] = None):
        """
        Args:
            data_root: Root directory of VideoAVE dataset
            domains: List of domains to load (e.g., ["beauty", "sports"])
            split: "train" or "test"
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_root = data_root
        self.split = split
        self.max_samples = max_samples
        self.samples = []
        
        # Determine which domains to load
        if domains is None:
            domains = self._get_all_domains()
        
        self._load_data(domains)
        logger.info(f"Loaded {len(self.samples)} attribute samples from {split} split")
    
    def _get_all_domains(self) -> List[str]:
        """Get all available domains from dataset"""
        split_dir = os.path.join(self.data_root, f"{self.split}_data")
        if not os.path.exists(split_dir):
            return []
        
        domains = []
        for fname in os.listdir(split_dir):
            if fname.endswith(f"_{self.split}.csv"):
                domain = fname.replace(f"_{self.split}.csv", "")
                domains.append(domain)
        return sorted(domains)
    
    def _load_data(self, domains: List[str]) -> None:
        """Load and flatten data from specified domains"""
        for domain in domains:
            csv_path = os.path.join(
                self.data_root, 
                f"{self.split}_data", 
                f"{domain}_{self.split}.csv"
            )
            
            if not os.path.exists(csv_path):
                logger.warning(f"Dataset file not found: {csv_path}")
                continue
            
            logger.info(f"Loading {domain} from {csv_path}")
            
            try:
                df = pd.read_csv(csv_path)
                
                for idx, row in df.iterrows():
                    if self.max_samples and idx >= self.max_samples:
                        break
                    
                    product_id = row.get("product_id", "")
                    video_url = row.get("video_url", "")
                    video_path = row.get("content_url", video_url)  # Use content_url if available
                    title = row.get("video_title", "") or row.get("parent_title", "")
                    
                    # Parse aspects (attributes)
                    aspects_str = row.get("aspects", "{}")
                    try:
                        if isinstance(aspects_str, str):
                            if not aspects_str or aspects_str.strip() == "":
                                aspects = {}
                            else:
                                # Try JSON first (standard format)
                                try:
                                    aspects = json.loads(aspects_str)
                                except json.JSONDecodeError:
                                    # Fall back to eval for Python dict format {'key': 'value'}
                                    try:
                                        aspects = eval(aspects_str)
                                        if not isinstance(aspects, dict):
                                            aspects = {}
                                    except Exception:
                                        logger.warning(f"Failed to parse aspects for product {product_id}: {aspects_str[:100]}")
                                        aspects = {}
                        else:
                            aspects = aspects_str if isinstance(aspects_str, dict) else {}
                    except Exception as e:
                        logger.warning(f"Error parsing aspects for product {product_id}: {e}")
                        aspects = {}
                    
                    # Flatten: create one sample per attribute
                    for attr_name, attr_value in aspects.items():
                        if attr_value:  # Skip empty values
                            sample = {
                                "product_id": product_id,
                                "video_path": video_path,
                                "category": domain,
                                "title": title,
                                "attr_name": attr_name,
                                "attr_value": str(attr_value),
                            }
                            self.samples.append(sample)
                
            except Exception as e:
                logger.error(f"Error loading {csv_path}: {e}")
                continue
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Return a single attribute sample"""
        return self.samples[idx].copy()
    
    def get_by_product(self, product_id: str) -> List[Dict]:
        """Get all attribute samples for a specific product"""
        return [s for s in self.samples if s["product_id"] == product_id]
    
    def get_unique_products(self) -> List[str]:
        """Get list of unique product IDs"""
        return list(set(s["product_id"] for s in self.samples))
    
    def get_unique_attributes(self) -> List[str]:
        """Get list of unique attribute names"""
        return list(set(s["attr_name"] for s in self.samples))


class DataLoader:
    """Simple batch data loader"""
    
    def __init__(self, dataset: VideoAVEAttrDataset, batch_size: int = 4, 
                 shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        
        if shuffle:
            import random
            random.shuffle(self.indices)
    
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield batch
    
    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
