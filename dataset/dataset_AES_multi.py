import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union

import gin
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.utils.data.dataset
import tqdm
from sklearn.cluster import KMeans


@gin.configurable
class AESFeaturesMulti(torch.utils.data.dataset.Dataset):
    """AES dataset providing Multiple Domain Labels simultaneously (Multi version)."""
    
    def __init__(
        self,
        data_path: str = '../track2',
        dataset_name: str = 'audiomos2025-track2-train_list_filter',
        ssl_model_name: str = 'w2v2_xlsr_2b',
        layer: int = 11,
        debug: bool = False,
        random_seed: int = 960,
        
        # Legacy args compatibility
        domain_grouping_strategy: str = 'none', 
        num_random_domains: int = 2,
    ):
        self._data_path = data_path
        self.dataset_name = dataset_name
        self._ssl_model_name = ssl_model_name
        self._layer = layer
        self._debug = debug
        self._random_seed = random_seed

        self._label_names = [
            'Production_Quality',
            'Production_Complexity',
            'Content_Enjoyment',
            'Content_Usefulness'
        ]

        # ----------- Load CSV -----------
        csv_path = os.path.join(data_path, f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self._df = pd.read_csv(csv_path)
        if debug:
            self._df = self._df[:200]

        # ----------- Load Labels + Features -----------
        self._labels = self._load_labels()
        self._features = self._load_features()
        
        # ----------- Pre-calculate ALL Domain Labels -----------
        self.all_domain_labels: Dict[str, np.ndarray] = {}
        
        # Only compute domains for training sets
        if 'train' in dataset_name:
            self._compute_all_domains()
            
    def _compute_all_domains(self):
        """計算所有預定義的 Domain Labels (K-Means, GT)"""
        print("[DatasetMulti] Pre-calculating ALL domain labels...")
        
        # 1. Prepare Features for Clustering
        pooled = [np.mean(f, axis=0) for f in self._features]
        X = np.stack(pooled)

        # 2. K-Means 2 (kmeans_2)
        kmeans2 = KMeans(n_clusters=2, random_state=self._random_seed, n_init=10)
        self.all_domain_labels['kmeans_2'] = kmeans2.fit_predict(X)

        # 3. K-Means 4 (kmeans_4)
        kmeans4 = KMeans(n_clusters=4, random_state=self._random_seed, n_init=10)
        self.all_domain_labels['kmeans_4'] = kmeans4.fit_predict(X)
        
        # 4. Ground Truth (gt)
        domain_map = {} 
        next_domain_id = 0
        total_samples = len(self._df)
        gt_labels = np.zeros(total_samples, dtype=int)

        for idx, path in enumerate(self._df['data_path']):
            try:
                normalized_path = os.path.normpath(path)
                parts = normalized_path.split('track2/')
                if len(parts) < 2:
                    source_folder = "unknown"
                else:
                    remaining_path = parts[1]
                    if remaining_path.startswith('/'): remaining_path = remaining_path[1:]
                    source_folder = remaining_path.split(os.path.sep)[0]
                
                if source_folder not in domain_map:
                    domain_map[source_folder] = next_domain_id
                    next_domain_id += 1
                
                gt_labels[idx] = domain_map[source_folder]
            except Exception:
                gt_labels[idx] = 0
        
        self.all_domain_labels['gt'] = gt_labels
        self.num_gt_domains = next_domain_id
        print(f"  -> Generated: kmeans_2, kmeans_4, gt ({next_domain_id} classes)")

    @property
    def features_shape(self) -> np.ndarray:
        return self._features[0].shape

    def get_all_domain_labels(self):
        return self.all_domain_labels

    def _load_labels(self) -> pd.DataFrame:
        return self._df[self._label_names]

    def _load_features(self):
        paths = self._df['data_path']
        if self._debug: paths = paths[:200]

        def load_single(path):
            feature_path_base = os.path.join(re.sub(
                r"track2/.*?/([^/]+)$", 
                f"track2/audio_feature_{self._ssl_model_name}_layer{self._layer}/",
                path
            ), os.path.basename(path)) 
            feature_path = re.sub(r'\.(flac|wav|mp3)$', '.npy', feature_path_base, flags=re.IGNORECASE)
            return np.load(feature_path)

        features = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for feat in tqdm.tqdm(ex.map(load_single, paths), total=len(paths), desc=f"Loading AES features ({self.dataset_name})"):
                features.append(feat)
        return features

    def __getitem__(self, idx):
        feat = self._features[idx]
        label = self._labels.iloc[idx].to_numpy(dtype=np.float32)
        
        domain_labels = {}
        if self.all_domain_labels:
            for key, labels_array in self.all_domain_labels.items():
                domain_labels[key] = labels_array[idx]
                
        return feat, label, domain_labels

    def __len__(self):
        return len(self._features)

    def collate_fn(self, batch):
        f_list, y_list, d_list = zip(*batch)
        f_tensor = torch.FloatTensor(np.array(f_list))
        y_tensor = torch.FloatTensor(np.array(y_list))
        
        domain_tensor_dict = {}
        if len(d_list) > 0 and d_list[0]: 
            keys = d_list[0].keys()
            for k in keys:
                values = [d[k] for d in d_list]
                domain_tensor_dict[k] = torch.LongTensor(np.array(values))
                
        return f_tensor, y_tensor, domain_tensor_dict

@gin.configurable
def get_dataloader(dataset, batch_size, num_workers, shuffle):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=shuffle, collate_fn=dataset.collate_fn
    )