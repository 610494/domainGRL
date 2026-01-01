import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import gin
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.utils.data.dataset
import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


@gin.configurable
class AESFeatures(torch.utils.data.dataset.Dataset):
    """AES dataset with multi-aspect labels and Domain Grouping support."""
    
    def __init__(
        self,
        data_path: str = '../track2',
        dataset_name: str = 'audiomos2025-track2-train_list_filter',
        ssl_model_name: str = 'w2v2_xlsr_2b',
        layer: int = 11,
        debug: bool = False,

        # ---- Domain Grouping ----
        domain_grouping_strategy: str = 'none',   # 'none', 'random', 'k-means', 'GT', 'hybrid'
        num_random_domains: int = 2,              # 兼作 K-means 的 k 值
        random_seed: int = 960,
        kmeans_metric: str = 'L2',                # 'L2' or 'cossim'
    ):
        self._data_path = data_path
        self.dataset_name = dataset_name
        self._ssl_model_name = ssl_model_name
        self._layer = layer
        self._debug = debug
        self._domain_grouping_strategy = domain_grouping_strategy
        self._random_seed = random_seed
        self._num_random_domains = num_random_domains
        self._kmeans_metric = kmeans_metric

        self._label_names = [
            'Production_Quality',
            'Production_Complexity',
            'Content_Enjoyment',
            'Content_Usefulness'
        ]

        # Need domain label or not
        self._return_domain = (domain_grouping_strategy != 'none')

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

        total_samples = len(self._df)
        self._num_domains_detected = num_random_domains # Default fallback

        # ----------- Domain Assignment -----------
        if domain_grouping_strategy == 'none':
            self._domain_labels = np.zeros(total_samples, dtype=int)

        elif domain_grouping_strategy == 'random':
            print(f"[Dataset] Using random {num_random_domains} domains.")
            rng = np.random.default_rng(random_seed)
            self._domain_labels = rng.integers(0, num_random_domains, total_samples)
            self._num_domains_detected = num_random_domains

        elif domain_grouping_strategy == 'k-means':
            self._domain_labels, self._num_domains_detected = self._get_kmeans_domains(
                n_clusters=num_random_domains, metric=kmeans_metric
            )
        
        elif domain_grouping_strategy == 'GT':
            self._domain_labels, self._num_domains_detected = self._get_gt_domains()

        elif domain_grouping_strategy == 'hybrid':
            print("[Dataset] Using 'hybrid' strategy.")
            # 1. K-Means labels (Quality & Usefulness) -> 設 k=8 (或使用 num_random_domains)
            # 這裡我們使用傳入的 num_random_domains 當作 k
            k_val = num_random_domains 
            kmeans_labels, n_kmeans = self._get_kmeans_domains(n_clusters=k_val, metric=kmeans_metric)
            
            # 2. GT labels (Complexity & Enjoyment)
            gt_labels, n_gt = self._get_gt_domains()

            # 3. Combine into (N, 4) matrix
            # idx 0: Quality -> Kmeans
            # idx 1: Complexity -> GT
            # idx 2: Enjoyment -> GT
            # idx 3: Usefulness -> Kmeans
            hybrid_labels = np.zeros((total_samples, 4), dtype=int)
            hybrid_labels[:, 0] = kmeans_labels
            hybrid_labels[:, 1] = gt_labels
            hybrid_labels[:, 2] = gt_labels
            hybrid_labels[:, 3] = kmeans_labels
            
            self._domain_labels = hybrid_labels
            
            # 回傳 dict 讓 Model 知道要建立幾個 Head
            self._num_domains_detected = {'kmeans': int(n_kmeans), 'gt': int(n_gt)}
            print(f"[Dataset] Hybrid domains ready. Config: {self._num_domains_detected}")

        else:
             if domain_grouping_strategy == 'sim_vs_live':
                 print("[Dataset] Warning: 'sim_vs_live' fallback to 2 random domains.")
                 self._domain_labels = np.random.randint(0, 2, total_samples)
                 self._num_domains_detected = 2
             else:
                 raise ValueError(f"Unknown strategy: {domain_grouping_strategy}")

    # ================= Helper Methods =================
    def _get_kmeans_domains(self, n_clusters, metric):
        print(f"[Dataset] Running K-means (k={n_clusters}, metric={metric})...")
        pooled = [np.mean(f, axis=0) for f in self._features]
        X = np.stack(pooled)
        if metric == 'cossim':
            X = normalize(X, norm='l2', axis=1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self._random_seed, n_init=10)
        labels = kmeans.fit_predict(X)
        return labels, n_clusters

    def _get_gt_domains(self):
        print("[Dataset] Extracting GT (Source Folder) domains...")
        domain_map = {} 
        next_domain_id = 0
        total_samples = len(self._df)
        labels = np.zeros(total_samples, dtype=int)

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

                labels[idx] = domain_map[source_folder]
            except Exception:
                labels[idx] = 0
        
        print(f"  -> GT Domains found: {next_domain_id}")
        return labels, next_domain_id

    @property
    def features_shape(self) -> np.ndarray:
        return self._features[0].shape
    
    @property
    def num_domains(self):
        return self._num_domains_detected

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
        if self._return_domain:
            # hybrid 模式下 domain_labels[idx] 是 array (4,)，collator 會自動處理
            return feat, label, self._domain_labels[idx]
        return feat, label

    def __len__(self):
        return len(self._features)

    def collate_fn(self, batch):
        if len(batch[0]) == 3:
            f, y, d = zip(*batch)
            return torch.FloatTensor(np.array(f)), torch.FloatTensor(np.array(y)), torch.LongTensor(np.array(d))
        else:
            f, y = zip(*batch)
            return torch.FloatTensor(np.array(f)), torch.FloatTensor(np.array(y))

@gin.configurable
def get_dataloader(dataset, batch_size, num_workers, shuffle):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=shuffle, collate_fn=dataset.collate_fn
    )