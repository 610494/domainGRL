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
        domain_grouping_strategy: str = 'none',   # 'none', 'random', 'k-means', 'GT'
        num_random_domains: int = 2,
        random_seed: int = 960,
        kmeans_metric: str = 'L2',                # 'L2' or 'cossim'
    ):
        self._data_path = data_path
        self.dataset_name = dataset_name
        self._ssl_model_name = ssl_model_name
        self._layer = layer
        self._debug = debug
        self._domain_grouping_strategy = domain_grouping_strategy

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
            indices = rng.permutation(total_samples)
            split_indices = np.array_split(indices, num_random_domains)
            domain_labels = np.zeros(total_samples, dtype=int)
            for i in range(num_random_domains):
                domain_labels[split_indices[i]] = i
            self._domain_labels = domain_labels
            self._num_domains_detected = num_random_domains

        elif domain_grouping_strategy == 'k-means':
            print(f"[Dataset] Using k-means domain grouping (k={num_random_domains}, metric={kmeans_metric}).")
            pooled = [np.mean(f, axis=0) for f in self._features]
            X = np.stack(pooled)

            if kmeans_metric == 'cossim':
                X = normalize(X, norm='l2', axis=1)
            
            kmeans = KMeans(n_clusters=num_random_domains, random_state=random_seed, n_init=10)
            self._domain_labels = kmeans.fit_predict(X)
            self._num_domains_detected = num_random_domains
        
        elif domain_grouping_strategy == 'GT':
            # Ground Truth strategy based on source folder name
            print("[Dataset] Using 'GT' (Source Folder) domain grouping strategy.")
            domain_map = {} 
            next_domain_id = 0
            domain_labels = np.zeros(total_samples, dtype=int)

            for idx, path in enumerate(self._df['data_path']):
                try:
                    normalized_path = os.path.normpath(path)
                    # 假設路徑結構包含 'track2/'
                    parts = normalized_path.split('track2/')
                    if len(parts) < 2:
                        source_folder = "unknown"
                    else:
                        # 取 track2 後的第一個目錄名
                        remaining_path = parts[1]
                        if remaining_path.startswith('/'): remaining_path = remaining_path[1:]
                        source_folder = remaining_path.split(os.path.sep)[0]
                    
                    if source_folder not in domain_map:
                        domain_map[source_folder] = next_domain_id
                        next_domain_id += 1
                        print(f"  -> Found Domain {domain_map[source_folder]}: {source_folder}")

                    domain_labels[idx] = domain_map[source_folder]
                except Exception:
                    domain_labels[idx] = 0
            
            self._domain_labels = domain_labels
            self._num_domains_detected = next_domain_id
            print(f"[Dataset] Total GT Domains detected: {self._num_domains_detected}")
        elif domain_grouping_strategy == 'TYPE':
            # Broad Category strategy: Audio vs Music vs Speech
            print("[Dataset] Using 'Type' (Audio/Music/Speech) domain grouping strategy.")

            # 定義分組規則：將資料夾名稱映射到 Type ID
            # 0: Audio (Environmental)
            # 1: Music
            # 2: Speech
            TYPE_MAPPING = {
                "unbalanced_train_segments": 0,
                
                "musiccaps": 1,
                "MUSDB18": 1,
                
                "cv-corpus-13.0-2023-03-09": 2,
                "LibriTTS": 2,
                "EARS": 2
            }
            
            # 用於 Debug 顯示的名稱
            TYPE_NAMES = {0: "Audio", 1: "Music", 2: "Speech"}

            domain_labels = np.zeros(total_samples, dtype=int)
            # 用集合來記錄實際有偵測到哪幾類
            detected_types = set()

            for idx, path in enumerate(self._df['data_path']):
                try:
                    # --- 路徑解析邏輯 (與 GT 相同) ---
                    normalized_path = os.path.normpath(path)
                    parts = normalized_path.split('track2/')
                    if len(parts) < 2:
                        source_folder = "unknown"
                    else:
                        remaining_path = parts[1]
                        if remaining_path.startswith('/'): remaining_path = remaining_path[1:]
                        source_folder = remaining_path.split(os.path.sep)[0]
                    # ----------------------------------

                    # 查表歸類
                    if source_folder in TYPE_MAPPING:
                        type_id = TYPE_MAPPING[source_folder]
                        domain_labels[idx] = type_id
                        detected_types.add(type_id)
                    else:
                        # 若出現預期外的資料夾，可選擇報錯或歸類到某一類 (這裡預設歸類為 Audio/0)
                        # print(f"Warning: Unknown source folder found: {source_folder}")
                        domain_labels[idx] = 0

                except Exception:
                    domain_labels[idx] = 0

            self._domain_labels = domain_labels
            # 固定為 3 類，或者是根據實際偵測到的數量
            self._num_domains_detected = 3 
            
            print(f"[Dataset] Total Type Domains detected: {self._num_domains_detected}")
            for tid in sorted(list(detected_types)):
                print(f"  -> Type {tid}: {TYPE_NAMES.get(tid, 'Unknown')}")
        # Fallback for validation/test sets to avoid errors if they don't use 'GT' but model expects domains
        # (Usually validation sets are mapped to 0 or handled separately)
        else: 
             # Allow sim_vs_live to map to random or just ignore if legacy arg passed
             if domain_grouping_strategy == 'sim_vs_live':
                 print("[Dataset] Warning: 'sim_vs_live' is not native to AES. Falling back to 2 random domains.")
                 self._domain_labels = np.random.randint(0, 2, total_samples)
                 self._num_domains_detected = 2
             else:
                 raise ValueError(f"Unknown strategy: {domain_grouping_strategy}")

    @property
    def features_shape(self) -> np.ndarray:
        return self._features[0].shape
    
    @property
    def num_domains(self) -> int:
        """Expose the number of domains for the model configuration."""
        return self._num_domains_detected

    def _load_labels(self) -> pd.DataFrame:
        return self._df[self._label_names]

    def _load_features(self):
        paths = self._df['data_path']
        if self._debug: paths = paths[:200]

        def load_single(path):
            # feature_path_base = path.replace('audio/', f'audio_feature_{self._ssl_model_name}_layer{self._layer}/')
            # /share/nas169/wago/AudioMOS/data/track2/audio_feature_w2v2_xlsr_2b_layer11/
            feature_path_base = os.path.join(re.sub(
                r"track2/.*?/([^/]+)$", 
                f"track2/audio_feature_{self._ssl_model_name}_layer{self._layer}/",
                path
            ), os.path.basename(path)) 
            # print(f'path: {path}')
            # print(f'feature_path_base: {feature_path_base}')
            # input("")
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