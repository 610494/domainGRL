import argparse
import json
import os
import re

import gin
import numpy as np
import torch
import torch.utils.data
# ==============================================================================
#                               IMPORTS
# ==============================================================================
# 假設這兩個檔案與此 script 在同一目錄
from dataset.dataset_AES import AESFeatures
from model.model import ProjectionHead
from tqdm import tqdm

# ==============================================================================
#                               CONFIG
# ==============================================================================
ASPECTS = ['Production_Quality', 'Production_Complexity', 'Content_Enjoyment', 'Content_Usefulness']

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for AES Multi-Domain Model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config.json or config.gin")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions.npz")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 覆蓋 Dataset 設定 (通常用於指定測試集 CSV)
    parser.add_argument("--dataset_name", type=str, default="audiomos2025-track2-test_list_filter", 
                        help="Name of the CSV file (without .csv) to use for inference")
    parser.add_argument("--data_path", type=str, default="../track2", help="Root path of the dataset")
    return parser.parse_args()

# ==============================================================================
#                            HELPER FUNCTIONS
# ==============================================================================

def extract_system_id(sample_id):
    """
    從 sample_id 或檔名中解析出 System ID。
    這對於 test.py 正確聚合 (Aggregate) 資料至關重要。
    
    假設格式範例: 
    1. 檔名: "VoiceMOS2022-track2-sys05-utt02.wav" -> 取 "sys05" (index 2)
    2. 檔名: "sys05-utt02.wav" -> 取 "sys05" (index 0)
    """
    s = str(sample_id)
    parts = s.split('-')
    
    # 策略 A: 如果是用 - 分隔的標準格式，且長度足夠，通常 System ID 在中間
    # 這裡預設採用你之前提到的 split('-')[2] 邏輯，這通常對應官方資料集格式
    if len(parts) >= 3:
        return parts[2]
    
    # 策略 B: 如果很短，可能是 "sys05-utt01"，取第一個
    if len(parts) >= 2:
        return parts[0]
        
    # 策略 C: 真的無法解析，就回傳原字串 (這會導致 aggregation 失敗，變成 3060 筆)
    return s

def save_results(output_dir, predictions, targets, sys_ids):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "predictions.npz")
    
    data_dict = {}
    
    # Save per-aspect predictions and targets
    for idx, asp in enumerate(ASPECTS):
        data_dict[f'{asp}_preds'] = predictions[:, idx]
        data_dict[f'{asp}_targets'] = targets[:, idx]
        
    # Save System IDs for aggregation
    data_dict['sys_ids'] = sys_ids
    
    np.savez(save_path, **data_dict)
    print(f"Saved results to {save_path}")

# ==============================================================================
#                               MAIN
# ==============================================================================

def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1. Load Config (Gin)
    # 使用 skip_unknown=True 忽略 TrainingLoopMulti
    if not args.config_path.endswith('.json'):
        gin.parse_config_file(args.config_path, skip_unknown=True)
    else:
        # 如果是 json，假設參數已經透過其他方式傳遞或預設，
        # 這裡僅做簡單讀取確認
        with open(args.config_path, 'r') as f:
            pass

    # 強制綁定 Dataset 參數 (確保讀取測試集)
    gin.bind_parameter('AESFeatures.dataset_name', args.dataset_name)
    gin.bind_parameter('AESFeatures.data_path', args.data_path)
    # 確保 debug 模式關閉，讀取全量資料
    try:
        gin.bind_parameter('AESFeatures.debug', False)
    except:
        pass # 如果 config 沒定義 debug 可能會報錯，忽略即可

    # 2. Initialize Dataset & Dataloader
    print(f"Initializing Dataset: {args.dataset_name}")
    dataset = AESFeatures() 
    # Shuffle 必須為 False 才能對應 ID
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, num_workers=4, shuffle=False, collate_fn=dataset.collate_fn
    )

    # 3. Extract & Convert System IDs
    # 這是解決 (3060 vs 36) shape mismatch 的關鍵
    if 'sample_id' in dataset._df.columns:
        raw_ids = dataset._df['sample_id'].values
    else:
        raw_ids = dataset._df['data_path'].apply(lambda x: os.path.basename(x)).values

    # 將檔名/Sample ID 轉換為 System ID
    sys_ids = np.array([extract_system_id(rid) for rid in raw_ids])
    
    unique_systems = np.unique(sys_ids)
    print(f"Samples: {len(sys_ids)} | Unique Systems: {len(unique_systems)}")
    # 如果這裡是 36 左右，代表解析成功

    # 5. Initialize Model
    model = ProjectionHead(
        in_shape=dataset.features_shape
    ).to(device)

    # 6. Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 7. Inference Loop
    all_preds = []
    all_targets = []

    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            features, labels = batch 
            features = features.to(device)
            
            # domain_weight=0.0 during inference
            out_tuple = model(features) 
            quality_out = out_tuple[0]
            
            # 處理輸出格式: 可能是 Tensor，或是 (mean, cov) tuple
            if isinstance(quality_out, (tuple, list)):
                preds = quality_out[0] # 取 mean
            else:
                preds = quality_out

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 8. Save Results
    save_results(args.output_dir, all_preds, all_targets, sys_ids)

if __name__ == "__main__":
    main()