#!/bin/bash

# ================= Configuration =================

# 1. 原始實驗的根目錄 (Search Root)
# 程式會去這些目錄下搜尋
SEARCH_DIRS=(
    # "/share/nas169/wago/MultiGauss/AES/results_aes_domain_grl/domainGRL_proba/DW0.1_DSk-means_DN4_L0.0"
    # "/share/nas169/wago/MultiGauss/AES/results_aes_domain_grl/domainGRL_proba/DW0.1_DSk-means_DN9_L0.0"
    # "/share/nas169/wago/MultiGauss/AES/results_aes_domain_grl/domainGRL_proba/DW0.3_DSk-means_DN9_L0.0"
    # "/share/nas169/wago/MultiGauss/AES/results_aes_domain_grl/domainGRL_proba/DW0.5_DSk-means_DN4_L0.0"
    # "/share/nas169/wago/MultiGauss/AES/results_aes_domain_grl/domainGRL_proba/DW0.5_DSk-means_DN9_L0.0"
    # "/share/nas169/wago/MultiGauss/AES/results_aes_domain_grl/domainGRL_proba"
    # "/share/nas169/wago/MultiGauss/AES/results_aes_domain_grl/domainGRL_proba/DW0.1_DSk-means_DN10_L0.0"
    # "/share/nas169/wago/MultiGauss/AES/results_aes_domain_grl/domainGRL_proba/DW0.1_DSrandom_DN10_L0.0"
    "/share/nas169/wago/MultiGauss/AES/results_aes_baselines_DR0.5/baseline_proba/lambda0.0"
    "/share/nas169/wago/MultiGauss/AES/results_aes_baselines_WD1e-3/baseline_proba/lambda0.0/seed1006"
    "/share/nas169/wago/MultiGauss/AES/results_aes_baselines_WD1e-4/baseline_proba/lambda0.0"
)

# 2. 預測結果要存去哪裡 (New Output Root)
# 結構會變成: PRED_ROOT_DIR/experiment_name/seedXXXX/predictions.npz
PRED_ROOT_DIR="evaluation_results_batch_w_pred/results_aes_domain_grl"

# 3. 指定要跑的 Seed
TARGET_SEED="1006"
# 1001 1005
# 4. 資料集設定 (Inference 要跑哪一份 CSV)
DATA_PATH="../track2"
DATASET_NAME="audiomos2025-track2-eval_list" # 不需要加 .csv

# 5. Checkpoint 檔名 (通常是 best_model.pt 或 checkpoint.pt)
CKPT_NAME="model_best_state_dict.pt" # 或 "best_model.pt"
CONFIG_NAME="config.gin" # 或 "config.gin"

# ================= Execution =================

echo "Starting Batch Inference for Seed: ${TARGET_SEED}"

for SEARCH_ROOT in "${SEARCH_DIRS[@]}"; do
    
    # 檢查目錄是否存在
    if [ ! -d "$SEARCH_ROOT" ]; then
        echo "Warning: Directory not found: $SEARCH_ROOT"
        continue
    fi

    # 在該目錄下搜尋包含指定 seed 的資料夾
    # 假設結構是: SEARCH_ROOT/experiment_name/seed1006/
    # find command 會找出所有符合 seed1006 的路徑
    find "$SEARCH_ROOT" -type d -name "seed${TARGET_SEED}" | while read EXP_SEED_DIR; do
        
        # 1. 準備路徑
        CKPT_PATH="${EXP_SEED_DIR}/${CKPT_NAME}"
        CFG_PATH="${EXP_SEED_DIR}/${CONFIG_NAME}"
        
        # 2. 檢查檔案是否存在
        if [ ! -f "$CKPT_PATH" ] || [ ! -f "$CFG_PATH" ]; then
            echo "[Skip] Missing checkpoint or config in: $EXP_SEED_DIR"
            continue
        fi

        # 3. 建構輸出路徑
        # 我們要把 SEARCH_ROOT 的上層結構保留下來可能有困難，
        # 這裡簡單做法是提取 experiment name (上一層目錄名)
        EXP_NAME=$(basename $(dirname "$EXP_SEED_DIR"))
        # 或者保留完整相對路徑
        # 這裡示範: PRED_ROOT_DIR / experiment_name / seedXXXX
        SAVE_DIR="${PRED_ROOT_DIR}/${EXP_NAME}/seed${TARGET_SEED}"
        NUM_DOMAINS=$(echo "$EXP_NAME" | grep -Po '(?<=DN)[0-9]+')

        echo "---------------------------------------------------"
        echo "Processing: $EXP_NAME (Seed ${TARGET_SEED})"
        echo "  Config: $CFG_PATH"
        echo "  Ckpt:   $CKPT_PATH"
        echo "  Output: $SAVE_DIR"
        
        # 4. 執行 Python Inference
        python inference_aes.py \
            --config_path "$CFG_PATH" \
            --checkpoint_path "$CKPT_PATH" \
            --output_dir "$SAVE_DIR" \
            --dataset_name "$DATASET_NAME" \
            --data_path "$DATA_PATH" \
            --num_domains "$NUM_DOMAINS"

        if [ $? -eq 0 ]; then
            echo "✅ Done."
        else
            echo "❌ Failed."
        fi

    done
done

echo "Batch Inference Completed."