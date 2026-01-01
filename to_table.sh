#!/bin/bash

# ================= Configuration =================

# 定義要跑的輸入目錄陣列
INPUT_DIRS=(
    # "results_aes_baselines/baseline_proba"
    "results_aes_domain_grl/domainGRL_proba"
    # "results_aes_domain_grl_covar_regu/domainGRL_proba"
    "results_aes_domain_grl_multi_new/domainGRL_proba"
    # "results_aes_baselines_regu/baseline_proba"
    # "results_aes_domain_grl_covar_regu/domainGRL_proba"
    # "results_aes_domain_grl_covar_regu_L1/domainGRL_proba"
    # "results_aes_domain_grl_multi/domainGRL_proba"
    # "results_ortho"
    # "results_supcon"
)

# 定義種子組合陣列
SEED_SETS=(
    "seed1006"
    # "seed1000 seed1001 seed1002 seed1003 seed1004"
    # "seed1001 seed1002 seed1003 seed1004 seed1005"
    # "seed1002 seed1003 seed1004 seed1005 seed1006"
    # "seed1003 seed1004 seed1005 seed1006 seed1007"
    # "seed1004 seed1005 seed1006 seed1007 seed1008"
    # "seed1005 seed1006 seed1007 seed1008 seed1009"
)

# 共用設定
SHOW_STD=true
COLSEP="2pt"
BASE_OUTPUT_ROOT="table_all" # 輸出最上層目錄

# =================================================

# 檢查 numpy (可選)
# pip install numpy > /dev/null 2>&1

# 雙層迴圈：遍歷所有 Input 目錄 和 所有 Seed 組合
for INPUT_DIR in "${INPUT_DIRS[@]}"; do
    
    # 檢查輸入目錄是否存在
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Warning: Input directory '$INPUT_DIR' does not exist. Skipping."
        continue
    fi

    # 自動解析 Dataset Name
    # 邏輯：取輸入路徑的父目錄名稱
    # 例如 input="results_aes_domain_grl_covar_regu/domainGRL_proba"
    # dirname -> results_aes_domain_grl_covar_regu
    # basename -> results_aes_domain_grl_covar_regu
    PARENT_DIR=$(dirname "$INPUT_DIR")
    DATASET_NAME=$(basename "$PARENT_DIR")

    for SEEDS in "${SEED_SETS[@]}"; do
        
        # 1. 處理 Output 資料夾命名
        # 將 seeds 變成字串後綴，例如 "seed1001 seed1002" -> "seed1001_seed1002"
        SEED_SUFFIX=$(echo "$SEEDS" | tr ' ' '_')
        OUTPUT_DIR="${BASE_OUTPUT_ROOT}/${DATASET_NAME}/${SEED_SUFFIX}"

        if [ ! -d "$OUTPUT_DIR" ]; then
            mkdir -p "$OUTPUT_DIR"
        fi

        # 2. 處理 Caption 顯示內容
        # 格式範例: (results_aes_domain_grl, Seeds: seed1001 seed1002)
        CAPTION_INFO="${DATASET_NAME}, Seeds: ${SEEDS}"

        # 3. 建立並執行指令
        CMD="python to_table.py --root \"$INPUT_DIR\" --outdir \"$OUTPUT_DIR\" --colsep \"$COLSEP\" --seeds $SEEDS --caption_info \"$CAPTION_INFO\""

        if [ "$SHOW_STD" = false ]; then
            CMD="$CMD --no_std"
        fi

        echo "---------------------------------------------------"
        echo "Processing: $DATASET_NAME"
        echo "Seeds: $SEEDS"
        echo "Output: $OUTPUT_DIR"
        
        eval $CMD
    done
done

echo "---------------------------------------------------"
echo "All tasks done."