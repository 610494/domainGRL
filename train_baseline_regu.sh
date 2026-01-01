#!/bin/bash
# 腳本名稱: run_aes_baselines.sh
# 目的: 執行 AES Dataset 的訓練實驗 (支援 Grid Search)

# 確保使用 UTF-8 輸出
export LANG=en_US.UTF-8

# 設定 GPU (請根據你的機器修改，例如 0 或 1)
export CUDA_VISIBLE_DEVICES=3

# =========================
# ====== User config ======
# =========================
N=10  # 每個組合執行的種子 (seed) 次數 (Seed 1001, 1002, 1003)
BASE_OUTPUT_ROOT="results_aes_baselines_regu" # 修改輸出目錄名稱以區分 AES
USE_WANDB=true   # 設 true 就會啟用 wandb logging

# 核心配置: 相關性正則化 Lambda 值的列表
# AES 的各個維度 (PQ, PC, CE, CU) 相關性可能與 NISQA 不同
# 你可以測試不同的權重，例如 (0.0 0.1 0.2)
LAMBDA_VALUES=(0.1 0.3 0.5 1.0)

# 要測試的 Loss Type 和其對應的 GIN 配置路徑
declare -A CONFIGS

# [注意] 如果你還沒有製作 AES 專用的 non_proba (MSE) GIN 檔，請先註解掉下面這行
# CONFIGS["baseline_non_proba"]="configs/baseline_non_proba.gin"

# [修改] 指向你剛剛修改好的 AES 版本 baseline.gin
# 假設 baseline.gin 在當前目錄，如果在 configs 資料夾下請保留 configs/ 前綴
CONFIGS["baseline_proba"]="configs/baseline.gin"

# =========================
# ====== Output Files =====
# =========================
SUMMARY_FILE="${BASE_OUTPUT_ROOT}/final_summary.txt"
RESULT_DIR="${BASE_OUTPUT_ROOT}/details"

# 建立結果資料夾
mkdir -p "$RESULT_DIR"

# 設定 WandB 旗標
if [ "$USE_WANDB" = true ]; then
    WANDB_FLAG="--use_wandb"
else
    WANDB_FLAG=""
fi

# 總實驗計數器 (僅供進度條顯示參考)
TOTAL_CONFIGS=$(( N * ${#CONFIGS[@]} * (${#LAMBDA_VALUES[@]} + 1) ))
CURRENT_RUN=0

echo "Running AES experiments..."
echo "Output Directory: $BASE_OUTPUT_ROOT"
echo "==================================="

# ----------------------------------------------------
# 函數: 執行單次實驗並記錄結果 (使用 SRCC 作為指標)
# ----------------------------------------------------
run_experiment() {
    # 參數: $1=Loss_Type, $2=Gin_Path, $3=Lambda, $4=Seed, $5=Corr_Use, $6=Experiment_Group
    LOSS_TYPE=$1
    GIN_PATH=$2
    LAMBDA=$3
    SEED=$4
    CORR_USE=$5
    EXP_GROUP=$6 
    
    # 設置儲存路徑
    SAVE_PATH="${BASE_OUTPUT_ROOT}/${LOSS_TYPE}/${EXP_GROUP}/seed${SEED}"
    
    CURRENT_RUN=$((CURRENT_RUN + 1))
    echo "[Run $CURRENT_RUN] Type: ${LOSS_TYPE} | Group: ${EXP_GROUP} | Seed: ${SEED} | Lambda: ${LAMBDA}"

    # 執行 Python 訓練腳本
    # 注意: train_baseline.py 會讀取 baseline.gin 中的 output_scale=4.5 來適配 AES
    python train_baseline.py \
        --gin_path "$GIN_PATH" \
        --save_path "$SAVE_PATH" \
        --seed "$SEED" \
        --covarreg_lambda "$LAMBDA" \
        $CORR_USE \
        $WANDB_FLAG
    
    # 讀取該次 checkpoint.json 的結果 (Best SRCC)
    CHECKPOINT_FILE="${SAVE_PATH}/checkpoint.json"
    RESULT_SRCC=$(python3 -c "
import json, sys
try:
    with open('$CHECKPOINT_FILE') as f:
        data = json.load(f)
        # 兼容性: 檢查 best_srcc 或 best_metric
        print(data.get('best_srcc', data.get('best_metric', 'N/A')))
except Exception:
    print('N/A')
")

    # 記錄結果到詳細結果檔案
    DETAIL_FILE="${RESULT_DIR}/${LOSS_TYPE}.txt"
    echo "Group=${EXP_GROUP}, Lambda=${LAMBDA}, Seed=${SEED}, Best_SRCC=${RESULT_SRCC}" >> "$DETAIL_FILE"

    echo "   -> Done. Best SRCC: ${RESULT_SRCC}"
}

# ----------------------------------------------------
# 階段 1: 遍歷所有 LOSS_TYPE
# ----------------------------------------------------
for LOSS_TYPE in "${!CONFIGS[@]}"; do
    GIN_PATH=${CONFIGS[$LOSS_TYPE]}
    
    echo "" | tee -a $SUMMARY_FILE
    echo "==================================================" | tee -a $SUMMARY_FILE
    echo "STARTING LOSS TYPE: $LOSS_TYPE" | tee -a $SUMMARY_FILE
    echo "Config: $GIN_PATH" | tee -a $SUMMARY_FILE
    echo "==================================================" | tee -a $SUMMARY_FILE

    # --------------------------------
    # # 階段 1.1: 執行 Baseline (Lambda=0.0 / Corr=False)
    # # --------------------------------
    # BASELINE_LAMBDA=0.0
    # BASELINE_CORR_USE=""
    # BASELINE_GROUP="lambda0.0" # 標記為 lambda0.0 作為基準
    
    # echo "--- Running Baseline (L=0.0) ---"
    # for ((i=1; i<=N; i++)); do
    #     SEED=$((999 + i))
    #     run_experiment "$LOSS_TYPE" "$GIN_PATH" "$BASELINE_LAMBDA" "$SEED" "$BASELINE_CORR_USE" "$BASELINE_GROUP"
    # done
    
    # -----------------------------------------------------------------------
    # 階段 1.2: 執行 Grid Search (covarreg_use=True)
    #           只有在 LOSS_TYPE 是 'baseline_proba' 且 Lambda > 0 時才執行
    # -----------------------------------------------------------------------
    if [ "$LOSS_TYPE" == "baseline_proba" ]; then
        GRID_CORR_USE="--covarreg_use"
        
        for LAMBDA in "${LAMBDA_VALUES[@]}"; do
            # 跳過 0.0，因為上面已經跑過 Baseline 了
            if (( $(echo "$LAMBDA == 0.0" | bc -l) )); then
                continue
            fi
            
            echo "--- Running Regularization Search (L=${LAMBDA}) ---"
            GRID_GROUP="lambda_${LAMBDA}"
            for ((i=1; i<=N; i++)); do
                SEED=$((999 + i))
                run_experiment "$LOSS_TYPE" "$GIN_PATH" "$LAMBDA" "$SEED" "$GRID_CORR_USE" "$GRID_GROUP"
            done
        done
    fi

    # --------------------------------
    # 階段 1.3: 簡單總結
    # --------------------------------
    echo "" | tee -a $SUMMARY_FILE
    echo "Summary for $LOSS_TYPE:" | tee -a $SUMMARY_FILE
    
    # 這裡使用簡單的 grep/awk 計算平均值
    if [ -f "${RESULT_DIR}/${LOSS_TYPE}.txt" ]; then
        cat "${RESULT_DIR}/${LOSS_TYPE}.txt" | awk -F', ' '{print $4}' | awk -F'=' '{sum+=$2; n++} END {print "  Avg SRCC across all runs: " sum/n}' | tee -a $SUMMARY_FILE
    else
        echo "  No results found." | tee -a $SUMMARY_FILE
    fi

done

# 最終輸出
echo ""
echo "=================================================="
echo "✅ All AES experiments finished!"
echo "📄 Results saved in: ${RESULT_DIR}/"
echo "📊 Summary file: $SUMMARY_FILE"
echo "=================================================="