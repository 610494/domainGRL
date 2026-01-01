#!/bin/bash
# 腳本名稱: run_aes_domain_grl.sh
# 目的: 針對 AES 資料集執行 Domain GRL 實驗 (GT, Random, K-Means)

export LANG=en_US.UTF-8
export CUDA_VISIBLE_DEVICES=0

# =========================
# ====== User config ======
# =========================
N=6
BASE_OUTPUT_ROOT="results_aes_domain_grl_covar_regu"
USE_WANDB=true

# Correlation Lambda (僅測試 0.0 和 0.1)
LAMBDA_VALUES=(0.1 0.3 0.5 1.0)

declare -A CONFIGS
CONFIGS["domainGRL_proba"]="configs/domainGRL_covar_regu.gin" # 確保 gin 檔名正確

# Domain GRL 參數
DOMAIN_WEIGHTS=(0.05 0.01 0.1 0.5) 

# [修改] 替換 sim_vs_live 為 GT (Ground Truth)
# GT: 依據資料夾名稱分組 (AES 適用)
# random / k-means: 演算法分組
DOMAIN_STRATEGIES=("random" "k-means" "GT")
DOMAIN_NUMS=(2 4 6 8 16) 

# =========================
# ====== Execution ======
# =========================
RESULT_DIR="${BASE_OUTPUT_ROOT}/details"
mkdir -p "$RESULT_DIR"

if [ "$USE_WANDB" = true ]; then WANDB_FLAG="--use_wandb"; else WANDB_FLAG=""; fi

echo "Running AES Domain GRL Experiments..."

run_experiment() {
    LOSS_TYPE=$1; GIN_PATH=$2; LAMBDA=$3; SEED=$4; CORR_USE=$5
    DW=$6; DN=$7; DS=$8; EXP_GROUP=$9
    
    SAVE_PATH="${BASE_OUTPUT_ROOT}/${LOSS_TYPE}/${EXP_GROUP}/seed${SEED}"
    echo "Running: ${EXP_GROUP} (Seed ${SEED})"

    # 執行 Python 腳本
    # 注意：如果 DS='GT'，Python 腳本會自動覆蓋 DN (domain_nums)
    python train_domainGRL_covar_regu_L2.py \
        --gin_path "$GIN_PATH" \
        --save_path "$SAVE_PATH" \
        --seed "$SEED" \
        --covarreg_lambda "$LAMBDA" \
        $CORR_USE \
        --domain_weight "$DW" \
        --domain_nums "$DN" \
        --domain_grouping_strategy "$DS" \
        $WANDB_FLAG

    # 讀取結果
    CHECKPOINT_FILE="${SAVE_PATH}/checkpoint.json"
    RESULT_SRCC=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_FILE')).get('best_srcc', 'N/A'))" 2>/dev/null || echo "N/A")
    
    echo "Group=${EXP_GROUP}, Best_SRCC=${RESULT_SRCC}" >> "${RESULT_DIR}/${LOSS_TYPE}.txt"
}

for LOSS_TYPE in "${!CONFIGS[@]}"; do
    GIN_PATH=${CONFIGS[$LOSS_TYPE]}
    
    for DW in "${DOMAIN_WEIGHTS[@]}"; do
        for DS in "${DOMAIN_STRATEGIES[@]}"; do
            
            # [修改] 處理 GT 策略
            if [ "$DS" == "GT" ]; then
                # GT 策略不需要指定 DN，這裡傳入 0 作為佔位符，Python 端會自動偵測
                # 我們只跑一次迴圈 (num_list=[0])
                NUM_LIST=(0)
            else
                # random 和 k-means 測試指定的 NUMs
                NUM_LIST=("${DOMAIN_NUMS[@]}")
            fi

            for DN in "${NUM_LIST[@]}"; do
                for LAMBDA in "${LAMBDA_VALUES[@]}"; do
                    GRL_CORR_USE=""
                    if (( $(echo "$LAMBDA > 0.0" | bc -l) )); then GRL_CORR_USE="--covarreg_use"; fi
                    
                    # 命名處理
                    DN_LABEL=$DN
                    if [ "$DS" == "GT" ]; then DN_LABEL="Auto"; fi
                    
                    GRL_GROUP="DW${DW}_DS${DS}_DN${DN_LABEL}_L${LAMBDA}"
                    
                    for ((i=1; i<=N; i++)); do
                        SEED=$((1003 + i))
                        run_experiment "$LOSS_TYPE" "$GIN_PATH" "$LAMBDA" "$SEED" "$GRL_CORR_USE" "$DW" "$DN" "$DS" "$GRL_GROUP"
                    done
                done
            done
        done
    done
done