#!/bin/bash
# 執行 Method 2: Supervised Contrastive Learning

export CUDA_VISIBLE_DEVICES=3
BASE_ROOT="results_supcon"
GIN_PATH="configs/domainGRL.gin"

# 固定使用 Single Domain 設置 (K-Means, K=16)


# 參數掃描
DOMAIN_NUMS=(2 4 6 8 16)
SUPCON_WEIGHTS=(0.01 0.1 0.5)
TEMPERATURES=(0.07 0.1 0.2)
SEEDS=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009)

echo "Starting Method 2 Experiments..."

for DOMAIN_NUM in "${DOMAIN_NUMS[@]}"; do
    for WEIGHT in "${SUPCON_WEIGHTS[@]}"; do
        for TEMP in "${TEMPERATURES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                EXP_NAME="K${DOMAIN_NUM}_SupConW${WEIGHT}_Temp${TEMP}"
                SAVE_PATH="${BASE_ROOT}/${EXP_NAME}/seed${SEED}"
                
                echo "Run: $EXP_NAME"
                
                python train_supcon.py \
                    --gin_path "$GIN_PATH" \
                    --save_path "$SAVE_PATH" \
                    --seed "$SEED" \
                    --supcon_weight "$WEIGHT" \
                    --supcon_temp "$TEMP" \
                    --domain_nums "$DOMAIN_NUM" \
                    --use_wandb
            done
        done
    done
done