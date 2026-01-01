#!/bin/bash
# 執行 Method 1: Orthogonal Disentanglement

export CUDA_VISIBLE_DEVICES=3
BASE_ROOT="results_ortho"
GIN_PATH="configs/domainGRL.gin"

# 參數掃描
DOMAIN_NUMS=(2 4 6 8 16)
ORTH_WEIGHTS=(0.01 0.1 1.0)
RECON_WEIGHTS=(0.1 0.5 1.0)
SEEDS=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009)

echo "Starting Method 1 Experiments..."

for DOMAIN_NUM in "${DOMAIN_NUMS[@]}"; do
    for ORTH in "${ORTH_WEIGHTS[@]}"; do
        for RECON in "${RECON_WEIGHTS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                
                EXP_NAME="K${DOMAIN_NUM}_Ortho${ORTH}_Recon${RECON}"
                SAVE_PATH="${BASE_ROOT}/${EXP_NAME}/seed${SEED}"
                
                echo "Run: $EXP_NAME"
                
                python train_ortho.py \
                    --gin_path "$GIN_PATH" \
                    --save_path "$SAVE_PATH" \
                    --seed "$SEED" \
                    --orth_weight "$ORTH" \
                    --recon_weight "$RECON" \
                    --domain_nums "$DOMAIN_NUM" \
                    --use_wandb
            done
        done
    done
done