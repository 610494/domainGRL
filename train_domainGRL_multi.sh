#!/bin/bash
# 腳本名稱: run_aes_domain_grl_multi.sh
# 目的: 搜尋不同的 Domain Branch 組合 (Multi-Domain Version)

export LANG=en_US.UTF-8
export CUDA_VISIBLE_DEVICES=0

# =========================
# ====== User config ======
# =========================
N=6  
BASE_OUTPUT_ROOT="results_aes_domain_grl_multi"
USE_WANDB=true

LAMBDA_VALUES=(0.0)

declare -A CONFIGS
CONFIGS["domainGRL_proba"]="configs/domainGRL_multi.gin"

DOMAIN_WEIGHTS=(0.05 0.01 0.1 0.5) 

# ==================================================
# ====== Domain Combinations Search Strategy ======
# ==================================================
TARGET_COMBINATIONS=(
    "KMeans2_GT"
    "KMeans4_GT"
    "All_Mixed"
)

# =========================
# ====== Execution ======
# =========================
RESULT_DIR="${BASE_OUTPUT_ROOT}/details"
mkdir -p "$RESULT_DIR"

if [ "$USE_WANDB" = true ]; then WANDB_FLAG="--use_wandb"; else WANDB_FLAG=""; fi

echo "Running AES Multi-Domain GRL Experiments..."

run_experiment() {
    LOSS_TYPE=$1; GIN_PATH=$2; LAMBDA=$3; SEED=$4; CORR_USE=$5
    DW=$6; COMBO_NAME=$7; JSON_CONFIG=$8; EXP_GROUP=$9
    
    SAVE_PATH="${BASE_OUTPUT_ROOT}/${LOSS_TYPE}/${EXP_GROUP}/seed${SEED}"
    echo "Running: ${EXP_GROUP} (Seed ${SEED})"
    echo "  Config: $JSON_CONFIG"

    # [Update] 傳入 --exp_tag "$COMBO_NAME"
    python train_domainGRL_multi.py \
        --gin_path "$GIN_PATH" \
        --save_path "$SAVE_PATH" \
        --seed "$SEED" \
        --covarreg_lambda "$LAMBDA" \
        $CORR_USE \
        --domain_weight "$DW" \
        --domain_configs_json "$JSON_CONFIG" \
        --exp_tag "$COMBO_NAME" \
        $WANDB_FLAG

    CHECKPOINT_FILE="${SAVE_PATH}/checkpoint.json"
    RESULT_SRCC=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_FILE')).get('best_srcc', 'N/A'))" 2>/dev/null || echo "N/A")
    
    echo "Group=${EXP_GROUP}, Best_SRCC=${RESULT_SRCC}" >> "${RESULT_DIR}/${LOSS_TYPE}.txt"
}

for LOSS_TYPE in "${!CONFIGS[@]}"; do
    GIN_PATH=${CONFIGS[$LOSS_TYPE]}
    
    for DW in "${DOMAIN_WEIGHTS[@]}"; do
        for COMBO_NAME in "${TARGET_COMBINATIONS[@]}"; do
            
            case $COMBO_NAME in
                "KMeans2")
                    JSON_CFG='{"kmeans_2": {"num_classes": 2, "weight": 1.0}}'
                    ;;
                "KMeans4")
                    JSON_CFG='{"kmeans_4": {"num_classes": 4, "weight": 1.0}}'
                    ;;
                "GT")
                    JSON_CFG='{"gt": {"num_classes": 26, "weight": 1.0}}'
                    ;;
                "KMeans2_GT")
                    JSON_CFG='{"kmeans_2": {"num_classes": 2, "weight": 0.5}, "gt": {"num_classes": 26, "weight": 0.5}}'
                    ;;
                "KMeans4_GT")
                    JSON_CFG='{"kmeans_4": {"num_classes": 4, "weight": 0.5}, "gt": {"num_classes": 26, "weight": 0.5}}'
                    ;;
                "All_Mixed")
                    JSON_CFG='{"kmeans_2": {"num_classes": 2, "weight": 0.33}, "kmeans_4": {"num_classes": 4, "weight": 0.33}, "gt": {"num_classes": 26, "weight": 0.33}}'
                    ;;
                *)
                    echo "Unknown combination: $COMBO_NAME"
                    continue
                    ;;
            esac

            for LAMBDA in "${LAMBDA_VALUES[@]}"; do
                GRL_CORR_USE=""
                if (( $(echo "$LAMBDA > 0.0" | bc -l) )); then GRL_CORR_USE="--covarreg_use"; fi
                
                GRL_GROUP="Combo_${COMBO_NAME}_DW${DW}_L${LAMBDA}"
                
                for ((i=1; i<=N; i++)); do
                    SEED=$((1003 + i))
                    run_experiment "$LOSS_TYPE" "$GIN_PATH" "$LAMBDA" "$SEED" "$GRL_CORR_USE" "$DW" "$COMBO_NAME" "$JSON_CFG" "$GRL_GROUP"
                done
            done
        done
    done
done