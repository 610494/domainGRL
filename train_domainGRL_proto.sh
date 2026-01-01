#!/bin/bash
# 腳本名稱: run_aes_domain_proto.sh
# 目的: 執行 AES Combined (GRL + Proto) 實驗

export LANG=en_US.UTF-8
export CUDA_VISIBLE_DEVICES=2

# =========================
# ====== User config ======
# =========================
N=4
BASE_OUTPUT_ROOT="results_aes_full_combined"
USE_WANDB=true

# Loss Type
LOSS_TYPE="combined_proba"
GIN_PATH="configs/domainGRL_proto.gin"
LAMBDA_VALUES=(0.0) # Correlation

# --- Grid Parameters ---
DOMAIN_WEIGHTS=(0.1)
DOMAIN_STRATEGIES=("random" "k-means" "GT") # K-Means optional
DOMAIN_NUMS=(6) # Used for random/kmeans
NUM_PROTOTYPES=(5)
PROTO_DIV_LAMBDAS=(0.2 0.3 0.4) 

# =========================
RESULT_DIR="${BASE_OUTPUT_ROOT}/details"
mkdir -p "$RESULT_DIR"

if [ "$USE_WANDB" = true ]; then WANDB_FLAG="--use_wandb"; else WANDB_FLAG=""; fi

echo "Running AES Combined (GRL + Proto) Experiments..."

run_experiment() {
    L=$1; SEED=$2; CORR=$3; DW=$4; DS=$5; DN=$6; PROTOS=$7; PD=$8; GRP=$9
    
    SAVE_PATH="${BASE_OUTPUT_ROOT}/${LOSS_TYPE}/${GRP}/seed${SEED}"
    echo "Running: ${GRP} (Seed ${SEED})"

    python train_domainGRL_proto.py \
        --gin_path "$GIN_PATH" \
        --save_path "$SAVE_PATH" \
        --seed "$SEED" \
        --covarreg_lambda "$L" \
        $CORR \
        --domain_weight "$DW" \
        --domain_nums "$DN" \
        --domain_grouping_strategy "$DS" \
        --num_prototypes "$PROTOS" \
        --proto_div_lambda "$PD" \
        $WANDB_FLAG

    # Check Result
    CHECKPOINT_FILE="${SAVE_PATH}/checkpoint.json"
    RESULT_SRCC=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_FILE')).get('best_srcc', 'N/A'))" 2>/dev/null || echo "N/A")
    echo "Group=${GRP}, PD=${PD}, DW=${DW}, Seed=${SEED}, Best_SRCC=${RESULT_SRCC}" >> "${RESULT_DIR}/${LOSS_TYPE}.txt"
}

for DW in "${DOMAIN_WEIGHTS[@]}"; do
    for DS in "${DOMAIN_STRATEGIES[@]}"; do
        
        if [ "$DS" == "GT" ]; then NUM_LIST=(0); else NUM_LIST=("${DOMAIN_NUMS[@]}"); fi

        for DN in "${NUM_LIST[@]}"; do
            for PROTOS in "${NUM_PROTOTYPES[@]}"; do
                for PD in "${PROTO_DIV_LAMBDAS[@]}"; do
                    for LAMBDA in "${LAMBDA_VALUES[@]}"; do
                        
                        CORR_USE=""
                        if (( $(echo "$LAMBDA > 0.0" | bc -l) )); then CORR_USE="--covarreg_use"; fi
                        
                        DN_LABEL=$DN
                        if [ "$DS" == "GT" ]; then DN_LABEL="Auto"; fi
                        
                        GRP="DW${DW}_DS${DS}_DN${DN_LABEL}_PD${PD}"
                        
                        for ((i=1; i<=N; i++)); do
                            SEED=$((999 + i))
                            run_experiment "$LAMBDA" "$SEED" "$CORR_USE" "$DW" "$DS" "$DN" "$PROTOS" "$PD" "$GRP"
                        done
                    done
                done
            done
        done
    done
done

echo "Done. Results in ${RESULT_DIR}"