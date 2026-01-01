#!/bin/bash
# 腳本名稱: run_proto_only.sh
# 目的: 執行 AES Prototype-Only (No GRL) 實驗

export LANG=en_US.UTF-8
export CUDA_VISIBLE_DEVICES=0

# =========================
# ====== User config ======
# =========================
N=1
BASE_OUTPUT_ROOT="results_aes_proto_only"
USE_WANDB=true

# Loss Type
LOSS_TYPE="proto_only_proba"
GIN_PATH="configs/proto_only.gin"

# Correlation Regularization
LAMBDA_VALUES=(0.0)

# Prototype Configs
NUM_PROTOTYPES=(10 5 20)
PROTO_DIV_LAMBDAS=(0.0 0.1 0.5) 

# =========================
# ====== Execution ======
# =========================
RESULT_DIR="${BASE_OUTPUT_ROOT}/details"
mkdir -p "$RESULT_DIR"

if [ "$USE_WANDB" = true ]; then WANDB_FLAG="--use_wandb"; else WANDB_FLAG=""; fi

echo "Running AES Proto-Only Experiments..."

run_experiment() {
    L=$1; SEED=$2; CORR=$3; PROTOS=$4; PD=$5; GRP=$6
    
    SAVE_PATH="${BASE_OUTPUT_ROOT}/${LOSS_TYPE}/${GRP}/seed${SEED}"
    echo "Running: ${GRP} (Seed ${SEED})"

    python train_proto_only.py \
        --gin_path "$GIN_PATH" \
        --save_path "$SAVE_PATH" \
        --seed "$SEED" \
        --covarreg_lambda "$L" \
        $CORR \
        --num_prototypes "$PROTOS" \
        --proto_div_lambda "$PD" \
        $WANDB_FLAG

    # Read Result
    CHECKPOINT_FILE="${SAVE_PATH}/checkpoint.json"
    RESULT_SRCC=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_FILE')).get('best_srcc', 'N/A'))" 2>/dev/null || echo "N/A")
    
    echo "Group=${GRP}, PD=${PD}, Seed=${SEED}, Best_SRCC=${RESULT_SRCC}" >> "${RESULT_DIR}/${LOSS_TYPE}.txt"
}

for PROTOS in "${NUM_PROTOTYPES[@]}"; do
    for PD in "${PROTO_DIV_LAMBDAS[@]}"; do
        for LAMBDA in "${LAMBDA_VALUES[@]}"; do
            
            CORR_USE=""
            if (( $(echo "$LAMBDA > 0.0" | bc -l) )); then CORR_USE="--covarreg_use"; fi
            
            GRP="NP${PROTOS}_PD${PD}_L${LAMBDA}"
            
            for ((i=1; i<=N; i++)); do
                SEED=$((999 + i))
                run_experiment "$LAMBDA" "$SEED" "$CORR_USE" "$PROTOS" "$PD" "$GRP"
            done
        done
    done
done

echo "Done. Results in ${RESULT_DIR}"