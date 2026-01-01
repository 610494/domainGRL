#!/bin/bash

# ================= Configuration =================
BASELINE="evaluation_results_batch_w_pred/results_aes_baselines/baseline_proba/lambda0.0"
TOP_K=-1
N_JOBS=8
SEEDS=(1006)
# SEEDS=(1000 1001 1002 1003 1004 1005 1006 1007 1008 1009)

SEARCH_DIRS=(
    # "evaluation_results_batch_w_pred/results_aes_baselines_regu/baseline_proba"
    # "evaluation_results_batch_w_pred/results_aes_domain_grl/domainGRL_proba"
    # "evaluation_results_batch_w_pred/results_ortho"
    # "evaluation_results_batch_w_pred/results_supcon"
    # "evaluation_results_batch_w_pred/results_aes_domain_grl_multi_new"
    # "evaluation_results_batch_w_pred/results_aes_domain_grl/DW0.5_DSTYPE_DN3_L0.0"
    # "evaluation_results_batch_w_pred/results_aes_domain_grl/DW0.3_DSTYPE_DN3_L0.0"
    # "evaluation_results_batch_w_pred/results_aes_domain_grl/DW0.1_DSTYPE_DN3_L0.0"
    "evaluation_results_batch_w_pred/baselines_WD1e-4"
    "evaluation_results_batch_w_pred/baselines_WD1e-3"
    "evaluation_results_batch_w_pred/baseline_DR0.5"
)

# 設定篩選模式:
# "none" -> 只看分數
# "mse"  -> MSE 顯著進步
# "srcc" -> SRCC 顯著進步 (新增)
# "all"  -> 全部指標顯著進步
FILTER_MODE="none"

# ================= Execution =================

echo "Starting Analysis with Filter: ${FILTER_MODE}"

for SEED in "${SEEDS[@]}"; do
    
    OUTPUT_FILE="evaluation_results_batch_w_pred/single_${FILTER_MODE}/table_top${TOP_K}_seed_${SEED}_${FILTER_MODE}.tex"
    
    python test.py \
        --seed $SEED \
        --baseline_path "$BASELINE" \
        --search_dirs "${SEARCH_DIRS[@]}" \
        --output_path "$OUTPUT_FILE" \
        --top_k $TOP_K \
        --filter_mode "$FILTER_MODE" \
        --n_jobs $N_JOBS

done