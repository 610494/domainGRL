#!/bin/bash
# 腳本名稱: run_aes_domain_grl_power_set.sh
# 目的: 自動遍歷所有 Domain Branch 的排列組合 (Multi-Head Only)
# 更新: 
#   1. GT num_classes 改為 6
#   2. 跳過只有 1 個 Head 的情況 (只跑組合)

export LANG=en_US.UTF-8
export CUDA_VISIBLE_DEVICES=3

# =========================
# ====== User config ======
# =========================
N=1
BASE_OUTPUT_ROOT="results_aes_domain_grl_multi_new"
USE_WANDB=true

# 定義所有可用的 Head 資訊 (名稱顯示用, JSON Key, Class數)
# 順序對應: Bit 0, Bit 1, Bit 2, ...
HEAD_NAMES=("KMeans2"  "KMeans4"  "KMeans6"  "KMeans8"  "KMeans16"  "GT")
HEAD_KEYS=("kmeans_2" "kmeans_4" "kmeans_6" "kmeans_8" "kmeans_16" "gt")
# [修改點 1] GT 的維度改為 6
HEAD_DIMS=(2          4          6          8          16          6)

# 總共有幾個 Head
NUM_HEADS=${#HEAD_NAMES[@]} # 6
# 計算總組合數 2^6 = 64
TOTAL_COMBOS=$((1 << NUM_HEADS))

LAMBDA_VALUES=(0.0)
DOMAIN_WEIGHTS=(0.1 0.5) 

declare -A CONFIGS
CONFIGS["domainGRL_proba"]="configs/domainGRL_multi.gin"

# =========================
# ====== Execution ======
# =========================
RESULT_DIR="${BASE_OUTPUT_ROOT}/details"
mkdir -p "$RESULT_DIR"

if [ "$USE_WANDB" = true ]; then WANDB_FLAG="--use_wandb"; else WANDB_FLAG=""; fi

echo "Running AES Power-Set GRL Experiments (Multi-Head Only)..."
echo "Total Heads: $NUM_HEADS"
echo "GT Dimension: 6"

run_experiment() {
	    LOSS_TYPE=$1; GIN_PATH=$2; LAMBDA=$3; SEED=$4; CORR_USE=$5
	        DW=$6; COMBO_NAME=$7; JSON_CONFIG=$8; EXP_GROUP=$9
		    
		    SAVE_PATH="${BASE_OUTPUT_ROOT}/${LOSS_TYPE}/${EXP_GROUP}/seed${SEED}"
		        
		        # if [ -f "${SAVE_PATH}/checkpoint.json" ]; then echo "Skipping $SAVE_PATH"; return; fi

			    echo "Running: ${EXP_GROUP} (Seed ${SEED})"
			        # echo "  Config: $JSON_CONFIG" 

				    python train_domainGRL_multi.py \
					            --gin_path "$GIN_PATH" \
						            --save_path "$SAVE_PATH" \
							            --seed "$SEED" \
								            --covarreg_lambda "$LAMBDA" \
									            $CORR_USE \
										            --domain_weight "$DW" \
											            --domain_configs_json "$JSON_CONFIG" \
												            --exp_tag "$COMBO_NAME" \
													            $WANDB_FLAG > /dev/null 2>&1 

				        CHECKPOINT_FILE="${SAVE_PATH}/checkpoint.json"
					    RESULT_SRCC=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_FILE')).get('best_srcc', 'N/A'))" 2>/dev/null || echo "N/A")
					        echo "Group=${EXP_GROUP}, Best_SRCC=${RESULT_SRCC}" >> "${RESULT_DIR}/${LOSS_TYPE}.txt"
					}

					for LOSS_TYPE in "${!CONFIGS[@]}"; do
						    GIN_PATH=${CONFIGS[$LOSS_TYPE]}
						        
						        for DW in "${DOMAIN_WEIGHTS[@]}"; do
								        
								        # =======================================================
									        # Loop: 自動生成所有排列組合
										        # =======================================================
											        for (( i=1; i<TOTAL_COMBOS; i++ )); do
													            
													            # --- 1. 分析當前 Bitwise 組合 ---
														                CURRENT_JSON_PARTS=()
																            CURRENT_NAME_PARTS=()
																	                ACTIVE_INDICES=()

																			            # 掃描每一個 Bit
																				                for (( j=0; j<NUM_HEADS; j++ )); do
																							                if (( (i >> j) & 1 )); then
																										                    ACTIVE_INDICES+=($j)
																												                    fi
																														                done

																																            NUM_ACTIVE=${#ACTIVE_INDICES[@]}

																																	                # [修改點 2] 如果只有一個 Head，跳過不跑
																																			            if [ "$NUM_ACTIVE" -lt 2 ]; then
																																					                    continue
																																							                fi

																																									            # --- 2. 計算權重 (平均分配) ---
																																										                WEIGHT=$(awk "BEGIN {print 1.0/$NUM_ACTIVE}")

																																												            # --- 3. 構建 JSON 與 Name ---
																																													                for idx in "${ACTIVE_INDICES[@]}"; do
																																																                KEY=${HEAD_KEYS[$idx]}
																																																		                DIM=${HEAD_DIMS[$idx]}
																																																				                NAME=${HEAD_NAMES[$idx]}
																																																						                
																																																						                # JSON 片段
																																																								                CURRENT_JSON_PARTS+=("\"$KEY\": {\"num_classes\": $DIM, \"weight\": $WEIGHT}")
																																																										                # 名稱片段
																																																												                CURRENT_NAME_PARTS+=("$NAME")
																																																														            done

																																																															                # 組合 JSON 字串
																																																																	            IFS=,; JSON_STR="{${CURRENT_JSON_PARTS[*]}}"; unset IFS
																																																																		                # 組合 名稱字串
																																																																				            IFS=_; COMBO_NAME="${CURRENT_NAME_PARTS[*]}"; unset IFS

																																																																					                # --- 4. 執行實驗 ---
																																																																							            for LAMBDA in "${LAMBDA_VALUES[@]}"; do
																																																																									                    GRL_CORR_USE=""
																																																																											                    if (( $(echo "$LAMBDA > 0.0" | bc -l) )); then GRL_CORR_USE="--covarreg_use"; fi
																																																																													                    
																																																																													                    GRL_GROUP="Combo_${COMBO_NAME}_DW${DW}_L${LAMBDA}"
																																																																															                    
																																																																															                    for ((k=1; k<=N; k++)); do
																																																																																		                        SEED=$((1005 + k))
																																																																																					                    run_experiment "$LOSS_TYPE" "$GIN_PATH" "$LAMBDA" "$SEED" "$GRL_CORR_USE" "$DW" "$COMBO_NAME" "$JSON_STR" "$GRL_GROUP"
																																																																																							                    done
																																																																																									                done
																																																																																											        done
																																																																																												    done
																																																																																											    done
