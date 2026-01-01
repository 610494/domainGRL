import glob
import os
from collections import defaultdict

import numpy as np
import scipy.stats
from joblib import Parallel, delayed
from tqdm import tqdm

# ==============================================================================
#                               CONFIG SECTION
# ==============================================================================

# 1. 功能開關
CALC_P_VALUE = False   # 是否計算 P-value (False 會快很多，且表格無星號)
USE_TOP_K    = True   # 是否只篩選 Top-K (False 會列出資料夾下所有實驗)

# 2. Seed 組合
SEED_CONFIGS = {
    "1": [1000, 1001, 1002, 1003, 1004],
    "2": [1001, 1002, 1003, 1004, 1005],
    "3": [1002, 1003, 1004, 1005, 1006],
    "4": [1003, 1004, 1005, 1006, 1007],
    "5": [1004, 1005, 1006, 1007, 1008],
    "6": [1005, 1006, 1007, 1008, 1009],
}

# 3. 參數設定
TOP_K_PER_DIR = 3     # 只有在 USE_TOP_K = True 時有效
BOOTSTRAP_ROUNDS = 1000
N_JOBS = -1
ALPHA = 0.05

BASELINE_PATH = "evaluation_results_batch_w_pred/results_aes_baselines/baseline_proba/lambda0.0"

# 這些是不同的「方法家族」，表格會依照這個順序分區塊顯示
SEARCH_DIRS = [
    # "evaluation_results_batch_w_pred/results_aes_baselines_regu/baseline_proba",
    "evaluation_results_batch_w_pred/results_aes_domain_grl_covar_regu/domainGRL_proba",
    "evaluation_results_batch_w_pred/results_aes_domain_grl/domainGRL_proba",
    # "evaluation_results_batch_w_pred/results_ortho",
    # "evaluation_results_batch_w_pred/results_supcon",
]

ASPECTS = ['Production_Quality', 'Production_Complexity', 'Content_Enjoyment', 'Content_Usefulness']
ASPECT_SHORT = ['PQ', 'PC', 'CE', 'CU']

# ==============================================================================
# Calculation Helpers
# ==============================================================================

def get_system_level_data(sys_ids, preds, targets):
    sys_p = defaultdict(list)
    sys_t = defaultdict(list)
    for s, p, t in zip(sys_ids, preds, targets):
        sys_p[s].append(p)
        sys_t[s].append(t)
    keys = sorted(sys_p.keys())
    agg_p = np.array([np.mean(sys_p[k]) for k in keys])
    agg_t = np.array([np.mean(sys_t[k]) for k in keys])
    return agg_p, agg_t

def calc_single_seed_metrics(p, t):
    if len(t) < 2: return 0.0, 0.0, 0.0
    mse = np.mean((t - p) ** 2)
    pcc = scipy.stats.pearsonr(t, p)[0]
    srcc = scipy.stats.spearmanr(t, p)[0]
    return mse, pcc, srcc

def vectorized_pcc(x, y):
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    numer = (x_centered * y_centered).sum(axis=1)
    denom = np.sqrt((x_centered**2).sum(axis=1)) * np.sqrt((y_centered**2).sum(axis=1))
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom!=0)

def fast_bootstrap_pooled(target, pred_base, pred_prop, n_resamples=BOOTSTRAP_ROUNDS):
    n = len(target)
    rng = np.random.RandomState(42)
    indices = rng.randint(0, n, size=(n_resamples, n))
    
    t_samp = target[indices]
    pb_samp = pred_base[indices]
    pp_samp = pred_prop[indices]
    
    mse_b = np.mean((t_samp - pb_samp)**2, axis=1)
    mse_p = np.mean((t_samp - pp_samp)**2, axis=1)
    p_mse = np.mean((mse_p - mse_b) >= 0)
    
    pcc_b = vectorized_pcc(t_samp, pb_samp)
    pcc_p = vectorized_pcc(t_samp, pp_samp)
    p_pcc = np.mean((pcc_p - pcc_b) <= 0)
    
    rank_t = t_samp.argsort(axis=1).argsort(axis=1)
    rank_pb = pb_samp.argsort(axis=1).argsort(axis=1)
    rank_pp = pp_samp.argsort(axis=1).argsort(axis=1)
    srcc_b = vectorized_pcc(rank_t, rank_pb)
    srcc_p = vectorized_pcc(rank_t, rank_pp)
    p_srcc = np.mean((srcc_p - srcc_b) <= 0)
    
    return p_mse, p_pcc, p_srcc

def load_seed_data(folder_path):
    npz_path = os.path.join(folder_path, "predictions.npz")
    if not os.path.exists(npz_path): return None
    try:
        data = np.load(npz_path)
        processed = {}
        sys_ids = data['sys_ids']
        for asp in ASPECTS:
            raw_p = data[f'{asp}_preds']
            raw_t = data[f'{asp}_targets']
            sys_p, sys_t = get_system_level_data(sys_ids, raw_p, raw_t)
            processed[asp] = {'p': sys_p, 't': sys_t}
        return processed
    except: return None

# ==============================================================================
# Scoring & Processing
# ==============================================================================

def calculate_method_score(method_paths):
    """計算 Macro Average SRCC 用於排序"""
    all_srccs = []
    valid_count = 0
    for path in method_paths:
        data = load_seed_data(path)
        if not data: continue
        valid_count += 1
        seed_srccs = []
        for asp in ASPECTS:
            p = data[asp]['p']
            t = data[asp]['t']
            if len(t) > 1:
                s = scipy.stats.spearmanr(t, p)[0]
                seed_srccs.append(s)
            else:
                seed_srccs.append(0)
        all_srccs.append(np.mean(seed_srccs))
    if valid_count == 0: return -1.0
    return np.mean(all_srccs)

def process_method_full(method_info, baseline_pooled):
    name, seed_paths, cat_idx = method_info
    
    method_data = {asp: {'metrics': [], 'pooled_p': [], 'pooled_t': []} for asp in ASPECTS}
    
    valid_seeds = 0
    for path in seed_paths:
        data = load_seed_data(path)
        if not data: continue
        valid_seeds += 1
        for asp in ASPECTS:
            p = data[asp]['p']
            t = data[asp]['t']
            m, pc, sr = calc_single_seed_metrics(p, t)
            method_data[asp]['metrics'].append([m, pc, sr])
            method_data[asp]['pooled_p'].append(p)
            method_data[asp]['pooled_t'].append(t)
            
    if valid_seeds == 0: return None
    
    row_cells = []
    
    for asp in ASPECTS:
        metrics_arr = np.array(method_data[asp]['metrics'])
        means = np.mean(metrics_arr, axis=0)
        stds = np.std(metrics_arr, axis=0)
        
        # === P-value 計算邏輯控制 ===
        if name == "Baseline" or not CALC_P_VALUE:
            # 如果不開啟計算，填入 1.0 (代表不顯著，不會有星號)
            p_values = [1.0, 1.0, 1.0]
        else:
            pool_p = np.concatenate(method_data[asp]['pooled_p'])
            pool_t = np.concatenate(method_data[asp]['pooled_t'])
            base_pool_p = baseline_pooled[asp]['p']
            p_values = fast_bootstrap_pooled(pool_t, base_pool_p, pool_p)

        def fmt(mean, std, p_val):
            # 只有當 CALC_P_VALUE = True 且 p < ALPHA 時才加星號
            if CALC_P_VALUE and p_val < ALPHA:
                sig = r"^*"
                return f"{mean:.3f} $\\pm$ {std:.3f}${sig}$"
            else:
                return f"{mean:.3f} $\\pm$ {std:.3f}"

        row_cells.append(fmt(means[0], stds[0], p_values[0]))
        row_cells.append(fmt(means[1], stds[1], p_values[1]))
        row_cells.append(fmt(means[2], stds[2], p_values[2]))
        
    return {
        'name': name.replace('_', r'\_'), 
        'cells': row_cells, 
        'cat_idx': cat_idx
    }

# ==============================================================================
# Main Logic
# ==============================================================================

def run_analysis_for_group(group_name, target_seeds):
    print(f"\n" + "="*60)
    print(f"Processing Group [{group_name}] Seeds: {target_seeds}")
    print(f"Settings: P-Value={CALC_P_VALUE}, Top-K={USE_TOP_K}")
    print("="*60)
    
    # 1. Baseline
    baseline_seeds_data = []
    baseline_paths = []
    for seed in target_seeds:
        path = os.path.join(BASELINE_PATH, f"seed{seed}")
        if os.path.exists(os.path.join(path, "predictions.npz")):
            baseline_paths.append(path)
            data = load_seed_data(path)
            if data: baseline_seeds_data.append(data)
    
    if len(baseline_seeds_data) != len(target_seeds):
        print(f"  [Error] Baseline incomplete. Skip.")
        return

    # 如果不需計算 P-value，這裡其實可以跳過，但為了代碼結構一致我們保留 pooled 結構
    baseline_pooled = {asp: {'p': [], 't': []} for asp in ASPECTS}
    if CALC_P_VALUE:
        for data in baseline_seeds_data:
            for asp in ASPECTS:
                baseline_pooled[asp]['p'].append(data[asp]['p'])
                baseline_pooled[asp]['t'].append(data[asp]['t'])
        for asp in ASPECTS:
            baseline_pooled[asp]['p'] = np.concatenate(baseline_pooled[asp]['p'])
            baseline_pooled[asp]['t'] = np.concatenate(baseline_pooled[asp]['t'])

    # 2. Scanning Candidates
    final_candidates = []
    final_candidates.append({'name': 'Baseline', 'paths': baseline_paths, 'cat_idx': 0})

    for idx, search_root in enumerate(SEARCH_DIRS, start=1):
        if not os.path.exists(search_root): continue
        
        category_name = os.path.basename(search_root)
        print(f"  Scanning: {category_name} ...")
        
        dir_candidates = []
        for current_path, dirs, files in os.walk(search_root):
            found_seeds_paths = []
            all_seeds_present = True
            for s in target_seeds:
                s_path = os.path.join(current_path, f"seed{s}")
                if f"seed{s}" in dirs and os.path.exists(os.path.join(s_path, "predictions.npz")):
                    found_seeds_paths.append(s_path)
                else:
                    all_seeds_present = False
                    break
            
            if all_seeds_present:
                exp_name = os.path.basename(current_path)
                # 只有在需要 Top-K 排序時才一定要算 score
                # 但為了方便 debug 顯示，我們還是都算一下 (代價很小)
                score = calculate_method_score(found_seeds_paths)
                dir_candidates.append({'name': exp_name, 'paths': found_seeds_paths, 'score': score})
                dirs[:] = []
        
        # === Top-K 邏輯控制 ===
        if USE_TOP_K:
            dir_candidates.sort(key=lambda x: x['score'], reverse=True)
            selected = dir_candidates[:TOP_K_PER_DIR]
        else:
            # 如果不篩選，就保留全部 (依然照分數排一下比較好看)
            dir_candidates.sort(key=lambda x: x['score'], reverse=True)
            selected = dir_candidates
            
        for cand in selected:
            cand['cat_idx'] = idx
            final_candidates.append(cand)

    # 3. Full Analysis
    print(f"\nRunning Analysis for {len(final_candidates)} methods...")
    task_list = [(c['name'], c['paths'], c['cat_idx']) for c in final_candidates]
    
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_method_full)(m, baseline_pooled) 
        for m in tqdm(task_list, desc=f"Group {group_name}")
    )
    rows = [r for r in results if r is not None]

    # 4. Generate LaTeX
    # 檔名自動帶上參數
    f_tag = f"Top{TOP_K_PER_DIR}" if USE_TOP_K else "All"
    if not CALC_P_VALUE: f_tag += "_NoPVal"
    
    output_filename = f"evaluation_results_batch_w_pred/table_{f_tag}_seeds_{group_name}.tex"
    generate_latex(rows, output_filename, target_seeds)

def generate_latex(rows, filename, seeds):
    latex = []
    latex.append(r"\begin{table*}[h!]")
    latex.append(r"\centering")
    latex.append(r"\tiny")
    latex.append(r"\setlength{\tabcolsep}{1.2pt}")
    
    # 動態調整 Caption
    caption = f"Results (Seeds: {seeds}). "
    if USE_TOP_K:
        caption += f"Showing Top-{TOP_K_PER_DIR} per category. "
    else:
        caption += "Showing all methods. "
        
    if CALC_P_VALUE:
        caption += r"$^*$ denotes $p < 0.05$ over Baseline."
    else:
        caption += "P-value calculation disabled."
        
    latex.append(r"\caption{" + caption + "}")
    
    col_def = "l" + ("ccc" * len(ASPECTS))
    latex.append(r"\begin{tabular}{" + col_def + r"}")
    latex.append(r"\toprule")
    
    header1 = r"\multirow{2}{*}{\textbf{Method}}"
    for short in ASPECT_SHORT:
        header1 += r" & \multicolumn{3}{c}{\textbf{" + short + r"}}"
    latex.append(header1 + r" \\")
    
    cmid = ""
    for i in range(len(ASPECTS)):
        start = 2 + i*3
        end = start + 2
        cmid += r" \cmidrule(lr){" + f"{start}-{end}" + r"}"
    latex.append(cmid)
    
    header2 = ""
    sub_h = r" & \textbf{MSE}$\downarrow$ & \textbf{PCC}$\uparrow$ & \textbf{SRCC}$\uparrow$"
    for _ in ASPECTS:
        header2 += sub_h
    latex.append(header2 + r" \\")
    latex.append(r"\midrule")
    
    rows.sort(key=lambda x: (x['cat_idx'], x['name']))
    
    for i, row in enumerate(rows):
        line = f"{row['name']}"
        for cell in row['cells']:
            line += f" & {cell}"
        latex.append(line + r" \\")
        
        # 智慧分隔線
        if i < len(rows) - 1:
            if row['cat_idx'] != rows[i+1]['cat_idx']:
                latex.append(r"\midrule")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    
    with open(filename, "w") as f:
        f.write("\n".join(latex))
    print(f"Table saved to {filename}")

def main():
    for name, seeds in SEED_CONFIGS.items():
        run_analysis_for_group(name, seeds)

if __name__ == "__main__":
    main()
