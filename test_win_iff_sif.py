import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import scipy.stats
from joblib import Parallel, delayed
from tqdm import tqdm

# ==============================================================================
#                               CONFIG
# ==============================================================================
ASPECTS = ['Production_Quality', 'Production_Complexity', 'Content_Enjoyment', 'Content_Usefulness']
ASPECT_SHORT = ['PQ', 'PC', 'CE', 'CU']
ALPHA = 0.05

# ==============================================================================
#                            HELPER FUNCTIONS
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

def run_ttest_mse(target, pred_base, pred_method):
    se_base = (target - pred_base) ** 2
    se_method = (target - pred_method) ** 2
    # Two-sided T-test
    _, p_val = scipy.stats.ttest_rel(se_base, se_method)
    return p_val

def run_williams_test(ground_truth, pred_base, pred_method, metric='pcc'):
    x = np.array(ground_truth).flatten()
    y = np.array(pred_base).flatten()
    z = np.array(pred_method).flatten()
    
    if metric == 'srcc':
        x = scipy.stats.rankdata(x)
        y = scipy.stats.rankdata(y)
        z = scipy.stats.rankdata(z)
    
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]
    
    n = len(x)
    det = 1 - r_xy**2 - r_xz**2 - r_yz**2 + 2 * r_xy * r_xz * r_yz
    if det < 0: det = 0
    
    numer = (r_xy - r_xz) * np.sqrt((n - 1) * (1 + r_yz))
    denom = np.sqrt(2 * (n - 1) / (n - 3) * det + ((r_xy + r_xz)**2 / 4) * ((1 - r_yz)**3))
    
    if denom == 0: return 1.0
    t_val = numer / denom
    p_val = 2 * (1 - scipy.stats.t.cdf(np.abs(t_val), n - 3))
    return p_val

def load_seed_data(folder_path):
    npz_path = os.path.join(folder_path, "predictions.npz")
    if not os.path.exists(npz_path): return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        processed = {}
        sys_ids = data['sys_ids']
        for asp in ASPECTS:
            raw_p = data[f'{asp}_preds']
            raw_t = data[f'{asp}_targets']
            sys_p, sys_t = get_system_level_data(sys_ids, raw_p, raw_t)
            processed[asp] = {'p': sys_p, 't': sys_t}
        return processed
    except: return None

def analyze_candidate(cand_path, exp_name, baseline_data):
    """
    載入單一候選人數據，計算指標並進行顯著性檢定
    """
    data = load_seed_data(cand_path)
    if not data: return None
    
    cand_res = {
        'name': exp_name,
        'path': cand_path,
        'srcc_avg': 0.0,
        'metrics': {}, 
        'pvals': {},   
        'pass_mse': False,  # Default False
        'pass_srcc': False, 
        'pass_all': False   
    }
    
    srcc_list = []
    
    # 計數器：統計有幾個 Aspect 是顯著贏過 Baseline 的
    cnt_mse_better = 0
    cnt_srcc_better = 0
    
    for asp in ASPECTS:
        p = data[asp]['p']
        t = data[asp]['t']
        
        # 1. Method Metrics
        mse, pcc, srcc = calc_single_seed_metrics(p, t)
        cand_res['metrics'][asp] = (mse, pcc, srcc)
        if len(t) > 1: srcc_list.append(srcc)
        
        # 2. Baseline Metrics (for comparison direction)
        base_p = baseline_data[asp]['p']
        base_t = baseline_data[asp]['t']
        
        if p.shape != base_p.shape:
            if asp == ASPECTS[0]: 
                print(f"[Skip] Shape mismatch for {exp_name}: Cand {p.shape} vs Base {base_p.shape}")
            return None
        
        base_mse_val = np.mean((base_t - base_p)**2)
        if len(base_t) > 1:
            base_srcc_val = scipy.stats.spearmanr(base_t, base_p)[0]
        else:
            base_srcc_val = 0.0
        
        # 3. P-values (Method vs Baseline)
        p_mse = run_ttest_mse(t, base_p, p)
        p_pcc = run_williams_test(t, base_p, p, metric='pcc')
        p_srcc = run_williams_test(t, base_p, p, metric='srcc')
        
        cand_res['pvals'][asp] = (p_mse, p_pcc, p_srcc)
        
        # 4. Check Significance Conditions (計數邏輯)
        # (A) MSE: p < 0.05 且 誤差更小
        mse_better_and_sig = 0
    mse_better_total = 0

    srcc_better_and_sig = 0
    srcc_better_total = 0

    for asp in ASPECTS:
        mse, pcc, srcc = cand_res['metrics'][asp]
        p_mse, p_pcc, p_srcc = cand_res['pvals'][asp]

        base_p = baseline_data[asp]['p']
        base_t = baseline_data[asp]['t']
        base_mse_val = np.mean((base_t - base_p)**2)
        base_srcc_val = scipy.stats.spearmanr(base_t, base_p)[0] if len(base_t) > 1 else 0.0

        # ---- MSE 判定 ----
        if mse < base_mse_val:             # 候選人贏 baseline
            mse_better_total += 1
            if p_mse < ALPHA:              # 贏且顯著
                mse_better_and_sig += 1

        # ---- SRCC 判定 ----
        if srcc > base_srcc_val:           # 候選人贏 baseline
            srcc_better_total += 1
            if p_srcc < ALPHA:             # 贏且顯著
                srcc_better_and_sig += 1

    # 5. Determine Pass/Fail（套用你的新邏輯）
    # 只要：所有贏的 Aspect 都顯著 AND 至少有 1 個贏且顯著
    cand_res['pass_mse'] = (mse_better_total > 0) and (mse_better_and_sig == mse_better_total)
    cand_res['pass_srcc'] = (srcc_better_total > 0) and (srcc_better_and_sig == srcc_better_total)

    # pass_all = MSE 與 SRCC 都通過
    cand_res['pass_all'] = cand_res['pass_mse'] and cand_res['pass_srcc']
    
    return cand_res

def generate_latex_row(cand_res, name_override=None):
    name = name_override if name_override else cand_res['name']
    name = name.replace('_', r'\_')
    
    cells = []
    for asp in ASPECTS:
        mse, pcc, srcc = cand_res['metrics'][asp]
        p_mse, p_pcc, p_srcc = cand_res['pvals'][asp]
        
        def fmt(val, p_val, is_baseline=False):
            if is_baseline: return f"{val:.3f}"
            p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            return f"{val:.3f} ({p_str})"

        is_base = (name_override == "Baseline")
        cells.append(fmt(mse, p_mse, is_base))
        cells.append(fmt(pcc, p_pcc, is_base))
        cells.append(fmt(srcc, p_srcc, is_base))
        
    return {'name': name, 'cells': cells}

def generate_latex_table(rows, filename, seed, top_k, filter_mode):
    latex = []
    latex.append(r"\begin{table*}[h!]")
    latex.append(r"\centering")
    latex.append(r"\tiny")
    latex.append(r"\setlength{\tabcolsep}{1.2pt}")
    
    caption = f"Top-{top_k} Results (Seed: {seed}, Filter: {filter_mode}). Format: Value (p-value)."
    latex.append(r"\caption{" + caption + r"}")
    
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
    
    for row in rows:
        line = f"{row['name']}"
        for cell in row['cells']:
            line += f" & {cell}"
        latex.append(line + r" \\")
        
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", encoding='utf-8') as f:
        f.write("\n".join(latex))
    print(f"Table saved to {filename}")

# ==============================================================================
#                               MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--baseline_path", type=str, required=True)
    parser.add_argument("--search_dirs", type=str, nargs='+', required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    # 更新 choices
    parser.add_argument("--filter_mode", type=str, default="none", choices=["none", "mse", "srcc", "all"], 
                        help="Filter: 'none' (Score only), 'mse' (MSE sig), 'srcc' (SRCC sig), 'all' (All sig)")
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args()

    print(f"\n" + "="*60)
    print(f"Analysis (Filter: {args.filter_mode}) | Seed: {args.seed}")
    print("="*60)

    # 1. Load Baseline
    baseline_path = os.path.join(args.baseline_path, f"seed{args.seed}")
    baseline_data = load_seed_data(baseline_path)
    if not baseline_data:
        print(f"[Error] Baseline not found: {baseline_path}")
        return

    # 2. Find Candidates
    cand_paths = []
    for root in args.search_dirs:
        for search_root in glob.glob(root) if '*' in root else [root]:
            if not os.path.exists(search_root): continue
            for current_path, dirs, files in os.walk(search_root):
                s_path = os.path.join(current_path, f"seed{args.seed}")
                if f"seed{args.seed}" in dirs and os.path.exists(os.path.join(s_path, "predictions.npz")):
                    cand_paths.append((current_path, os.path.basename(current_path), s_path))
                    dirs[:] = [] 

    print(f"Found {len(cand_paths)} potential candidates. Analyzing...")

    # 3. Analyze Candidates
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(analyze_candidate)(path, name, baseline_data) 
        for _, name, path in tqdm(cand_paths, desc="Analyzing")
    )
    results = [r for r in results if r is not None]

    # 4. Filter
    filtered_cands = []
    
    for r in results:
        if r['path'] == baseline_path: continue 
        
        if args.filter_mode == 'mse':
            if r['pass_mse']: filtered_cands.append(r)
        elif args.filter_mode == 'srcc':   # 新增選項
            if r['pass_srcc']: filtered_cands.append(r)
        elif args.filter_mode == 'all':
            if r['pass_all']: filtered_cands.append(r)
        else:
            filtered_cands.append(r) 

    # Sort by SRCC (High to Low)
    filtered_cands.sort(key=lambda x: x['srcc_avg'], reverse=True)
    
    print(f"\nCandidates passing filter '{args.filter_mode}': {len(filtered_cands)} / {len(results)}")
    
    if args.top_k > 0:
        top_k_cands = filtered_cands[:args.top_k]
    else:
        top_k_cands = filtered_cands
    
    # 5. Output
    base_res = analyze_candidate(baseline_path, "Baseline", baseline_data)
    for asp in base_res['pvals']: base_res['pvals'][asp] = (1.0, 1.0, 1.0)
    
    final_rows = [generate_latex_row(base_res, name_override="Baseline")]
    for c in top_k_cands:
        final_rows.append(generate_latex_row(c))

    generate_latex_table(final_rows, args.output_path, args.seed, args.top_k, args.filter_mode)

if __name__ == "__main__":
    main()