import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

# 定義 Aspect 顯示順序與縮寫
ASPECTS = ["Production_Quality", "Production_Complexity", "Content_Enjoyment", "Content_Usefulness"]
ASPECT_MAP = {
    "Production_Quality": "PQ",
    "Production_Complexity": "PC",
    "Content_Enjoyment": "CE",
    "Content_Usefulness": "CU"
}
DISPLAY_ASPECTS = ["PQ", "PC", "CE", "CU", "Average"]
METRICS = ["MSE", "LCC", "SRCC"]

def parse_log_file(log_path, best_epoch):
    dev_results = defaultdict(lambda: defaultdict(dict))
    eval_results = defaultdict(lambda: defaultdict(dict))
    
    current_epoch = -1
    found_best_epoch = False
    lines_matched = 0
    
    epoch_pattern = re.compile(r"INFO:root:Epoch\s+(\d+)\s+-")
    data_pattern = re.compile(r"INFO:root:\[(.*?)\]\[(.*?)\]\s+MSE=([0-9\.]+)\s+(?:LCC|PCC)=([0-9\.]+)\s+SRCC=([0-9\.]+)")

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    if current_epoch == best_epoch:
                        found_best_epoch = True
                    continue

                data_match = data_pattern.search(line)
                if data_match:
                    lines_matched += 1
                    dataset_tag = data_match.group(1)
                    aspect_full = data_match.group(2)
                    
                    if aspect_full not in ASPECT_MAP: continue 
                    aspect_short = ASPECT_MAP[aspect_full]

                    vals = {
                        "MSE": float(data_match.group(3)),
                        "LCC": float(data_match.group(4)), 
                        "SRCC": float(data_match.group(5))
                    }

                    if "dev_list" in dataset_tag:
                        if current_epoch == best_epoch:
                            dev_results[aspect_short] = vals
                    
                    if "eval_list" in dataset_tag:
                        eval_results[aspect_short] = vals
        
        if not found_best_epoch:
            # print(f"    [DEBUG] Log read, but Best Epoch {best_epoch} was NOT found in logs.")
            pass
        elif len(dev_results) == 0:
            pass 
        
    except Exception as e:
        print(f"    [Error] reading {log_path}: {e}")
        return None, None

    return dev_results, eval_results

def calculate_average_aspect(metrics_data_dict):
    avg_metrics = {}
    real_aspects = ["PQ", "PC", "CE", "CU"]
    
    for m in METRICS:
        vals = []
        for a in real_aspects:
            v = metrics_data_dict.get(a, {}).get(m, None)
            if v is not None:
                vals.append(v)
        if vals:
            avg_metrics[m] = np.mean(vals)
        else:
            avg_metrics[m] = 0.0
    return avg_metrics

# 修改：新增 caption_info 參數
def generate_latex_styled(data_dict, set_name, caption_info="", show_std=True, colsep="2pt"):
    def get_cmidrules(start_col_idx, num_blocks):
        rules = ""
        current = start_col_idx
        for _ in range(num_blocks):
            end = current + 2
            rules += f"\\cmidrule(lr){{{current}-{end}}} "
            current += 3
        return rules

    header_definitions = "\\providecommand{\\npm}{\\mkern-3mu\\pm\\mkern-3mu}\n"

    col_def = "l" + "c" + ("ccc" * len(DISPLAY_ASPECTS))
    
    # 組合 Caption: Set Name + Extra Info
    full_caption = f"{set_name}"
    if caption_info:
        full_caption += f" ({caption_info})"

    content = header_definitions
    content += "\\begin{table*}[h!]\n\\centering\n\\tiny\n" 
    content += f"\\setlength{{\\tabcolsep}}{{{colsep}}}\n"
    content += f"\\caption{{Detailed results on {full_caption}.}}\n"
    content += f"\\begin{{tabular}}{{{col_def}}}\n\\toprule\n"
    
    header1 = "\\multirow{2}{*}{\\textbf{Method}} & \\multirow{2}{*}{\\textbf{Seed}}"
    for a in DISPLAY_ASPECTS:
        header1 += f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{a}}}}}"
    header1 += " \\\\\n"
    
    header_cmid = get_cmidrules(3, len(DISPLAY_ASPECTS)) + "\n"
    
    header3 = " & " 
    for _ in DISPLAY_ASPECTS:
        for m in METRICS:
            arrow = "$\\downarrow$" if m == "MSE" else "$\\uparrow$"
            header3 += f" & \\textbf{{{m}}}{arrow}"
    header3 += " \\\\\n\\midrule\n"
    
    content += header1 + header_cmid + header3

    sorted_methods = sorted(data_dict.keys())
    
    for method in sorted_methods:
        seeds = sorted(data_dict[method].keys())
        for i, seed in enumerate(seeds):
            row_str = ""
            method_clean = method.replace("_", "\\_")
            method_disp = f"\\textbf{{{method_clean}}}" if i == 0 else ""
            
            row_str += f"{method_disp} & {seed}"
            
            current_seed_data = data_dict[method][seed]
            avg_vals = calculate_average_aspect(current_seed_data)
            
            for a in DISPLAY_ASPECTS:
                if a == "Average":
                    vals = avg_vals
                else:
                    vals = current_seed_data.get(a, {})
                    
                for m in METRICS:
                    v = vals.get(m, 0.0)
                    row_str += f" & {v:.4f}"
            
            content += row_str + " \\\\\n"
        content += "\\midrule\n"

    content += "\\bottomrule\n\\end{tabular}\n"
    content += "\\end{table*}\n\n"

    # Aggregated
    col_def_agg = "l" + ("ccc" * len(DISPLAY_ASPECTS))
    
    content += "\\begin{table*}[h!]\n\\centering\n\\tiny\n"
    content += f"\\setlength{{\\tabcolsep}}{{{colsep}}}\n" 
    content += f"\\caption{{Aggregated results on {full_caption} (Mean $\\pm$ Std).}}\n"
    content += f"\\begin{{tabular}}{{{col_def_agg}}}\n\\toprule\n"
    
    header1_agg = "\\multirow{2}{*}{\\textbf{Method}}"
    for a in DISPLAY_ASPECTS:
        header1_agg += f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{a}}}}}"
    header1_agg += " \\\\\n"
    
    header_cmid_agg = get_cmidrules(2, len(DISPLAY_ASPECTS)) + "\n"
    
    header3_agg = " "
    for _ in DISPLAY_ASPECTS:
        for m in METRICS:
            arrow = "$\\downarrow$" if m == "MSE" else "$\\uparrow$"
            header3_agg += f" & \\textbf{{{m}}}{arrow}"
    header3_agg += " \\\\\n\\midrule\n"
    
    content += header1_agg + header_cmid_agg + header3_agg

    for method in sorted_methods:
        seeds = data_dict[method].keys()
        method_clean = method.replace('_', '\\_')
        row_str = f"\\textbf{{{method_clean}}}"
        
        avg_data_per_seed = []
        for seed in seeds:
            avg_data_per_seed.append(calculate_average_aspect(data_dict[method][seed]))

        for a in DISPLAY_ASPECTS:
            for m in METRICS:
                vals = []
                for idx, seed in enumerate(seeds):
                    if a == "Average":
                        v = avg_data_per_seed[idx].get(m, None)
                    else:
                        v = data_dict[method][seed].get(a, {}).get(m, None)
                    
                    if v is not None: vals.append(v)
                
                if vals:
                    mu = np.mean(vals)
                    sigma = np.std(vals)
                    
                    if show_std:
                        row_str += f" & {mu:.3f} $\\npm$ {sigma:.3f}" 
                    else:
                        row_str += f" & {mu:.3f}"
                else:
                    row_str += " & -"
        
        content += row_str + " \\\\\n"

    content += "\\bottomrule\n\\end{tabular}\n"
    content += "\\end{table*}\n"
    
    return content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Input root directory")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for tex files")
    parser.add_argument("--no_std", action='store_true', help="Do not show std")
    parser.add_argument("--colsep", type=str, default="2pt", help="LaTeX tabcolsep value")
    parser.add_argument("--seeds", nargs="+", type=str, help="List of seeds to include")
    # 修改：新增參數接收 Caption 額外資訊
    parser.add_argument("--caption_info", type=str, default="", help="Extra info to append to caption (e.g. dataset name)")
    
    args = parser.parse_args()

    show_std = not args.no_std 
    all_dev_data = defaultdict(dict)
    all_eval_data = defaultdict(dict)

    print(f"Scanning: {args.root}")
    if args.seeds:
        print(f"Filtering Seeds: {args.seeds}")
    
    found_any_log = False

    for root, dirs, files in os.walk(args.root):
        if "train.log" in files:
            current_folder_path = root
            seed_name = os.path.basename(current_folder_path)
            
            if args.seeds and seed_name not in args.seeds:
                continue

            found_any_log = True
            
            if "checkpoint.json" not in files:
                continue
                
            try:
                ckpt_path = os.path.join(root, "checkpoint.json")
                with open(ckpt_path, 'r') as f:
                    try:
                        ckpt_data = json.load(f)
                        best_epoch = ckpt_data.get("epoch")
                    except json.JSONDecodeError:
                        print(f"  [Skip] Invalid JSON in {ckpt_path}")
                        continue
                
                if best_epoch is None: continue

                parent_folder_path = os.path.dirname(current_folder_path)
                method_name = os.path.basename(parent_folder_path)
                if not method_name: method_name = "Root"

                log_path = os.path.join(root, "train.log")
                dev_res, eval_res = parse_log_file(log_path, best_epoch)

                if dev_res and eval_res:
                    all_dev_data[method_name][seed_name] = dev_res
                    all_eval_data[method_name][seed_name] = eval_res
                    # print(f"  [Success] Parsed: Method='{method_name}', Seed='{seed_name}'")
                else:
                    print(f"  [Fail] Parsing failed for {root}")

            except Exception as e:
                print(f"  [Error] Processing {root}: {e}")

    if not found_any_log:
        if args.seeds:
            print(f"\n[WARNING] No logs found for seeds: {args.seeds}. Check path or seed names.")
        else:
            print("\n[WARNING] No 'train.log' found.")

    # 產生 LaTeX 時傳入 caption_info
    if all_dev_data:
        tex = generate_latex_styled(all_dev_data, "Dev Set", caption_info=args.caption_info, show_std=show_std, colsep=args.colsep)
        out_path = os.path.join(args.outdir, "tables_dev.tex")
        with open(out_path, "w", encoding='utf-8') as f: f.write(tex)
        print(f"\nSaved: {out_path}")

    if all_eval_data:
        tex = generate_latex_styled(all_eval_data, "Eval Set", caption_info=args.caption_info, show_std=show_std, colsep=args.colsep)
        out_path = os.path.join(args.outdir, "tables_eval.tex")
        with open(out_path, "w", encoding='utf-8') as f: f.write(tex)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()