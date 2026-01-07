import torch
import os
import glob
import datetime
import re

def extract_date_from_filename(filename):
    # Try common patterns
    # 2025-08-10
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    # 20250903
    match = re.search(r'(\d{8})', filename)
    if match:
        s = match.group(1)
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return "Unknown"

def get_file_size_mb(path):
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)

def count_parameters(state_dict):
    total_params = 0
    for tensor in state_dict.values():
        if hasattr(tensor, 'numel'):
            total_params += tensor.numel()
    return total_params

def generate_report(base_dir, output_file):
    # Filter out _delete files
    all_files = sorted(glob.glob(os.path.join(base_dir, "*.ckpt")))
    files = [f for f in all_files if not f.endswith("_delete")]
    
    report_lines = []
    report_lines.append("# NSSR SHHS Checkpoint Report")
    report_lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"\nTotal Valid Checkpoints analyzed: {len(files)}")
    report_lines.append(f"(Excluded {len(all_files) - len(files)} files marked for deletion)\n")
    
    # Summary Table
    report_lines.append("## Summary Table")
    report_lines.append("| Date | Model | Params (M) | Best Val Loss | Epoch | PL Ver | Filename |")
    report_lines.append("|---|---|---|---|---|---|---|")
    
    details = []
    
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        print(f"Processing {i+1}/{len(files)}: {filename}")
        
        try:
            checkpoint = torch.load(file_path, map_location='cpu')
            
            # Basic Info
            date = extract_date_from_filename(filename)
            file_size = get_file_size_mb(file_path)
            pl_version = checkpoint.get('pytorch-lightning_version', 'N/A')
            
            # Metadata
            hparams = checkpoint.get('hyper_parameters', {})
            model_name = hparams.get('model_name', 'Unknown')
            epoch = checkpoint.get('epoch', 'Unknown')
            global_step = checkpoint.get('global_step', 'Unknown')
            
            # Parameter Count
            state_dict = checkpoint.get('state_dict', {})
            param_count = count_parameters(state_dict)
            param_count_m = param_count / 1_000_000
            
            # Best Score
            best_score = "N/A"
            callbacks = checkpoint.get('callbacks', {})
            for key, val in callbacks.items():
                if 'ModelCheckpoint' in key and isinstance(val, dict):
                    if 'best_model_score' in val:
                        score = val['best_model_score']
                        if hasattr(score, 'item'):
                            best_score = f"{score.item():.4f}"
                        else:
                            best_score = str(score)
            
            if best_score == "N/A":
                match = re.search(r'val_loss=([\d\.]+)', filename)
                if match:
                    best_score = f"{match.group(1)} (std)"

            # Add to Summary
            report_lines.append(f"| {date} | {model_name} | {param_count_m:.2f}M | {best_score} | {epoch} | {pl_version} | `{filename}` |")
            
            # Add to Details
            details.append(f"### {i+1}. {filename}")
            details.append(f"- **Date**: {date}")
            details.append(f"- **Model Architecture**: {model_name}")
            details.append(f"- **Parameters**: {param_count_m:.2f} Million ({param_count:,})")
            details.append(f"- **File Size**: {file_size:.2f} MB")
            details.append(f"- **PL Version**: {pl_version}")
            details.append(f"- **Training Status**: Epoch {epoch}, Step {global_step}")
            details.append(f"- **Best Val Loss**: {best_score}")
            
            if hparams:
                details.append("- **Hyperparameters**:")
                for k, v in hparams.items():
                    details.append(f"  - `{k}`: {v}")
            else:
                details.append("- **Hyperparameters**: None found")
                
            # Optimizer info if available
            opt_states = checkpoint.get('optimizer_states', [])
            if opt_states:
                details.append(f"- **Optimizer**: State dictionary present ({len(opt_states)} optimizer(s))")

            details.append("")
            
        except Exception as e:
            report_lines.append(f"| {date if 'date' in locals() else 'Unknown'} | Error | - | - | - | - | `{filename}` |")
            details.append(f"### {i+1}. {filename}")
            details.append(f"**ERROR**: {str(e)}")
            details.append("")

    report_lines.append("\n## Detailed Reports")
    report_lines.extend(details)
    
    with open(output_file, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"Report generated at {output_file}")

if __name__ == "__main__":
    base_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/"
    output_path = os.path.join(base_dir, "CHECKPOINT_REPORT.md")
    generate_report(base_dir, output_path)
