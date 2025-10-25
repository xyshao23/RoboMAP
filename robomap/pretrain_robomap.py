#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main training script for the RoboMAP project.

This script is the primary entry point for starting a pretraining run.
It parses command-line arguments, loads the main configuration,
dynamically determines data sources and output paths based on the config,
and then hands off control to the Pretrain_RoboMAP_Palligemma pipeline class.
"""

# --- 1. Standard Library Imports ---
import argparse
import os
import sys
import yaml
import datetime

# --- 2. Local (robomap) Imports ---
# Import the main pipeline class
from robomap.pipeline import Pretrain_RoboMAP_Palligemma


# --- 3. Main Execution Block ---
if __name__ == "__main__":

    # --- 1. Argument Parsing (Compatible with your bash script) ---
    parser = argparse.ArgumentParser(description="Pretraining script for RoboMAP")

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the training configuration YAML file (e.g., configs/config.example.yaml)"
    )
    parser.add_argument(
        "--branches",
        type=int,
        required=True,
        help="Script mode (This script only supports '2' for pretraining)"
    )
    parser.add_argument(
        "--vis_flag",
        type=str,
        default=None,
        help="Flag passed to the pipeline (usage depends on pipeline implementation)"
    )
    args = parser.parse_args()

    # --- 2. Mode Check ---
    # This script is now focused only on pretraining (branches=2)
    if args.branches != 2:
        print(f"Error: This script 'pretrain_robodata.py' is only for pretraining (branches=2).", file=sys.stderr)
        print(f"You specified --branches {args.branches}. Please run the correct script (e.g., eval.py or visualize.py) for other modes.", file=sys.stderr)
        sys.exit(1)

    print("--- [Mode: Pretraining (branches=2)] ---")

    # --- 3. Load Configuration File ---
    print(f"Loading configuration from: {args.config_path}")
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        # 捕获 YAML 解析错误，例如你之前遇到的引号和括号错误
        print(f"Error parsing YAML file {args.config_path}: {e}", file=sys.stderr)
        print("  > Please check for syntax errors like missing quotes '\"' or curly braces '}' in your config.", file=sys.stderr)
        sys.exit(1)

    # --- 4. Dynamically Load Data Sources from Config ---
    print("--- Loading data sources from config file ---")
    all_data_sources = []

    # [修改 1] 检查 data_root1, data_root2, 和 data_paths
    if 'data_root1' not in config or 'data_root2' not in config or 'data_paths' not in config:
        print(f"Error: 'data_root1', 'data_root2', or 'data_paths' not found in config file '{args.config_path}'", file=sys.stderr)
        sys.exit(1)

    # [修改 2] 加载并解析 *所有* 的 data_root
    try:
        resolved_data_root1 = os.path.expanduser(config['data_root1'])
        resolved_data_root2 = os.path.expanduser(config['data_root2'])
    except Exception as e:
        print(f"Error expanding data_root paths: {e}", file=sys.stderr)
        sys.exit(1)

    # [修改 3] 创建一个包含所有已解析根路径的字典，用于格式化
    # 这样 .format() 就能同时访问 {data_root1} 和 {data_root2}
    format_context = {
        'data_root1': resolved_data_root1,
        'data_root2': resolved_data_root2
    }
    print(f"  > Loaded data_root1: {resolved_data_root1}")
    print(f"  > Loaded data_root2: {resolved_data_root2}")


    # 检查 'sampling_config' (保持不变)
    if 'sampling_config' not in config:
        print(f"Error: 'sampling_config' not found in config file '{args.config_path}'", file=sys.stderr)
        sys.exit(1)

    # 循环遍历 'sampling_config' 中的键 (保持不变)
    for source_key in config['sampling_config'].keys(): # Iterate only over keys intended for sampling
        if source_key not in config['data_paths']:
            print(f"Warning: Data source '{source_key}' defined in 'sampling_config' but not found in 'data_paths'. Skipping.")
            continue

        paths = config['data_paths'][source_key]
        try:
            # [修改 4] 使用字典解包 (**) 将 format_context 传递给 .format()
            # 配置文件中的路径字符串现在必须使用 {data_root1} 或 {data_root2}
            source_dict = {
                'json_path': paths['json_path'].format(**format_context),
                'image_root': paths['image_root'].format(**format_context)
            }
            
            # 验证路径是否存在 (保持不变)
            if not os.path.exists(source_dict['json_path']):
                print(f"Warning: JSON path not found for '{source_key}': {source_dict['json_path']}. Skipping source.")
                continue
            if not os.path.exists(source_dict['image_root']):
                print(f"Warning: Image root not found for '{source_key}': {source_dict['image_root']}. Skipping source.")
                continue

            all_data_sources.append(source_dict)
            print(f"  > Added data source: '{source_key}'")
        
        except KeyError as e:
            # [修改 5] 提供更详细的错误信息
            if str(e) == "'data_root'":
                print(f"Error: Data source '{source_key}' is still using the old '{{data_root}}' format.", file=sys.stderr)
                print(f"  > Please update config to use '{{data_root1}}' or '{{data_root2}}'.", file=sys.stderr)
            elif str(e) in ("'json_path'", "'image_root'"):
                print(f"Error: Missing 'json_path' or 'image_root' key for data source '{source_key}' in config.", file=sys.stderr)
            else:
                # 捕获其他缺失的键 (例如, 配置文件中写了 {data_root3})
                print(f"Error: Missing format key {e} for data source '{source_key}'. Check config.", file=sys.stderr)
            sys.exit(1)
            
        except Exception as e:
            print(f"Error processing paths for '{source_key}': {e}", file=sys.stderr)
            sys.exit(1)

    # 检查是否成功加载了任何数据源 (保持不变)
    if not all_data_sources:
        print("Error: No valid data sources were loaded. Please check config file paths and contents.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Successfully configured {len(all_data_sources)} data sources.")

    # --- 5. Determine Output Path ---
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dynamically generate the experiment name
    sampling_config = config['sampling_config']
    ratio_parts = [
        f"{name[0]}{int(cfg['ratio'] * 10)}"
        for name, cfg in sampling_config.items()
        if cfg.get('ratio', 0) > 0 # Safely check ratio
    ]
    ratio_parts.sort()
    ratio_str = "".join(ratio_parts)

    # Build experiment name using safe .get() calls
    exp_name = (
        f"{config.get('exp_name', 'robomap_exp')}_"
        f"ratio-{ratio_str}_"
        f"loss-{config.get('loss_type', 'unknown')}_"
        f"up-{config.get('upsample_method', 'unknown')}_"
        f"bs{config.get('bs', 'N')}_"
        f"lr{config.get('lr', 'N')}_"
        f"epochs{config.get('num_train_epochs', 'N')}"
    )

    # Ensure output_dir exists in config
    base_output_dir_config = config.get("output_dir", "./outputs") # Default to ./outputs
    base_output_dir = os.path.expanduser(base_output_dir_config) # Handle '~'
    output_path = os.path.join(base_output_dir, exp_name, current_time)

    print(f"Experiment name: {exp_name}")
    print(f"Base output directory: {base_output_dir}")
    print(f"Full output path for this run: {output_path}")

    # Ensure output_path exists (creates intermediate dirs if needed)
    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {output_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 6. Initialize and Launch Pretraining Pipeline ---
    try:
        print("\nInitializing pretraining pipeline...")
        pipeline = Pretrain_RoboMAP_Palligemma(
            pretrain=True,
            config_path=args.config_path # Pipeline loads its own config
        )

        print("\nStarting pipeline.pretrain()...")

        # Determine freeze_vision_tower from config, default to True
        freeze_vision_tower = config.get("freeze_vision_tower", True)

        pipeline.pretrain(
            args=args,
            output_path=output_path, # Pass the calculated output path
            data_sources=all_data_sources, # Pass the validated data sources
            freeze_vision_tower=freeze_vision_tower
        )

        print("\n--- Pretraining finished successfully. ---")

    except Exception as e:
        print(f"\n--- An error occurred during pipeline execution: ---", file=sys.stderr)
        import traceback
        traceback.print_exc()
        print(f"----------------------------------------------------", file=sys.stderr)
        sys.exit(1)
