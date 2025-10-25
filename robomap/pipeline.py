# robomap/pipeline.py

# --- 1. Standard Library Imports ---
import os
import yaml
from typing import List, Dict, Any

# --- 2. Third-Party Imports ---
import torch
from torch.utils.data import WeightedRandomSampler
from transformers import TrainingArguments, AutoProcessor

# --- 3. Local (robomap) Imports ---
# Import models, datasets, and utils from other files in this package
from .model import RoboMAP_Paligemma
from .dataset import RoboMAPDataset, DataCollator
from .utils import (
    load_all_params,
    count_parameters,
    init_weights,
    VisualizationCallback,
    BalancedTrainer
)


class Pretrain_RoboMAP_Palligemma:
    """
    The main pipeline class for training and evaluating the RoboMAP model.
    
    This class handles:
    - Loading configurations and models.
    - Setting up datasets and data samplers.
    - Configuring the Hugging Face Trainer.
    - Running the training loop (`pretrain`).
    - Running inference (`test_inference`).
    """
    def __init__(self, pretrain: bool, config_path: str):
        """
        Initializes the pipeline.
        
        Args:
            pretrain (bool): True if in training mode, False if in inference mode.
            config_path (str): Path to the main YAML configuration file.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # --- [CRITICAL REFACTOR] Load paths from config, DO NOT hardcode ---
        # Ensure these keys exist in your YAML file (e.g., config.example.yaml)
        self.base_model_id = self.config.get("base_model_path", "/mnt/diskc/shaoxy/model/paligemma-3b-mix-224")
        self.finetuned_checkpoint_path = self.config.get("finetuned_checkpoint_path", None)
        
        print(f"[__init__] Loading Processor from base model: {self.base_model_id}")
        self.processor = AutoProcessor.from_pretrained(self.base_model_id)
        print("[__init__] Processor loaded successfully.")

        if not pretrain:
            # --- Inference Mode Setup ---
            self.device = "cuda"
            # Use 'checkpoint_dir' from config for inference
            checkpoint_path = self.config.get("checkpoint_dir", None)
            if not checkpoint_path:
                raise ValueError("'checkpoint_dir' not specified in config for inference mode.")

            print(f"[Inference Mode] Loading base model: {self.base_model_id}")
            self.pretrained_model = RoboMAP_Paligemma.from_pretrained(
                self.base_model_id, trust_remote_code=True
            )
            
            print(f"[Inference Mode] Loading and applying fine-tuned weights from: {checkpoint_path}")
            all_params = load_all_params(checkpoint_path)
            missing_keys, unexpected_keys = self.pretrained_model.load_state_dict(
                all_params, strict=False
            )
            
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            del all_params
            
            self.pretrained_model.to(self.device)
            print("[Inference Mode] Model ready.")

    # (You removed 'test_inference' in your last prompt, so it is omitted here)

    def pretrain(self, args: Any, output_path: str, data_sources: List[Dict], 
                 freeze_vision_tower: bool = True):
        """
        Main training function.
        Sets up datasets, weighted sampler, model, and trainer, then starts training.
        """
        
        # --- 1. Setup Datasets ---
        # Use self.config loaded during __init__
        sampling_config = self.config['sampling_config']
        
        ratio_parts = [f"{name[0]}{int(cfg['ratio'] * 10)}" for name, cfg in sampling_config.items()]
        ratio_parts.sort()
        ratio_str = "".join(ratio_parts)

        # Create a dynamic cache directory name based on sampling ratios
        dynamic_cache_dir = f"./data/processed_alldata_{ratio_str}"
        print(f"--- Using dynamic cache directory: {dynamic_cache_dir} ---")

        print("--- Loading training dataset ---")
        train_dataset = RoboMAPDataset(
            data_sources=data_sources,
            mode='train',
            test_split_ratio=0.005, # Use a small split for test
            cache_dir=dynamic_cache_dir 
        )

        print("\n--- Loading test dataset (for visualization) ---")
        test_dataset = RoboMAPDataset(
            data_sources=data_sources,
            mode='test',
            cache_dir=dynamic_cache_dir 
        )
        print("-" * 20)

        collate_fn = DataCollator(self.processor)

        # --- 2. Setup Weighted Sampler ---
        print("\n--- Creating universal weighted sampler for training set ---")
        
        total_ratio = sum(cfg['ratio'] for cfg in sampling_config.values())
        if not 0.999 < total_ratio < 1.001:
            print(f"[Warning] Sum of sampling ratios in config is {total_ratio:.2f}, "
                  f"not 1.0. Please check your sampling_config.")

        # Dynamically separate indices for all data sources
        source_indices = {name: [] for name in sampling_config}
        for i, sample in enumerate(train_dataset.samples):
            sample_flag = sample['flag']
            for name, config in sampling_config.items():
                if sample_flag in config['flags']:
                    source_indices[name].append(i)
                    break # Sample matched to a source

        source_counts = {name: len(indices) for name, indices in source_indices.items()}

        manual_epoch_length = self.config.get("num_samples_per_epoch", "auto")

        if isinstance(manual_epoch_length, int) and manual_epoch_length > 0:
            num_samples_per_epoch = manual_epoch_length
            print(f"\n--- [Manual] Using epoch length from config: {num_samples_per_epoch} ---")
        else:
            print("\n--- [Auto] 'num_samples_per_epoch' not specified. Calculating automatically... ---")
            base_dataset_name = 'robodata' # Base dataset for ratio calculation
            base_count = source_counts.get(base_dataset_name, 0)

            if base_count == 0:
                num_samples_per_epoch = len(train_dataset)
                print(f"[Warning] Base dataset '{base_dataset_name}' has 0 samples. "
                      f"Using total dataset length {num_samples_per_epoch} as epoch length.")
            else:
                epoch_ratio = self.config.get("epoch_ratio", 2.0)
                num_samples_per_epoch = int(base_count * float(epoch_ratio))
                print(f"--- Auto-calculated epoch length: {num_samples_per_epoch} "
                      f"( = '{base_dataset_name}' count {base_count} * {epoch_ratio}) ---")

        print("\n--- Data Source Statistics ---")
        for name, count in source_counts.items():
            print(f" - {name} (flags: {sampling_config[name]['flags']}): {count} samples")

        # Calculate weights for all samples
        weights = torch.zeros(len(train_dataset), dtype=torch.float)
        print("\n--- Calculating Sample Weights ---")
        for name, config in sampling_config.items():
            count = source_counts[name]
            ratio = config['ratio']
            indices = source_indices[name]
            
            if count > 0:
                # Weight per sample = (total_ratio_for_source / num_samples_in_source)
                weight_per_sample = ratio / count
                print(f"  [Calculating] Source '{name}':")
                print(f"    - Target Ratio: {ratio}")
                print(f"    - Sample Count: {count}")
                print(f"    - Calculated weight_per_sample: {weight_per_sample:.8f}")
                weights[torch.tensor(indices, dtype=torch.long)] = weight_per_sample
            else:
                print(f"  [Info] Source '{name}' has 0 samples in the training set. Skipping weight assignment.")
        
        print("\n--- Weight Tensor Sanity Check ---")
        num_nonzero_weights = torch.count_nonzero(weights)
        print(f"- Number of non-zero weights: {num_nonzero_weights} / {len(train_dataset)}")
        total_weight_sum = weights.sum()
        print(f"- Sum of all weights: {total_weight_sum.item():.4f} (Should be ~1.0)")
        print(f"- Contains negative weights: {torch.any(weights < 0)}")
        print(f"- Contains NaN values: {torch.isnan(weights).any()}")

        if weights.sum() <= 0:
            print(f"\n[Fatal Error] Sum of weights is {weights.sum()}. Cannot create sampler.")
            raise ValueError("Sum of weights must be positive to create a valid sampler.")
        else:
            sampler = WeightedRandomSampler(
                weights=weights, 
                num_samples=num_samples_per_epoch, 
                replacement=True
            )
            print("\n--- Universal weighted sampler created successfully ---")

        # --- 3. Load Model ---
        print(f"\n--- [Step 1] Loading full model architecture and weights from base: {self.base_model_id} ---")
        model = RoboMAP_Paligemma.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True # Required for custom RoboMAP_Paligemma class
        )
        print("--- Base model loaded successfully ---")

        if self.finetuned_checkpoint_path is not None:
            print(f"\n--- [Step 2] Loading and overriding weights from fine-tuned checkpoint: {self.finetuned_checkpoint_path} ---")
            finetuned_params = load_all_params(self.finetuned_checkpoint_path)
            
            missing_keys, unexpected_keys = model.load_state_dict(finetuned_params, strict=False)
            
            print("--- Weight override complete ---")
            print(f"[Info] {len(missing_keys)} keys were in the model but not in the checkpoint (expected for decoder):")
            print(f"  -> Missing Keys (sample): {missing_keys[:5]}...")
            print(f"[Info] {len(unexpected_keys)} keys were in the checkpoint but not in the model (should be empty):")
            print(f"  -> Unexpected Keys: {unexpected_keys}")
            del finetuned_params
        else:
            print("[Info] No fine-tuned checkpoint provided. Training from base model.")
        
        # --- 4. Configure Model for Training ---
        
        # Count parameters of the new head
        count_parameters(model.heatmap_decoder, module_name="Upsampling Head (model.heatmap_decoder)")

        # Cast heatmap decoder to float32 for stability
        model.heatmap_decoder = model.heatmap_decoder.to(torch.float32)
        model.gradient_checkpointing_enable()

        # Apply custom weight initialization ONLY to the new decoder head
        print("\n--- Applying custom weight initialization to the new decoder head (model.heatmap_decoder) ---")
        model.heatmap_decoder.apply(init_weights)
        print("--- Decoder head initialization complete ---")

        # Freeze specified parts of the model
        freeze_names = ["lm_head", "embed_tokens"]
        if freeze_vision_tower:
            freeze_names.append("vision_tower")
        
        print(f"--- Freezing parameters containing names: {freeze_names} ---")
        for name, param in model.named_parameters():
            if any(freeze_name in name for freeze_name in freeze_names):
                param.requires_grad_(False)
                
        # --- 5. Setup Trainer ---
        lr = float(self.config["lr"])
        bs = self.config["bs"]

        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            learning_rate=lr,
            warmup_steps=self.config["warmup_steps"],
            lr_scheduler_type="cosine",    # Use cosine decay scheduler
            max_grad_norm=1.0,             # Enable gradient clipping
            weight_decay=0.1,              # Increase weight decay for regularization
            optim="adamw_torch_fused",     # Use the fused optimizer
            bf16=True,                     # Enable BF16 mixed precision
            logging_steps=self.config["logging_steps"],
            logging_strategy="steps",
            save_strategy="steps",
            save_steps=self.config["save_steps"],
            save_total_limit=self.config["save_total_limit"],
            dataloader_num_workers=self.config["dataloader_num_workers"],
            dataloader_pin_memory=True,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            ddp_find_unused_parameters=False
        )

        # Instantiate visualization callback
        vis_callback = VisualizationCallback(
            processor=self.processor,
            test_dataset=test_dataset,
            output_dir=training_args.output_dir,
            every_n_steps=self.config.get("visualization_steps", 250)
        )
            
        # Use the custom BalancedTrainer
        trainer = BalancedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
            callbacks=[vis_callback],
            train_sampler=sampler  # Pass the custom weighted sampler
        )
        
        print("\n--- Starting Training ---")
        trainer.train()
        print("--- Training Finished ---")
