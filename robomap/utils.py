# robomap/utils.py

import os
import textwrap
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from transformers import Trainer, TrainerCallback, AutoProcessor, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from torch.utils.data import Sampler

# ===================================================================
#                      1. Data Processing Utils
# ===================================================================

def generate_hm_from_pt(pt, res, sigma, thres_sigma_times=3):
    """
    Pytorch code to generate heatmaps from point. Points with values less than
    thres are made 0
    :type pt: torch.FloatTensor of size (num_pt, 2)
    :type res: int or (int, int)
    :param sigma: scalar standard deviation, or ``(sigma_x, sigma_y)`` for
        an anisotropic Gaussian. If it is -1, generate a one-hot heatmap.
    :type sigma: float
    :type thres: float
    """
    num_pt, x = pt.shape
    assert x == 2

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    _hmx = torch.arange(0, resy).to(pt.device)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    _hmy = torch.arange(0, resx).to(pt.device)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1)

    pt = pt.view([num_pt, 1, 1, 2])
    if isinstance(sigma, (tuple, list)):
        sigma_tensor = torch.tensor(
            sigma, dtype=hm.dtype, device=hm.device
        ).view(1, 1, 1, 2)
        normalized_distance = (hm - pt) / sigma_tensor
        hm = torch.exp(-0.5 * torch.sum(normalized_distance**2, -1))
    else:
        hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2)))
    thres = np.exp(-1 * (thres_sigma_times**2) / 2)
    hm[hm < thres] = 0.0

    hm /= torch.sum(hm, (1, 2), keepdim=True) + 1e-6

    # TODO: make a more efficient version
    if sigma == -1:
        _hm = hm.view(num_pt, resx * resy)
        hm = torch.zeros((num_pt, resx * resy), device=hm.device)
        temp = torch.arange(num_pt).to(hm.device)
        hm[temp, _hm.argmax(-1)] = 1

    return hm


# ===================================================================
#                      2. Model & Weight Utils
# ===================================================================

def load_all_params(checkpoint_dir: str) -> dict:
    """
    Loads model parameters from a checkpoint directory.
    
    This function handles loading standard Hugging Face checkpoints,
    DeepSpeed checkpoints, etc., and returns a single state_dict.
    
    Args:
        checkpoint_dir (str): Path to the checkpoint directory.
        
    Returns:
        dict: A consolidated state_dict.
    """
    
    # --- TODO: Replace this with your actual weight loading logic ---
    # This is a placeholder for loading a standard HF checkpoint.
    # If you use DeepSpeed or FSDP, you will need a more complex
    # consolidation script here.
    
    hf_checkpoint_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    if os.path.exists(hf_checkpoint_file):
        print(f"[load_all_params] Loading standard HF checkpoint from: {hf_checkpoint_file}")
        return torch.load(hf_checkpoint_file, map_location="cpu")
    else:
        # Add logic for DeepSpeed/FSDP sharded weights if needed
        raise FileNotFoundError(
            f"Could not find 'pytorch_model.bin' in {checkpoint_dir}. "
            f"Please implement your specific checkpoint loading logic in 'robomap/utils.py'."
        )
    # --- End of TODO section ---


def count_parameters(model: nn.Module, module_name: str = ""):
    """
    Counts and prints the total and trainable parameters of a model module.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n--- Parameter Count for: {module_name} ---")
    print(f"Total params:     {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")
    print(f"----------------------------------------")


def init_weights(m: nn.Module):
    """
    Applies Kaiming (Conv) or Xavier (Linear) initialization to a module.
    This is typically applied to newly added layers (like the decoder head).
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ===================================================================
#                      3. Visualization Utils
# ===================================================================
def visualize_all_in_one(
    original_image: Image.Image,
    low_res_map: torch.Tensor,
    predicted_heatmap: torch.Tensor,
    gt_heatmap: torch.Tensor,
    save_path: str,
    prompt: str = ""
) -> None:
    """
    Generates and saves a 2x2 grid visualizing the model's performance.
    
    The grid contains:
    - (0, 0): Original Image
    - (0, 1): PCA visualization of the 16x16 low-res feature map
    - (1, 0): Predicted heatmap (spatial softmax) overlaid on the image
    - (1, 1): Ground truth heatmap overlaid on the image
    
    Args:
        original_image: The original PIL.Image.
        low_res_map: The 16x16 feature map from the model.
            Shape: (1, C, H, W) e.g., (1, 4096, 16, 16)
        predicted_heatmap: The raw logits from the heatmap decoder.
            Shape: (1, 1, 224, 224)
        gt_heatmap: The ground truth heatmap tensor.
            Shape: (1, 224, 224) or (224, 224)
        save_path: The full path to save the resulting PNG file.
        prompt: The text prompt used for this sample.
    """
    
    # --- 1. Process Images and Feature Maps ---
    
    # Resize original image
    image = original_image.convert("RGB").resize((224, 224))
    
    # PCA for 16x16 feature map
    B, C, H, W = low_res_map.shape
    tokens_reshaped = low_res_map.squeeze(0).permute(1, 2, 0).reshape(H * W, C).cpu().detach().float().numpy()
    
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(tokens_reshaped)
    # Normalize PCA features to [0, 1] for visualization
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    
    pca_map = pca_features.reshape(H, W, 3)
    # Upsample PCA map to 224x224 using nearest-neighbor to show the blocks
    pca_map_upsampled = cv2.resize(pca_map, (224, 224), interpolation=cv2.INTER_NEAREST)

    # --- 2. Process Predicted Heatmap ---
    
    # Squeeze and convert to numpy
    pred_heatmap_np = predicted_heatmap.squeeze().cpu().detach().float().numpy()
    
    # Apply a spatial softmax to normalize the logits into a probability distribution
    pred_heatmap_flat = pred_heatmap_np.flatten()
    pred_heatmap_softmax = np.exp(pred_heatmap_flat) / np.sum(np.exp(pred_heatmap_flat))
    pred_heatmap_softmax = pred_heatmap_softmax.reshape(224, 224)

    # --- 3. Process Ground Truth Heatmap ---
    
    gt_heatmap_np = gt_heatmap.squeeze().cpu().detach().float().numpy()
    # Normalize GT heatmap to [0, 1] for visualization
    if gt_heatmap_np.max() > 0:
        gt_heatmap_np = (gt_heatmap_np - gt_heatmap_np.min()) / (gt_heatmap_np.max() - gt_heatmap_np.min())

    # --- 4. Plotting ---
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    if prompt:
        fig.suptitle(textwrap.fill(f'Prompt: "{prompt}"', width=70), fontsize=14, y=0.95)
    
    # Top-left: Original Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Top-right: PCA of 16x16 Feature Map
    axes[0, 1].imshow(pca_map_upsampled)
    axes[0, 1].set_title('16x16 Feature Map (PCA)')
    axes[0, 1].axis('off')

    # Bottom-left: Predicted Heatmap
    axes[1, 0].imshow(image)
    axes[1, 0].imshow(pred_heatmap_softmax, cmap='jet', alpha=0.5)
    axes[1, 0].set_title('Predicted Heatmap (Softmax)')
    axes[1, 0].axis('off')

    # Bottom-right: Ground Truth Heatmap
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(gt_heatmap_np, cmap='jet', alpha=0.5)
    axes[1, 1].set_title('Ground Truth Heatmap')
    axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust for suptitle
    plt.savefig(save_path, dpi=300)
    plt.close()


# ===================================================================
#                      4. Trainer & Callback Utils
# ===================================================================

class VisualizationCallback(TrainerCallback):
    """
    A callback that generates and saves visualizations periodically during training.
    [Updated] Directly uses pre-calculated heatmaps provided by the Dataset.
    """
    def __init__(self, processor, test_dataset, output_dir, every_n_steps=500):
        self.processor = processor
        
        # Select a fixed set of indices for consistent observation
        self.vis_indices = [0, 50, 100, 150, 200, 250, 300, 350, 400] 
        self.vis_indices = [i for i in self.vis_indices if i < len(test_dataset)]
        
        # self.test_samples is now a list containing multiple sample dictionaries
        self.test_samples = [test_dataset[i] for i in self.vis_indices]
        
        self.output_dir = os.path.join(output_dir, "visualizations")
        self.every_n_steps = every_n_steps
        os.makedirs(self.output_dir, exist_ok=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero and state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            print(f"\n--- Generating visualizations for step {state.global_step} ({len(self.test_samples)} samples) ---")
            
            model.eval() # Switch to evaluation mode

            for i, sample in enumerate(self.test_samples):
                
                image = sample["image"]
                prompt = sample["prompt"]
                
                # --- [CoreChange] Directly get the pre-calculated ground truth heatmap from the sample ---
                # Our new Dataset returns 'heatmap' already as a (1, H, W) tensor
                gt_heatmap_tensor = sample["heatmap"]

                # --- Model Inference (Unchanged) ---
                inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    # Call model forward, but don't pass heatmaps, so loss isn't calculated
                    outputs = model(**inputs)

                # --- Call Visualization Function (Unchanged) ---
                save_path = os.path.join(self.output_dir, f"step_{state.global_step}_sample_{self.vis_indices[i]}.png")
                
                visualize_all_in_one(
                    original_image=image,
                    low_res_map=outputs["low_res_feature_map"],
                    predicted_heatmap=outputs["predicted_heatmap"],
                    gt_heatmap=gt_heatmap_tensor, # <--- Pass the ground truth tensor obtained directly from the sample
                    save_path=save_path,
                    prompt=prompt
                )
            
            model.train() # Switch back to training mode
            print(f"--- Visualizations saved to: {self.output_dir} ---\n")


class BalancedTrainer(Trainer):
    """
    A custom Hugging Face Trainer that accepts a pre-initialized
    `train_sampler` in its constructor.
    
    This is necessary for using a WeightedRandomSampler, as the default
    Trainer logic creates its own sampler which would override it.
    """
    def __init__(self, *args, train_sampler: Optional[Sampler] = None, **kwargs):
        """
        Initializes the Trainer.
        
        Args:
            *args: Positional arguments passed to the base Trainer.
            train_sampler (Sampler, optional): The pre-initialized sampler
                (e.g., WeightedRandomSampler) to be used for training.
            **kwargs: Keyword arguments passed to the base Trainer.
        """
        super().__init__(*args, **kwargs)
        # Store the custom sampler in a unique attribute
        self.custom_train_sampler = train_sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        Overrides the default method to create the training DataLoader.
        
        This is the core modification: it explicitly uses the
        `self.custom_train_sampler` provided during __init__ instead of
        letting the Trainer create a default one.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        # This is the key change:
        # We build the DataLoader using our custom sampler.
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=self.custom_train_sampler, # <-- Use our custom sampler
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
