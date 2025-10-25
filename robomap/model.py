# robomap/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

# Third-party libraries
from einops import rearrange
from transformers import PaliGemmaForConditionalGeneration


class AdaptiveHeatmapDecoder(nn.Module):
    """
    An adaptive heatmap decoder based on the "ConvexUpSample" principle.
    
    This module upsamples a low-resolution feature map into a high-resolution
    heatmap. It follows a parallel "What" and "How" branch structure:
    1.  The "What" branch determines the content (M_low) to be upsampled.
    2.  The "How" branch dynamically generates the upsampling kernels (W) 
        for each spatial location.
    
    This version introduces a bottleneck layer (1x1 conv) at the
    beginning of each branch. This significantly reduces the parameter count
    and computational cost, making it a parameter-efficient replacement
    for earlier versions.
    """
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 up_ratio: int, 
                 up_kernel: int = 3, 
                 bottleneck_dim: int = 128, 
                 mask_scale: float = 0.1):
        """
        Initializes the AdaptiveHeatmapDecoder module.

        Args:
            in_dim (int): Number of input channels (e.g., 3072).
            out_dim (int): Number of output channels (e.g., 1 for a single heatmap).
            up_ratio (int): The upsampling factor (e.g., 14).
            up_kernel (int): The size of the local kernel for unfolding (e.g., 3).
                             Must be an odd number.
            bottleneck_dim (int): The intermediate channel dimension after the
                                  1x1 conv bottleneck (e.g., 128 or 256).
            mask_scale (float): A scaling factor for the kernel-generating
                                branch before softmax.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.up_ratio = up_ratio
        self.up_kernel = up_kernel
        self.mask_scale = mask_scale
        
        assert (self.up_kernel % 2) == 1, "up_kernel must be an odd number."

        # --- Bottleneck Layers (Parameter-Efficiency) ---
        # 1x1 convs to reduce channel dimension from in_dim to bottleneck_dim
        self.content_bottleneck = nn.Conv2d(in_dim, bottleneck_dim, kernel_size=1) # "What" branch
        self.kernel_bottleneck = nn.Conv2d(in_dim, bottleneck_dim, kernel_size=1)  # "How" branch

        # --- "What" Branch (Content Generation) ---
        # Predicts the low-resolution content (M_low) from the bottlenecked features
        self.content_net = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim * 2, out_dim, kernel_size=3, padding=1)
        )

        # --- "How" Branch (Kernel Generation) ---
        # Predicts the upsampling kernels (W) from the bottlenecked features
        # The output dimension is (up_ratio^2 * up_kernel^2)
        kernel_dim = (self.up_ratio**2) * (self.up_kernel**2)
        self.kernel_net = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_dim * 2, kernel_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input low-resolution features.
                              Shape: (bs, in_dim, h, w)
        
        Returns:
            torch.Tensor: The upsampled high-resolution heatmap.
                          Shape: (bs, out_dim, h * up_ratio, w * up_ratio)
        """
        bs, c, h, w = x.shape
        assert c == self.in_dim, f"Input channel {c} does not match in_dim {self.in_dim}"

        # 1. Apply bottlenecks to reduce feature dimension
        # (bs, in_dim, h, w) -> (bs, bottleneck_dim, h, w)
        x_content = self.content_bottleneck(x)
        x_kernel = self.kernel_bottleneck(x)

        # 2. Generate low-resolution content (M_low) and upsampling kernels (W)
        content_low_res = self.content_net(x_content)   # Shape: (bs, out_dim, h, w)
        upsample_kernels = self.kernel_net(x_kernel)    # Shape: (bs, kernel_dim, h, w)

        # --- The following logic performs the kernel-based upsampling ---

        # 3. Prepare upsampling kernels
        # kernel_dim = (up_ratio^2 * up_kernel^2)
        # Scale and apply softmax to ensure kernels sum to 1 (convex combination)
        upsample_kernels = self.mask_scale * upsample_kernels
        # (bs, kernel_dim, h, w) -> (bs, 1, k^2, r, r, h, w)
        upsample_kernels = upsample_kernels.view(
            bs, 1, self.up_kernel**2, self.up_ratio, self.up_ratio, h, w
        )
        upsample_kernels = torch.softmax(upsample_kernels, dim=2)
        
        # 4. Prepare local content patches (unfolding)
        # Extract local (up_kernel x up_kernel) patches from the low-res content
        content_unfolded = F.unfold(
            content_low_res,
            kernel_size=self.up_kernel,
            padding=self.up_kernel // 2
        )
        # Shape: (bs, out_dim * up_kernel^2, h*w)
        
        # Reshape to align for broadcasting with kernels
        # (bs, out_dim, k^2, 1, 1, h, w)
        content_patches = content_unfolded.view(
            bs, self.out_dim, self.up_kernel**2, 1, 1, h, w
        )

        # 5. Perform weighted sum (convex upsampling)
        # Broadcasting:
        #   content_patches:  (bs, out_dim, k^2, 1, 1, h, w)
        # * upsample_kernels: (bs, 1,       k^2, r, r, h, w)
        # -> Result:         (bs, out_dim, k^2, r, r, h, w)
        #
        # Summing over dim=2 (k^2) performs the weighted average for each
        # high-resolution pixel in the (r, r) patch.
        weighted_sum = torch.sum(content_patches * upsample_kernels, dim=2)
        # Shape: (bs, out_dim, r, r, h, w)

        # 6. Reshape to final high-resolution output
        # This is a "depth-to-space" or "pixel-shuffle" like operation
        
        # (bs, out_dim, r, r, h, w) -> (bs, out_dim, h, r, w, r)
        out_permuted = weighted_sum.permute(0, 1, 4, 2, 5, 3) 
        
        # (bs, out_dim, h, r, w, r) -> (bs, out_dim, h*r, w*r)
        output_high_res = out_permuted.reshape(
            bs, self.out_dim, h * self.up_ratio, w * self.up_ratio
        )

        return output_high_res
    

class RoboMAP_Paligemma(PaliGemmaForConditionalGeneration):
    """
    A custom PaliGemma model adapted for point-based robot tasks (RoboMAP).

    This class inherits from PaliGemmaForConditionalGeneration and appends a
    custom heatmap decoder (e.g., AdaptiveHeatmapDecoder) to the model.
    It processes inputs through the VLM, extracts the image patch features
    from the final hidden state, reshapes them into a 2D feature map,
    and then upsamples this map using the attached decoder to produce
    a high-resolution heatmap (logits).

    If ground truth 'heatmaps' are provided during the forward pass,
    it computes and returns a loss.
    """
    def __init__(self, config):
        """
        Initializes the model and the custom heatmap decoder head.

        Args:
            config: The model configuration object, expected to contain
                    PaliGemma configs and custom keys like 'upsample_method'
                    and 'bottleneck_dim'.
        """
        super().__init__(config)

        self.vlm_dim = config.hidden_size
        
        # --- Initialize the Heatmap Decoder Head ---
        upsample_method = getattr(config, 'upsample_method', 'convex') # Default to 'convex'
        
        if upsample_method == 'convex':
            # Read bottleneck_dim from config, default to 512
            bottleneck_dim = getattr(config, 'bottleneck_dim', 512)
            print(f"[RoboMAP_Paligemma] Initializing AdaptiveHeatmapDecoder "
                  f"(bottleneck_dim={bottleneck_dim}).")
            
            self.heatmap_decoder = AdaptiveHeatmapDecoder(
                in_dim=self.vlm_dim,
                out_dim=1,          # Output a single-channel heatmap
                up_ratio=14,        # Upsample from 16x16 to 224x224 (16*14=224)
                up_kernel=3,
                bottleneck_dim=bottleneck_dim
            )
        else:
            # Placeholder for other potential decoder types
            raise NotImplementedError(f"Upsample method '{upsample_method}' is not implemented.")

        # Architectural constants
        self.patch_grid_size = 16  # Assumes 16x16=256 patches per image

        # --- Loss Function ---
        # Note: Using CrossEntropyLoss for a (N, C, 1) target is non-standard.
        # This assumes the target heatmap contains soft labels or probabilities.
        # For standard 0/1 heatmaps, nn.BCEWithLogitsLoss is more common.
        self.heatmap_loss_fn = nn.CrossEntropyLoss(reduction="none")

        # --- Inference & Debug ---
        # Threshold for inference, not used in forward()
        self.final_thres = 0.1 
        # Counter for debugging visualizations, not used in forward()
        self.debug_counter = 0


    def forward(self, 
                input_ids: torch.Tensor, 
                pixel_values: torch.Tensor, 
                attention_mask: torch.Tensor, 
                heatmaps: Optional[torch.Tensor] = None, 
                **kwargs) -> Dict[str, Any]:
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Token IDs.
                Shape: (batch_size, seq_len)
            pixel_values (torch.Tensor): Input images, pre-processed.
                Shape: (batch_size * num_images, 3, 224, 224)
            attention_mask (torch.Tensor): Attention mask for token IDs.
                Shape: (batch_size, seq_len)
            heatmaps (Optional[torch.Tensor]): Ground truth heatmaps for training.
                Shape: (batch_size, 1, 224, 224)
            **kwargs: Additional arguments passed to the super().forward()

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "predicted_heatmap": The raw logits from the decoder.
                  Shape: (batch_size, num_images, 224, 224)
                - "low_res_feature_map": The 2D feature map fed to the decoder.
                  Shape: (batch_size * num_images, vlm_dim, 16, 16)
                - "loss": (Optional) The computed heatmap loss if 'heatmaps'
                  was provided.
        """
        
        # --- 1. Get VLM Hidden States ---
        # Pass inputs through the base PaliGemma model
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True  # Ensure we get all hidden states
        )
        
        # Get the last hidden state
        hidden_states = outputs.hidden_states
        vlm_output = hidden_states[-1]  # Shape: (batch_size, seq_len, vlm_dim)

        # --- 2. Extract and Reshape Image Tokens ---
        image_token_features_list = []
        batch_size, _, H_img, W_img = pixel_values.shape
        
        # TODO: This assumes num_images=1, which is common. 
        # If num_images > 1, this logic needs review.
        num_images = 1 
        
        # Calculate the expected number of tokens per image
        num_tokens_per_image = self.patch_grid_size * self.patch_grid_size # 16*16 = 256

        # Loop through batch to handle variable-length padding
        for i in range(batch_size):
            current_mask = attention_mask[i]
            current_output = vlm_output[i]
            
            # Find all non-padded tokens
            non_padded_indices = torch.nonzero(current_mask != 0, as_tuple=True)[0]
            non_padded_output = current_output[non_padded_indices]
            
            # This is a strong assumption:
            # It assumes the first 'num_tokens_per_image' tokens after
            # the prompt are the image tokens we want.
            expected_min_len = num_tokens_per_image * num_images
            assert non_padded_output.shape[0] >= expected_min_len, \
                f"Batch {i} has only {non_padded_output.shape[0]} non-padded tokens, " \
                f"but {expected_min_len} are required for {num_images} image(s)."
            
            # Extract the image tokens
            image_tokens = non_padded_output[:expected_min_len]
            image_token_features_list.append(image_tokens)

        # Stack the extracted tokens
        # Shape: (batch_size, 256, vlm_dim)
        image_token_features = torch.stack(image_token_features_list) 

        # Reshape flat tokens back into a 2D spatial feature map
        # (b, (1 h w), dim) -> (b, dim, 1, h, w)
        image_features_spatial = rearrange(
            image_token_features, 'b (c h w) d -> b d c h w', 
            c=num_images, h=self.patch_grid_size, w=self.patch_grid_size
        ).contiguous()

        # (b, dim, 1, h, w) -> (b, 1, dim, h, w) -> (b*1, dim, h, w)
        low_res_feature_map = (
            image_features_spatial.transpose(1, 2)
            .clone()
            .view(
                batch_size * num_images, self.vlm_dim, 
                self.patch_grid_size, self.patch_grid_size
            )
        )
        # Final shape: (batch_size * num_images, vlm_dim, 16, 16)

        # --- 3. Decode Heatmap ---
        # Pass low-res features through the adaptive decoder
        # (b*num_img, vlm_dim, 16, 16) -> (b*num_img, 1, 224, 224)
        heatmap_logits = self.heatmap_decoder(low_res_feature_map)

        # Clamp logits for numerical stability before loss calculation
        heatmap_logits = torch.clamp(heatmap_logits, min=-15, max=15)
        
        # Reshape back to match batch_size and num_images
        # (b*num_img, 1, 224, 224) -> (b, num_img, 224, 224)
        heatmap_logits_reshaped = heatmap_logits.view(batch_size, num_images, H_img, W_img)

        # --- 4. Calculate Loss (if training) ---
        loss = None
        if heatmaps is not None:
            # Reshape logits to (bs, num_pixels, 1)
            # This is a non-standard shape for CrossEntropyLoss
            logits_flat = heatmap_logits.view(
                batch_size * num_images, 1, H_img * W_img
            ).transpose(1, 2)
            # Shape: (bs*num_img, 50176, 1)

            # Prepare ground truth targets
            # Assuming 'heatmaps' shape is (bs, 1, 224, 224)
            # and num_images is 1
            targets_flat = heatmaps.view(
                batch_size, 1, H_img * W_img
            ).transpose(1, 2).clone().to(logits_flat.device)
            # Shape: (bs, 50176, 1)
            
            # Calculate loss.
            # This loss fn expects logits and targets of this specific shape.
            loss_per_pixel = self.heatmap_loss_fn(logits_flat, targets_flat)
            loss = loss_per_pixel.mean()

        # --- 5. Prepare Output ---
        output = {
            # Return the raw logits; activation (e.g., sigmoid)
            # should be applied during inference.
            "predicted_heatmap": heatmap_logits_reshaped,
            "low_res_feature_map": low_res_feature_map
        }
        
        if loss is not None:
            output["loss"] = loss
            
        return output
