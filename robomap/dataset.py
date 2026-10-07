# robomap/dataset.py

import os
import json
import ast
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import List, Dict, Any

# --- Third-party Libraries ---
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor

from .label_utils import shape_aware_sigmas
from .utils import generate_hm_from_pt

class RoboMAPDataset(Dataset):
    """
    A robust dataset class for RoboMAP.
    
    This class integrates loading, filtering, cleaning, stratified sampling,
    and caching for multiple data sources (RoboData, COCO, OXE, etc.).
    It intelligently parses different annotation formats, including normalized
    coordinates and pixel coordinates, and prepares them for model training.
    """
    def __init__(self, data_sources: list,
                 mode: str = 'train',
                 test_split_ratio: float = 0.01,
                 random_state: int = 42,
                 force_resplit: bool = False,
                 cache_dir: str = "processed_alldata_v1"):
        """
        Initializes the dataset.

        Args:
            data_sources (list): A list of data source configurations.
                Each element is a dict: {'json_path': str, 'image_root': str}.
            mode (str): 'train' or 'test'.
            test_split_ratio (float): The fraction of data to reserve for the test split.
            random_state (int): Random seed for reproducible train/test splits.
            force_resplit (bool): If True, ignore existing cache and re-process all data.
            cache_dir (str): Directory to store and load the processed/split data cache.
        """

        self.mode = mode
        os.makedirs(cache_dir, exist_ok=True)
        train_cache_path = os.path.join(cache_dir, "train_split.json")
        test_cache_path = os.path.join(cache_dir, "test_split.json")

        if force_resplit or not (os.path.exists(train_cache_path) and os.path.exists(test_cache_path)):
            print(f"Cache not found or 'force_resplit' is True. Starting full data processing pipeline...")
            all_samples = self._load_and_parse_all(data_sources)
            
            # Stratify split by data source to ensure both train and test
            # have a representative sample of all data types.
            stratify_labels = [sample.get('source', 'unknown') for sample in all_samples]
            train_samples, test_samples = train_test_split(
                all_samples, 
                test_size=test_split_ratio, 
                random_state=random_state, 
                stratify=stratify_labels
            )
            
            print(f"Saving splits to '{cache_dir}' directory...")
            with open(train_cache_path, 'w') as f: json.dump(train_samples, f, indent=4)
            with open(test_cache_path, 'w') as f: json.dump(test_samples, f, indent=4)
            print("Cache saved successfully.")
        else:
            print(f"Detected existing cache in '{cache_dir}'. Loading...")

        load_path = train_cache_path if self.mode == 'train' else test_cache_path
        if self.mode in ['train', 'test']:
            with open(load_path, 'r') as f:
                self.samples = json.load(f)
            print(f"Successfully loaded {self.mode} split with {len(self.samples)} samples.")
        else:
            raise ValueError(f"Invalid mode. Must be 'train' or 'test', but got '{self.mode}'")
        
        # --- Define Augmentation Pipeline ---
        # Only apply augmentations in 'train' mode
        if mode == 'train':
            self.augmentation_pipeline = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
            ])
        else:
            # For 'test' or 'validation' mode, no random augmentations
            self.augmentation_pipeline = None

    def _load_and_parse_all(self, data_sources: list) -> list:
        """
        Helper function to load, parse, filter, and clean all data sources.
        """
        temp_samples = []
        # Use defaultdict for convenient counting of new flags
        flag_counts = defaultdict(int)

        print(f"Loading, validating, and parsing raw annotations from {len(data_sources)} data sources...")
        for source_info in data_sources:
            json_path = source_info['json_path']
            image_root = source_info.get('image_root') # Use .get() for safety

            # If image_root is not provided, default to the JSON file's directory
            if not image_root:
                image_root = os.path.dirname(json_path)
                print(f"[Info] 'image_root' not provided for '{os.path.basename(json_path)}', "
                      f"defaulting to its directory: {image_root}")

            print(f"--> Processing file: {json_path}")
            print(f"    > Image Root: {image_root}")
            
            try:
                with open(json_path, "r") as f:
                    raw_data = json.load(f)
                
                parser_func = None
                filename_lower = os.path.basename(json_path).lower()

                # Dispatch to the correct parser based on filename conventions
                if 'detection' in filename_lower:
                    parser_func = self._parse_coco_entry
                elif any(keyword in filename_lower for keyword in ['fractal', 'bc_z', 'bridge', 'language_table']):
                    parser_func = self._parse_oxe_entry
                elif 'roborefit' in filename_lower:
                    parser_func = self._parse_roborefit_entry   
                elif 'reasoning' in filename_lower: 
                    parser_func = self._parse_refspatial_entry
                else:
                    # Default parser
                    parser_func = self._parse_robodata_entry

                # Pass image_root to the specific parser function
                processed = parser_func(raw_data, flag_counts, image_root)
                temp_samples.extend(processed)
            except Exception as e:
                print(f"   [ERROR] Fatal error processing {json_path}: {e}. Skipping file.")

        # --- Print Final Statistics ---
        print("\n" + "="*50)
        print("✅ All files parsed and validated. Data statistics:")
        total_valid = 0
        for flag, count in flag_counts.items():
            print(f"  - {flag}: {count} valid samples")
            total_valid += count
        print(f"--- \n  - Total: {total_valid} clean and valid samples available.")
        print("="*50 + "\n")

        return temp_samples

    def _parse_robodata_entry(self, raw_data_list, flag_counts, image_root):
        """Parses the default robodata JSON format."""
        processed_samples = []
        # Filter out ambiguous prompts
        ambiguous_words = ["next to", "near", "around"]

        for entry in raw_data_list:
            if len(entry.get("conversations", [])) < 2:
                continue

            human_prompt_full = entry["conversations"][0]["value"]
            prompt_lower = human_prompt_full.lower()
            if any(word in prompt_lower for word in ambiguous_words):
                continue

            gpt_label_str = entry["conversations"][1]["value"]

            try:
                # Extract the core prompt, including the <image> tag
                core_prompt = human_prompt_full.split("Your answer should be")[0].strip()
                if "<image>" not in core_prompt:
                    continue # Skip if <image> tag is missing
            except IndexError:
                continue # Skip if format is unexpected

            temp_sample = {
                "id": entry.get("id", "unknown/unknown"),
                "image_path": os.path.join(image_root, entry["image"]),
                "prompt": core_prompt,
                "raw_label": gpt_label_str,
                "source": "robodata",
                "flag": "detection_0"
            }

            # --- Validation and Counting ---
            # Only add sample if coordinates are valid
            if self._parse_coordinates(temp_sample["raw_label"], temp_sample["flag"]):
                processed_samples.append(temp_sample)
                flag_counts[temp_sample["flag"]] += 1

        return processed_samples
    
    def _parse_coco_entry(self, raw_data_list, flag_counts, image_root):
        """Parses COCO-style JSON format."""
        processed_samples = []
        
        # String templates to identify and clean prompts
        matched_string1 = "<image>\nPlease provide the bounding box coordinate of the region this sentence describes: "
        matched_string1_ = "Please provide the bounding box coordinate of the region this sentence describes: "
        matched_string2 = " Format the result as a list of tuples, i.e. [(x1, y1, w1, h1), (x2, y2, w2, h2), ...], where x and y are the normalized pixel locations of the object centers, and w and h are the normalized object widths and heights. All values of x, y, w, and h should be between 0 and 1."

        for entry in raw_data_list:
            conversations = entry.get("conversations", [])
            for i in range(1, len(conversations), 2):
                text = conversations[i-1]["value"]
                gt_answer = conversations[i]["value"]
                prompt, flag = "", None

                if matched_string1 in text or matched_string1_ in text:
                    flag = 'detection_1'
                    prompt = text.replace(matched_string1, "").replace(matched_string1_, "").replace("<image>\n", "").strip()
                elif matched_string2 in text:
                    flag = 'detection_2'
                    prompt = text.replace(matched_string2, "").replace("<image>\n", "").strip()
                
                if flag:
                    temp_sample = {
                        "id": entry.get("id", "unknown/unknown"),
                        "image_path": os.path.join(image_root, entry["image"]),
                        "prompt": f"<image>\n{prompt}",
                        "raw_label": gt_answer,
                        "source": "coco",
                        "flag": flag
                    }
                    
                    # --- Validation and Counting ---
                    # [Optimization] Only call _parse_coordinates once.
                    if self._parse_coordinates(temp_sample["raw_label"], temp_sample["flag"]):
                        processed_samples.append(temp_sample)
                        flag_counts[temp_sample["flag"]] += 1
        
        return processed_samples
    
    def _parse_oxe_entry(self, raw_data_list, flag_counts, image_root):
        """
        Parses OXE (e.g., fractal) format JSON, which uses pixel coordinates.
        This revised version splits multi-turn conversations into multiple
        independent samples.
        """
        print(f">>> Using revised _parse_oxe_entry parser...")
        processed_samples = []
        for entry in raw_data_list:
            conversations = entry.get("conversations", [])
            
            # Ensure at least one pair of dialogues and an even number
            if len(conversations) < 2 or len(conversations) % 2 != 0:
                print(f"--- Skipping malformed entry (ID: {entry.get('id', 'N/A')}), "
                      f"insufficient or odd number of conversation turns.")
                continue
            
            # Iterate through each human-gpt pair
            for i in range(0, len(conversations), 2):
                human_turn = conversations[i]
                gpt_turn = conversations[i+1]

                if human_turn.get("from") != "human" or gpt_turn.get("from") != "gpt":
                    continue # Skip if roles are not human/gpt

                human_prompt_full = human_turn["value"]
                gpt_label_str = gpt_turn["value"]
                core_prompt = human_prompt_full # Keep the full prompt

                # Generate a unique ID for each split sample
                sample_id = f"{entry.get('id', 'oxe/unknown')}_{i // 2}"

                temp_sample = {
                    "id": sample_id,
                    "image_path": os.path.join(image_root, entry["image"]),
                    "prompt": f"<image>\n{core_prompt}",
                    "raw_label": gpt_label_str,
                    "source": "oxe",
                    "flag": "oxe_action_points"
                }
                
                # Validate coordinate format
                if self._parse_coordinates(temp_sample["raw_label"], temp_sample["flag"]):
                    processed_samples.append(temp_sample)
                    flag_counts[temp_sample["flag"]] += 1
                    
        return processed_samples
    
    def _parse_refspatial_entry(self, raw_data_list, flag_counts, image_root):
        """Parses RefSpatial format JSON."""
        print(f">>> Using _parse_refspatial_entry parser...")
        processed_samples = []
        for entry in raw_data_list:
            if len(entry.get("conversations", [])) < 2:
                continue
            
            human_prompt_full = entry["conversations"][0]["value"]
            gpt_label_str = entry["conversations"][1]["value"]
            core_prompt = human_prompt_full # Keep the full prompt

            image_filename = entry['image']

            # Handle cases where image_filename is a list
            if isinstance(image_filename, list):
                image_filename = image_filename[0]

            temp_sample = {
                "id": entry.get("id", "refspatial/unknown"),
                "image_path": os.path.join(image_root, image_filename),
                "prompt": f"<image>\n{core_prompt}",
                "raw_label": gpt_label_str,
                "source": "refspatial",
                "flag": "refspatial"
            }
            
            if self._parse_coordinates(temp_sample["raw_label"], temp_sample["flag"]):
                processed_samples.append(temp_sample)
                flag_counts[temp_sample["flag"]] += 1
                
        return processed_samples
    
    def _parse_roborefit_entry(self, raw_data_list, flag_counts, image_root):
        """Parses Roborefit format JSON, which uses pixel coordinates."""
        print(f">>> Using _parse_roborefit_entry parser...")
        processed_samples = []
        for entry in raw_data_list:
            if len(entry.get("conversations", [])) < 2:
                continue
            
            human_prompt_full = entry["conversations"][0]["value"]
            gpt_label_str = entry["conversations"][1]["value"]
            core_prompt = human_prompt_full # Keep the full prompt

            temp_sample = {
                "id": entry.get("id", "roborefit/unknown"),
                "image_path": os.path.join(image_root, entry["image"]),
                "prompt": f"<image>\n{core_prompt}",
                "raw_label": gpt_label_str,
                "source": "roborefit",
                "flag": "roborefit"
            }
            
            # Validate coordinate format
            if self._parse_coordinates(temp_sample["raw_label"], temp_sample["flag"]):
                processed_samples.append(temp_sample)
                flag_counts[temp_sample["flag"]] += 1
                
        return processed_samples
    
    def _parse_coordinates(self, gt_answer_str: str, flag: str) -> list:
        """
        [Robust V2] Parses coordinate strings.
        - For robodata, coco, refspatial: returns list of [x, y] normalized coordinates.
        - For oxe, roborefit: returns list of [x, y] PIXEL coordinates.
        - Returns an empty list [] if parsing or validation fails.
        """
        try:
            # Safely evaluate the string representation (e.g., "[[0.5, 0.5]]")
            raw_coords = ast.literal_eval(gt_answer_str)
            if not isinstance(raw_coords, list): return []
        except (ValueError, SyntaxError):
            return [] # String was not a valid Python literal
        
        # --- PIXEL Coordinate Handlers (OXE, Roborefit) ---
        # These flags expect pixel coordinates. We just validate the format
        # and return them as-is. Normalization happens in __getitem__.
        if flag == 'oxe_action_points' or flag == 'roborefit':
            # Expects a single point: [x, y]
            if (isinstance(raw_coords, list) and len(raw_coords) == 2 
                and all(isinstance(n, (int, float)) for n in raw_coords)):
                # Wrap it in a list to be consistent: [[x, y]]
                return [raw_coords]
            else:
                # print(f"Coordinate parsing error for flag '{flag}': {gt_answer_str}")
                return []

        # --- NORMALIZED Coordinate Handlers (Robodata, COCO, RefSpatial) ---
        elif flag in ['detection_0', 'detection_1', 'detection_2', 'refspatial']:
            temp_normalized_points = []
            
            if flag in ['detection_0', 'refspatial']:
                # Expects: [x, y] or [[x1, y1], [x2, y2], ...]
                if raw_coords and isinstance(raw_coords[0], (list, tuple)):
                    temp_normalized_points.extend(raw_coords)
                elif raw_coords and len(raw_coords) == 2:
                    temp_normalized_points.append(raw_coords)
            
            elif flag == 'detection_1':
                # Expects: [xmin, ymin, xmax, ymax] or [[xmin, ymin, xmax, ymax]]
                box_coords = raw_coords[0] if raw_coords and isinstance(raw_coords[0], list) else raw_coords
                if box_coords and len(box_coords) == 4:
                    xmin, ymin, xmax, ymax = box_coords
                    temp_normalized_points.append([(xmin + xmax) / 2, (ymin + ymax) / 2])
            
            elif flag == 'detection_2':
                # Expects: [[x, y, w, h], [x, y, w, h], ...] (from COCO)
                for item in raw_coords:
                    if isinstance(item, (list, tuple)) and len(item) == 4:
                        temp_normalized_points.append([item[0], item[1]]) # Use center (x, y)

            # Final validation for all normalized points
            valid_points = []
            for p in temp_normalized_points:
                if (isinstance(p, (list, tuple)) and len(p) == 2 
                    and 0 <= p[0] <= 1 and 0 <= p[1] <= 1):
                    valid_points.append(p)
            return valid_points
        
        else:
            # print(f"Unknown flag '{flag}' in _parse_coordinates.")
            return []

    def _parse_heatmap_annotations(self, gt_answer_str: str, flag: str) -> list:
        """Parse centers together with the Gaussian shape for supervision.

        Point-like annotations retain the isotropic ``sigma=2`` target used
        in the paper. COCO bounding boxes additionally carry an anisotropic
        ``(sigma_x, sigma_y)`` computed from their aspect ratio.
        """
        points = self._parse_coordinates(gt_answer_str, flag)
        if not points:
            return []

        isotropic_annotations = [
            {"point": [float(point[0]), float(point[1])], "sigma": (2.0, 2.0)}
            for point in points
        ]

        try:
            raw_coords = ast.literal_eval(gt_answer_str)
        except (ValueError, SyntaxError):
            return []

        if flag == "detection_1":
            box_coords = (
                raw_coords[0]
                if raw_coords and isinstance(raw_coords[0], (list, tuple))
                else raw_coords
            )
            if (
                not isinstance(box_coords, (list, tuple))
                or len(box_coords) != 4
                or not all(isinstance(value, (int, float)) for value in box_coords)
            ):
                return []

            xmin, ymin, xmax, ymax = box_coords
            try:
                sigma = shape_aware_sigmas(xmax - xmin, ymax - ymin)
            except ValueError:
                return []
            isotropic_annotations[0]["sigma"] = sigma
            return isotropic_annotations[:1]

        if flag == "detection_2":
            annotations = []
            for item in raw_coords:
                if (
                    not isinstance(item, (list, tuple))
                    or len(item) != 4
                    or not all(isinstance(value, (int, float)) for value in item)
                ):
                    continue

                x, y, width, height = item
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    continue
                try:
                    sigma = shape_aware_sigmas(width, height)
                except ValueError:
                    continue
                annotations.append({"point": [float(x), float(y)], "sigma": sigma})
            return annotations

        return isotropic_annotations

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Retrieves a single processed sample.
        """
        sample_info = self.samples[idx]
        
        # --- 1. Load and Augment Image ---
        image_pil = Image.open(sample_info["image_path"]).convert("RGB")
        
        # Apply color augmentation only in training mode
        if self.mode == 'train' and self.augmentation_pipeline:
            image_pil = self.augmentation_pipeline(image_pil)
        
        # --- 2. Parse Coordinates and Generate Heatmap ---
        W, H = image_pil.size 
        h, w = 224, 224 # Target model input size

        try:
            annotations = self._parse_heatmap_annotations(
                sample_info["raw_label"], sample_info["flag"]
            )
        except Exception:
            annotations = [] # Fail-safe

        points_list_final = []
        sigmas_final = []
        for annotation in annotations:
            point = annotation["point"]
            if sample_info["flag"] in ['oxe_action_points', 'roborefit']:
                # Pixel coordinates are normalized against the source image.
                point = [point[0] / W, point[1] / H]

            if 0 <= point[0] <= 1 and 0 <= point[1] <= 1:
                points_list_final.append(point)
                sigmas_final.append(annotation["sigma"])

        if not points_list_final:
            # No valid points, return empty heatmap and tensor
            gt_heatmap = torch.zeros(h, w)
            points_tensor = torch.empty(0, 2)
        else:
            # Generate heatmap from valid points
            points_tensor = torch.tensor(points_list_final, dtype=torch.float32)
            # Scale normalized points to target resolution (224x224)
            points_scaled = points_tensor * torch.tensor([w, h], dtype=torch.float32)

            all_heatmaps = [
                generate_hm_from_pt(
                    point.unsqueeze(0),
                    res=(w, h),
                    sigma=sigma,
                )[0]
                for point, sigma in zip(points_scaled, sigmas_final)
            ]
            # Combine heatmaps from multiple points by taking the max
            gt_heatmap = torch.stack(all_heatmaps, dim=0).amax(dim=0)

        # --- 3. Finalize Output ---
        return {
            "image": image_pil,
            "prompt": sample_info["prompt"], 
            "heatmap": gt_heatmap,
            "points": points_tensor  # Normalized [0,1] points
        }
    
    def get_class_counts(self):
        """Utility function to get counts of each data flag."""
        return Counter([s['flag'] for s in self.samples])


@dataclass
class DataCollator(object):
    """
    Data collator for supervised fine-tuning.
    
    It takes a batch of samples from the RoboMAPDataset, processes
    the text and images using the provided processor, and also
    collates the ground truth heatmaps into a single tensor.
    """
    processor: AutoProcessor 

    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Processes a list of samples into a single batch.

        Args:
            data: A list of dictionary samples, where each sample
                  is from RoboMAPDataset (e.g., {"image": ..., 
                  "prompt": ..., "heatmap": ...}).

        Returns:
            A dictionary containing the batched and processed
            tensors, including 'input_ids', 'pixel_values',
            'attention_mask', and the collated 'heatmaps'.
        """
        # Extract prompts and images from the batch
        texts = [ex["prompt"] for ex in data]
        images = [ex["image"] for ex in data]
        
        # --- [Added] Collect and stack heatmaps ---
        heatmaps = [ex["heatmap"] for ex in data]
        
        # Stack into a single batch tensor: (batch_size, height, width)
        batched_heatmaps = torch.stack(heatmaps, dim=0)
        # ------------------------------------------

        # Process text and images using the HuggingFace processor
        # This handles tokenization, image preprocessing, and padding
        tokens = self.processor(
            text=texts, 
            images=images, 
            return_tensors="pt", 
            padding="longest"
        )
        
        # Add the batched heatmaps to the final output dictionary
        # The model's forward() method will expect this key.
        tokens["heatmaps"] = batched_heatmaps

        return tokens
