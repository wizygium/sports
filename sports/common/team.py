from typing import Generator, Iterable, List, Optional, Dict, Union, Tuple
import numpy as np
import pickle
import os
import supervision as sv
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipModel, SiglipConfig
from peft import get_peft_model, LoraConfig, TaskType

# Default to a high-quality SigLIP model. 
# SigLIP 2 supports native aspect ratios (NaFlex).
DEFAULT_MODEL_PATH = 'google/siglip-so400m-patch14-384' 
import cv2
from PIL import Image 

class TeamClassifier:
    """
    A classifier that uses SigLIP (v1/v2) for zero-shot text clustering 
    and LoRA fine-tuning for team specialization.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32, model_path: str = DEFAULT_MODEL_PATH):
        self.device = device
        self.batch_size = batch_size
        self.model_path = model_path
        
        print(f"Loading SigLIP model: {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = SiglipModel.from_pretrained(model_path).to(device)
        self.model.eval()
        
        self.cluster_model = None
        self.prompts = None
        self.label_map = {}
        self.ref_class_id = -1 # ID for referee if identified
        
    def _get_image_embeddings(self, crops: List[np.ndarray], desc: str = "Embedding extraction") -> torch.Tensor:
        """Generating embeddings for a list of images."""
        if not crops:
            return torch.empty(0, self.model.config.vision_config.hidden_size).to(self.device)
            
        # Convert CV2 (BGR) to PIL (RGB)
        # Note: sv.cv2_to_pillow might not handle BGR->RGB if just wrapping Image.fromarray
        # We explicitly convert to ensure correct colors for CLIP/SigLIP
        pil_images = []
        for c in crops:
            if c is not None and c.size > 0:
                img = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                # If image is too small, resize to avoid Transformers ambiguity with (1, 1, 3) vs (3, 1, 1)
                if img.width < 32 or img.height < 32:
                    img = img.resize((32, 32), Image.BILINEAR)
                pil_images.append(img)
            else:
                pil_images.append(Image.new("RGB", (32, 32)))
        
        # SigLIP processor handles aspect ratios well (NaFlex in v2)
        # do_resize=True, size={"height": H, "width": W} usually
        embeddings = []
        
        # Process in batches
        for i in range(0, len(pil_images), self.batch_size):
            batch = pil_images[i : i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # Get vision features using the vision_model's distinct pooler or mean pool
                vision_outputs = self.model.vision_model(**inputs)
                # Use pooled output if available, else mean of last hidden state
                if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
                    batch_embs = vision_outputs.pooler_output
                else:
                    batch_embs = vision_outputs.last_hidden_state.mean(dim=1)
                
                # Project projection if needed (SigLIP usually projects vision to common space)
                # The full SiglipModel computes this in get_image_features usually
                batch_embs = self.model.vision_embedder(batch_embs) if hasattr(self.model, "vision_embedder") else batch_embs
                
                # Normalize
                batch_embs = batch_embs / batch_embs.norm(dim=-1, keepdim=True)
                embeddings.append(batch_embs)
                
        return torch.cat(embeddings)

    def compute_text_probs(self, crops: List[np.ndarray], candidate_prompts: List[str]) -> np.ndarray:
        """
        Compute probability of each crop belonging to each text prompt.
        """
        if not crops or not candidate_prompts:
            return np.array([])
            
        # 1. Encode Text
        inputs = self.processor(text=candidate_prompts, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
                text_embs = text_outputs.pooler_output
            else:
                text_embs = text_outputs.last_hidden_state.mean(dim=1)
                
            # Assume model has text_embedder or projections aligned? 
            # Standard SiglipModel 'get_text_features' does normalization
            # We can just call model.get_text_features if available via wrapping, 
            # but simpler to rely on the forward pass if we passed both?
            # Actually, let's just use the convenience methods if possible, 
            # but usually we want to cache text features.
            
            # Using standard SiglipModel forward pass logic:
            # text_features = output.text_embeds ... 
            # But we are accessing sub-models. Let's use the full model forward for text.
            pass

        # Cleaner way: Encode text once using the model's helper
        with torch.no_grad():
            text_inputs = self.processor(text=candidate_prompts, padding=True, return_tensors="pt").to(self.device)
            text_features = self.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # shape: [n_prompts, dim]

        # 2. Encode Images
        img_features = self._get_image_embeddings(crops, desc="Zero-shot calc") # shape: [n_crops, dim]
        
        # 3. Compute Similarity (SigLIP logit scale)
        # logit_scale = self.model.logit_scale.exp()
        # logits_per_image = logit_scale * img_features @ text_features.t()
        # probs = logits_per_image.softmax(dim=1)
        
        # We just want relative probabilities/similarities
        sims = img_features @ text_features.t()
        probs = sims.softmax(dim=1).cpu().numpy()
        
        return probs

    def fit_prompts(self, crops: List[np.ndarray], prompts_map: Dict[str, str]) -> None:
        """
        Initialize classification based on text prompts.
        
        Args:
            prompts_map: Dict mapping Class Name -> Prompt Text
                         e.g. {"Team A": "red jersey", "Ref": "yellow shirt"}
        """
        self.prompts = list(prompts_map.values())
        self.class_names = list(prompts_map.keys())
        self.label_map = {i: name for i, name in enumerate(self.class_names)}
        
        # We don't 'fit' a cluster model in this mode, we use the text features as centroids
        print(f"Initialized with prompts: {self.prompts}")

    def fit_kmeans(self, crops: List[np.ndarray], k: int = 2) -> None:
        """Fit K-Means clustering."""
        print("Extracting features for K-Means...")
        embs = self._get_image_embeddings(crops).cpu().numpy()
        
        print(f"Fitting K-Means (k={k})...")
        self.cluster_model = KMeans(n_clusters=k, n_init=10)
        self.cluster_model.fit(embs)
        # Default labels (integers)
        self.label_map = {i: i for i in range(k)}

    def save(self, path: str):
        """Save the fitted classifier (KMeans model + label map) to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'cluster_model': self.cluster_model,
                'label_map': self.label_map,
                'prompts': self.prompts
            }, f)
        print(f"TeamClassifier saved to {path}")

    def load(self, path: str) -> bool:
        """Load a fitted classifier from disk."""
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.cluster_model = data.get('cluster_model')
                self.label_map = data.get('label_map', {})
                self.prompts = data.get('prompts')
            print(f"TeamClassifier loaded from {path}")
            return True
        except Exception as e:
            print(f"Failed to load TeamClassifier: {e}")
            return False

    def get_cluster_confidences(self, crops: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (labels, confidences) for the given crops.
        Confidence is defined as -1 * distance_to_centroid (closer is better).
        Or we can normalize it. For outlier filtering, raw distance is fine.
        """
        if not self.cluster_model:
            raise ValueError("Model not fitted")
            
        embs = self._get_image_embeddings(crops).cpu().numpy()
        # [N, n_clusters] distances
        dists = self.cluster_model.transform(embs)
        
        # Label is index of min distance
        labels = np.argmin(dists, axis=1)
        
        # Distance to assigned centroid
        min_dists = dists[np.arange(len(dists)), labels]
        
        # Confidence: We can treat negated distance as score (higher is better)
        # But for filtering we just want the distance values.
        return labels, min_dists

    def assign_clusters_to_prompts(self, crops: List[np.ndarray], prompts_map: Dict[str, str], debug_dir: str = None) -> None:
        """
        Assign K-Means clusters to text prompts using Hungarian Algorithm (1-to-1 matching).
        Must call fit_kmeans first.
        """
        if not self.cluster_model:
            raise ValueError("Must call fit_kmeans before assigning prompts.")
            
        print(f"Assigning {self.cluster_model.n_clusters} clusters to {len(prompts_map)} prompts: {prompts_map}")
        
        # 1. Predict Cluster for each crop
        embs = self._get_image_embeddings(crops).cpu().numpy()
        cluster_labels = self.cluster_model.predict(embs)
        
        # DEBUG: Save sample crops for each cluster
        if debug_dir:
            import os
            import cv2
            os.makedirs(debug_dir, exist_ok=True)
            for k in range(self.cluster_model.n_clusters):
                cluster_indices = np.where(cluster_labels == k)[0]
                if len(cluster_indices) > 0:
                    # Pick up to 16 random samples
                    samples_idx = np.random.choice(cluster_indices, min(len(cluster_indices), 16), replace=False)
                    sample_crops = [crops[i] for i in samples_idx]
                    
                    # Create a grid (4x4)
                    rows = []
                    for r in range(0, len(sample_crops), 4):
                        row_imgs = sample_crops[r:r+4]
                        # Resize to same height/width for stacking
                        target_h, target_w = 128, 128
                        resized = [cv2.resize(img, (target_w, target_h)) for img in row_imgs]
                        # Pad row if needed
                        while len(resized) < 4:
                            resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))
                        rows.append(np.hstack(resized))
                    grid = np.vstack(rows)
                    cv2.imwrite(f"{debug_dir}/cluster_{k}_samples.jpg", grid)
                    print(f"Saved {debug_dir}/cluster_{k}_samples.jpg")

        # 2. Compute Text Probs for each crop
        prompt_texts = list(prompts_map.values())
        class_names = list(prompts_map.keys())
        probs = self.compute_text_probs(crops, prompt_texts) # [N, n_prompts]
        
        # 3. Vote per cluster (Average Probability)
        # {cluster_idx: [avg_prob_prompt_0, avg_prob_prompt_1, ...]}
        # We construct a Cost Matrix for Linear Sum Assignment
        # Rows: Clusters, Cols: Prompts
        # We want to MAXIMIZE probability, so Cost = 1 - Prob (or negative prob)
        
        n_clusters = self.cluster_model.n_clusters
        n_prompts = len(prompt_texts)
        cost_matrix = np.zeros((n_clusters, n_prompts))
        
        # Accumulate probabilities
        for i, label in enumerate(cluster_labels):
            cost_matrix[label] += probs[i]
            
        # Normalize by count to get average prob
        counts = np.bincount(cluster_labels, minlength=n_clusters)
        print("\n--- CLUSTER PROBABILITIES ---")
        for i in range(n_clusters):
            if counts[i] > 0:
                cost_matrix[i] /= counts[i]
            
            # Detailed Logging
            probs_str = ", ".join([f"{p:.4f}" for p in cost_matrix[i]])
            print(f"Cluster {i} (n={counts[i]}): [{probs_str}] -> Prompts: {class_names}")

        # 4. Hungarian Algorithm (Linear Sum Assignment)
        # scipy.optimize.linear_sum_assignment finds min cost.
        # We want max prob, so pass negative probs.
        from scipy.optimize import linear_sum_assignment
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        
        self.prompts = prompt_texts
        self.label_map = {}
        
        print("\n--- HUNGARIAN ASSIGNMENT ---")
        for r, c in zip(row_ind, col_ind):
            prompt_name = class_names[c]
            prob = cost_matrix[r, c]
            self.label_map[r] = prompt_name
            print(f"  Cluster {r} -> {prompt_name} (Avg Prob for this prompt: {prob:.4f})")
            
        print(f"Final Label Map: {self.label_map}\n")

    def fit(self, crops: List[np.ndarray]) -> None:
        """Backward compatibility wrapper for fit_kmeans."""
        self.fit_kmeans(crops)

    def train_lora(self, tracklets: Dict[int, List[np.ndarray]], labels: Dict[int, int], epochs: int = 5):
        """
        Fine-tune the model using LoRA on labeled tracklets.
        """
        print(f"Starting LoRA training on {len(labels)} tracklets for {epochs} epochs...")
        
        # 1. Setup LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"] 
        )
        try:
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        except ValueError as e:
            print(f"Warning: Could not apply LoRA: {e}")
            return

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Augmentations
        from torchvision import transforms
        transform = transforms.Compose([
            # Input is already PIL from sv.cv2_to_pillow
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(), # Convert to Tensor
            transforms.ToPILImage() # Convert back to PIL for processor
        ])
        
        # 2. Data Prep
        train_data = []
        for tid, label in labels.items():
            if tid not in tracklets: continue
            images = tracklets[tid]
            # Subsample
            if len(images) > 10:
                indices = np.linspace(0, len(images)-1, 10, dtype=int)
                images = [images[i] for i in indices]
            for img in images:
                train_data.append((img, label))
        
        if not train_data:
            print("No training data available.")
            return

        class_indices = list(set(labels.values()))
        self.model.vision_model.train()
        
        for epoch in range(epochs):
            np.random.shuffle(train_data)
            total_loss = 0
            batch_size = 16
            
            for i in range(0, len(train_data), batch_size):
                batch_items = train_data[i:i+batch_size]
                
                # Apply Augmentation
                batch_imgs = []
                for x in batch_items:
                    # x[0] is BGR numpy
                    rgb = sv.cv2_to_pillow(x[0]) # PIL
                    # Convert to tensor for transform? Or transform accepts PIL
                    # ColorJitter accepts PIL
                    aug_img = transform(rgb)
                    batch_imgs.append(aug_img)

                batch_lbls = torch.tensor([x[1] for x in batch_items]).to(self.device)
                
                inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
                
                # CRITICAL FIX: Use get_image_features to match inference path.
                # Previously we used vision_model directly which might skip projection layers.
                embs = self.model.get_image_features(**inputs)
                
                # Normalize (SigLIP/CLIP standard)
                embs = embs / embs.norm(dim=-1, keepdim=True)
                
                if self.prompts:
                    # Anchor to text features (frozen)
                    with torch.no_grad():
                        text_inputs = self.processor(text=self.prompts, padding=True, return_tensors="pt").to(self.device)
                        text_anchors = self.model.get_text_features(**text_inputs)
                        text_anchors = text_anchors / text_anchors.norm(dim=-1, keepdim=True)
                    
                    logits = embs @ text_anchors.t() * self.model.logit_scale.exp()
                    loss = F.cross_entropy(logits, batch_lbls)
                else:
                    # Placeholder if no prompts
                    loss = torch.tensor(0.0, requires_grad=True).to(self.device)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
            
        self.model = self.model.merge_and_unload()
        self.model.eval()

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict labels for crops.
        """
        # Preference: K-Means (Visual Cluster) if available, unless Prompts + LoRA is more trusted?
        # If we did 'assign_clusters_to_prompts', self.prompts is SET, but we want to return Cluster IDs?
        # No, predict usually returns Index (0, 1).
        
        # "Hybrid": Use K-Means model for prediction (Visual consistency), 
        # self.label_map handles the semantic name.
        if self.cluster_model:
             embs = self._get_image_embeddings(crops).cpu().numpy()
             raw_preds = self.cluster_model.predict(embs)
             
             # Apply Hungarian Assignment Map if available
             if self.label_map:
                 # Map raw cluster ID -> Assigned Prompt Index
                 # label_map keys are int (cluster id), values are int (prompt index)
                 # We must ensure types match. 
                 # Let's assume label_map is Dict[int, int or str]. 
                 # If it maps to string names, we might need to map back to index if downstream expects int?
                 # No, downstream uses whatever we return. 
                 # If label_map values are Prompt Names ("white jersey"), then we return strings?
                 # Wait, assign_clusters_to_prompts sets self.label_map using `prompt_name` (string).
                 # But V8 logs showed it mapping to integers?
                 # V8 Log: "Final Label Map: {0: 1, 1: 0}" -> These are integers.
                 # Ah, in assign_clusters_to_prompts:
                 #   prompt_name = class_names[c] (could be index if class_names is list of string?)
                 # Let's check assign_clusters_to_prompts implementation.
                 
                 # To be safe, we map valid keys.
                 return np.array([self.label_map.get(p, p) for p in raw_preds])
             
             return raw_preds
        elif self.prompts:
            probs = self.compute_text_probs(crops, self.prompts)
            return np.argmax(probs, axis=1)
        else:
            return np.array([0] * len(crops))

