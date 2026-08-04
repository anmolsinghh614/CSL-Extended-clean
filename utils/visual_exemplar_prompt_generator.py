import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

try:
    import clip
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    MODELS_AVAILABLE = True
except ImportError:
    print("Warning: CLIP and/or BLIP models not available. Install with:")
    print("pip install git+https://github.com/openai/CLIP.git")
    print("pip install transformers pillow")
    MODELS_AVAILABLE = False


@dataclass
class ImageExemplar:
    """Container for image exemplar data."""
    image_path: str
    clip_embedding: torch.Tensor
    class_id: int
    distance_to_prototype: float
    caption: Optional[str] = None


class VisualExemplarPromptGenerator:
    """
    Implements Option 3: Use Nearest Visual Exemplars + Caption
    
    Finds nearest images to class prototypes in CLIP space and generates
    captions using BLIP for semantic prompt generation.
    """
    
    # Photographic modifiers used to expand a caption into many distinct prompts.
    PROMPT_MODIFIERS = [
        "high resolution photograph",
        "sharp focus, natural lighting",
        "daylight, outdoors",
        "close-up view",
        "wide shot, plain background",
        "overcast day",
        "side view",
        "front view",
        "detailed texture",
        "soft shadows",
        "clear background",
        "golden hour lighting",
    ]
    
    def __init__(self,
                 memory_manager,
                 clip_model_name: str = "ViT-B/32",
                 blip_model_name: str = "Salesforce/blip-image-captioning-base",
                 device: str = "cuda",
                 cache_dir: str = "./clip_cache",
                 pixel_mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
                 pixel_std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010),
                 use_blip: bool = True,
                 use_clip: bool = True,
                 require_models: bool = False):
        """
        Initialize the visual exemplar prompt generator.
        
        Args:
            memory_manager: MemoryManager instance with trained prototypes
            clip_model_name: CLIP model to use for embeddings
            blip_model_name: BLIP model for captioning
            device: Device for computation
            cache_dir: Directory to cache CLIP embeddings
            pixel_mean: Normalization mean used by the classifier's transform, needed to
                recover displayable pixels before handing images to CLIP/BLIP
            pixel_std: Normalization std used by the classifier's transform
            use_blip: Whether to load BLIP for exemplar captioning
            use_clip: Whether to load CLIP for prompt ranking
            require_models: Raise if the captioning/ranking models cannot be loaded.
                When False the generator degrades to class-name templates instead, so a
                missing optional dependency cannot abort a training run.
        """
        self.memory_manager = memory_manager
        self.device = device
        self.cache_dir = cache_dir
        self.pixel_mean = torch.tensor(pixel_mean).view(3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(3, 1, 1)
        os.makedirs(cache_dir, exist_ok=True)
        
        if not MODELS_AVAILABLE and require_models:
            raise ImportError("Required models not available. Please install CLIP and transformers.")
        
        self.clip_model = None
        self.clip_preprocess = None
        self.blip_processor = None
        self.blip_model = None
        
        if MODELS_AVAILABLE and use_clip:
            try:
                self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
                self.clip_model.eval()
                print(f"Loaded CLIP model: {clip_model_name}")
            except Exception as e:
                if require_models:
                    raise
                print(f"Warning: could not load CLIP ({e}). Prompt ranking disabled.")
                self.clip_model = None
        
        if MODELS_AVAILABLE and use_blip:
            try:
                self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
                self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
                self.blip_model = self.blip_model.to(device)
                self.blip_model.eval()
                print(f"Loaded BLIP model: {blip_model_name}")
            except Exception as e:
                if require_models:
                    raise
                print(f"Warning: could not load BLIP ({e}). Exemplar captioning disabled.")
                self.blip_model = None
        
        # Storage for image data and embeddings
        self.image_database = {}  # class_id -> List[ImageExemplar]
        self.clip_embeddings_cache = {}  # image_path -> clip_embedding
        
    @property
    def captioning_available(self) -> bool:
        return self.blip_model is not None
    
    @property
    def ranking_available(self) -> bool:
        return self.clip_model is not None
    
    def _tensor_to_pil(self, image: torch.Tensor, upscale_to: int = 224):
        """
        Convert a normalized CHW image tensor into a PIL image.
        
        Args:
            image: Image tensor, optionally with a leading batch dimension
            upscale_to: Side length to resize to. CLIP and BLIP are trained on ~224px
                inputs, so passing 32x32 tensors straight through yields poor captions.
        """
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.dim() != 3 or image.shape[0] != 3:
            return None
        
        image = image.detach().float().cpu()
        
        # Undo the classifier's normalization when the tensor is still normalized
        if image.min() < 0:
            image = image * self.pixel_std + self.pixel_mean
        
        image = torch.clamp(image, 0, 1)
        image_pil = Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype(np.uint8))
        
        if upscale_to and image_pil.size[0] < upscale_to:
            image_pil = image_pil.resize((upscale_to, upscale_to), Image.BICUBIC)
        
        return image_pil
    
    def build_image_database(self, 
                           dataloader,
                           max_images_per_class: int = 1000,
                           save_cache: bool = True) -> None:
        """
        Build database of images with CLIP embeddings for each class.
        
        Args:
            dataloader: DataLoader providing (images, labels, paths) tuples
            max_images_per_class: Maximum images to process per class
            save_cache: Whether to save CLIP embeddings to disk
        """
        print("Building image database with CLIP embeddings...")
        
        # Initialize database
        for class_id in range(self.memory_manager.memory_bank.num_classes):
            self.image_database[class_id] = []
        
        # Track counts per class
        class_counts = defaultdict(int)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Handle different dataloader formats
                if len(batch) == 3:
                    images, labels, image_paths = batch
                elif len(batch) == 2:
                    images, labels = batch
                    image_paths = [f"batch_{batch_idx}_idx_{i}" for i in range(len(images))]
                else:
                    raise ValueError("Dataloader must provide (images, labels) or (images, labels, paths)")
                
                # Process each image in batch
                for i, (image, label) in enumerate(zip(images, labels)):
                    class_id = label.item()
                    
                    # Skip if we have enough images for this class
                    if class_counts[class_id] >= max_images_per_class:
                        continue
                    
                    image_path = image_paths[i] if isinstance(image_paths[i], str) else f"batch_{batch_idx}_idx_{i}"
                    
                    # Compute CLIP embedding
                    clip_embedding = self._compute_clip_embedding(image, image_path)
                    
                    if clip_embedding is not None:
                        # Create exemplar entry
                        exemplar = ImageExemplar(
                            image_path=image_path,
                            clip_embedding=clip_embedding,
                            class_id=class_id,
                            distance_to_prototype=0.0  # Will be computed later
                        )
                        
                        self.image_database[class_id].append(exemplar)
                        class_counts[class_id] += 1
                
                if batch_idx % 100 == 0:
                    total_images = sum(len(exemplars) for exemplars in self.image_database.values())
                    print(f"Processed {batch_idx} batches, {total_images} images indexed")
        
        # Save cache if requested
        if save_cache:
            self._save_clip_cache()
        
        total_images = sum(len(exemplars) for exemplars in self.image_database.values())
        print(f"Image database built: {total_images} images across {len(self.image_database)} classes")
    
    def _compute_clip_embedding(self, image: torch.Tensor, image_path: str) -> Optional[torch.Tensor]:
        """Compute CLIP embedding for an image."""
        # Check cache first
        if image_path in self.clip_embeddings_cache:
            return self.clip_embeddings_cache[image_path]
        
        if not self.ranking_available:
            return None
        
        try:
            image_pil = self._tensor_to_pil(image)
            if image_pil is None:
                print(f"Warning: Unexpected image shape {tuple(image.shape)}")
                return None
            
            # Preprocess and encode with CLIP
            image_input = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                clip_embedding = self.clip_model.encode_image(image_input)
                clip_embedding = F.normalize(clip_embedding, dim=1)
            
            # Cache the embedding
            self.clip_embeddings_cache[image_path] = clip_embedding.squeeze(0).cpu()
            
            return clip_embedding.squeeze(0).cpu()
            
        except Exception as e:
            print(f"Warning: Failed to compute CLIP embedding for {image_path}: {e}")
            return None
    
    
    CIFAR10_CLASS_NAMES = {
        0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
        5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
    }
    
    def generate_prompts(self,
                         class_idx: int,
                         exemplar_features: torch.Tensor = None,
                         num_prompts: int = 5,
                         temperature: float = 0.8,
                         class_name: str = None) -> List[str]:
        """
        Generate class-name template prompts for a single class.
        
        This is the fallback used when no exemplar images are available. Prefer
        `generate_prompts_from_exemplars`, which conditions the prompts on the images the
        memory bank actually selected.
        
        Args:
            class_idx: Class ID to generate prompts for
            exemplar_features: Unused here; accepted for backward compatibility
            num_prompts: Number of prompts to return
            temperature: Unused here; accepted for backward compatibility
            class_name: Human-readable class name (falls back to the CIFAR-10 names)
        
        Returns:
            List of prompts for the class
        """
        if class_name is None:
            class_name = self.CIFAR10_CLASS_NAMES.get(class_idx, f"class_{class_idx}")
        
        base_prompts = [
            f"a photo of a {class_name}",
            f"a high quality image of a {class_name}",
            f"a clear photograph of a {class_name}",
            f"a {class_name} in a natural setting",
            f"a detailed view of a {class_name}",
        ]
        
        return self._expand_to_count(base_prompts, class_name, num_prompts)
    
    def generate_prompts_from_exemplars(self,
                                        class_idx: int,
                                        class_name: str,
                                        exemplar_images: List[torch.Tensor],
                                        num_prompts: int = 5,
                                        temperature: float = 0.8) -> Tuple[List[str], List[str]]:
        """
        Generate prompts conditioned on the class's nearest visual exemplars.
        
        This is the Option 3 path: the memory bank prototype selects which real training
        images best represent the class, BLIP describes those images, and CLIP scores the
        resulting prompt candidates against the exemplars so the prompts that survive are
        the ones that actually look like the class.
        
        Args:
            class_idx: Class ID
            class_name: Human-readable class name
            exemplar_images: Normalized image tensors chosen as this class's exemplars
            num_prompts: Number of prompts to return
            temperature: Controls how much of the ranked candidate pool is sampled from.
                0 keeps strictly the top-scoring prompts; higher values sample more widely
                for diversity.
        
        Returns:
            (prompts, captions) — the selected prompts and the raw exemplar captions
        """
        captions = []
        if self.captioning_available:
            for image in exemplar_images:
                caption = self._generate_blip_caption(image)
                if caption:
                    captions.append(caption.strip().lower())
        
        candidates = self._build_prompt_candidates(class_name, captions)
        
        if not candidates:
            return self.generate_prompts(class_idx, num_prompts=num_prompts,
                                         class_name=class_name), captions
        
        ranked = self._rank_prompts_by_clip(candidates, exemplar_images, temperature)
        selected = ranked[:num_prompts]
        
        if len(selected) < num_prompts:
            selected = self._expand_to_count(selected, class_name, num_prompts)
        
        return selected, captions
    
    def _build_prompt_candidates(self, class_name: str, captions: List[str]) -> List[str]:
        """Build a deduplicated candidate pool from exemplar captions plus templates."""
        candidates = [
            f"a photo of a {class_name}",
            f"a high quality photograph of a {class_name}",
            f"a clear detailed photo of a {class_name}",
            f"a {class_name} in a natural setting",
        ]
        
        for caption in captions:
            descriptors = self._extract_descriptors(caption, class_name)
            
            # Anchor every caption-derived prompt on the class name: BLIP captions of
            # 32x32 images are often vague, and an unanchored prompt can drift to a
            # different object entirely, which would poison the synthetic samples.
            if class_name in caption:
                candidates.append(f"a photo of {caption}")
            else:
                candidates.append(f"a photo of a {class_name}, {caption}")
            
            if descriptors:
                candidates.append(f"a photo of a {class_name}, {descriptors}")
                for modifier in self.PROMPT_MODIFIERS:
                    candidates.append(f"a photo of a {class_name}, {descriptors}, {modifier}")
        
        for modifier in self.PROMPT_MODIFIERS:
            candidates.append(f"a photo of a {class_name}, {modifier}")
        
        return self._deduplicate(candidates)
    
    def _expand_to_count(self, prompts: List[str], class_name: str, num_prompts: int) -> List[str]:
        """Pad a prompt list up to num_prompts by appending photographic modifiers."""
        expanded = list(prompts)
        
        for modifier in self.PROMPT_MODIFIERS:
            if len(expanded) >= num_prompts:
                break
            expanded.append(f"a photo of a {class_name}, {modifier}")
        
        # Still short (a very large num_prompts): cycle modifiers over existing prompts
        modifier_idx = 0
        while len(expanded) < num_prompts and prompts:
            base = prompts[modifier_idx % len(prompts)]
            modifier = self.PROMPT_MODIFIERS[modifier_idx % len(self.PROMPT_MODIFIERS)]
            expanded.append(f"{base}, {modifier}")
            modifier_idx += 1
            if modifier_idx > num_prompts * 4:
                break
        
        return self._deduplicate(expanded)[:num_prompts]
    
    @staticmethod
    def _deduplicate(prompts: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        unique = []
        for prompt in prompts:
            key = prompt.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(prompt.strip())
        return unique
    
    def _rank_prompts_by_clip(self,
                              candidates: List[str],
                              exemplar_images: List[torch.Tensor],
                              temperature: float = 0.0) -> List[str]:
        """
        Order candidate prompts by CLIP similarity to the exemplar images.
        
        Falls back to the original ordering when CLIP is unavailable or no exemplar
        image could be embedded.
        """
        if not self.ranking_available or not exemplar_images:
            return candidates
        
        try:
            image_embeddings = []
            for image in exemplar_images:
                image_pil = self._tensor_to_pil(image)
                if image_pil is None:
                    continue
                image_input = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.clip_model.encode_image(image_input)
                image_embeddings.append(F.normalize(embedding.float(), dim=1))
            
            if not image_embeddings:
                return candidates
            
            image_centroid = F.normalize(torch.cat(image_embeddings).mean(dim=0, keepdim=True), dim=1)
            
            tokens = clip.tokenize(candidates, truncate=True).to(self.device)
            with torch.no_grad():
                text_embeddings = self.clip_model.encode_text(tokens)
            text_embeddings = F.normalize(text_embeddings.float(), dim=1)
            
            scores = (text_embeddings @ image_centroid.T).squeeze(1)
            
            if temperature and temperature > 0:
                # Sample without replacement, weighted by similarity, so repeated rounds
                # do not regenerate the exact same prompt set.
                weights = torch.softmax(scores / max(temperature, 1e-3), dim=0)
                order = torch.multinomial(weights, num_samples=len(candidates), replacement=False)
            else:
                order = torch.argsort(scores, descending=True)
            
            return [candidates[i] for i in order.tolist()]
        
        except Exception as e:
            print(f"Warning: CLIP prompt ranking failed ({e}). Using unranked candidates.")
            return candidates
    
    
    def find_nearest_exemplars_for_tail_classes(self, 
                                              k_exemplars: int = 5,
                                              use_memory_prototypes: bool = True) -> Dict[int, List[ImageExemplar]]:
        """
        Find k nearest image exemplars for each tail class based on CLIP similarity.
        
        Args:
            k_exemplars: Number of nearest exemplars per class
            use_memory_prototypes: Whether to use memory bank prototypes vs CLIP prototypes
            
        Returns:
            Dictionary mapping tail class_id to list of nearest exemplars
        """
        tail_classes = self.memory_manager.get_tail_classes()
        nearest_exemplars = {}
        
        print(f"Finding nearest exemplars for {len(tail_classes)} tail classes...")
        
        for class_id in tail_classes:
            if class_id not in self.image_database or not self.image_database[class_id]:
                print(f"Warning: No images found for tail class {class_id}")
                continue
            
            # Get prototype embedding
            if use_memory_prototypes:
                # Use memory bank prototype (from CSL training)
                memory_prototype = self.memory_manager.memory_bank.get_prototype(class_id)
                
                # Project memory prototype to CLIP space (if dimensions differ)
                if memory_prototype.shape[0] != 512:  # CLIP ViT-B/32 embedding size
                    # Simple projection - in practice you might want a learned mapping
                    prototype_embedding = self._project_to_clip_space(memory_prototype)
                else:
                    prototype_embedding = F.normalize(memory_prototype, dim=0)
            else:
                # Compute CLIP prototype as average of class images
                prototype_embedding = self._compute_clip_prototype(class_id)
            
            if prototype_embedding is None:
                continue
            
            # Compute distances to all images in this class
            exemplars = self.image_database[class_id]
            
            for exemplar in exemplars:
                clip_emb = exemplar.clip_embedding.to(self.device)
                prototype_emb = prototype_embedding.to(self.device)
                
                # Cosine similarity (higher = more similar)
                similarity = F.cosine_similarity(clip_emb, prototype_emb, dim=0)
                # Convert to distance (lower = more similar)
                exemplar.distance_to_prototype = 1.0 - similarity.item()
            
            # Sort by distance and take k nearest
            sorted_exemplars = sorted(exemplars, key=lambda x: x.distance_to_prototype)
            nearest_exemplars[class_id] = sorted_exemplars[:k_exemplars]
        
        return nearest_exemplars
    
    def _project_to_clip_space(self, memory_feature: torch.Tensor) -> torch.Tensor:
        """
        Project memory bank feature to CLIP embedding space.
        Simple linear projection - could be replaced with learned mapping.
        """
        # Simple random projection for demo (replace with learned mapping in practice)
        if not hasattr(self, '_projection_matrix'):
            input_dim = memory_feature.shape[0]
            clip_dim = 512  # CLIP ViT-B/32 dimension
            
            # Create a fixed random projection matrix
            torch.manual_seed(42)  # For reproducibility
            self._projection_matrix = torch.randn(input_dim, clip_dim) / np.sqrt(input_dim)
            self._projection_matrix = self._projection_matrix.to(self.device)
        
        projected = torch.matmul(memory_feature.to(self.device), self._projection_matrix)
        return F.normalize(projected, dim=0)
    
    def _compute_clip_prototype(self, class_id: int) -> Optional[torch.Tensor]:
        """Compute CLIP prototype as average of class images."""
        if class_id not in self.image_database or not self.image_database[class_id]:
            return None
        
        embeddings = [exemplar.clip_embedding for exemplar in self.image_database[class_id]]
        if not embeddings:
            return None
        
        # Average embeddings
        stacked_embeddings = torch.stack(embeddings)
        prototype = torch.mean(stacked_embeddings, dim=0)
        return F.normalize(prototype, dim=0)
    
    def generate_captions_for_exemplars(self, 
                                      nearest_exemplars: Dict[int, List[ImageExemplar]],
                                      dataloader) -> Dict[int, List[str]]:
        """
        Generate BLIP captions for nearest exemplars.
        
        Args:
            nearest_exemplars: Output from find_nearest_exemplars_for_tail_classes
            dataloader: DataLoader to retrieve actual images
            
        Returns:
            Dictionary mapping class_id to list of captions
        """
        print("Generating BLIP captions for exemplars...")
        
        # Build path to image mapping from dataloader
        path_to_image = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if len(batch) == 3:
                    images, labels, image_paths = batch
                elif len(batch) == 2:
                    images, labels = batch
                    image_paths = [f"batch_{batch_idx}_idx_{i}" for i in range(len(images))]
                else:
                    continue
                
                for i, image in enumerate(images):
                    path = image_paths[i]
                    path_to_image[path] = image
        
        # Generate captions
        class_captions = {}
        
        for class_id, exemplars in nearest_exemplars.items():
            captions = []
            
            for exemplar in exemplars:
                # Get image
                if exemplar.image_path in path_to_image:
                    image = path_to_image[exemplar.image_path]
                    caption = self._generate_blip_caption(image)
                    
                    if caption:
                        captions.append(caption)
                        exemplar.caption = caption
                else:
                    print(f"Warning: Could not find image for path {exemplar.image_path}")
            
            if captions:
                class_captions[class_id] = captions
        
        return class_captions
    
    def _generate_blip_caption(self, image: torch.Tensor) -> Optional[str]:
        """Generate caption for a single image using BLIP."""
        if not self.captioning_available:
            return None
        
        try:
            image_pil = self._tensor_to_pil(image)
            if image_pil is None:
                return None
            
            # Generate caption with BLIP
            inputs = self.blip_processor(image_pil, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            print(f"Warning: Failed to generate BLIP caption: {e}")
            return None
    
    def create_semantic_prompts(self, 
                              class_captions: Dict[int, List[str]],
                              class_names: Dict[int, str],
                              prompt_templates: Optional[List[str]] = None) -> Dict[int, List[str]]:
        """
        Create semantic prompts from generated captions.
        
        Args:
            class_captions: Generated captions per class
            class_names: Class name mapping
            prompt_templates: Optional templates for prompt formatting
            
        Returns:
            Dictionary mapping class_id to semantic prompts
        """
        if prompt_templates is None:
            prompt_templates = [
                "A photo of a {}",
                "{}",
                "A detailed image of a {} that is {}",
                "A {} with {}"
            ]
        
        semantic_prompts = {}
        
        for class_id, captions in class_captions.items():
            class_name = class_names.get(class_id, f"class_{class_id}")
            prompts = []
            
            for caption in captions:
                # Clean caption
                clean_caption = caption.strip().lower()
                
                # Extract descriptive parts from caption
                descriptors = self._extract_descriptors(clean_caption, class_name)
                
                # Generate prompts using templates
                for template in prompt_templates:
                    if "{}" in template:
                        if template.count("{}") == 1:
                            prompt = template.format(class_name)
                        elif template.count("{}") == 2 and descriptors:
                            prompt = template.format(class_name, descriptors)
                        else:
                            continue
                    else:
                        prompt = template
                    
                    prompts.append(prompt)
                
                # Also use the raw caption as a prompt
                prompts.append(f"A photograph showing {clean_caption}")
            
            # Remove duplicates while preserving order
            unique_prompts = []
            seen = set()
            for prompt in prompts:
                if prompt not in seen:
                    unique_prompts.append(prompt)
                    seen.add(prompt)
            
            semantic_prompts[class_id] = unique_prompts
        
        return semantic_prompts
    
    STOPWORDS = {'the', 'and', 'with', 'that', 'this', 'there', 'its', 'his', 'her',
                 'are', 'was', 'were', 'has', 'have', 'for', 'from', 'into', 'onto'}
    
    def _extract_descriptors(self, caption: str, class_name: str) -> str:
        """Extract descriptive phrases from caption."""
        caption = caption.strip().lower()
        
        # Remove common prefixes
        prefixes = ["a photo of", "an image of", "a picture of", "this is", "there is"]
        for prefix in prefixes:
            if caption.startswith(prefix):
                caption = caption[len(prefix):].strip()
        
        class_name_lower = class_name.lower()
        descriptors = []
        for word in caption.split():
            word = word.strip('.,;:!?"\'')
            if len(word) <= 2 or word in self.STOPWORDS:
                continue
            # Drop the class name itself; it is already anchored in the prompt template
            if word == class_name_lower or word == f"{class_name_lower}s":
                continue
            descriptors.append(word)
        
        return " ".join(descriptors[:6])
    
    def save_prompts_json(self, 
                         semantic_prompts: Dict[int, List[str]],
                         class_names: Dict[int, str],
                         filepath: str,
                         include_exemplar_info: bool = True) -> None:
        """Save generated prompts to JSON file."""
        output_data = {
            'metadata': {
                'method': 'Visual Exemplar + BLIP Captioning (Option 3)',
                'num_tail_classes': len(semantic_prompts),
                'total_prompts': sum(len(prompts) for prompts in semantic_prompts.values()),
                'clip_available': self.ranking_available,
                'blip_available': self.captioning_available,
            },
            'prompts': {}
        }
        
        for class_id, prompts in semantic_prompts.items():
            class_data = {
                'class_name': class_names.get(class_id, f"class_{class_id}"),
                'prompts': prompts,
                'prompt_count': len(prompts)
            }
            
            if include_exemplar_info and class_id in self.image_database:
                exemplars = self.image_database[class_id]
                class_data['exemplar_info'] = {
                    'total_images': len(exemplars),
                    'avg_distance_to_prototype': np.mean([e.distance_to_prototype for e in exemplars])
                }
            
            output_data['prompts'][str(class_id)] = class_data
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Semantic prompts saved to: {filepath}")
    
    def _save_clip_cache(self) -> None:
        """Save CLIP embeddings cache to disk."""
        cache_file = os.path.join(self.cache_dir, "clip_embeddings_cache.pt")
        torch.save(self.clip_embeddings_cache, cache_file)
        print(f"CLIP embeddings cache saved to: {cache_file}")
    
    def _load_clip_cache(self) -> None:
        """Load CLIP embeddings cache from disk."""
        cache_file = os.path.join(self.cache_dir, "clip_embeddings_cache.pt")
        if os.path.exists(cache_file):
            self.clip_embeddings_cache = torch.load(cache_file)
            print(f"Loaded CLIP embeddings cache from: {cache_file}")
    
    def run_full_pipeline(self,
                         dataloader,
                         class_names: Dict[int, str],
                         k_exemplars: int = 5,
                         max_images_per_class: int = 1000,
                         output_dir: str = "./option3_outputs") -> Dict[int, List[str]]:
        """
        Run the complete Option 3 pipeline.
        
        Args:
            dataloader: Training data loader
            class_names: Mapping of class_id to class_name
            k_exemplars: Number of exemplars per tail class
            max_images_per_class: Max images to process per class
            output_dir: Directory to save outputs
            
        Returns:
            Generated semantic prompts for tail classes
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Build image database with CLIP embeddings
        print("Step 1: Building image database...")
        self.build_image_database(dataloader, max_images_per_class)
        
        # Step 2: Find nearest exemplars for tail classes  
        print("Step 2: Finding nearest exemplars...")
        nearest_exemplars = self.find_nearest_exemplars_for_tail_classes(k_exemplars)
        
        # Step 3: Generate captions for exemplars
        print("Step 3: Generating BLIP captions...")
        class_captions = self.generate_captions_for_exemplars(nearest_exemplars, dataloader)
        
        # Step 4: Create semantic prompts
        print("Step 4: Creating semantic prompts...")
        semantic_prompts = self.create_semantic_prompts(class_captions, class_names)
        
        # Step 5: Save results
        print("Step 5: Saving results...")
        
        # Save prompts
        prompts_file = os.path.join(output_dir, "option3_semantic_prompts.json")
        self.save_prompts_json(semantic_prompts, class_names, prompts_file)
        
        # Save detailed exemplar analysis
        analysis_file = os.path.join(output_dir, "option3_exemplar_analysis.json")
        self._save_exemplar_analysis(nearest_exemplars, class_captions, analysis_file)
        
        print(f"Option 3 pipeline complete! Results saved to {output_dir}")
        return semantic_prompts
    
    def _save_exemplar_analysis(self, 
                               nearest_exemplars: Dict[int, List[ImageExemplar]],
                               class_captions: Dict[int, List[str]],
                               filepath: str) -> None:
        """Save detailed analysis of exemplars and captions."""
        analysis = {
            'summary': {
                'total_tail_classes': len(nearest_exemplars),
                'classes_with_captions': len(class_captions),
                'avg_exemplars_per_class': np.mean([len(exemplars) for exemplars in nearest_exemplars.values()])
            },
            'class_details': {}
        }
        
        for class_id, exemplars in nearest_exemplars.items():
            captions = class_captions.get(class_id, [])
            
            analysis['class_details'][str(class_id)] = {
                'num_exemplars': len(exemplars),
                'num_captions': len(captions),
                'avg_distance_to_prototype': np.mean([e.distance_to_prototype for e in exemplars]),
                'captions': captions,
                'exemplar_distances': [e.distance_to_prototype for e in exemplars]
            }
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)