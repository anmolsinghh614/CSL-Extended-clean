import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
from pathlib import Path
try:
    import open_clip
    CLIP_AVAILABLE = 'open_clip'
except ImportError:
    try:
        import clip
        CLIP_AVAILABLE = 'clip'
    except ImportError:
        print("Warning: No CLIP library available. Quality assessment will be disabled.")
        CLIP_AVAILABLE = None


class TailClassImageGenerator:
    """
    Generates synthetic images for tail classes using semantic prompts
    derived from CSL memory bank features.
    """
    
    def __init__(self, 
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda",
                 seed: Optional[int] = None):
        """
        Initialize the image generation pipeline.
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
            device: Device to run inference on
            seed: Random seed for reproducible generation
        """
        self.device = device
        self.seed = seed
        
        # Load Stable Diffusion pipeline
        print("Loading Stable Diffusion pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # Disable for research use
            requires_safety_checker=False
        ).to(device)
        
        # Use DPM solver for faster generation
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Load CLIP for quality assessment
        if CLIP_AVAILABLE == 'open_clip':
            print("Loading OpenCLIP for quality assessment...")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai', device=device
            )
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        elif CLIP_AVAILABLE == 'clip':
            print("Loading CLIP for quality assessment...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.clip_tokenizer = clip.tokenize
        else:
            print("CLIP not available - quality assessment disabled")
            self.clip_model = None
            self.clip_preprocess = None
            self.clip_tokenizer = None
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        print("Image generation pipeline ready!")
    
    def generate_images_for_class(self,
                                class_id: int,
                                prompts: List[str],
                                num_images_per_prompt: int = 4,
                                num_inference_steps: int = 25,
                                guidance_scale: float = 7.5,
                                height: int = 512,
                                width: int = 512) -> List[Image.Image]:
        """
        Generate synthetic images for a specific tail class.
        
        Args:
            class_id: Class identifier
            prompts: List of text prompts for the class
            num_images_per_prompt: Number of images to generate per prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            height, width: Image dimensions
            
        Returns:
            List of generated PIL Images
        """
        generated_images = []
        
        for i, prompt in enumerate(prompts):
            print(f"Generating {num_images_per_prompt} images for class {class_id}, prompt {i+1}/{len(prompts)}")
            
            # Generate batch of images
            images = self.pipe(
                prompt,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=torch.Generator(device=self.device).manual_seed(self.seed + i) if self.seed else None
            ).images
            
            generated_images.extend(images)
        
        return generated_images
    
    def generate_tail_class_dataset(self,
                                  prompts_dict: Dict[int, List[str]],
                                  class_names: Dict[int, str],
                                  output_dir: str = "./synthetic_tail_images",
                                  images_per_class: int = 20,
                                  **generation_kwargs) -> Dict[int, List[str]]:
        """
        Generate complete synthetic dataset for all tail classes.
        
        Args:
            prompts_dict: Dictionary mapping class_id to list of prompts
            class_names: Dictionary mapping class_id to class_name
            output_dir: Directory to save generated images
            images_per_class: Target number of images per class
            **generation_kwargs: Additional arguments for generate_images_for_class
            
        Returns:
            Dictionary mapping class_id to list of saved image paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        generated_paths = {}
        metadata = {
            'generation_config': generation_kwargs,
            'total_classes': len(prompts_dict),
            'images_per_class': images_per_class,
            'classes': {}
        }
        
        for class_id, prompts in prompts_dict.items():
            class_name = class_names.get(class_id, f"class_{class_id}")
            class_dir = Path(output_dir) / f"class_{class_id}_{class_name.replace(' ', '_')}"
            class_dir.mkdir(exist_ok=True)
            
            print(f"\n=== Generating images for {class_name} (Class {class_id}) ===")
            
            # Calculate images per prompt
            images_per_prompt = max(1, images_per_class // len(prompts))
            
            # Generate images
            generated_images = self.generate_images_for_class(
                class_id=class_id,
                prompts=prompts,
                num_images_per_prompt=images_per_prompt,
                **generation_kwargs
            )
            
            # Save images and collect paths
            saved_paths = []
            for i, image in enumerate(generated_images):
                image_path = class_dir / f"synthetic_{class_id}_{i:04d}.jpg"
                image.save(image_path, quality=95)
                saved_paths.append(str(image_path))
            
            generated_paths[class_id] = saved_paths
            
            # Update metadata
            metadata['classes'][class_id] = {
                'class_name': class_name,
                'prompts': prompts,
                'num_images': len(saved_paths),
                'image_paths': saved_paths
            }
            
            print(f"Saved {len(saved_paths)} images for {class_name}")
        
        # Save metadata
        metadata_path = Path(output_dir) / "generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nGeneration complete! Metadata saved to: {metadata_path}")
        return generated_paths
    
    def assess_image_quality(self, 
                           image: Image.Image, 
                           prompt: str) -> Dict[str, float]:
        """
        Assess quality of generated image using CLIP similarity.
        
        Args:
            image: Generated PIL Image
            prompt: Text prompt used for generation
            
        Returns:
            Dictionary containing quality metrics
        """
        if self.clip_model is None:
            return {
                'clip_similarity': 0.5,  # Default neutral score
                'prompt': prompt,
                'quality_score': 0.5
            }
        
        with torch.no_grad():
            # Preprocess image for CLIP
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text based on CLIP implementation
            if CLIP_AVAILABLE == 'open_clip':
                text_input = self.clip_tokenizer([prompt]).to(self.device)
            else:  # original clip
                text_input = self.clip_tokenizer([prompt]).to(self.device)
            
            # Get embeddings
            if CLIP_AVAILABLE == 'open_clip':
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
            else:  # original clip
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
            
            # Compute similarity
            clip_similarity = torch.cosine_similarity(image_features, text_features).item()
            
            # Additional quality metrics could be added here
            # (e.g., FID, IS, aesthetic scores)
            
            return {
                'clip_similarity': clip_similarity,
                'prompt': prompt,
                'quality_score': clip_similarity  # Simple quality score for now
            }
    
    def filter_high_quality_images(self,
                                 generated_paths: Dict[int, List[str]],
                                 prompts_dict: Dict[int, List[str]],
                                 quality_threshold: float = 0.25) -> Dict[int, List[str]]:
        """
        Filter generated images based on CLIP similarity scores.
        
        Args:
            generated_paths: Dict mapping class_id to list of image paths
            prompts_dict: Dict mapping class_id to list of prompts
            quality_threshold: Minimum CLIP similarity score to keep image
            
        Returns:
            Filtered dictionary of high-quality image paths
        """
        filtered_paths = {}
        quality_stats = {}
        
        for class_id, image_paths in generated_paths.items():
            class_prompts = prompts_dict.get(class_id, [""])
            high_quality_paths = []
            scores = []
            
            for i, image_path in enumerate(image_paths):
                # Load image
                image = Image.open(image_path)
                
                # Get corresponding prompt
                prompt_idx = i // (len(image_paths) // max(1, len(class_prompts)))
                prompt = class_prompts[min(prompt_idx, len(class_prompts) - 1)]
                
                # Assess quality
                quality_metrics = self.assess_image_quality(image, prompt)
                scores.append(quality_metrics['clip_similarity'])
                
                # Keep if above threshold
                if quality_metrics['quality_score'] >= quality_threshold:
                    high_quality_paths.append(image_path)
            
            filtered_paths[class_id] = high_quality_paths
            quality_stats[class_id] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'kept_ratio': len(high_quality_paths) / len(image_paths),
                'kept_count': len(high_quality_paths),
                'total_count': len(image_paths)
            }
            
            print(f"Class {class_id}: Kept {len(high_quality_paths)}/{len(image_paths)} "
                  f"images (mean CLIP score: {np.mean(scores):.3f})")
        
        return filtered_paths, quality_stats
    
    def create_training_dataset(self,
                              filtered_paths: Dict[int, List[str]],
                              output_file: str = "./synthetic_training_data.txt") -> str:
        """
        Create training dataset file compatible with existing data loaders.
        
        Args:
            filtered_paths: Dict mapping class_id to list of high-quality image paths
            output_file: Path to output dataset file
            
        Returns:
            Path to created dataset file
        """
        with open(output_file, 'w') as f:
            for class_id, image_paths in filtered_paths.items():
                for image_path in image_paths:
                    # Format: image_path class_id
                    f.write(f"{image_path} {class_id}\n")
        
        total_images = sum(len(paths) for paths in filtered_paths.values())
        print(f"Created training dataset with {total_images} synthetic images")
        print(f"Dataset file: {output_file}")
        
        return output_file


def main_generation_pipeline(memory_manager,
                           feature_to_text_mapper,
                           class_names: Dict[int, str],
                           output_dir: str = "./synthetic_tail_dataset"):
    """
    Complete pipeline: Memory Bank -> Prompts -> Images -> Training Data
    """
    print("=== TAIL CLASS SYNTHETIC IMAGE GENERATION PIPELINE ===\n")
    
    # Step 1: Generate semantic prompts
    print("Step 1: Generating semantic prompts from memory bank...")
    tail_prompts = feature_to_text_mapper.generate_prompts_for_tail_classes(
        memory_manager=memory_manager,
        class_names=class_names,
        num_prompts_per_class=3
    )
    
    # Save prompts
    prompts_file = f"{output_dir}/tail_class_prompts.json"
    feature_to_text_mapper.save_prompts_to_file(tail_prompts, class_names, prompts_file)
    
    # Step 2: Generate images
    print("\nStep 2: Generating synthetic images...")
    generator = TailClassImageGenerator(device="cuda", seed=42)
    
    generated_paths = generator.generate_tail_class_dataset(
        prompts_dict=tail_prompts,
        class_names=class_names,
        output_dir=f"{output_dir}/images",
        images_per_class=24,  # Generate 24 images per tail class
        num_inference_steps=25,
        guidance_scale=7.5
    )
    
    # Step 3: Quality filtering
    print("\nStep 3: Filtering high-quality images...")
    filtered_paths, quality_stats = generator.filter_high_quality_images(
        generated_paths=generated_paths,
        prompts_dict=tail_prompts,
        quality_threshold=0.22
    )
    
    # Step 4: Create training dataset
    print("\nStep 4: Creating training dataset...")
    dataset_file = generator.create_training_dataset(
        filtered_paths=filtered_paths,
        output_file=f"{output_dir}/synthetic_tail_training.txt"
    )
    
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Generated synthetic images for {len(tail_prompts)} tail classes")
    print(f"Final dataset: {dataset_file}")
    
    return dataset_file, quality_stats