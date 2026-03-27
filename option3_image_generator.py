"""
Generate synthetic images from Option 3 semantic prompts using diffusion models.
FIXED: Better progress tracking, memory management, and performance
"""

import torch
import json
import os
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
from datetime import datetime
import sys

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"Diffusers not available ({type(e).__name__}): {e}")
    print("SD image generation will be disabled. Install/fix with: pip install diffusers transformers accelerate")
    DIFFUSERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("Install OpenAI: pip install openai")
    OPENAI_AVAILABLE = False


class Option3ImageGenerator:
    """Generate synthetic images from Option 3 semantic prompts."""
    
    def __init__(self, 
                 model_type: str = "stable_diffusion",
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda",
                 output_dir: str = "./generated_images"):
        """
        Initialize image generator.
        
        Args:
            model_type: "stable_diffusion", "dalle", or "local"
            model_id: Model identifier for Hugging Face or local path
            device: Device for inference
            output_dir: Directory to save generated images
        """
        self.model_type = model_type
        self.model_id = model_id
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the selected model
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the selected diffusion model."""
        if self.model_type == "stable_diffusion":
            self._initialize_stable_diffusion()
        elif self.model_type == "dalle":
            self._initialize_dalle()
        elif self.model_type == "local":
            self._initialize_local_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _initialize_stable_diffusion(self):
        """Initialize Stable Diffusion model."""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Diffusers not available. Install with: pip install diffusers transformers accelerate")
        
        print(f"Loading Stable Diffusion model: {self.model_id}")
        
        # Load the pipeline with memory-efficient options
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True,         # Lazy loading — don't load all weights into RAM at once
                variant="fp16" if self.device == "cuda" else None  # Download fp16 weights directly
            )
        except Exception as e:
            # Fallback without variant (some models don't have fp16 variant)
            print(f"  Note: fp16 variant not available, loading default: {e}")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
        
        # Optimize for speed and memory
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        if self.device == "cuda":
            # Use sequential CPU offload — only loads one component to GPU at a time
            # This uses ~2GB VRAM instead of ~4GB for the full model
            try:
                self.pipeline.enable_sequential_cpu_offload()
                print("✓ Sequential CPU offload enabled (low VRAM mode)")
            except Exception:
                # Fallback to standard GPU loading
                self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_attention_slicing()
            # Enable xformers for faster inference if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("✓ xformers memory efficient attention enabled")
            except:
                pass
        
        print("✓ Stable Diffusion loaded successfully")
        
    def generate_batch(self, 
                  prompts: List[str],
                  class_idx: int,
                  num_images_per_prompt: int = 1,
                  batch_size: int = 1,
                  guidance_scale: float = 7.5,
                  num_inference_steps: int = 50,
                  image_size: int = 512) -> List[str]:
        """
        Generate batch of images for given prompts.
        
        Args:
            prompts: List of text prompts
            class_idx: Class ID for organizing outputs
            num_images_per_prompt: Images to generate per prompt
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of diffusion steps
            image_size: Output image size (SD works best at 512)
        
        Returns:
            List of paths to generated images
        """
        generated_paths = []
    
        for prompt_idx, prompt in enumerate(prompts):
            for img_idx in range(num_images_per_prompt):
                # Generate image at SD-native resolution (512×512)
                # Images are later resized to dataset resolution (e.g. 32×32) during preprocessing
                image = self.pipeline(
                    prompt,
                    height=image_size,
                    width=image_size,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]
            
                # Save image
                filename = f"class{class_idx}_prompt{prompt_idx}_img{img_idx}.png"
                save_path = os.path.join(self.output_dir, filename)
                image.save(save_path)
                generated_paths.append(save_path)
    
        return generated_paths
    def _initialize_dalle(self):
        """Initialize DALL-E API."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")
        print("✓ DALL-E API initialized (remember to set your API key)")
    
    def _initialize_local_model(self):
        """Initialize local diffusion model."""
        print(f"Loading local model from: {self.model_id}")
        pass
    
    def generate_images_from_prompts(self,
                                   prompts_file: str,
                                   images_per_prompt: int = 10,
                                   image_size: int = 512,
                                   inference_steps: int = 20,
                                   guidance_scale: float = 7.5) -> Dict[int, List[str]]:
        """
        Generate images from Option 3 prompts file.
        
        Args:
            prompts_file: Path to JSON file with Option 3 prompts
            images_per_prompt: Number of images to generate per prompt
            image_size: Output image size (512, 768, 1024)
            inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Dictionary mapping class_id to list of generated image paths
        """
        # Load prompts
        with open(prompts_file, 'r') as f:
            prompt_data = json.load(f)
        
        prompts_dict = prompt_data['prompts']
        generated_images = {}
        
        # Calculate total images for progress tracking
        total_images = sum(
            len(class_data['prompts']) * images_per_prompt 
            for class_data in prompts_dict.values()
        )
        current_image = 0
        
        print(f"\nGenerating {total_images} images for {len(prompts_dict)} tail classes...")
        print(f"Config: {image_size}x{image_size}, {inference_steps} steps, guidance={guidance_scale}")
        print("="*70)
        
        for class_id_str, class_data in prompts_dict.items():
            class_id = int(class_id_str)
            class_name = class_data['class_name']
            prompts = class_data['prompts']
            
            print(f"\n[Class {class_id}/{len(prompts_dict)}] {class_name}")
            print("-"*70)
            
            class_dir = os.path.join(self.output_dir, f"class_{class_id}_{class_name}")
            os.makedirs(class_dir, exist_ok=True)
            
            class_image_paths = []
            
            for prompt_idx, prompt in enumerate(prompts):
                # Generate multiple images per prompt
                for img_idx in range(images_per_prompt):
                    current_image += 1
                    
                    # Progress indicator
                    progress_pct = (current_image / total_images) * 100
                    print(f"[{current_image}/{total_images} | {progress_pct:.1f}%] "
                          f"Prompt {prompt_idx+1}/{len(prompts)}, Image {img_idx+1}/{images_per_prompt}", 
                          end='', flush=True)
                    
                    image_path = self._generate_single_image(
                        prompt=prompt,
                        class_id=class_id,
                        class_name=class_name,
                        prompt_idx=prompt_idx,
                        img_idx=img_idx,
                        class_dir=class_dir,
                        image_size=image_size,
                        inference_steps=inference_steps,
                        guidance_scale=guidance_scale
                    )
                    
                    if image_path:
                        class_image_paths.append(image_path)
                        print(" ✓", flush=True)
                    else:
                        print(" ✗", flush=True)
                    
                    # Clear GPU cache periodically
                    if self.device == "cuda" and current_image % 5 == 0:
                        torch.cuda.empty_cache()
            
            generated_images[class_id] = class_image_paths
            print(f"\n✓ Class {class_id} complete: {len(class_image_paths)}/{len(prompts) * images_per_prompt} images generated")
        
        # Save generation summary
        self._save_generation_summary(generated_images, prompts_file)
        
        return generated_images
    
    def _generate_single_image(self,
                             prompt: str,
                             class_id: int,
                             class_name: str,
                             prompt_idx: int,
                             img_idx: int,
                             class_dir: str,
                             image_size: int,
                             inference_steps: int,
                             guidance_scale: float) -> Optional[str]:
        """Generate a single image from prompt."""
        
        filename = f"class_{class_id}_prompt_{prompt_idx}_img_{img_idx}.png"
        image_path = os.path.join(class_dir, filename)
        
        try:
            if self.model_type == "stable_diffusion":
                image = self._generate_with_stable_diffusion(
                    prompt, image_size, inference_steps, guidance_scale
                )
            elif self.model_type == "dalle":
                image = self._generate_with_dalle(prompt, image_size)
            elif self.model_type == "local":
                image = self._generate_with_local_model(prompt, image_size)
            else:
                return None
            
            # Save image
            if image:
                image.save(image_path)
                return image_path
            
        except Exception as e:
            print(f"\n    ✗ Error: {e}", flush=True)
            
        return None
    
    def _generate_with_stable_diffusion(self,
                                      prompt: str,
                                      image_size: int,
                                      inference_steps: int,
                                      guidance_scale: float) -> Optional[Image.Image]:
        """Generate image using Stable Diffusion."""
        try:
            # Disable progress bar for cleaner output
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    height=image_size,
                    width=image_size,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(
                        np.random.randint(0, 2**31-1)
                    ),
                    # Disable progress bar
                    callback=None,
                    callback_steps=None
                )
            
            return result.images[0]
            
        except Exception as e:
            raise Exception(f"Stable Diffusion error: {e}")
    
    def _generate_with_dalle(self, prompt: str, image_size: int) -> Optional[Image.Image]:
        """Generate image using DALL-E API."""
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size=f"{image_size}x{image_size}"
            )
            
            import requests
            import io
            image_url = response['data'][0]['url']
            image_response = requests.get(image_url)
            image = Image.open(io.BytesIO(image_response.content))
            return image
            
        except Exception as e:
            raise Exception(f"DALL-E error: {e}")
    
    def _generate_with_local_model(self, prompt: str, image_size: int) -> Optional[Image.Image]:
        """Generate image using local model."""
        print(f"Local model generation for: {prompt}")
        return None
    
    def _save_generation_summary(self, generated_images: Dict[int, List[str]], prompts_file: str):
        """Save summary of generated images."""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'source_prompts_file': prompts_file,
            'model_type': self.model_type,
            'model_id': self.model_id,
            'total_classes': len(generated_images),
            'total_images': sum(len(paths) for paths in generated_images.values()),
            'classes': {}
        }
        
        for class_id, image_paths in generated_images.items():
            summary['classes'][str(class_id)] = {
                'num_images': len(image_paths),
                'image_paths': image_paths
            }
        
        summary_path = os.path.join(self.output_dir, 'generation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Generation summary saved to: {summary_path}")


def quick_generate_test_images():
    """Quick function to generate images from your test results."""
    
    print("\n" + "="*70)
    print("OPTION 3: SYNTHETIC IMAGE GENERATION")
    print("="*70)
    
    # Check if test prompts exist
    test_prompts_file = "./option3_test_results/test_prompts.json"
    if not os.path.exists(test_prompts_file):
        print(f"✗ Test prompts file not found: {test_prompts_file}")
        print("Run option3_fixed.py first to generate prompts")
        return
    
    # Initialize generator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    generator = Option3ImageGenerator(
        model_type="stable_diffusion",
        model_id="runwayml/stable-diffusion-v1-5",
        device=device,
        output_dir="./option3_generated_images"
    )
    
    # FAST SETTINGS FOR TESTING
    generated_images = generator.generate_images_from_prompts(
        prompts_file=test_prompts_file,
        images_per_prompt=2,     # Only 2 images per prompt for testing
        image_size=512,          # SD-native resolution for best quality
        inference_steps=20,      # 20 steps is good balance
        guidance_scale=7.5
    )
    
    # Print results
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    
    total_images = sum(len(paths) for paths in generated_images.values())
    print(f"✓ Successfully generated {total_images} images for {len(generated_images)} tail classes")
    
    for class_id, image_paths in sorted(generated_images.items()):
        print(f"  • Class {class_id}: {len(image_paths)} images")
    
    print(f"\n✓ Images saved to: ./option3_generated_images/")
    print(f"✓ Summary: ./option3_generated_images/generation_summary.json")


def generate_production_quality():
    """Generate high-quality images for final use."""
    
    print("\n" + "="*70)
    print("PRODUCTION QUALITY IMAGE GENERATION")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    generator = Option3ImageGenerator(
        model_type="stable_diffusion",
        model_id="runwayml/stable-diffusion-v1-5",
        device=device,
        output_dir="./option3_production_images"
    )
    
    prompts_file = "./option3_test_results/test_prompts.json"
    
    # HIGH QUALITY SETTINGS
    generator.generate_images_from_prompts(
        prompts_file=prompts_file,
        images_per_prompt=10,    # 10 variations per prompt
        image_size=512,          # Standard high quality
        inference_steps=50,      # More steps = better quality
        guidance_scale=7.5
    )


def create_balanced_dataset():
    """Create a balanced dataset by combining generated images with original data."""
    
    print("\n" + "="*70)
    print("CREATING BALANCED DATASET")
    print("="*70)
    
    # Load generation summary
    summary_file = "./option3_generated_images/generation_summary.json"
    if not os.path.exists(summary_file):
        print("✗ No generation summary found. Generate images first.")
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Create balanced dataset structure
    balanced_dir = "./balanced_dataset"
    os.makedirs(balanced_dir, exist_ok=True)
    
    for class_id_str, class_data in summary['classes'].items():
        class_id = int(class_id_str)
        num_generated = class_data['num_images']
        
        # Create class directory
        class_dir = os.path.join(balanced_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy generated images
        generated_dir = os.path.join(class_dir, "generated")
        os.makedirs(generated_dir, exist_ok=True)
        
        import shutil
        for i, image_path in enumerate(class_data['image_paths']):
            if os.path.exists(image_path):
                dest_path = os.path.join(generated_dir, f"generated_{i}.png")
                shutil.copy2(image_path, dest_path)
        
        print(f"✓ Class {class_id}: {num_generated} synthetic images added")
    
    print(f"\n✓ Balanced dataset created in: {balanced_dir}")


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_generate_test_images()
        elif sys.argv[1] == "production":
            generate_production_quality()
        elif sys.argv[1] == "balance":
            create_balanced_dataset()
        else:
            print(f"Unknown command: {sys.argv[1]}")
    else:
        print("\n" + "="*70)
        print("OPTION 3 IMAGE GENERATOR")
        print("="*70)
        print("\nUsage:")
        print("  python option3_image_generator.py test        # Quick test (2 imgs/prompt)")
        print("  python option3_image_generator.py production  # High quality (10 imgs/prompt)")
        print("  python option3_image_generator.py balance     # Create balanced dataset")
        print("\nOr import and use the Option3ImageGenerator class directly")
        print("="*70)