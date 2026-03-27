#!/usr/bin/env python3
"""
Generate synthetic images for CIFAR-10 tail classes using the prompts we just created.
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import json
import os
from pathlib import Path
import time

def generate_images_for_cifar10():
    """Generate synthetic images using the prompts from test_prompt_generation_cifar10.py"""
    
    print("Generating Synthetic Images for CIFAR-10 Tail Classes")
    print("="*55)
    
    # Check if prompts file exists
    prompts_file = "./cifar10_tail_class_prompts.json"
    if not Path(prompts_file).exists():
        print(f"Error: {prompts_file} not found!")
        print("Run 'python test_prompt_generation_cifar10.py' first")
        return
    
    # Load the generated prompts
    with open(prompts_file, 'r') as f:
        prompt_data = json.load(f)
    
    print(f"Loaded prompts for {len(prompt_data['prompts'])} classes")
    
    # Setup device and check GPU memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory:.1f}GB")
        if gpu_memory < 6:
            print("Warning: <6GB GPU memory. Generation will be slower.")
    
    # Load Stable Diffusion pipeline
    print("Loading Stable Diffusion pipeline...")
    print("Note: First run will download ~5GB of models - this may take 10-20 minutes")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Use faster scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("Stable Diffusion loaded successfully")
        
    except Exception as e:
        print(f"Error loading Stable Diffusion: {e}")
        print("This usually means insufficient GPU memory or network issues")
        return
    
    # Generation parameters
    # NOTE: Stable Diffusion was trained at 512×512 and generates unusable images at 32×32.
    # Generate at 512×512 (SD-native), then resize to 32×32 during preprocessing/feature extraction.
    num_images_per_prompt = 4
    num_inference_steps = 25
    guidance_scale = 7.5
    height = 512
    width = 512
    
    print(f"Generation settings:")
    print(f"  Images per prompt: {num_images_per_prompt}")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Image size: {width}x{height} (resized to 32x32 during preprocessing)")
    
    # Create output directory
    output_dir = Path("./synthetic_cifar10_images")
    output_dir.mkdir(exist_ok=True)
    
    # Generate images for each class
    total_generated = 0
    generation_stats = {}
    
    for class_id_str, class_info in prompt_data['prompts'].items():
        class_id = int(class_id_str)
        class_name = class_info['class_name']
        prompts = class_info['prompts']
        
        print(f"\nGenerating images for Class {class_id}: {class_name}")
        print(f"Using {len(prompts)} prompts")
        
        # Create class directory
        class_dir = output_dir / f"class_{class_id}_{class_name}"
        class_dir.mkdir(exist_ok=True)
        
        class_images = []
        
        for prompt_idx, prompt in enumerate(prompts):
            print(f"  Prompt {prompt_idx+1}: '{prompt[:50]}...'")
            
            start_time = time.time()
            
            try:
                # Generate images
                images = pipe(
                    prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=torch.Generator(device=device).manual_seed(42 + prompt_idx)
                ).images
                
                # Save images
                for img_idx, image in enumerate(images):
                    image_filename = f"synthetic_{class_id}_{prompt_idx}_{img_idx}.jpg"
                    image_path = class_dir / image_filename
                    image.save(image_path, quality=95)
                    class_images.append(str(image_path))
                
                generation_time = time.time() - start_time
                print(f"    Generated {len(images)} images in {generation_time:.1f}s")
                
            except Exception as e:
                print(f"    Error generating images: {e}")
                continue
        
        total_generated += len(class_images)
        generation_stats[class_id] = {
            'class_name': class_name,
            'images_generated': len(class_images),
            'image_paths': class_images
        }
        
        print(f"  Total images for {class_name}: {len(class_images)}")
    
    # Save generation metadata
    metadata = {
        'generation_settings': {
            'model': 'runwayml/stable-diffusion-v1-5',
            'num_images_per_prompt': num_images_per_prompt,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'image_size': f"{width}x{height}",
            'device': device
        },
        'classes': generation_stats,
        'total_images': total_generated
    }
    
    metadata_file = output_dir / "generation_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGeneration Complete!")
    print(f"="*30)
    print(f"Total images generated: {total_generated}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved: {metadata_file}")
    
    # Show breakdown by class
    print(f"\nBreakdown by class:")
    for class_id, stats in generation_stats.items():
        print(f"  Class {class_id} ({stats['class_name']}): {stats['images_generated']} images")
    
    # Create simple training dataset file
    print(f"\nCreating training dataset file...")
    dataset_file = output_dir / "synthetic_cifar10_training.txt"
    with open(dataset_file, 'w') as f:
        for class_id, stats in generation_stats.items():
            for image_path in stats['image_paths']:
                f.write(f"{image_path} {class_id}\n")
    
    print(f"Dataset file created: {dataset_file}")
    print(f"Format: image_path class_id")
    
    print(f"\nNext steps:")
    print(f"1. Review generated images in: {output_dir}")
    print(f"2. Check image quality manually")  
    print(f"3. Use {dataset_file} for training augmentation")
    
    return str(output_dir), str(dataset_file)

if __name__ == "__main__":
    generate_images_for_cifar10()