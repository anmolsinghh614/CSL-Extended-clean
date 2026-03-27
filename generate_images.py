#!/usr/bin/env python3
"""
Generate synthetic images for tail classes using the complete pipeline.
Run this after successfully running first_real_test.py
"""

import torch
import json
from pathlib import Path
from utils.memory_manager import MemoryManager
from utils.feature_to_text import FeatureToTextMapper
from utils.image_diffusion_pipeline import TailClassImageGenerator, main_generation_pipeline
from models import ResNet50  # Import your model


def load_class_names(dataset_name):
    """
    Load class names for your dataset.
    You'll need to create these mappings for your specific dataset.
    """
    if dataset_name == "imagenet":
        # For ImageNet-LT, you'll need to create this mapping
        # This is just an example - replace with your actual class names
        class_names = {}
        try:
            # Try to load from file if you have it
            with open('imagenet_class_names.json', 'r') as f:
                class_names = {int(k): v for k, v in json.load(f).items()}
        except FileNotFoundError:
            # Fallback: create generic names
            print("⚠️  No class names file found. Using generic names.")
            print("💡 Tip: Create 'imagenet_class_names.json' with actual class names for better prompts")
            class_names = {i: f"imagenet_class_{i}" for i in range(1000)}
    
    elif dataset_name == "inaturalist":
        # For iNaturalist, create similar mapping
        class_names = {}
        try:
            with open('inaturalist_class_names.json', 'r') as f:
                class_names = {int(k): v for k, v in json.load(f).items()}
        except FileNotFoundError:
            print("⚠️  No class names file found. Using generic names.")
            print("💡 Tip: Create 'inaturalist_class_names.json' with actual class names")
            # iNaturalist typically has fewer classes
            class_names = {i: f"species_{i}" for i in range(8142)}
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return class_names


def main():
    # Configuration - adjust these for your setup
    dataset_name = "imagenet"  # or "inaturalist"
    model_name = "resnet50"
    num_classes = 1000  # Adjust based on your dataset
    images_per_tail_class = 12  # Number of synthetic images per tail class
    
    print(f"🎨 Generating Synthetic Images for {dataset_name} Tail Classes")
    print("="*60)
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 GPU Memory: {gpu_memory:.1f}GB")
        if gpu_memory < 6:
            print("⚠️  Warning: <6GB GPU memory. Consider reducing batch size or using CPU.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    print("\n📋 Step 1: Loading Model and Memory Bank...")
    
    # Initialize model
    model = ResNet50(num_classes=num_classes).to(device)
    
    # Initialize memory manager
    memory_manager = MemoryManager(
        model=model,
        num_classes=num_classes,
        capacity_per_class=256,
        alpha_base=0.1,
        tail_threshold_percentile=20.0,
        device=device,
        save_dir=f'./memory_checkpoints/{dataset_name}_{model_name}'
    )
    
    # Load latest memory checkpoint
    loaded_path = memory_manager.load_latest_memory(f"{dataset_name}_{model_name}")
    if not loaded_path:
        print("❌ No memory bank checkpoint found!")
        print("💡 First train your model with: python main.py --use_memory_bank")
        return
    
    print(f"✅ Loaded memory bank from: {loaded_path}")
    
    print("\n📋 Step 2: Loading Class Names...")
    
    # Load class names
    class_names = load_class_names(dataset_name)
    print(f"✅ Loaded {len(class_names)} class names")
    
    print("\n📋 Step 3: Initializing Feature-to-Text Mapper...")
    
    # Initialize feature-to-text mapper
    mapper = FeatureToTextMapper(
        feature_dim=model.get_feature_dim(),
        device=device
    )
    print("✅ Feature-to-text mapper initialized")
    
    print("\n📋 Step 4: Running Complete Generation Pipeline...")
    print("⏳ This will take a while (downloading models + generating images)...")
    
    # Run complete pipeline
    try:
        dataset_file, quality_stats = main_generation_pipeline(
            memory_manager=memory_manager,
            feature_to_text_mapper=mapper,
            class_names=class_names,
            output_dir=f"./synthetic_tail_dataset_{dataset_name}_{model_name}"
        )
        
        print("\n🎉 Generation Complete!")
        print(f"📁 Dataset file: {dataset_file}")
        
        # Print quality statistics
        print("\n📊 Quality Statistics:")
        total_kept = sum(stats['kept_count'] for stats in quality_stats.values())
        total_generated = sum(stats['total_count'] for stats in quality_stats.values())
        avg_score = sum(stats['mean_score'] for stats in quality_stats.values()) / len(quality_stats)
        
        print(f"   Generated: {total_generated} images")
        print(f"   High-quality: {total_kept} images ({total_kept/total_generated*100:.1f}%)")
        print(f"   Average CLIP score: {avg_score:.3f}")
        
        # Show some example classes
        print(f"\n📋 Example Generated Classes (first 5):")
        for i, (class_id, stats) in enumerate(list(quality_stats.items())[:5]):
            class_name = class_names.get(class_id, f"class_{class_id}")
            print(f"   {class_id:3d}: {class_name:30s} -> {stats['kept_count']:2d} images (score: {stats['mean_score']:.3f})")
        
        print(f"\n✅ Success! Your synthetic tail class dataset is ready.")
        print(f"📁 Check the images in: ./synthetic_tail_dataset_{dataset_name}_{model_name}/images/")
        
        print(f"\n🚀 Next Steps:")
        print(f"1. Inspect generated images visually")
        print(f"2. Integrate into training with synthetic augmentation")
        print(f"3. Measure performance improvements on tail classes")
        
    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        print("💡 Common issues:")
        print("   - Internet connection needed for model downloads")
        print("   - Insufficient GPU memory (try CPU or reduce batch size)")
        print("   - Disk space (each image ~100KB, tail classes * 12 images)")


if __name__ == "__main__":
    main()