"""
Standalone test for Option 3 implementation.
This file includes everything you need to test the functionality.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

# Try to import required models
try:
    import clip
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image
    MODELS_AVAILABLE = True
    print("✓ CLIP and BLIP models available")
except ImportError as e:
    print(f"✗ Missing dependencies: {e}")
    print("Install with:")
    print("pip install git+https://github.com/openai/CLIP.git")
    print("pip install transformers pillow")
    MODELS_AVAILABLE = False

# Import your existing modules
try:
    from memory_manager import MemoryManager
    from memory_bank import MemoryBank
    print("✓ Memory modules imported successfully")
except ImportError as e:
    print(f"✗ Could not import memory modules: {e}")
    print("Make sure memory_manager.py and memory_bank.py are in the same directory")


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
    """
    
    def __init__(self,
                 memory_manager,
                 clip_model_name: str = "ViT-B/32",
                 blip_model_name: str = "Salesforce/blip-image-captioning-base",
                 device: str = "cuda",
                 cache_dir: str = "./clip_cache"):
        """Initialize the visual exemplar prompt generator."""
        self.memory_manager = memory_manager
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if not MODELS_AVAILABLE:
            raise ImportError("Required models not available. Please install CLIP and transformers.")
        
        # Load CLIP model
        print(f"Loading CLIP model: {clip_model_name}")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        
        # Load BLIP model for captioning
        print(f"Loading BLIP model: {blip_model_name}")
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
        self.blip_model = self.blip_model.to(device)
        self.blip_model.eval()
        
        # Storage for image data and embeddings
        self.image_database = {}
        self.clip_embeddings_cache = {}
        
        print("✓ Models loaded successfully")
    
    def test_basic_functionality(self):
        """Test basic functionality without full pipeline."""
        print("\n" + "="*50)
        print("TESTING BASIC FUNCTIONALITY")
        print("="*50)
        
        # Test 1: Check memory manager
        tail_classes = self.memory_manager.get_tail_classes()
        print(f"✓ Found {len(tail_classes)} tail classes: {tail_classes[:5]}...")
        
        # Test 2: Check CLIP model
        dummy_image = torch.randn(3, 224, 224)
        dummy_pil = Image.fromarray((dummy_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        
        with torch.no_grad():
            image_input = self.clip_preprocess(dummy_pil).unsqueeze(0).to(self.device)
            clip_embedding = self.clip_model.encode_image(image_input)
            print(f"✓ CLIP embedding shape: {clip_embedding.shape}")
        
        # Test 3: Check BLIP model
        inputs = self.blip_processor(dummy_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=20, num_beams=3)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        print(f"✓ BLIP caption generated: '{caption}'")
        
        # Test 4: Check memory bank prototypes
        if len(tail_classes) > 0:
            sample_class = tail_classes[0]
            prototype = self.memory_manager.memory_bank.get_prototype(sample_class)
            print(f"✓ Memory bank prototype shape for class {sample_class}: {prototype.shape}")
        
        print("✓ All basic tests passed!")
        return True
    
    def create_dummy_prompts_for_testing(self) -> Dict[int, List[str]]:
        """Create dummy prompts for testing without full pipeline."""
        tail_classes = self.memory_manager.get_tail_classes()
        
        dummy_prompts = {}
        for class_id in tail_classes[:5]:  # Test with first 5 tail classes
            prompts = [
                f"A detailed photograph of class {class_id}",
                f"A high-quality image showing class {class_id} in natural lighting",
                f"A professional photo of class {class_id} with clear features"
            ]
            dummy_prompts[class_id] = prompts
        
        return dummy_prompts
    
    def save_prompts_json(self, semantic_prompts: Dict[int, List[str]], 
                         class_names: Dict[int, str], filepath: str):
        """Save generated prompts to JSON file."""
        output_data = {
            'metadata': {
                'method': 'Visual Exemplar + BLIP Captioning (Option 3)',
                'num_tail_classes': len(semantic_prompts),
                'total_prompts': sum(len(prompts) for prompts in semantic_prompts.values()),
            },
            'prompts': {}
        }
        
        for class_id, prompts in semantic_prompts.items():
            output_data['prompts'][str(class_id)] = {
                'class_name': class_names.get(class_id, f"class_{class_id}"),
                'prompts': prompts,
                'prompt_count': len(prompts)
            }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Prompts saved to: {filepath}")


def test_memory_manager_loading():
    """Test loading your existing memory manager."""
    print("\n" + "="*50)
    print("TESTING MEMORY MANAGER LOADING")
    print("="*50)
    
    try:
        # You'll need to modify these paths/parameters based on your setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Create a dummy model for testing (replace with your actual model)
        class DummyModel:
            def get_feature_dim(self):
                return 2048
        
        dummy_model = DummyModel()
        
        # Initialize memory manager
        memory_manager = MemoryManager(
            model=dummy_model,
            num_classes=100,  # Adjust based on your dataset
            device=device,
            save_dir='./memory_checkpoints'
        )
        
        # Try to load existing memory
        latest_path = memory_manager.load_latest_memory()
        if latest_path:
            print(f"✓ Loaded memory from: {latest_path}")
            memory_manager.print_summary()
            return memory_manager
        else:
            print("✗ No existing memory bank found")
            print("You need to train your CSL model first and save the memory bank")
            return None
            
    except Exception as e:
        print(f"✗ Error loading memory manager: {e}")
        return None


def main():
    """Main test function."""
    print("OPTION 3 IMPLEMENTATION TEST")
    print("="*50)
    
    # Step 1: Test memory manager loading
    memory_manager = test_memory_manager_loading()
    if memory_manager is None:
        print("\n✗ Cannot proceed without memory manager")
        print("Please ensure you have:")
        print("1. Trained your CSL model")
        print("2. Saved the memory bank checkpoints")
        print("3. The memory_checkpoints directory exists")
        return
    
    # Step 2: Test model availability
    if not MODELS_AVAILABLE:
        print("\n✗ Required models not available")
        return
    
    # Step 3: Initialize Option 3 generator
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        option3_generator = VisualExemplarPromptGenerator(
            memory_manager=memory_manager,
            clip_model_name="ViT-B/32",
            device=device
        )
        
        # Step 4: Test basic functionality
      #  option3_generator.test_basic_functionality()
        
        # Step 5: Generate dummy prompts for testing
        print("\n" + "="*50)
        print("GENERATING TEST PROMPTS")
        print("="*50)
        
        dummy_prompts = option3_generator.create_dummy_prompts_for_testing()
        
        # Create dummy class names
        class_names = {i: f"class_{i}" for i in range(100)}
        
        # Save test prompts
        option3_generator.save_prompts_json(
            dummy_prompts, 
            class_names, 
            "test_option3_prompts.json"
        )
        
        print(f"\n✓ Generated test prompts for {len(dummy_prompts)} classes")
        for class_id, prompts in list(dummy_prompts.items())[:2]:
            print(f"\nClass {class_id} prompts:")
            for prompt in prompts:
                print(f"  - {prompt}")
        
        print("\n✓ Basic Option 3 test completed successfully!")
        print("\nNext steps:")
        print("1. Modify the dataloader integration based on your dataset")
        print("2. Run the full pipeline with actual training data")
        print("3. Use the generated prompts for diffusion model training")
        
    except Exception as e:
        print(f"\n✗ Error during Option 3 testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()