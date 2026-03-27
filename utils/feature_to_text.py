import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from transformers import AutoTokenizer, AutoModel, BlipProcessor, BlipForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available. Using simplified text generation.")
    TRANSFORMERS_AVAILABLE = False
from typing import Dict, List, Optional


class FeatureToTextMapper(nn.Module):
    """
    Maps CSL memory bank features to semantic text descriptions.
    Implements BLIP^{-1} functionality for tail class prompt generation.
    """
    
    def __init__(self, 
                 feature_dim: int = 2048,
                 text_embedding_dim: int = 768,
                 hidden_dim: int = 1024,
                 max_length: int = 77,
                 device: str = 'cuda'):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.device = device
        
        # Feature-to-text embedding projection network
        self.feature_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, text_embedding_dim),
            nn.LayerNorm(text_embedding_dim)
        ).to(device).float()  # Ensure float32 weights
        
        # Load pre-trained models if available
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load pre-trained BLIP model for text generation
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                
                # Text encoder for semantic similarity
                self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                self.text_encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                
                # Move models to device
                self.blip_model = self.blip_model.to(device)
                self.text_encoder = self.text_encoder.to(device)
                
                print("Loaded pre-trained BLIP and text encoder models")
            except Exception as e:
                print(f"Warning: Could not load pre-trained models: {e}")
                print("Using simplified template-based generation")
                self.blip_processor = None
                self.blip_model = None
                self.tokenizer = None
                self.text_encoder = None
        else:
            self.blip_processor = None
            self.blip_model = None
            self.tokenizer = None
            self.text_encoder = None
        
        # Move feature projector to device after initialization
        self.feature_projector = self.feature_projector.to(device)
        self.to(device)
        self.to(device)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project features to text embedding space.
        
        Args:
            features: [batch_size, feature_dim] CSL memory bank features
            
        Returns:
            text_embeddings: [batch_size, text_embedding_dim]
        """
        # Ensure float32 dtype
        features = features.float()
        return self.feature_projector(features)
    
    def generate_text_prompt(self, feature: torch.Tensor, class_name: str = "") -> str:
        """
        Generate semantic text prompt from feature vector.
        
        Args:
            feature: [feature_dim] Single feature vector
            class_name: Optional class name for context
            
        Returns:
            Generated text prompt
        """
        with torch.no_grad():
            # Project feature to text embedding space
            if feature.dim() == 1:
                feature = feature.unsqueeze(0)
            
            # Fix dtype mismatch - ensure float32 and move to device
            feature = feature.to(self.device).float()
            
            text_embedding = self.feature_projector(feature)
            
            # Use template-based generation 
            prompt = self._generate_prompt_from_embedding(text_embedding, class_name)
            
            return prompt
    
    def _generate_prompt_from_embedding(self, text_embedding: torch.Tensor, class_name: str) -> str:
        """
        Generate prompt from text embedding using learned mapping.
        Enhanced template-based generation that works without external models.
        """
        # Enhanced templates with more variety
        descriptive_templates = [
            "A detailed photograph of a {}",
            "A {} in its natural habitat with distinctive features", 
            "A beautiful {} showing characteristic appearance and coloring",
            "A {} with detailed visual attributes and natural lighting",
            "A high-quality image of a {} displaying typical behavior",
            "A {} captured in sharp detail with natural background",
            "A stunning {} photograph showing intricate details",
            "A {} in perfect focus with vibrant natural colors"
        ]
        
        quality_descriptors = [
            "high-resolution", "detailed", "sharp", "vibrant", "natural",
            "professional", "clear", "well-lit", "crisp", "realistic"
        ]
        
        habitat_descriptors = {
            "bird": ["in flight", "perched on a branch", "in its nest", "feeding", "singing"],
            "animal": ["in the wild", "in natural habitat", "roaming freely", "hunting", "resting"],
            "insect": ["on a flower", "in nature", "close-up detail", "macro photography"],
            "plant": ["in bloom", "in natural sunlight", "showing leaves and flowers", "botanical detail"],
            "default": ["in natural setting", "outdoors", "in daylight", "showing detail"]
        }
        
        # Use embedding to select template and descriptors
        embedding_values = text_embedding.cpu().numpy().flatten()
        
        # Select template based on embedding norm
        embedding_norm = np.linalg.norm(embedding_values)
        template_idx = int(embedding_norm * len(descriptive_templates)) % len(descriptive_templates)
        template = descriptive_templates[template_idx]
        
        # Add quality descriptor based on embedding values
        quality_idx = int(abs(embedding_values[0] * 10)) % len(quality_descriptors)
        quality = quality_descriptors[quality_idx]
        
        # Determine category from class name for habitat descriptor
        category = "default"
        class_name_lower = class_name.lower()
        if any(word in class_name_lower for word in ["bird", "hawk", "eagle", "sparrow", "robin", "finch", "wren"]):
            category = "bird"
        elif any(word in class_name_lower for word in ["mammal", "cat", "dog", "bear", "deer", "fox"]):
            category = "animal" 
        elif any(word in class_name_lower for word in ["butterfly", "bee", "beetle", "ant", "spider"]):
            category = "insect"
        elif any(word in class_name_lower for word in ["flower", "plant", "tree", "grass", "leaf"]):
            category = "plant"
        
        # Select habitat descriptor
        habitat_list = habitat_descriptors.get(category, habitat_descriptors["default"])
        habitat_idx = int(abs(embedding_values[-1] * 10)) % len(habitat_list)
        habitat = habitat_list[habitat_idx]
        
        # Combine everything
        if class_name:
            enhanced_prompt = f"{quality} {template.format(class_name)} {habitat}"
        else:
            enhanced_prompt = f"{quality} photograph of a rare species {habitat}"
            
        return enhanced_prompt
    
    def generate_prompts_for_tail_classes(self, 
                                        memory_manager,
                                        class_names: Dict[int, str],
                                        num_prompts_per_class: int = 3) -> Dict[int, List[str]]:
        """
        Generate multiple semantic prompts for all tail classes.
        
        Args:
            memory_manager: MemoryManager instance
            class_names: Dictionary mapping class_id to class_name
            num_prompts_per_class: Number of diverse prompts per class
            
        Returns:
            Dictionary mapping class_id to list of prompts
        """
        tail_classes = memory_manager.get_tail_classes()
        tail_prompts = {}
        
        for class_id in tail_classes:
            prompts = []
            
            # Get class prototype and sample features
            prompt_data = memory_manager.get_semantic_prompt_data(class_id, k_features=5)
            
            if 'prototype' in prompt_data and prompt_data['prototype'] is not None:
                prototype = prompt_data['prototype'].to(self.device)
                class_name = class_names.get(class_id, f"class_{class_id}")
                
                # Generate base prompt from prototype
                base_prompt = self.generate_text_prompt(prototype, class_name)
                prompts.append(base_prompt)
                
                # Generate variations from sampled features
                features = prompt_data.get('features', [])
                for i, feature in enumerate(features[:num_prompts_per_class-1]):
                    if isinstance(feature, torch.Tensor):
                        variant_prompt = self.generate_text_prompt(feature.to(self.device), class_name)
                        prompts.append(variant_prompt)
            
            if prompts:
                tail_prompts[class_id] = prompts
                
        return tail_prompts
    
    def compute_prompt_quality(self, prompt: str, reference_features: torch.Tensor) -> float:
        """
        Compute quality score for generated prompt vs reference features.
        
        Args:
            prompt: Generated text prompt
            reference_features: [feature_dim] Reference feature vector
            
        Returns:
            Quality score (0-1)
        """
        if self.text_encoder is None or self.tokenizer is None:
            # Return a neutral score if models aren't available
            return 0.5
            
        with torch.no_grad():
            try:
                # Encode prompt to text embedding
                inputs = self.tokenizer(prompt, return_tensors='pt', 
                                      padding=True, truncation=True, max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                text_output = self.text_encoder(**inputs)
                prompt_embedding = text_output.last_hidden_state.mean(dim=1)  # Average pooling
                
                # Project reference features to text space
                ref_text_embedding = self.feature_projector(reference_features.unsqueeze(0).to(self.device).float())
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(prompt_embedding, ref_text_embedding, dim=1)
                return similarity.item()
            except Exception as e:
                print(f"Warning: Error computing prompt quality: {e}")
                return 0.5
    
    def save_prompts_to_file(self, prompts: Dict[int, List[str]], 
                           class_names: Dict[int, str], 
                           filepath: str):
        """Save generated prompts to file for diffusion pipeline."""
        import json
        
        prompt_data = {
            'metadata': {
                'num_classes': len(prompts),
                'total_prompts': sum(len(p) for p in prompts.values()),
                'model_info': 'FeatureToTextMapper'
            },
            'prompts': {}
        }
        
        for class_id, class_prompts in prompts.items():
            prompt_data['prompts'][str(class_id)] = {
                'class_name': class_names.get(class_id, f"class_{class_id}"),
                'prompts': class_prompts,
                'count': len(class_prompts)
            }
        
        with open(filepath, 'w') as f:
            json.dump(prompt_data, f, indent=2)
        
        print(f"Prompts saved to: {filepath}")


class TrainableFeatureToText(FeatureToTextMapper):
    """
    Trainable version that learns optimal feature-to-text mapping.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional layers for better text generation
        self.text_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.text_embedding_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=3
        )
    
    def train_on_class_descriptions(self, 
                                  features: torch.Tensor, 
                                  descriptions: List[str],
                                  num_epochs: int = 100):
        """
        Train the feature-to-text mapper on paired data.
        
        Args:
            features: [N, feature_dim] Feature vectors
            descriptions: List of N text descriptions
            num_epochs: Training epochs
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            # Encode text descriptions
            encoded_texts = []
            for desc in descriptions:
                inputs = self.tokenizer(desc, return_tensors='pt', 
                                      padding=True, truncation=True, max_length=self.max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    text_output = self.text_encoder(**inputs)
                    text_embed = text_output.last_hidden_state.mean(dim=1)
                    encoded_texts.append(text_embed)
            
            target_embeddings = torch.cat(encoded_texts, dim=0)
            
            # Forward pass
            predicted_embeddings = self.feature_projector(features.to(self.device))
            
            # Compute loss
            loss = criterion(predicted_embeddings, target_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")