"""
Phase 3: Feature-Space DDPM Model
Generates ResNet feature vectors directly without needing images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings for timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FeatureDDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model for ResNet Feature Space
    
    Generates synthetic feature vectors [512] directly
    No image generation required!
    """
    
    def __init__(self, 
                 feature_dim=512,
                 num_classes=10,
                 hidden_dim=1024,
                 num_layers=4,
                 num_timesteps=1000,
                 beta_schedule='cosine'):
        """
        Args:
            feature_dim: Dimension of ResNet features (512 for ResNet32)
            num_classes: Number of classes (10 for CIFAR-10)
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            num_timesteps: Diffusion timesteps (T)
            beta_schedule: 'linear' or 'cosine'
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        
        # ========== Noise Schedule ==========
        if beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            self.betas = self._linear_beta_schedule(num_timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for closed form
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
        # Register as buffers (move to GPU automatically)
        self.register_buffer('betas_buffer', self.betas)
        self.register_buffer('sqrt_alphas_cumprod_buffer', self.sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod_buffer', self.sqrt_one_minus_alphas_cumprod)
        
        # ========== Noise Prediction Network ε_θ ==========
        # Timestep embedding
        time_dim = 128
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, time_dim)
        
        # Main MLP for noise prediction
        layers = []
        input_dim = feature_dim + time_dim + time_dim  # features + time + class
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        ])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, feature_dim))
        
        self.noise_pred_net = nn.Sequential(*layers)
        
        print(f"FeatureDDPM initialized:")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Num classes: {num_classes}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Timesteps: {num_timesteps}")
        print(f"  Beta schedule: {beta_schedule}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule from Improved DDPM paper
        Better than linear for feature generation
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """Linear schedule (original DDPM)"""
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def q_sample(self, f_0, t, noise=None):
        """
        Forward diffusion process: Add noise to clean features
        q(f_t | f_0) = N(f_t; sqrt(ᾱ_t) * f_0, (1 - ᾱ_t) * I)
        
        Args:
            f_0: Clean features [batch_size, feature_dim]
            t: Timesteps [batch_size]
            noise: Optional pre-generated noise
        
        Returns:
            f_t: Noisy features [batch_size, feature_dim]
        """
        if noise is None:
            noise = torch.randn_like(f_0)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, f_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, f_0.shape)
        
        return sqrt_alphas_cumprod_t * f_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def _extract(self, a, t, x_shape):
        """Extract values from a based on timestep t"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def predict_noise(self, f_t, t, class_ids):
        """
        Predict noise ε_θ(f_t, t, c)
        
        Args:
            f_t: Noisy features [batch_size, feature_dim]
            t: Timesteps [batch_size]
            class_ids: Class labels [batch_size]
        
        Returns:
            predicted_noise: [batch_size, feature_dim]
        """
        # Timestep embedding
        t_emb = self.time_mlp(t)
        
        # Class embedding
        c_emb = self.class_embed(class_ids)
        
        # Concatenate all inputs
        x = torch.cat([f_t, t_emb, c_emb], dim=1)
        
        # Predict noise
        return self.noise_pred_net(x)
    
    def forward(self, f_0, class_ids):
        """
        Training forward pass: Compute denoising loss
        
        Args:
            f_0: Clean features [batch_size, feature_dim]
            class_ids: Class labels [batch_size]
        
        Returns:
            loss: MSE loss between true and predicted noise
        """
        batch_size = f_0.shape[0]
        device = f_0.device
        
        # Sample random timesteps for each sample in batch
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
        
        # Sample noise
        noise = torch.randn_like(f_0)
        
        # Add noise to features (forward diffusion)
        f_t = self.q_sample(f_0, t, noise)
        
        # Predict noise
        predicted_noise = self.predict_noise(f_t, t, class_ids)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, f_t, t, class_ids):
        """
        Single reverse diffusion step
        Sample from p(f_{t-1} | f_t)
        
        Args:
            f_t: Current noisy features [batch_size, feature_dim]
            t: Current timestep [batch_size]
            class_ids: Class labels [batch_size]
        
        Returns:
            f_{t-1}: Less noisy features
        """
        betas_t = self._extract(self.betas, t, f_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, f_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, f_t.shape)
        
        # Predict noise
        predicted_noise = self.predict_noise(f_t, t, class_ids)
        
        # Compute mean of p(f_{t-1} | f_t)
        model_mean = sqrt_recip_alphas_t * (
            f_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] > 0:
            # Add noise (except at t=0)
            posterior_variance_t = self._extract(self.posterior_variance, t, f_t.shape)
            noise = torch.randn_like(f_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean
    
    @torch.no_grad()
    def sample(self, class_ids, device='cuda'):
        """
        Generate synthetic features via reverse diffusion
        
        Args:
            class_ids: [batch_size] class labels to generate
            device: Device to use
        
        Returns:
            f_0: Generated clean features [batch_size, feature_dim]
        """
        batch_size = len(class_ids)
        
        # Start from pure Gaussian noise
        f_t = torch.randn(batch_size, self.feature_dim, device=device)
        
        # Reverse diffusion process
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            f_t = self.p_sample(f_t, t, class_ids)
        
        return f_t
    
    @torch.no_grad()
    def sample_with_confidence(self, class_ids, class_prototypes, device='cuda'):
        """
        Generate features and compute confidence scores
        
        Args:
            class_ids: [batch_size] class labels
            class_prototypes: Dict[class_id -> prototype tensor]
            device: Device to use
        
        Returns:
            features: [batch_size, feature_dim]
            confidences: [batch_size] confidence scores [0, 1]
        """
        # Generate features
        features = self.sample(class_ids, device)
        
        # Compute confidence as cosine similarity to class prototype
        confidences = []
        for i, class_id in enumerate(class_ids):
            prototype = class_prototypes[class_id.item()].to(device)
            feat = features[i]
            
            # Normalize vectors
            feat_norm = F.normalize(feat.unsqueeze(0), dim=1)
            proto_norm = F.normalize(prototype.unsqueeze(0), dim=1)
            
            # Cosine similarity [-1, 1]
            similarity = (feat_norm * proto_norm).sum()
            
            # Map to [0, 1]
            confidence = (similarity + 1) / 2
            confidences.append(confidence)
        
        confidences = torch.stack(confidences)
        
        return features, confidences
    
    @torch.no_grad()
    def sample_fast(self, class_ids, num_steps=50, device='cuda'):
        """
        Fast sampling using DDIM (fewer steps)
        
        Args:
            class_ids: [batch_size] class labels
            num_steps: Number of steps (< num_timesteps for speed)
            device: Device to use
        
        Returns:
            f_0: Generated features [batch_size, feature_dim]
        """
        # Select subset of timesteps
        timesteps = torch.linspace(0, self.num_timesteps - 1, num_steps, dtype=torch.long, device=device)
        
        batch_size = len(class_ids)
        f_t = torch.randn(batch_size, self.feature_dim, device=device)
        
        for i in reversed(range(len(timesteps))):
            t = timesteps[i].repeat(batch_size)
            f_t = self.p_sample(f_t, t, class_ids)
        
        return f_t


if __name__ == "__main__":
    # Test the model
    print("Testing FeatureDDPM...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = FeatureDDPM(
        feature_dim=512,
        num_classes=10,
        hidden_dim=1024,
        num_layers=4,
        num_timesteps=1000
    ).to(device)
    
    # Test training forward pass
    batch_size = 16
    f_0 = torch.randn(batch_size, 512, device=device)
    class_ids = torch.randint(0, 10, (batch_size,), device=device)
    
    loss = model(f_0, class_ids)
    print(f"\nTraining loss: {loss.item():.4f}")
    
    # Test sampling
    print("\nTesting sampling...")
    class_ids_sample = torch.tensor([6, 7, 8, 9], device=device)
    
    # Mock prototypes
    class_prototypes = {i: torch.randn(512) for i in range(10)}
    
    features, confidences = model.sample_with_confidence(
        class_ids_sample, 
        class_prototypes, 
        device=device
    )
    
    print(f"Generated features shape: {features.shape}")
    print(f"Confidences: {confidences}")
    print("\n✅ Model test passed!")