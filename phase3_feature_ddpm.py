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
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            betas = self._linear_beta_schedule(num_timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register every schedule tensor as a buffer so it follows the module to the
        # compute device; indexing a CPU-resident schedule with device-resident timesteps
        # forces a host sync on every one of the thousands of reverse-diffusion steps.
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer(
            'posterior_variance',
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        
        # ========== Feature standardization ==========
        # The diffusion forward process assumes roughly zero-mean, unit-variance data.
        # Raw penultimate ResNet activations are post-ReLU (non-negative, with per-dimension
        # scales in the tens), so training directly on them makes the model's noise
        # predictions meaningless. These buffers hold the statistics used to map real
        # features into the diffusion space and generated samples back out; they default to
        # the identity transform so an unconfigured model behaves as before.
        self.register_buffer('feature_mean', torch.zeros(feature_dim))
        self.register_buffer('feature_std', torch.ones(feature_dim))
        self.register_buffer('features_are_non_negative', torch.tensor(False))
        
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
    
    def set_feature_stats(self, features: torch.Tensor, non_negative: bool = True) -> None:
        """
        Fit the standardization statistics from a sample of real features.
        
        Args:
            features: Real features [num_samples, feature_dim]
            non_negative: Whether the source features are post-ReLU. When True, generated
                samples are clamped at zero so they stay on the real feature manifold.
        """
        features = features.detach().to(torch.float32)
        mean = features.mean(dim=0)
        std = features.std(dim=0)
        # Dead feature dimensions have zero variance; leaving them at zero would divide by
        # ~0 and turn tiny numerical noise into enormous inputs.
        std = torch.where(std < 1e-4, torch.ones_like(std), std)
        
        self.feature_mean = mean.to(self.feature_mean.device)
        self.feature_std = std.to(self.feature_std.device)
        self.features_are_non_negative = torch.tensor(bool(non_negative),
                                                      device=self.features_are_non_negative.device)
    
    def standardize(self, features: torch.Tensor) -> torch.Tensor:
        """Map real features into the diffusion space."""
        return (features - self.feature_mean) / self.feature_std
    
    def destandardize(self, features: torch.Tensor) -> torch.Tensor:
        """Map generated samples back into the real feature space."""
        features = features * self.feature_std + self.feature_mean
        if bool(self.features_are_non_negative):
            features = features.clamp_min(0.0)
        return features
    
    def _extract(self, a, t, x_shape):
        """Extract values from a based on timestep t"""
        batch_size = t.shape[0]
        out = a.to(t.device).gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
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
        
        # Work in the standardized space the noise schedule assumes
        f_0 = self.standardize(f_0)
        
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
    def p_sample(self, f_t, t, t_prev, class_ids, eta=0.0, clip_denoised=4.0):
        """
        Single reverse step from timestep t to timestep t_prev.
        
        Uses the generalized DDIM update, which is valid for any gap between t and t_prev.
        The ancestral DDPM update is only correct for t_prev = t - 1; applying it across a
        strided schedule under-removes noise at every step and the samples diverge.
        
        Args:
            f_t: Current noisy features [batch_size, feature_dim]
            t: Current timestep [batch_size]
            t_prev: Target timestep [batch_size]; negative values mean "fully denoised"
            class_ids: Class labels [batch_size]
            eta: 0 gives deterministic DDIM, 1 reproduces the DDPM posterior variance
            clip_denoised: Clamp the predicted clean sample to this many standard deviations.
                Without it a single bad noise prediction compounds over the remaining steps.
        
        Returns:
            f_{t_prev}: Less noisy features
        """
        alpha_bar_t = self._extract(self.alphas_cumprod, t, f_t.shape)
        
        # t_prev < 0 denotes the clean sample, where the cumulative product is 1
        safe_prev = t_prev.clamp_min(0)
        alpha_bar_prev = torch.where(
            (t_prev < 0).reshape(-1, *((1,) * (f_t.dim() - 1))),
            torch.ones_like(alpha_bar_t),
            self._extract(self.alphas_cumprod, safe_prev, f_t.shape)
        )
        
        predicted_noise = self.predict_noise(f_t, t, class_ids)
        
        # Recover the implied clean sample
        f_0_pred = (f_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        if clip_denoised:
            f_0_pred = f_0_pred.clamp(-clip_denoised, clip_denoised)
        
        sigma = eta * torch.sqrt(
            (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).clamp_min(1e-8)
        ) * torch.sqrt((1.0 - alpha_bar_t / alpha_bar_prev.clamp_min(1e-8)).clamp_min(0.0))
        
        direction = torch.sqrt((1.0 - alpha_bar_prev - sigma ** 2).clamp_min(0.0)) * predicted_noise
        f_prev = torch.sqrt(alpha_bar_prev) * f_0_pred + direction
        
        if eta > 0:
            f_prev = f_prev + sigma * torch.randn_like(f_t)
        
        return f_prev
    
    @torch.no_grad()
    def sample(self, class_ids, device='cuda', num_steps=None, eta=0.0):
        """
        Generate synthetic features via reverse diffusion
        
        Args:
            class_ids: [batch_size] class labels to generate
            device: Device to use
            num_steps: Optional number of reverse steps. Defaults to the full schedule;
                a smaller value strides the schedule for faster sampling.
            eta: Stochasticity of the sampler (0 = deterministic DDIM)
        
        Returns:
            f_0: Generated clean features [batch_size, feature_dim], in real feature space
        """
        batch_size = len(class_ids)
        class_ids = class_ids.to(device)
        
        if num_steps is None or num_steps >= self.num_timesteps:
            timesteps = list(range(self.num_timesteps - 1, -1, -1))
        else:
            timesteps = torch.linspace(
                self.num_timesteps - 1, 0, max(1, int(num_steps))
            ).round().to(torch.long).tolist()
        
        # Start from pure Gaussian noise
        f_t = torch.randn(batch_size, self.feature_dim, device=device)
        
        for step_idx, timestep in enumerate(timesteps):
            prev_timestep = timesteps[step_idx + 1] if step_idx + 1 < len(timesteps) else -1
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            t_prev = torch.full((batch_size,), prev_timestep, device=device, dtype=torch.long)
            f_t = self.p_sample(f_t, t, t_prev, class_ids, eta=eta)
        
        return self.destandardize(f_t)
    
    @torch.no_grad()
    def sample_batched(self, class_ids, device='cuda', num_steps=None, batch_size=256, eta=0.0):
        """
        Generate features in chunks so large requests do not blow up memory.
        
        Args:
            class_ids: [num_samples] class labels
            device: Device to use
            num_steps: Optional number of reverse steps
            batch_size: Maximum samples generated per reverse-diffusion pass
            eta: Stochasticity of the sampler
        
        Returns:
            features: [num_samples, feature_dim] in real feature space
        """
        class_ids = class_ids.to(device)
        chunks = []
        for start in range(0, len(class_ids), batch_size):
            chunks.append(self.sample(class_ids[start:start + batch_size], device, num_steps, eta))
        
        if not chunks:
            return torch.empty(0, self.feature_dim, device=device)
        return torch.cat(chunks, dim=0)
    
    @torch.no_grad()
    def sample_with_confidence(self, class_ids, class_prototypes, device='cuda',
                               num_steps=None, batch_size=256, eta=0.0):
        """
        Generate features and compute confidence scores
        
        Confidence is the cosine similarity between a generated feature and its class
        prototype, rescaled to [0, 1]. It is used to discard samples that fall outside the
        real feature distribution before they are trained on.
        
        Args:
            class_ids: [batch_size] class labels
            class_prototypes: Dict[class_id -> prototype tensor]
            device: Device to use
            num_steps: Optional number of reverse steps
            batch_size: Maximum samples generated per reverse-diffusion pass
        
        Returns:
            features: [batch_size, feature_dim]
            confidences: [batch_size] confidence scores [0, 1]
        """
        class_ids = class_ids.to(device)
        features = self.sample_batched(class_ids, device, num_steps, batch_size, eta)
        
        # Stack prototypes into a lookup table so confidence is one batched operation
        prototype_table = torch.zeros(self.num_classes, self.feature_dim, device=device)
        for class_id, prototype in class_prototypes.items():
            prototype_table[int(class_id)] = prototype.detach().to(device).flatten()
        
        prototypes = prototype_table[class_ids]
        similarity = F.cosine_similarity(features, prototypes, dim=1)
        confidences = (similarity + 1) / 2
        
        return features, confidences
    
    @torch.no_grad()
    def sample_fast(self, class_ids, num_steps=50, device='cuda'):
        """
        Fast sampling using a strided subset of the noise schedule (DDIM).
        
        Args:
            class_ids: [batch_size] class labels
            num_steps: Number of steps (< num_timesteps for speed)
            device: Device to use
        
        Returns:
            f_0: Generated features [batch_size, feature_dim]
        """
        return self.sample(class_ids, device=device, num_steps=num_steps)


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