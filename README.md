# Memory-Conditioned Diffusion Model for Long-Tail Hallucination

<p align="center">
  <b>Improving Tail Class Accuracy via Memory-Guided Synthetic Data Generation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-1.12+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/CIFAR--10-LT-orange.svg" alt="CIFAR-10-LT">
</p>

---

## Overview

Deep learning models struggle on **long-tailed datasets** where a few "head" classes dominate the training data while many "tail" classes have very few samples. This leads to models that predict head classes well but fail on rare tail classes.

This framework tackles this problem through a **6-step iterative pipeline** that:
1. Detects underperforming tail classes automatically using a Memory Bank
2. Generates high-quality synthetic data for those classes using two complementary methods
3. Fine-tunes the model on combined real + synthetic data to boost tail class accuracy

The key innovation is using **both image-space generation** (Stable Diffusion) and **feature-space generation** (DDPM) simultaneously — the former provides visual diversity while the latter generates features directly in the model's latent space, bypassing image quality limitations.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Class-Sensitive Loss (CSL)** | Focal-style auxiliary loss that adaptively boosts gradient flow for tail classes |
| **Memory Bank** | EMA prototypes + Reservoir sampling to track evolving class representations |
| **Feature-Space DDPM** | Generates 512-dim ResNet feature vectors directly — no image generation needed |
| **Visual Exemplar Prompting** | BLIP captioning + CLIP similarity to create accurate Stable Diffusion prompts |
| **Homogeneous Batch Sampling** | Custom sampler ensuring batches contain either images (4D) or features (2D), never mixed |
| **Iterative Self-Improvement** | Closed-loop pipeline: train → evaluate → generate → retrain until convergence |
| **Multi-Dataset Support** | CIFAR-10-LT, CIFAR-100-LT, Tiny ImageNet, ImageNet-LT, iNaturalist |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Create Imbalanced Dataset                                  │
│  ┌─────────────────────────────────────────────┐                    │
│  │ CIFAR-10 (50K) ──→ Exponential decay ──→ Long-tail version     │
│  │ Class 0: 5000 samples ... Class 9: ~500 samples (ratio=10)     │
│  └─────────────────────────────────────────────┘                    │
│                          │                                          │
│  Step 2: Train with Memory Bank + CSL Loss                          │
│  ┌─────────────────────────────────────────────┐                    │
│  │ ResNet-32 ──→ CSL Loss (CE + focal tail boost)                 │
│  │ Memory Bank fills with EMA prototypes + reservoir exemplars    │
│  │ Automatically identifies tail classes (bottom 30%)             │
│  └─────────────────────────────────────────────┘                    │
│                          │                                          │
│  ┌──────── Iterative Improvement Loop (3 rounds) ────────┐         │
│  │                                                        │         │
│  │  Step 3: Generate Prompts (BLIP + CLIP)               │         │
│  │  ┌──────────────────────────────────────────┐         │         │
│  │  │ Memory exemplars → BLIP captions                  │         │
│  │  │ → CLIP-filtered → SD prompts per tail class       │         │
│  │  └──────────────────────────────────────────┘         │         │
│  │                     │                                  │         │
│  │  Step 4: Generate Images (Stable Diffusion)           │         │
│  │  ┌──────────────────────────────────────────┐         │         │
│  │  │ Prompts → SD v1.5 → 512×512 images                │         │
│  │  │ (resized to 32×32 during feature extraction)       │         │
│  │  └──────────────────────────────────────────┘         │         │
│  │                     │                                  │         │
│  │  Step 5: Extract Features + Train DDPM                │         │
│  │  ┌──────────────────────────────────────────┐         │         │
│  │  │ SD images → ResNet → 512-dim features             │         │
│  │  │ Real features → Train DDPM → Generate more        │         │
│  │  │ Combine: SD features + DDPM features              │         │
│  │  └──────────────────────────────────────────┘         │         │
│  │                     │                                  │         │
│  │  Step 6: Fine-tune with Synthetic Data                │         │
│  │  ┌──────────────────────────────────────────┐         │         │
│  │  │ Real images + Synthetic features → Classifier     │         │
│  │  │ Homogeneous batches (images OR features)          │         │
│  │  │ CSL loss maintains tail-class boosting            │         │
│  │  └──────────────────────────────────────────┘         │         │
│  │                                                        │         │
│  └────────────────────────────────────────────────────────┘         │
│                                                                     │
│  Final Report: Per-class accuracy, tail improvements, plots         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, 6GB+ VRAM)
- ~5GB disk space for Stable Diffusion model weights (downloaded automatically)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/memory-conditioned-diffusion.git
cd memory-conditioned-diffusion

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate
# Activate (Linux/Mac)
source .venv/bin/activate

# Install PyTorch (select your CUDA version from https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework, ResNet models, CIFAR-10 dataset |
| `diffusers` | Stable Diffusion pipeline |
| `transformers` | BLIP captioning, CLIP embeddings |
| `accelerate` | Efficient model loading |
| `numpy` | Numerical operations |
| `matplotlib`, `seaborn` | Visualization and plots |
| `tqdm` | Progress bars |
| `Pillow` | Image I/O |

---

## Quick Start

### Run a Quick Test

Verify the pipeline works end-to-end with minimal resources (2 epochs, small batches):

```bash
python run.py test
```

### Run Full Training (Recommended)

Execute the complete pipeline with optimized hyperparameters:

```bash
python run.py full
```

**Default settings** (optimized for CIFAR-10):
| Parameter | Value | Description |
|-----------|-------|-------------|
| Imbalance Ratio | 10 | Moderate long-tail distribution |
| Initial Epochs | 50 | Memory bank training |
| Synthetic Epochs | 15 | Fine-tuning with augmented data |
| DDPM Hidden Dim | 512 | Matches ResNet feature dimension |
| DDPM Training Steps | 10,000 | Feature generation quality |
| Features per Class | 300 | Synthetic features per tail class |
| Improvement Rounds | 3 | Iterative generate-retrain cycles |

### Custom Configuration

```bash
python run.py custom --imbalance 20 --epochs 40 --synthetic-epochs 10 --rounds 2 --ddpm
```

### Benchmarks (Multiple Datasets)

```bash
python run.py benchmark --datasets cifar10 cifar100
```

### Other Commands

```bash
python run.py status   # Show training progress and latest results
python run.py clean    # Remove all generated outputs (checkpoints, images, logs)
```

---

## Project Structure

```
.
├── run.py                              # CLI entry point
├── orchestrator.py                     # Main 6-step pipeline controller
├── requirements.txt                    # Python dependencies
├── README.md
│
├── models/
│   ├── __init__.py
│   ├── resnet.py                       # ResNet-18/32/50/101 (CIFAR-adapted)
│   └── resnext.py                      # ResNeXt-50/101
│
├── utils/
│   ├── __init__.py
│   ├── csl_loss.py                     # Class-Sensitive Learning loss function
│   ├── memory_bank.py                  # EMA prototypes + Reservoir sampling
│   ├── memory_manager.py              # Memory-model integration layer
│   ├── visual_exemplar_prompt_generator.py  # BLIP+CLIP prompt generation
│   ├── feature_to_text.py             # Feature → text description mapping
│   ├── image_diffusion_pipeline.py    # Image diffusion utilities
│   ├── synthetic_training_integration.py   # Synthetic data integration
│   ├── option3version2.py             # Option 3 prompt generator v2
│   └── plot_utils.py                  # Loss/accuracy plotting
│
├── phase3_feature_ddpm.py             # Feature-space DDPM model
├── train_feature_ddpm.py              # DDPM training and feature extraction
├── hybrid_synthetic_pipeline.py       # Hybrid real+synthetic feature pipeline
├── option3_image_generator.py         # Stable Diffusion image generation
├── generate_cifar10_images.py         # CIFAR-10 standalone image generator
├── generate_images.py                 # General image generation script
│
├── dataset_configs.py                 # Dataset configuration registry
├── benchmark_runner.py                # Multi-dataset benchmark runner
│
└── dataloaders/
    ├── __init__.py
    ├── cifar_100_loader.py            # CIFAR-100-LT loader
    ├── tiny_imagenet_loader.py        # Tiny ImageNet loader
    ├── imagenet_lt_loader.py          # ImageNet-LT loader
    ├── inaturalist_loader.py          # iNaturalist loader
    ├── ImageNet_LT/                   # ImageNet-LT metadata
    └── Inaturalist18/                 # iNaturalist metadata
```

---

## How It Works

### Class-Sensitive Learning (CSL) Loss

The CSL loss extends standard Cross-Entropy with an auxiliary focal-style penalty that:
- Applies **3× higher weight** to tail class samples
- Uses **adaptive weighting** — increases the push if a tail class's predictions are declining
- Includes **inverse-frequency class balancing** in the CE component

```python
# From utils/csl_loss.py
total_loss = cross_entropy_loss + additional_term
# where additional_term = focal_weight * (-log(p_correct)) * sample_weight * 0.1
```

### Memory Bank

Each class maintains:
- **EMA Prototype**: Running average of all features seen for that class
- **Reservoir Buffer**: Fixed-capacity buffer of diverse exemplar features (256 per class)
- **Tail Classification**: Dynamic head/medium/tail labeling based on frequency percentiles

### Feature-Space DDPM

Instead of generating images, the DDPM generates **512-dimensional feature vectors** directly in ResNet's embedding space:
- Uses cosine noise schedule (better than linear for features)
- Class-conditioned generation via learned class embeddings
- Supports fast DDIM sampling for inference

### Synthetic Data Integration

Step 6 uses a **homogeneous batch sampler** that creates batches containing either:
- **Image batches** → processed through full ResNet backbone
- **Feature batches** → fed directly to the classifier head (`model.fc`)

This avoids dimension mismatches from mixing 4D image tensors with 2D feature vectors.

---

## Output & Results

After a full run, check the `results/` directory:

| File | Contents |
|------|----------|
| `final_report_<timestamp>.json` | Per-class accuracy, tail improvements, synthetic sample counts |
| `results_visualization_<timestamp>.png` | 4-panel plot: loss curve, accuracy, tail evolution, class distribution |
| `tail_analysis_<timestamp>.json` | Detailed tail class analysis |

The final report shows tail class improvements like:
```
Tail Class Improvements:
  truck:  45.20% → 67.80% (+22.60%)
  ship:   52.10% → 71.30% (+19.20%)
  horse:  48.90% → 65.40% (+16.50%)
```

---

## Supported Datasets

| Dataset | Classes | Image Size | Loader |
|---------|---------|-----------|--------|
| CIFAR-10-LT | 10 | 32×32 | Built-in (torchvision) |
| CIFAR-100-LT | 100 | 32×32 | `dataloaders/cifar_100_loader.py` |
| Tiny ImageNet | 200 | 64×64 | `dataloaders/tiny_imagenet_loader.py` |
| ImageNet-LT | 1000 | 224×224 | `dataloaders/imagenet_lt_loader.py` |
| iNaturalist 2018 | 8142 | 224×224 | `dataloaders/inaturalist_loader.py` |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA Out of Memory** | Reduce `batch_size` in config, or use `run.py test` for lighter settings |
| **Stable Diffusion download fails** | Ensure internet connection; model is ~5GB on first run |
| **Feature dimension mismatch** | Verify `feature_dim` matches your model's output (512 for ResNet32/34) |
| **Process killed silently** | OOM killer; monitor RAM usage, reduce `num_workers` |
| **Low improvement on tail classes** | Enable DDPM (`--ddpm`), increase `features_per_class`, or try lower `imbalance_ratio` |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{csl-memory-diffusion,
  title={Memory-Conditioned Diffusion Model for Improving Long-Tail Recognition},
  year={2026},
  note={B.Tech Thesis Project}
}
```

---

## License

This project is released for academic and research purposes.
