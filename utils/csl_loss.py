import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AdditionalTermLayer(nn.Module):
    """
    Class-Sensitive Learning (CSL) additional term.
    
    Provides an auxiliary loss that boosts gradient flow for tail classes
    by penalizing the model when it under-predicts rare classes.
    
    Uses fully differentiable softmax probabilities (not argmax) and
    computes a weighted focal-style penalty that adaptively adjusts
    based on class frequency and prediction confidence.
    """
    def __init__(self, target_class_index, num_classes):
        super(AdditionalTermLayer, self).__init__()
        self.target_class_index = target_class_index  # tail class indices
        self.num_classes = num_classes
        self.last_epoch = -1
        
        # Epoch-level tracking (for monitoring, NOT for gradient computation)
        self.epoch_pred_counts = None
        self.prev_epoch_pred_counts = None
        self.batch_count = 0

    def forward(self, inputs, true_labels, epoch):
        """
        Compute CSL additional term using differentiable operations.
        
        Args:
            inputs: Model logits [batch_size, num_classes]
            true_labels: Ground truth labels [batch_size]
            epoch: Current epoch number
        
        The term works by:
        1. Computing softmax probabilities (differentiable, unlike argmax)
        2. For each tail class sample, adding a penalty that encourages
           higher confidence on the correct class
        3. Using adaptive per-class weights: tail classes get higher weight
        """
        # --- Epoch reset for monitoring ---
        if epoch != self.last_epoch:
            if self.epoch_pred_counts is not None:
                self.prev_epoch_pred_counts = self.epoch_pred_counts.clone()
            self.epoch_pred_counts = torch.zeros(self.num_classes)
            self.batch_count = 0
            self.last_epoch = epoch

        inputs = torch.nan_to_num(inputs)
        self.batch_count += 1
        
        batch_size = inputs.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # --- Update prediction tracking (detached, for monitoring only) ---
        with torch.no_grad():
            preds = torch.argmax(inputs, dim=-1)
            for i in range(self.num_classes):
                self.epoch_pred_counts[i] += (preds == i).sum().item()

        # --- Differentiable CSL penalty ---
        # Step 1: Get softmax probabilities (fully differentiable)
        probs = F.softmax(inputs, dim=1)  # [B, C]
        
        # Step 2: Get the probability assigned to the true class for each sample
        # This is differentiable w.r.t. inputs
        true_class_probs = probs.gather(1, true_labels.unsqueeze(1)).squeeze(1)  # [B]
        
        # Step 3: Build per-sample weights — tail classes get higher weight
        # This creates a weighting mask (detached — weights themselves don't need grad)
        with torch.no_grad():
            sample_weights = torch.ones(batch_size, device=inputs.device)
            for i, label in enumerate(true_labels):
                label_int = label.item()
                if label_int in self.target_class_index:
                    # Tail class: boost weight to encourage correct predictions
                    sample_weights[i] = 3.0
                    
                    # Additional reinforcement: if this tail class was improving
                    # across epochs, slightly reduce the push (it's learning);
                    # if it was declining, increase the push
                    if self.prev_epoch_pred_counts is not None:
                        prev = self.prev_epoch_pred_counts[label_int]
                        curr = self.epoch_pred_counts[label_int]
                        if prev > 0 and curr < prev:
                            # Tail class predictions declining → push harder
                            sample_weights[i] = 4.0
                        elif prev > 0 and curr > prev:
                            # Tail class predictions improving → moderate push
                            sample_weights[i] = 2.0

        # Step 4: Compute focal-style penalty
        # -log(p_correct) is just CE, but we weight it by (1 - p_correct)
        # so samples the model is already confident on contribute less,
        # and confused tail samples contribute more
        focal_weight = (1.0 - true_class_probs).detach()  # [B], detached so it's a weight
        penalty = -torch.log(true_class_probs + 1e-7) * focal_weight * sample_weights
        
        # Step 5: Average and scale down to be auxiliary (not dominate CE)
        additional_term = penalty.mean() * 0.1
        
        return additional_term


class CSLLossFunc(nn.Module):
    def __init__(self, target_class_index, num_classes, samples_per_class=None):
        """
        Class-Sensitive Learning loss: Cross Entropy + CSL additional term.
        
        Args:
            target_class_index: List of tail class indices to boost
            num_classes: Total number of classes
            samples_per_class: Optional list/array of sample counts per class.
                If provided, computes inverse-frequency weights for CE loss
                to counteract class imbalance.
        """
        super(CSLLossFunc, self).__init__()
        self.num_classes = num_classes
        self.additional_term_layer = AdditionalTermLayer(target_class_index, num_classes)
        
        # Compute class-balanced weights from sample distribution
        if samples_per_class is not None:
            samples = np.array(samples_per_class, dtype=np.float64)
            total = samples.sum()
            # Inverse-frequency weighting: rare classes get higher weight
            weights = total / (num_classes * samples + 1e-8)
            # Normalize so weights average to 1.0 (keeps loss scale stable)
            weights = weights / weights.mean()
            self.register_buffer('class_weights', torch.tensor(weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, y_true, y_pred, epoch):
        # Clean predictions globally to avoid downstream ripples
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        
        # Core loss: Class-balanced Cross Entropy
        cross_entropy_loss = F.cross_entropy(y_pred, y_true, weight=self.class_weights)
        
        # CSL additional term for tail class boosting
        additional_term = self.additional_term_layer(y_pred, y_true, epoch)
        
        total_loss = cross_entropy_loss + additional_term
        
        return total_loss