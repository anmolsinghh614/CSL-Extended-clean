import torch
import torch.nn as nn
import torch.nn.functional as F

class LDALLossFunc(nn.Module):
    """
    Learning-Dynamics Aware Loss (LDAL) for Long-Tailed Classification.
    Reference: Shinde et al., "Beyond Static Costs: Learning-Dynamics Aware Loss Functions
               for Long-Tailed Classification", arXiv:2607.25830 (2026).
    
    Total Loss:
        L = L_CE + (1/C) * sum_i [ (gamma_i * N_tilde_i + r_i)^2 / (denom_i + epsilon) ]
    
    Where:
        - L_CE: Standard Cross-Entropy loss.
        - N_tilde_i: Normalized differentiable soft prediction count (C / B) * sum_b p_i^(b).
        - gamma_i: Dynamic class importance combining semantic scale S_i and Shannon entropy H_i,
                   capped by tau (default 5.0).
        - r_i: Inter-epoch regularizer (+alpha if minority class predictions decreased, -alpha if increased).
        - denom_i: Mean squared deviation between batch logits and class one-hot encoding (with stop-gradient).
    """
    def __init__(self, target_minority_classes, num_classes, tau=5.0, alpha=2.0, epsilon=1.0):
        super(LDALLossFunc, self).__init__()
        self.num_classes = num_classes
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        
        # Target minority classes U
        target_mask = torch.zeros(num_classes, dtype=torch.bool)
        for c in (target_minority_classes or []):
            if 0 <= int(c) < num_classes:
                target_mask[int(c)] = True
        self.register_buffer('target_mask', target_mask)
        
        # Inter-epoch prediction tracking: M^(t) and M^(t-1)
        self.register_buffer('curr_M', torch.zeros(num_classes, dtype=torch.float32))
        self.register_buffer('prev_M', torch.zeros(num_classes, dtype=torch.float32))
        self.has_prev_epoch = False
        
        # Intra-epoch accumulators for semantic scale (S_i) and Shannon entropy (H_i)
        self.register_buffer('norm_sum', torch.zeros(num_classes, dtype=torch.float32))
        self.register_buffer('norm_count', torch.zeros(num_classes, dtype=torch.float32))
        self.register_buffer('ent_sum', torch.zeros(num_classes, dtype=torch.float32))
        self.register_buffer('ent_count', torch.zeros(num_classes, dtype=torch.float32))
        
        self.last_epoch = -1

    def _reset_epoch_statistics(self, epoch):
        """Reset running statistics at epoch boundaries (Algorithm 1, Line 4)."""
        if self.last_epoch >= 0:
            self.prev_M.copy_(self.curr_M)
            self.has_prev_epoch = True
        
        self.curr_M.zero_()
        self.norm_sum.zero_()
        self.norm_count.zero_()
        self.ent_sum.zero_()
        self.ent_count.zero_()
        self.last_epoch = epoch

    def forward(self, y_true, y_pred, epoch):
        """
        Compute CE + LDAL loss.
        
        Args:
            y_true: Ground truth class labels [batch_size]
            y_pred: Model logits (pre-softmax) [batch_size, num_classes]
            epoch: Current training epoch index
            
        Returns:
            Scalar loss tensor with full computational graph attached.
        """
        # Epoch boundary check
        if epoch != self.last_epoch:
            self._reset_epoch_statistics(epoch)
            
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        batch_size = y_pred.size(0)
        if batch_size == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
            
        device = y_pred.device
        C = self.num_classes
        B = float(batch_size)
        
        # 1. Standard Cross-Entropy Loss
        ce_loss = F.cross_entropy(y_pred, y_true)
        
        # 2. Differentiable Soft Prediction Count: N_tilde_i = (C / B) * sum_b p_i^(b)
        probs = F.softmax(y_pred, dim=-1)  # [B, C], keeps autograd history
        N_tilde = (C / B) * torch.sum(probs, dim=0)  # [C]
        
        # 3. Update cumulative hard prediction counts M^(t) (detached)
        with torch.no_grad():
            preds = torch.argmax(y_pred, dim=-1)
            bincounts = torch.bincount(preds, minlength=C).float()
            self.curr_M += bincounts
            
            # 4. Accumulate per-class logit norms and prediction entropy
            # Logit norm ||z_b|| for sample b
            logit_norms = torch.norm(y_pred, p=2, dim=-1)  # [B]
            
            # Shannon entropy: - sum_k p_k * log(p_k) with stop-gradient
            p_sg = probs.detach()
            p_clamped = torch.clamp(p_sg, min=1e-12)
            sample_entropy = -torch.sum(p_sg * torch.log(p_clamped), dim=-1)  # [B]
            
            # Accumulate into class bins using true labels
            ones = torch.ones_like(logit_norms)
            self.norm_sum.index_add_(0, y_true, logit_norms)
            self.norm_count.index_add_(0, y_true, ones)
            self.ent_sum.index_add_(0, y_true, sample_entropy)
            self.ent_count.index_add_(0, y_true, ones)
            
            # 5. Compute Semantic Scale S_i and Entropy H_i
            # S_i = ( (1 / |F_i|) * sum ||z|| )^2
            avg_norms = self.norm_sum / torch.clamp(self.norm_count, min=1.0)
            S = avg_norms ** 2  # [C]
            H = self.ent_sum / torch.clamp(self.ent_count, min=1.0)  # [C]
            
            # Handle classes with no observed samples yet in this epoch
            S[self.norm_count == 0] = 0.0
            H[self.ent_count == 0] = 0.0
            
            # 6. Dynamic Importance Weight: gamma_i = min( S_i / (1 + max_j(S_j) * H_i), tau )
            S_max = S.max() + 1e-6
            gamma = S / (1.0 + S_max * H)
            gamma = torch.clamp(gamma, max=self.tau)  # [C]
            
            # 7. Inter-Epoch Regularizer r_i
            r = torch.zeros(C, device=device)
            if self.has_prev_epoch:
                diff = self.curr_M - self.prev_M
                # Target minority classes: +alpha if count decreased, -alpha if increased
                r = torch.where(self.target_mask & (diff < 0), torch.full_like(r, self.alpha), r)
                r = torch.where(self.target_mask & (diff > 0), torch.full_like(r, -self.alpha), r)
                
            # 8. Normalized Denominator with Stop-Gradient:
            # denom_i = (1 / BC) * sum_{b, k} (sg(z_k^(b)) - (e_i)_k)^2 + epsilon
            # Mathematically: (1 / C) * [ mean_b(||z_b||^2) - 2 * mean_b(z_{b, i}) + 1 ] + epsilon
            z_sg = y_pred.detach()
            mean_sq_norm = torch.mean(torch.sum(z_sg ** 2, dim=-1))  # scalar
            mean_z_i = torch.mean(z_sg, dim=0)                       # [C]
            denom = (mean_sq_norm - 2.0 * mean_z_i + 1.0) / float(C) + self.epsilon  # [C]
            denom = torch.clamp(denom, min=1e-6)
            
        # 9. LDAL Auxiliary Penalty: (1 / C) * sum_i [ (gamma_i * N_tilde_i + r_i)^2 / denom_i ]
        # Gradient propagates through N_tilde_i
        numerator = (gamma * N_tilde + r) ** 2  # [C]
        ldal_penalty = (1.0 / float(C)) * torch.sum(numerator / denom)
        
        total_loss = ce_loss + ldal_penalty
        return total_loss
