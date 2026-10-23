import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditionalTermLayer(nn.Module):
    """
    Learning-Dynamics Aware Loss (LDAL) auxiliary term.

    Computes a dynamic, class-rebalancing penalty that adapts every epoch
    based on semantic scale (logit norms), prediction entropy, and
    inter-epoch reinforcement.

    Modes:
      'full'            -> Full LDAL
      'no_entropy'      -> gamma_i = S_i / (1 + max(S))    [H_i removed]
      'no_regularizer'  -> r_i = 0 always
      'no_semantic'     -> gamma_i = 1 / (1 + H_i)         [S_i removed]
    """
    def __init__(self, target_class_index, num_classes, mode='full', alpha=2.0):
        super(AdditionalTermLayer, self).__init__()
        self.target_class_index = target_class_index
        self.num_classes = num_classes
        self.mode = mode
        self.alpha = alpha  # reinforcement magnitude
        self.previous_epoch_class_predictions = None
        self.current_epoch_class_predictions = None
        self.current_epoch = -1
        self.norm_sum = None
        self.norm_count = None
        self.entropy_sum = None
        self.entropy_count = None
        self._target_mask = None
        self._device = None

    def _ensure_target_mask(self, device):
        if self._target_mask is None or self._device != device:
            mask = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
            for idx in self.target_class_index:
                mask[idx] = True
            self._target_mask = mask
            self._device = device

    def forward(self, inputs, true_labels, epoch):
        inputs = torch.nan_to_num(inputs)
        device = inputs.device
        B, C = inputs.shape

        # --- Epoch transition: reset accumulators ---
        if self.current_epoch != epoch:
            if self.current_epoch_class_predictions is not None:
                self.previous_epoch_class_predictions = self.current_epoch_class_predictions.clone()
            self.current_epoch_class_predictions = torch.zeros(C, device=device)
            self.norm_sum = torch.zeros(C, device=device)
            self.norm_count = torch.zeros(C, device=device)
            self.entropy_sum = torch.zeros(C, device=device)
            self.entropy_count = torch.zeros(C, device=device)
            self.current_epoch = epoch

        self._ensure_target_mask(device)

        # =============================================================
        # 1. SOFT PREDICTIONS — normalized for class-count invariance
        #    Without (C/B): softmax probabilities sum to B (batch size),
        #    making the loss scale with batch size. Normalizing ensures
        #    each class contributes ~1.0 regardless of B or C.
        # =============================================================
        probs = F.softmax(inputs, dim=-1)                      # [B, C]
        batch_soft_predictions = probs.sum(dim=0) * (C / B)    # [C], ~1.0 per class

        # =============================================================
        # 2. HARD PREDICTIONS — for epoch-level tracking only
        # =============================================================
        with torch.no_grad():
            hard_preds = inputs.argmax(dim=-1)                 # [B]
            hard_counts = torch.zeros(C, device=device)
            hard_counts.scatter_add_(0, hard_preds, torch.ones(B, device=device))
            self.current_epoch_class_predictions += hard_counts

        # =============================================================
        # 3. ACCUMULATE NORM STATISTICS
        # =============================================================
        with torch.no_grad():
            sample_norms = inputs.detach().norm(dim=-1)        # [B]
            ones_B = torch.ones(B, device=device)
            self.norm_sum.scatter_add_(0, true_labels, sample_norms)
            self.norm_count.scatter_add_(0, true_labels, ones_B)

        # =============================================================
        # 4. SEMANTIC SCALES
        # =============================================================
        with torch.no_grad():
            safe_norm_count = self.norm_count.clamp(min=1)
            avg_magnitudes = self.norm_sum / safe_norm_count    # [C]
            semantic_scales = avg_magnitudes ** 2               # [C]
            semantic_scales[self.norm_count == 0] = 0.0
            max_semantic_scale = semantic_scales.max() + 1e-6

        # =============================================================
        # 5. CLASS ENTROPIES
        # =============================================================
        with torch.no_grad():
            probs_det = probs.detach()
            sample_entropies = -torch.sum(
                probs_det * torch.log(probs_det + 1e-6), dim=-1
            )                                                   # [B]
            self.entropy_sum.scatter_add_(0, true_labels, sample_entropies)
            self.entropy_count.scatter_add_(0, true_labels, ones_B)
            safe_ent_count = self.entropy_count.clamp(min=1)
            class_entropies = self.entropy_sum / safe_ent_count # [C]
            class_entropies[self.entropy_count == 0] = 0.0

        # =============================================================
        # 6. DYNAMIC GAMMAS — MODE-DEPENDENT, CLAMPED
        #    gamma_i = S_i / (1 + max(S) * H_i)
        #    Clamp at 5.0 to prevent the positive feedback loop:
        #    gamma grows with logit norms -> larger LDAL grad ->
        #    larger logits -> even larger gamma -> explosion.
        # =============================================================
        with torch.no_grad():
            if self.mode == 'no_entropy':
                dynamic_gammas = semantic_scales / (1.0 + max_semantic_scale)
            elif self.mode == 'no_semantic':
                dynamic_gammas = 1.0 / (1.0 + class_entropies)
            else:  # 'full' or 'no_regularizer'
                dynamic_gammas = semantic_scales / (1.0 + max_semantic_scale * class_entropies)
            dynamic_gammas = torch.clamp(dynamic_gammas, max=5.0)

        # =============================================================
        # 7. REINFORCEMENT TERMS — MODE-DEPENDENT
        # =============================================================
        with torch.no_grad():
            reinforcement_terms = torch.zeros(C, device=device)
            if self.mode != 'no_regularizer' and self.previous_epoch_class_predictions is not None:
                cur = self.current_epoch_class_predictions
                prev = self.previous_epoch_class_predictions
                mask = self._target_mask
                reinforcement_terms[(cur > prev) & mask] = -self.alpha
                reinforcement_terms[(cur < prev) & mask] = self.alpha

        # =============================================================
        # 8. LDAL LOSS — normalized denominator
        #    Denominator is normalized by (B * C) so the loss magnitude
        #    is invariant to both batch size and number of classes.
        # =============================================================
        numerators = (dynamic_gammas * batch_soft_predictions + reinforcement_terms) ** 2

        detached = inputs.detach()                              # [B, C]
        total_sq = (detached ** 2).sum()                        # scalar
        col_sums = detached.sum(dim=0)                          # [C]
        denominators = (total_sq - 2.0 * col_sums + B) / (B * C) + 1.0  # [C]

        additional_term = (numerators / denominators).sum() / C

        return additional_term


class CustomLossWithLDAL(nn.Module):
    """Combined CE + LDAL loss."""
    def __init__(self, target_class_index, num_classes, mode='full', alpha=2.0):
        super(CustomLossWithLDAL, self).__init__()
        self.additional_term_layer = AdditionalTermLayer(
            target_class_index, num_classes, mode=mode, alpha=alpha
        )

    def forward(self, y_true, y_pred, epoch):
        cross_entropy_loss = F.cross_entropy(y_pred, y_true.squeeze().long())
        additional_term = self.additional_term_layer(y_pred, y_true, epoch)
        total_loss = cross_entropy_loss + additional_term
        return total_loss


class CEOnlyLoss(nn.Module):
    """Plain cross-entropy for baseline comparisons."""
    def forward(self, y_true, y_pred, epoch):
        return F.cross_entropy(y_pred, y_true.squeeze().long())
