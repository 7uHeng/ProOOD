import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PrototypeModule(nn.Module):
    """
    Modular prototype learning block for semantic scene completion.
    Supports:
        - EMA-updated class prototypes (non-empty classes only)
        - Contrastive loss between features and prototypes
        - Prototype-guided feature enhancement with soft attention
    """

    def __init__(
        self,
        embed_dims,
        class_names,
        ema_momentum=0.05,
        prototype_warmup_steps=750,
        use_ema=True,
        quality_threshold=0.3,
        min_initialized_ratio=0.7,
        min_update_step=500,
    ):
        """
        Args:
            embed_dims (int): Dimension of input features (before projection)
            class_names (list): List of class names, e.g., ['empty', 'car', ...]
            ema_momentum (float): Momentum for EMA update
            prototype_warmup_steps (int): Warm-up steps before using prototypes
            use_ema (bool): Whether to use EMA for updating prototypes
            quality_threshold (float): Minimum avg prototype quality to enable enhancement
            min_initialized_ratio (float): Min ratio of initialized prototypes to activate
            min_update_step (int): Minimum steps before starting prototype updates
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.ema_momentum = ema_momentum
        self.prototype_warmup_steps = prototype_warmup_steps
        self.use_ema = use_ema
        self.quality_threshold = quality_threshold
        self.min_initialized_ratio = min_initialized_ratio
        self.min_update_step = min_update_step
        
        # Filter non-empty classes
        self.non_empty_class_ids = [i for i, name in enumerate(class_names) if name != "empty"]
        self.num_prototype_classes = len(self.non_empty_class_ids)
        self.class_names = class_names
        self.n_classes = len(class_names)

        # Mapping: original class ID -> prototype index
        self.prototype_class_mapping = {
            cls_id: idx for idx, cls_id in enumerate(self.non_empty_class_ids)
        }

        # Prototype dimension
        self.prototype_dim = embed_dims

        # Register buffers
        self.register_buffer("prototypes", torch.zeros(self.num_prototype_classes, self.prototype_dim))
        self.register_buffer("prototype_initialized", torch.zeros(self.num_prototype_classes).bool())
        self.register_buffer("prototype_quality", torch.zeros(self.num_prototype_classes))  # Inverse variance

        # Projection from prototype space to full feature space
        self.prototype_proj = nn.Sequential(
            nn.Linear(self.prototype_dim, 64),
            nn.ReLU(inplace=True)
        )
        
        # Learnable parameters for adaptive fusion
        self.temp = 0.2   # Softmax temperature
        self.alpha_scalar = nn.Parameter(torch.tensor(0.2))
        
        # Training step counter
        self.register_buffer("prototype_step", torch.tensor(0))

    def forward(self, vox_feats_full, target=None):
        """
        Forward pass during training to update prototypes.
        """
        if self.training and target is not None:
            self.prototype_step += 1  # Count steps in all forwards
            if self.should_update_prototype():  # Check if min update steps reached
                self.update_prototypes(vox_feats_full, target)
        return {}

    def update_prototypes(self, vox_feats_full, target):
        """
        Update prototypes using EMA on non-empty valid classes.

        Args:
            vox_feats_full (Tensor): [B, C, H, W, Z]
            target (Tensor): [B, H, W, Z]
        """
        B, C, H, W, Z = vox_feats_full.shape
        device = vox_feats_full.device

        with torch.no_grad():
            feats_flat = vox_feats_full.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # [B, N, C]
            target_flat = target.reshape(B, -1)  # [B, N]

            for b in range(B):
                unique_cls = torch.unique(target_flat[b])
                for cls_id_int in unique_cls.cpu().numpy():
                    if cls_id_int in (0, 255) or cls_id_int not in self.prototype_class_mapping:
                        continue

                    proto_idx = self.prototype_class_mapping[cls_id_int]
                    mask = (target_flat[b] == cls_id_int)
                    if mask.sum() > 0:
                        cls_feats = feats_flat[b][mask]  # [K, C]
                        mean_feat = cls_feats.mean(dim=0)

                        # Update prototype quality (inverse of variance)
                        if cls_feats.shape[0] > 1:
                            var = torch.var(cls_feats, dim=0).mean()
                            self.prototype_quality[proto_idx] = \
                                0.9 * self.prototype_quality[proto_idx] + 0.1 * (1.0 / (var.item() + 1e-5))

                        # Initialize or update via EMA
                        if not self.prototype_initialized[proto_idx]:
                            self.prototypes[proto_idx] = mean_feat
                            self.prototype_initialized[proto_idx] = True
                        elif self.use_ema:
                            self.prototypes[proto_idx] = (
                                (1 - self.ema_momentum) * self.prototypes[proto_idx] +
                                self.ema_momentum * mean_feat
                            )

    def should_use_enhancement(self):
        """Determine whether to apply prototype-guided enhancement."""
        if self.prototype_step < self.prototype_warmup_steps:
            return False
        if not self.use_ema or not self.prototype_initialized.any():
            return False
        init_ratio = self.prototype_initialized.float().mean().item()
        avg_quality = self.prototype_quality[self.prototype_initialized].mean().item()
        return init_ratio >= self.min_initialized_ratio and avg_quality >= self.quality_threshold

    def should_update_prototype(self):
        """
        Determine whether to start updating prototypes (e.g., after initial stabilization).
        Different from enhancement usage.
        """
        if not self.use_ema:
            return False
        return self.prototype_step >= self.min_update_step

    def prototype_contrastive_loss(self, feats_flat, target_flat, temperature=0.1):
        """
        Compute contrastive loss between voxel features and prototypes.

        Args:
            feats_flat (Tensor): Flattened features [N, C], C = embed_dims
            target_flat (Tensor): Labels [N]
            temperature (float): Softmax temperature

        Returns:
            loss (Tensor): Scalar
        """
        device = feats_flat.device
        prots = self.prototypes  # [K, C]
        K = prots.shape[0]
        if K == 0:
            return torch.tensor(0.0, device=device)

        valid_mask = (target_flat != 255) & (target_flat != 0)
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        feats = feats_flat[valid_mask]  # [M, C]
        labels = target_flat[valid_mask].long()

        label_to_proto_idx = torch.full((256,), -1, device=device, dtype=torch.long)
        for cls_id, idx in self.prototype_class_mapping.items():
            label_to_proto_idx[cls_id] = idx

        proto_indices = label_to_proto_idx[labels]  # [M]
        pos_prots = prots[proto_indices]  # [M, C]

        # L2 normalize
        feats = F.normalize(feats, p=2, dim=1)
        prots = F.normalize(prots, p=2, dim=1)
        pos_prots = F.normalize(pos_prots, p=2, dim=1)

        # Similarity scores
        sim_all = torch.mm(feats, prots.t()) / temperature  # [M, K]
        sim_pos = (feats * pos_prots).sum(dim=1) / temperature  # [M]

        log_prob = sim_pos - torch.log(torch.exp(sim_all).sum(dim=1))
        return -log_prob.mean()

    def enhance_features(self, vox_feats_diff, masked_idx, unmasked_idx, aux_occ_logit=None):
        """
        Semantic-guided prototype enhancement with clean residual connection.
        Enhances only masked voxels that are predicted as non-empty.

        Args:
            vox_feats_diff (Tensor): [B, C, H, W, Z]
            masked_idx (np.ndarray): [1, M]
            unmasked_idx (np.ndarray): [1, U]
            aux_occ_logit (Tensor): [B, n_cls, H, W, Z], semantic logits

        Returns:
            Tensor: [B, C, H, W, Z], enhanced features
        """
        if not self.use_ema or not self.should_use_enhancement():
            return vox_feats_diff

        B, C, H, W, Z = vox_feats_diff.shape
        device = vox_feats_diff.device
        N = H * W * Z

        # === 1. Get valid prototypes ===
        if not self.prototype_initialized.any():
            return vox_feats_diff

        prots = self.prototypes[self.prototype_initialized]  # [K, C]
        prots_norm = F.normalize(prots, p=2, dim=1)

        # === 2. Flatten input features ===
        feats = vox_feats_diff.view(B, C, -1)  # [B, C, N]
        feats_norm = F.normalize(feats, p=2, dim=1)  # [B, C, N]

        # === 3. Soft attention from prototypes ===
        sim = torch.einsum('bci,kc->bki', feats_norm, prots_norm)  # [B, K, N]
        weights = F.softmax(sim / self.temp, dim=1)  # [B, K, N]
        enhanced_feats = torch.einsum('bkn,kc->bnc', weights, prots)  # [B, N, C]

        # === 4. Build semantic-guided enhancement mask ===
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[torch.from_numpy(masked_idx[0]).long().to(device)] = True  # masked locations

        if aux_occ_logit is not None:
            pred_nonempty = (aux_occ_logit.argmax(dim=1) != 0)  # [B, H, W, Z]
            pred_nonempty = pred_nonempty.view(B, -1)  # [B, N]

        enhance_mask = mask.unsqueeze(0) & pred_nonempty  # [B, N]

        # === 5. Clean residual enhancement ===
        final_feats = feats.permute(0, 2, 1).clone()  # [B, N, C]
        if enhance_mask.any():
            final_feats[enhance_mask] += self.alpha_scalar * enhanced_feats[enhance_mask]

        # === 6. Reshape back ===
        return final_feats.permute(0, 2, 1).view(B, C, H, W, Z)