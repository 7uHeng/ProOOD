import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TailVoxelSelector(nn.Module):
    """Prototype-guided tail voxel selection module"""
    
    def __init__(self, prototype_module, class_names, tail_class_names, embed_dims, n_classes):
        """
        Args:
            prototype_module: prototype module
            class_names: list of all class names
            tail_class_names: list of tail class names
            embed_dims: input feature dimension (SDB output dim)
            n_classes: total number of classes
        """
        super().__init__()
        self.prototype_module = prototype_module
        self.class_names = class_names
        self.n_classes = n_classes
        
        # Get tail class IDs (exclude empty class)
        self.tail_class_ids = [
            i for i, name in enumerate(class_names) 
            if name in tail_class_names and name != "empty"
        ]
        
        # Create mapping from tail class to prototype indices
        self.tail_prototype_indices = []
        for cls_id in self.tail_class_ids:
            if cls_id in prototype_module.prototype_class_mapping:
                self.tail_prototype_indices.append(
                    prototype_module.prototype_class_mapping[cls_id]
                )
        self.tail_prototype_indices = torch.tensor(
            self.tail_prototype_indices, dtype=torch.long
        )
        # Tail voxel refined prediction MLP
        self.tail_mlp = nn.Sequential(
            nn.LayerNorm(embed_dims // 2),
            nn.Linear(embed_dims // 2, n_classes),
        )
        self.proto_adapter = nn.Sequential(
            nn.Linear(embed_dims, embed_dims // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 4, embed_dims)
        )
        self.register_buffer("min_tail_voxels", torch.tensor(10))  # prevent empty selection

    def compute_tail_certainty(self, vox_feats):
        """
        Compute tail voxel certainty scores
        
        Args:
            vox_feats: SDB output features [B, C, H, W, Z]
        
        Returns:
            tail_certainty: tail certainty scores [B, N]
            tail_similarity: tail similarity matrix [B, N, num_tail_classes]
        """
        if len(self.tail_prototype_indices) == 0:
            return None, None
            
        B, C, H, W, Z = vox_feats.shape
        device = vox_feats.device
        
        # Get tail class prototypes
        with torch.no_grad():
            prots = self.prototype_module.prototypes[self.tail_prototype_indices]
            prots = F.normalize(prots, p=2, dim=1)
        
        # Flatten features
        feats_flat = vox_feats.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        feats_norm = F.normalize(feats_flat, p=2, dim=2)
        
        # Compute similarity with tail prototypes [B, N, K_tail]
        tail_sim = torch.einsum('bnc,kc->bnk', feats_norm, prots)
        
        # Compute tail certainty (highest - second highest)
        top2_sim, _ = torch.topk(tail_sim, 2, dim=2)  # [B, N, 2]
        tail_certainty = top2_sim[:, :, 0] - top2_sim[:, :, 1]  # [B, N]
        
        return tail_certainty, tail_sim

    def select_tail_voxels(self, vox_feats, num_important):
        """
        Select tail voxels (top-1 + top-2 strategy)
        
        Args:
            vox_feats: SDB output features [B, C, H, W, Z]
            num_important: number of important voxels [B]
        
        Returns:
            tail_indices: selected voxel coordinates [N_selected, 4] (b, z, y, x)
        """
        B, C, H, W, Z = vox_feats.shape
        device = vox_feats.device
        total_voxels = H * W * Z
        
        # Get tail certainty and similarity
        tail_certainty, tail_sim = self.compute_tail_certainty(vox_feats)
        if tail_certainty is None or tail_sim is None:
            return None
        
        # Initialize selected voxel indices
        selected_indices = []
        
        # Process each batch separately
        for b in range(B):
            k = min(int(num_important[b].item()), total_voxels)
            if k < self.min_tail_voxels:
                continue
                
            # Get tail similarity for current batch
            batch_sim = tail_sim[b]  # [N, K_tail]
            
            # Strategy 1: select top-1 tail voxels
            top1_sim, top1_idx = torch.max(batch_sim, dim=1)  # [N]
            _, top1_order = torch.sort(top1_sim, descending=True)
            selected_top1 = top1_order[:k]
            
            # Check if K is reached
            if len(selected_top1) >= k:
                selected_indices.append(selected_top1)
                continue
                
            # Strategy 2: supplement with top-2 tail voxels
            remaining = k - len(selected_top1)
            if remaining > 0:
                # Exclude already selected voxels
                mask = torch.ones(total_voxels, dtype=torch.bool, device=device)
                mask[selected_top1] = False
                
                # Get second highest similarity
                masked_sim = batch_sim[mask]
                _, top2_idx = torch.topk(masked_sim, 2, dim=1)
                top2_sim = torch.gather(masked_sim, 1, top2_idx[:, 1:2]).squeeze(1)
                
                # Select voxels with highest second similarity
                _, top2_order = torch.sort(top2_sim, descending=True)
                selected_top2 = torch.nonzero(mask, as_tuple=False).squeeze(1)[top2_order[:remaining]]
                
                # Merge selections
                selected_indices.append(torch.cat([selected_top1, selected_top2]))
        
        if not selected_indices:
            return None
            
        # Merge all batch selections
        all_selected = torch.cat(selected_indices)
        
        # Convert to 3D coordinates
        z = all_selected % Z
        y = (all_selected // Z) % W
        x = all_selected // (W * Z)
        
        # Create coordinates with batch indices
        batch_idx = []
        pos = 0
        for b in range(B):
            k = min(int(num_important[b].item()), total_voxels)
            if k >= self.min_tail_voxels:
                count = min(k, len(all_selected[pos:]))
                batch_idx.append(torch.full((count,), b, dtype=torch.long, device=device))
                pos += count
        
        batch_idx = torch.cat(batch_idx)
        tail_indices = torch.stack([batch_idx, x, y, z], dim=1)
        
        return tail_indices

    def compute_tail_loss(self, vox_feats, tail_indices, target):
        """
        Compute tail voxel loss
        
        Args:
            vox_feats: SDB output features [B, C, H, W, Z]
            tail_indices: selected voxel coordinates [N, 4] (b, x, y, z)
            target: ground truth labels [B, H, W, Z]
        
        Returns:
            loss_tail: tail voxel loss scalar
        """
        if tail_indices is None or len(tail_indices) == 0:
            return torch.tensor(0.0, device=vox_feats.device)
        
        # Extract tail voxel features
        B, C, H, W, Z = vox_feats.shape
        feats_selected = vox_feats[
            tail_indices[:, 0],  # batch
            :,                   # channel
            tail_indices[:, 1],  # x
            tail_indices[:, 2],  # y
            tail_indices[:, 3]   # z
        ]  # [N_selected, C]
        
        # Refined prediction
        tail_pred = self.tail_mlp(feats_selected)  # [N_selected, n_classes]
        
        # Extract ground truth labels
        target_selected = target[
            tail_indices[:, 0],
            tail_indices[:, 1],
            tail_indices[:, 2],
            tail_indices[:, 3]
        ]  # [N_selected]
        
        # Compute CE loss (ignore 255)
        valid_mask = (target_selected != 255)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=vox_feats.device)
        
        loss = F.cross_entropy(
            tail_pred[valid_mask],
            target_selected[valid_mask].long(),
            ignore_index=255,
            reduction='mean'
        )
        
        return loss
    
    def enhance_tail_features(self, vox_feats, tail_indices):
        """
        Enhance features using residual connection:
            feat = feat + adapter(prototypes^T @ similarity)

        Args:
            vox_feats: [B, C, H, W, Z]
            tail_indices: [N, 4] (b, x, y, z)

        Returns:
            enhanced_feats: [B, C, H, W, Z]
        """
        if tail_indices is None or len(tail_indices) == 0:
            return vox_feats

        B, C, H, W, Z = vox_feats.shape
        device = vox_feats.device
        enhanced_feats = vox_feats.clone()  # don't modify original

        with torch.no_grad():
            prots = self.prototype_module.prototypes[self.tail_prototype_indices]  # [K_tail, C]
            prots_norm = F.normalize(prots, p=2, dim=1)

        # Extract selected voxel features
        coords_b = tail_indices[:, 0]
        coords_x = tail_indices[:, 1]
        coords_y = tail_indices[:, 2]
        coords_z = tail_indices[:, 3]

        selected_feats = enhanced_feats[coords_b, :, coords_x, coords_y, coords_z]  # [N, C]
        feats_norm = F.normalize(selected_feats, p=2, dim=1)

        # Compute similarity and generate weights
        sim = torch.mm(feats_norm, prots_norm.t())  # [N, K_tail]
        weights = F.softmax(sim * 10.0, dim=1)  # temperature scaling

        # Weighted aggregation of prototype vectors
        proto_signal = torch.mm(weights, prots)  # [N, C]

        # Map to residual space via adapter (non-linear transformation)
        residual = self.proto_adapter(proto_signal)  # [N, C]

        # Residual injection: f' = f + Δf
        enhanced_feats[coords_b, :, coords_x, coords_y, coords_z] += residual

        return enhanced_feats