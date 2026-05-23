import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, builder
from projects.mmdet3d_plugin.proood.utils.header import Header, SparseHeader
from projects.mmdet3d_plugin.proood.modules.sgb import SGB
from projects.mmdet3d_plugin.proood.modules.sdb import SDB
from projects.mmdet3d_plugin.proood.modules.flosp import FLoSP
from projects.mmdet3d_plugin.proood.utils.lovasz_losses import lovasz_softmax
from projects.mmdet3d_plugin.proood.utils.ssc_loss import sem_scal_loss, geo_scal_loss, CE_ssc_loss
from projects.mmdet3d_plugin.proood.prooodmodule.prototype import PrototypeModule
from projects.mmdet3d_plugin.proood.prooodmodule.tailvoxelselector import TailVoxelSelector

@HEADS.register_module()
class ProOODHead(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        embed_dims,
        scale_2d_list,
        pts_header_dict,
        depth=3,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        save_flag=False,
        use_ema_prototype=True,
        ema_momentum=0.05,
        prototype_warmup_steps=750,
        ood_flag=False,
        **kwargs
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.real_w = 51.2
        self.real_h = 51.2
        self.embed_dims = embed_dims
        self.use_ema_prototype = use_ema_prototype
        
        if kwargs.get('dataset', 'semantickitti') == 'semantickitti':
            self.class_names = [
                "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle",
                "person", "bicyclist", "motorcyclist", "road",
                "parking", "sidewalk", "other-ground", "building", "fence",
                "vegetation", "trunk", "terrain", "pole", "traffic-sign",
            ]
            self.class_weights = torch.from_numpy(np.array([
                0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557,
                0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786
            ]))
            self.tail_class_names = [
                "bicycle", "motorcycle", "truck", "other-vehicle",
                "person", "bicyclist", "motorcyclist", "pole", 
                "traffic-sign", "other-ground", "trunk"
            ]
            self.dataset = 1
            prototype_warmup_steps = 750
            min_update_step = 500
        elif kwargs.get('dataset', 'semantickitti') == 'kitti360':
            self.class_names = [
                'empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
                'person', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
                'fence', 'vegetation', 'terrain', 'pole', 'traffic-sign',
                'other-structure', 'other-object'
            ]
            self.class_weights = torch.from_numpy(np.array([
                0.464, 0.595, 0.865, 0.871, 0.717, 0.657, 0.852, 0.541, 0.602,
                0.567, 0.607, 0.540, 0.636, 0.513, 0.564, 0.701, 0.774, 0.580, 0.690
            ]))
            self.tail_class_names = [
                'bicycle', 'motorcycle', 'truck','other-object',
                'person', 'pole', 'traffic-sign'
            ]
            self.dataset = 2
            prototype_warmup_steps = 750
            min_update_step = 500
        self.n_classes = len(self.class_names)
        
        self.input_prototype_module = PrototypeModule(
            embed_dims=self.embed_dims,
            class_names=self.class_names,
            ema_momentum=ema_momentum,
            prototype_warmup_steps=prototype_warmup_steps,
            use_ema=use_ema_prototype,
            min_update_step=min_update_step,
        )
        
        self.output_prototype_module = PrototypeModule(
            embed_dims=self.embed_dims//2,
            class_names=self.class_names,
            ema_momentum=ema_momentum,
            prototype_warmup_steps=prototype_warmup_steps,
            use_ema=use_ema_prototype,
            min_update_step=min_update_step,
        )
        
        self.tail_voxel_selector = TailVoxelSelector(
            prototype_module=self.input_prototype_module,
            class_names=self.class_names,
            tail_class_names=self.tail_class_names,
            embed_dims=self.embed_dims,  
            n_classes=self.n_classes
        )

        self.aux_occ_head = nn.Sequential(
            nn.Conv3d(self.embed_dims, self.embed_dims // 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.embed_dims // 2, self.n_classes, kernel_size=1, stride=1, padding=0)
        )
        
        self.flosp = FLoSP(scale_2d_list)
        self.bottleneck = nn.Conv3d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1)
        self.sgb = SGB(sizes=[self.bev_h, self.bev_w, self.bev_z], channels=self.embed_dims)
        self.mlp_prior = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims//2),
            nn.LayerNorm(self.embed_dims//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.embed_dims//2, self.embed_dims)
        )
        occ_channel = 8 if pts_header_dict.get('guidance', False) else 0
        self.sdb = SDB(channel=self.embed_dims + occ_channel, out_channel=self.embed_dims//2, depth=depth)
        self.occ_header = nn.Sequential(
            SDB(channel=self.embed_dims, out_channel=self.embed_dims//2, depth=1),
            nn.Conv3d(self.embed_dims//2, 1, kernel_size=3, padding=1)
        )
        self.sem_header = SparseHeader(self.n_classes, feature=self.embed_dims)
        self.ssc_header = Header(self.n_classes, feature=self.embed_dims//2)
        self.pts_header = builder.build_head(pts_header_dict)
        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag
        self.ood_flag = ood_flag

    def forward(self, mlvl_feats, img_metas, target=None):
        """Forward function."""
        out = {}
        x3d = self.flosp(mlvl_feats, img_metas)  # bs, c, nq
        bs, c, _ = x3d.shape
        x3d = self.bottleneck(x3d.reshape(bs, c, self.bev_h, self.bev_w, self.bev_z))
        occ = self.occ_header(x3d).squeeze(1)
        out["occ"] = occ
        x3d = x3d.reshape(bs, c, -1)
        pts_out = self.pts_header(mlvl_feats, img_metas, target)
        pts_occ = pts_out['occ_logit'].squeeze(1)
        proposal = (pts_occ > 0).float().detach().cpu().numpy()
        out['pts_occ'] = pts_occ
        if proposal.sum() < 2:
            proposal = np.ones_like(proposal)
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1) > 0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1) == 0)).astype(np.int32)
        vox_coords = self.get_voxel_indices()
        seed_feats = x3d[0, :, vox_coords[unmasked_idx[0], 3]].permute(1, 0)
        seed_coords = vox_coords[unmasked_idx[0], :3]
        coords_torch = torch.from_numpy(np.concatenate(
            [np.zeros_like(seed_coords[:, :1]), seed_coords], axis=1)).to(seed_feats.device)
        seed_feats_desc = self.sgb(seed_feats, coords_torch)
        sem = self.sem_header(seed_feats_desc)
        out["sem_logit"] = sem
        out["coords"] = seed_coords
        vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=x3d.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats_desc
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mlp_prior(x3d[0, :, vox_coords[masked_idx[0], 3]].permute(1, 0))
        vox_feats_diff = vox_feats_flatten.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims).permute(3, 0, 1, 2).unsqueeze(0)
        
        aux_occ_logit = self.aux_occ_head(vox_feats_diff)  # [B, n_classes, H, W, Z]
        out['aux_occ_logit'] = aux_occ_logit
        
        vox_feats_input = vox_feats_diff.clone()
        out['vox_feats_input'] = vox_feats_input                
        vox_feats_diff = self.input_prototype_module.enhance_features(vox_feats_diff, masked_idx, unmasked_idx, aux_occ_logit)

        proposal_tensor = torch.from_numpy(proposal).bool().to(x3d.device)  # [B, H, W, Z]
        num_important = proposal_tensor.reshape(bs, -1).sum(dim=1)  
        if self.dataset == 1:
            num_important = (num_important.float() * 0.02).long() 
        else:
            num_important = (num_important.float() * 0.0078).long() 
        num_important = num_important.clamp(min=self.tail_voxel_selector.min_tail_voxels.item())

        if self.output_prototype_module.should_use_enhancement():
            tail_indices = self.tail_voxel_selector.select_tail_voxels(
                vox_feats_diff, num_important
            )
            out['tail_indices'] = tail_indices
            if tail_indices is not None:
                vox_feats_diff = self.tail_voxel_selector.enhance_tail_features(vox_feats_diff, tail_indices)        

        if self.pts_header.guidance:
            vox_feats_diff = torch.cat([vox_feats_diff, pts_out['occ_x']], dim=1)
        vox_feats_diff = self.sdb(vox_feats_diff)
        vox_feats = vox_feats_diff.clone()
        vox_feats_full = self.up_scale_2(vox_feats_diff)
        ssc_dict = self.ssc_header(vox_feats_full)  # mlp
        out.update(ssc_dict)
        out['vox_feats_full'] = vox_feats_full  # [1, C, H, W, Z], C = embed_dims//2
        out['vox_feats'] = vox_feats
        
        return out

    def step(self, out_dict, target, img_metas, step_type):
        ssc_pred = out_dict["ssc_logit"]
        sc_pred = out_dict["pts_occ"]       

        if self.training and target is not None and 'vox_feats_full' in out_dict and 'vox_feats_input' in out_dict:
            target_2 = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device) 
            vox_feats_full = out_dict['vox_feats_full']
            vox_feats_input = out_dict['vox_feats_input']
            self.output_prototype_module(vox_feats_full, target)

        if step_type == "train":
            sem_pred_2 = out_dict["sem_logit"]
            coords = out_dict['coords']
            sp_target_2 = target_2.clone()[0, coords[:, 0], coords[:, 1], coords[:, 2]]
            loss_dict = dict()
            class_weight = self.class_weights.type_as(target)
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
                loss_dict['loss_ssc'] = loss_ssc * 3
            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal
            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict['loss_geo_scal'] = loss_geo_scal
            ssc_aux_pred = out_dict['aux_occ_logit']    
                
            if self.CE_ssc_loss:
                loss_ssc_aux = CE_ssc_loss(ssc_aux_pred, target_2, class_weight)
                loss_dict['loss_ssc_aux'] = loss_ssc_aux * 0.2
            if self.sem_scal_loss:
                loss_sem_scal_aux = sem_scal_loss(ssc_aux_pred, target_2)
                loss_dict['loss_sem_scal_aux'] = loss_sem_scal_aux * 0.2
            if self.geo_scal_loss:
                loss_geo_scal_aux = geo_scal_loss(ssc_aux_pred, target_2)
                loss_dict['loss_geo_scal_aux'] = loss_geo_scal_aux * 0.2              
                      
            loss_sem = lovasz_softmax(F.softmax(sem_pred_2, dim=1), sp_target_2, ignore=255)
            loss_sem += F.cross_entropy(sem_pred_2, sp_target_2.long(), ignore_index=255)
            loss_dict['loss_sem'] = loss_sem
            ones = torch.ones_like(target_2).to(target_2.device)
            target_2_binary = torch.where(torch.logical_or(target_2 == 255, target_2 == 0), target_2, ones)
            loss_occ = F.binary_cross_entropy(out_dict['occ'].sigmoid()[target_2_binary != 255],
                                              target_2_binary[target_2_binary != 255].float())
            loss_dict['loss_occ'] = loss_occ
            loss_dict['loss_pts'] = F.binary_cross_entropy(out_dict['pts_occ'].sigmoid()[target_2_binary != 255],
                                                           target_2_binary[target_2_binary != 255].float())
                 
            if self.use_ema_prototype and 'vox_feats_full' in out_dict and self.output_prototype_module.should_use_enhancement():
                vox_feats_full = out_dict['vox_feats_full']
                B, C, H, W, Z = vox_feats_full.shape
                feats_flat = vox_feats_full.permute(0, 2, 3, 4, 1).reshape(-1, C)
                target_flat = target.reshape(-1)
                loss_proto_con = self.output_prototype_module.prototype_contrastive_loss(feats_flat, target_flat)
                loss_dict['loss_proto_con'] = loss_proto_con
            
            if 'tail_indices' in out_dict and out_dict['tail_indices'] is not None and self.input_prototype_module.should_use_enhancement():
                loss_tail = self.tail_voxel_selector.compute_tail_loss(
                    out_dict['vox_feats'], 
                    out_dict['tail_indices'], 
                    target_2
                )
                loss_dict['loss_tail'] = loss_tail
            
            return loss_dict

        elif step_type == "val" or step_type == "test":
            result = dict()
            pred_binary = (sc_pred > 0).long()
            result['output_voxels'] = ssc_pred
            result['target_voxels'] = target
            if self.ood_flag:
                y_pred = ssc_pred.detach().cpu().numpy()
                predicted_classes = np.argmax(y_pred, axis=1)  # (B, H, W, D)
                empty_mask = (predicted_classes == 0)  # (B, H, W, D)
                # ========================
                # Global Prototype OOD 
                # ========================
                vox_feats_full = out_dict['vox_feats_full']
                vox_feats_full = F.interpolate(vox_feats_full, size=[256,256,32], mode='trilinear', align_corners=False).contiguous()
                feats_full = vox_feats_full  # [1, C, H, W, Z], C = embed_dims//2
                mask_features = feats_full
                mask_target = ssc_pred
                mask_ = F.softmax(mask_target, dim=1)
                mask = mask_.argmax(dim=1)  # [1, H, W, Z]
                
                output_prototype_module = self.output_prototype_module
                prototypes = output_prototype_module.prototypes          # [K, C]
                prototype_initialized = output_prototype_module.prototype_initialized  # [K]
                prototype_class_mapping = output_prototype_module.prototype_class_mapping  # {cls_id: proto_idx}

                if not prototype_class_mapping:
                    global_ood_score = torch.zeros_like(mask_features[:, 0])  # [1, H, W, Z]
                else:
                    max_cls_id = max(prototype_class_mapping.keys())
                    lookup_table = torch.full((max_cls_id + 1,), -1, dtype=torch.long, device=prototypes.device)
                    for cls_id, proto_idx in prototype_class_mapping.items():
                        lookup_table[cls_id] = proto_idx

                    pred_labels = mask.long().squeeze(0)  # [H, W, Z]
                    global_proto_map = torch.zeros((pred_labels.size(0), pred_labels.size(1), pred_labels.size(2), prototypes.shape[1]),
                                                   device=prototypes.device)  # [H, W, Z, C]

                    proto_indices = lookup_table[pred_labels]  # [H, W, Z]
                    valid_mask = (proto_indices >= 0)  # [H, W, Z]

                    if valid_mask.any():
                        valid_proto_indices = proto_indices[valid_mask]  # [M]
                        valid_prototypes = prototypes[valid_proto_indices]  # [M, C]
                        global_proto_map.view(-1, prototypes.shape[1])[valid_mask.view(-1)] = valid_prototypes

                    feat_map = mask_features.permute(0, 2, 3, 4, 1).squeeze(0)  # [H, W, Z, C]

                    cos_sim = torch.nn.functional.cosine_similarity(feat_map, global_proto_map, dim=-1)  # [H, W, Z]
                    global_ood_score = 1.0 - cos_sim  # [H, W, Z]           

                    global_ood_scoremin_val = global_ood_score.min()
                    global_ood_scoremax_val = global_ood_score.max()
                    if global_ood_scoremax_val > global_ood_scoremin_val:
                        global_ood_score = (global_ood_score - global_ood_scoremin_val) / (global_ood_scoremax_val - global_ood_scoremin_val)          

                # ========================
                # local Prototype OOD
                # ========================
                top2_values, top2_indices = torch.topk(mask_, 2, dim=1)  # [1, 2, H, W, Z]
                difference = top2_values[:, 0] - top2_values[:, 1]        # [1, H, W, Z]
                scores = 1.0 - difference                                 # [1, H, W, Z]

                vis_mask = scores >= 0.9
                mask = mask_.argmax(dim=1)

                for_query = []

                for i in range(1, self.n_classes):
                    mask_cls = (mask == i)
                    mask_cls = mask_cls & (~vis_mask)

                    if mask_cls.sum() == 0:
                        query_ = mask_features.new_zeros((1, mask_features.shape[1]))  # [1, C]
                    else:
                        cls_features = mask_features * mask_cls.float().unsqueeze(1)
                        sum_feat = cls_features.sum(dim=(2, 3, 4))  # [1, C]
                        sum_mask = mask_cls.sum(dim=(1, 2, 3))
                        query_ = sum_feat / (sum_mask + 1e-8)       # [1, C]

                    for_query.append(query_)
                local_prototypes = torch.cat(for_query, dim=0)  # [n_classes-1, C]          

                pred_labels = mask.long()  # [1, H, W, Z]
                valid_mask = (pred_labels > 0)  # [1, H, W, Z]

                proto_indices = pred_labels - 1  # [1, H, W, Z]

                prototype_map = torch.zeros((pred_labels.size(1), pred_labels.size(2), pred_labels.size(3), local_prototypes.shape[1]),
                                            device=local_prototypes.device)  # [H, W, Z, C]

                valid_flat_mask = valid_mask.view(-1)  # [N]
                if valid_flat_mask.any():
                    flat_indices = proto_indices.view(-1)  # [N]
                    valid_indices = flat_indices[valid_flat_mask]  # [M]
                    valid_prototypes = local_prototypes[valid_indices]  # [M, C]
                    prototype_map.view(-1, local_prototypes.shape[1])[valid_flat_mask] = valid_prototypes

                feat_map = mask_features.permute(0, 2, 3, 4, 1).squeeze(0)  # [H, W, Z, C]

                cos_sim = torch.nn.functional.cosine_similarity(feat_map, prototype_map, dim=-1)  # [H, W, Z]
                local_ood_score = 1.0 - cos_sim  # [H, W, Z]
                local_ood_score = local_ood_score.unsqueeze(0)
                normalized_local_ood_score = torch.zeros_like(local_ood_score)

                for cls_id in range(1, self.n_classes):
                    cls_mask = (pred_labels == cls_id)  # [1, H, W, Z]
                    if cls_mask.sum() > 0:
                        cls_scores = local_ood_score[cls_mask]
                        if len(cls_scores) > 1:
                            min_val = cls_scores.min()
                            max_val = cls_scores.max()
                            if max_val > min_val:
                                normalized_cls_scores = (cls_scores - min_val) / (max_val - min_val)
                                normalized_local_ood_score[cls_mask] = normalized_cls_scores
                            else:
                                normalized_local_ood_score[cls_mask] = 0.0
                        else:
                            normalized_local_ood_score[cls_mask] = 0.0
                    else:
                        pass
                local_ood_score = normalized_local_ood_score 

                # ========================                        
                 # local logits OOD
                # ========================              
                if self.dataset == 1:
                    class_masks = {i: (predicted_classes == i) for i in range(1, 20)}
                else:
                    class_masks = {i: (predicted_classes == i) for i in range(1, 19)}
                ood_pred_all = torch.zeros_like(ssc_pred[:, 0, :, :, :])  # (B, H, W, D)
                local_ood = torch.zeros_like(ood_pred_all) # (B, H, W, D)
                for class_idx, mask in class_masks.items():
                    mask_tensor = torch.tensor(mask, device=ssc_pred.device).bool()
                    mask_tensor = mask_tensor.unsqueeze(1)  # (B, 1, H, W, D)
                    class_logits = (ssc_pred * mask_tensor).squeeze(0)  # (n_classes, H, W, D)
                    valid_mask = mask_tensor.squeeze(0)  # (H, W, D)
                    local_ood_logit = get_cosine_similarity(class_logits) # (bs, depth, height, width)
                    mask_squeezed = mask_tensor.squeeze(1)  # (B, H, W, D)
                    local_ood[mask_squeezed] = local_ood_logit.unsqueeze(0)[mask_squeezed]   # (B, H, W, D)    

                local_oodmin_val = local_ood.min()
                local_oodmax_val = local_ood.max()
                if local_oodmax_val > local_oodmin_val:
                    local_ood = (local_ood - local_oodmin_val) / (local_oodmax_val - local_oodmin_val)                      

                fused_ood = torch.max(torch.stack([
                    local_ood,
                    global_ood_score.unsqueeze(0),
                    local_ood_score
                ], dim=0), dim=0)[0]
                
                # ood_pred_all = torch.zeros_like(ssc_pred[:, 0, :, :, :])
                # if self.dataset == 1:
                #     region_classes = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
                # else:
                #     region_classes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  
                # for class_idx, mask in class_masks.items():
                #     mask_tensor = torch.tensor(mask, device=fused_ood.device).bool()
                #     if class_idx in region_classes:
                #         fused_ood[mask_tensor] = fused_ood[mask_tensor]/2 
                        
                empty_mask_expanded = torch.tensor(empty_mask, device=ssc_pred.device)  # (B, H, W, D)
                min_ood_score = fused_ood.min()
                fused_ood[empty_mask_expanded] = min_ood_score
                # # 10. Normalization (optional but recommended)
                # min_val = fused_ood.min()
                # max_val = fused_ood.max()
                # if max_val > min_val:
                #     fused_ood = (fused_ood - min_val) / (max_val - min_val)
                    
                result['ood_pred'] = fused_ood     

            # ood_pred = fused_ood.clone()
            # min_val = ood_pred.min()
            # max_val = ood_pred.max()
            # ood_pred = (ood_pred - min_val) / (max_val - min_val)
            # ood_scores_mapped = torch.tensor((ood_pred.cpu().numpy() * 255).astype(np.uint8))# Map to [0, 255] and convert to integer
            # zero_mask_mapped = ood_scores_mapped == 0
            # ood_scores_mapped[zero_mask_mapped] += 1
            # ood_scores_mapped[empty_mask_expanded] =0
            # ood_scores_tensor = torch.tensor(ood_scores_mapped, dtype=torch.int64).unsqueeze(0)
            # y_pred = ood_scores_tensor.detach().cpu().numpy()       
            # self.save_ood_pred(img_metas, y_pred)                 

            if self.save_flag:
                y_pred = ssc_pred.detach().cpu().numpy()
                # y_pred = np.argmax(y_pred, axis=1)
                self.save_pred(img_metas, y_pred)
            return result

    def training_step(self, out_dict, target, img_metas):
        return self.step(out_dict, target, img_metas, "train")

    def validation_step(self, out_dict, target, img_metas):
        return self.step(out_dict, target, img_metas, "val")

    def get_voxel_indices(self):
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / self.bev_h
        vol_bnds = np.zeros((3, 2))
        vol_bnds[:, 0] = vox_origin
        vol_bnds[:, 1] = vox_origin + np.array(scene_size)
        vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0] * vol_dim[1] * vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1), idx], axis=0).astype(int).T
        return vox_coords

    def save_pred(self, img_metas, y_pred):
        """Convert predicted labels to KITTI-40 format and save."""
        # Mapping from model class index to KITTI-40 label
        y_pred[y_pred == 10] = 44
        y_pred[y_pred == 11] = 48
        y_pred[y_pred == 12] = 49
        y_pred[y_pred == 13] = 50
        y_pred[y_pred == 14] = 51
        y_pred[y_pred == 15] = 70
        y_pred[y_pred == 16] = 71
        y_pred[y_pred == 17] = 72
        y_pred[y_pred == 18] = 80
        y_pred[y_pred == 19] = 81
        y_pred[y_pred == 1] = 10
        y_pred[y_pred == 2] = 11
        y_pred[y_pred == 3] = 15
        y_pred[y_pred == 4] = 18
        y_pred[y_pred == 5] = 20
        y_pred[y_pred == 6] = 30
        y_pred[y_pred == 7] = 31
        y_pred[y_pred == 8] = 32
        y_pred[y_pred == 9] = 40
        pred_folder = os.path.join("/root/autodl-tmp/proood", "sequences", img_metas[0]['sequence_id'], "predictions")
        os.makedirs(pred_folder, exist_ok=True)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))
    def save_ood_pred(self, img_metas, y_pred):
        # save predictions
        pred_folder = os.path.join("/root/autodl-tmp/pred_ood_prood_stu", "sequences", img_metas[0]['sequence_id'], "predictions") 
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label")) 

def get_cosine_similarity(logit):
    mean_logit = torch.mean(logit, dim=0, keepdim=True)
    cosine_sim = torch.nn.functional.cosine_similarity(logit, mean_logit, dim=0)
    return 1 - cosine_sim