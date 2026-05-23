# -------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# -------------------------------------------------------------------
#  Modified by Jianbiao Mei
# -------------------------------------------------------------------

import argparse
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.proood.apis.test import custom_multi_gpu_test, custom_single_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp


import numpy as np
# def ece_score(py, y_test, n_bins=15):
#     py = np.array(py)
#     y_test = np.array(y_test)
#     if y_test.ndim > 1:
#         y_test = np.argmax(y_test, axis=1)
#     py_index = np.argmax(py, axis=1)
#     py_value = []
#     for i in range(py.shape[0]):
#         py_value.append(py[i, py_index[i]])
#     py_value = np.array(py_value)
#     acc, conf = np.zeros(n_bins), np.zeros(n_bins)
#     Bm = np.zeros(n_bins)
#     for m in range(n_bins):
#         a, b = m / n_bins, (m + 1) / n_bins
#         for i in range(py.shape[0]):
#             if py_value[i] > a and py_value[i] <= b:
#                 Bm[m] += 1
#                 if py_index[i] == y_test[i]:
#                     acc[m] += 1
#                 conf[m] += py_value[i]
#         if Bm[m] != 0:
#             acc[m] = acc[m] / Bm[m]
#             conf[m] = conf[m] / Bm[m]
#     ece = 0
#     for m in range(n_bins):
#         ece += Bm[m] * np.abs((acc[m] - conf[m]))
#     return ece / sum(Bm)
# def ece_score_segmentation(py, y_test, ignore_labels=None, n_bins=15):
#     """
#     ECE calculation for semantic segmentation tasks, can exclude specific labels (e.g., 0 and 255).
#     
#     Args:
#         py: Model prediction probability matrix, shape (num_samples, num_classes)
#         y_test: Ground truth labels, shape (num_samples,) or (num_samples, num_classes) one-hot
#         ignore_labels: List of labels to ignore, e.g., [0, 255]
#         n_bins: Number of bins
#     """
#     py = np.array(py)
#     y_test = np.array(y_test)
#     
#     # Convert y_test to class indices
#     if y_test.ndim > 1:
#         y_test = np.argmax(y_test, axis=1)
#     
#     # Get predicted class and corresponding max probability for each sample
#     py_index = np.argmax(py, axis=1)
#     py_value = py[np.arange(py.shape[0]), py_index]  # More efficient approach
#     
#     # Default ignore label list
#     if ignore_labels is None:
#         ignore_labels = []
#     
#     # Create mask to mark samples that should be kept (i.e., ground truth labels not in ignore list)
#     # Also, to be fair, should we also exclude cases where predicted classes are in the ignore list? Usually filter based on ground truth labels only.
#     valid_mask = ~np.isin(y_test, ignore_labels)
#     
#     # Apply mask, keep only valid samples
#     y_test_valid = y_test[valid_mask]
#     py_index_valid = py_index[valid_mask]
#     py_value_valid = py_value[valid_mask]
#     
#     # If no valid samples, return NaN
#     if len(y_test_valid) == 0:
#         return np.nan
#     
#     n_valid = len(py_value_valid)
#     # Initialize bin arrays
#     acc, conf = np.zeros(n_bins), np.zeros(n_bins)
#     Bm = np.zeros(n_bins)
#     for m in range(n_bins):
#         a, b = m / n_bins, (m + 1) / n_bins
#         for i in range(n_valid):
#             if py_value_valid[i] > a and py_value_valid[i] <= b:
#                 Bm[m] += 1
#                 if py_index_valid[i] == y_test_valid[i]:
#                     acc[m] += 1
#                 conf[m] += py_value_valid[i]
#         if Bm[m] != 0:
#             acc[m] = acc[m] / Bm[m]
#             conf[m] = conf[m] / Bm[m]
#     ece = 0
#     for m in range(n_bins):
#         ece += Bm[m] * np.abs((acc[m] - conf[m]))
#     return ece / sum(Bm)
def ece_score_segmentation(py, y_test, ignore_labels=None, n_bins=15):
    """Vectorized accelerated version (recommended, results identical to V1, extremely high performance)."""
    py = np.array(py)
    py_max = np.max(py, axis=-1, keepdims=True)
    py_exp = np.exp(py - py_max)  # Prevent numerical overflow
    py = py_exp / np.sum(py_exp, axis=-1, keepdims=True)    
    
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    
    py_index = np.argmax(py, axis=1)
    py_value = py[np.arange(py.shape[0]), py_index]
    
    if ignore_labels is None:
        ignore_labels = []
    valid_mask = ~np.isin(y_test, ignore_labels)
    
    y_test_valid = y_test[valid_mask]
    py_index_valid = py_index[valid_mask]
    py_value_valid = py_value[valid_mask]
    
    if len(y_test_valid) == 0:
        return np.nan
    
    # --- Performance core: vectorized binning ---
    bin_edges = np.linspace(0, 1, n_bins + 1)
    # Assign each sample to a bin, interval is (left, right]
    bin_indices = np.digitize(py_value_valid, bin_edges[1:-1], right=True)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece, total = 0.0, 0
    for m in range(n_bins):
        mask = (bin_indices == m)
        Bm = np.sum(mask)
        if Bm > 0:
            acc = np.mean(py_index_valid[mask] == y_test_valid[mask])
            conf = np.mean(py_value_valid[mask])
            ece += Bm * np.abs(acc - conf)
            total += Bm
    return ece / total if total > 0 else np.nan
def ece_score_geometric(py_logits, y_true, ignore_label=255, n_bins=15):
    """
    Compute ECE for geometric occupancy (binary classification: empty vs occupied).
    Args:
        py_logits: Model predicted logits, shape (num_samples, 20)
                    Column 0: logits for "empty"
                    Columns 1-19: logits for 19 occupied classes
        y_true: Original semantic labels, shape (num_samples,)
        ignore_label: Label to ignore (e.g., 255)
        n_bins: Number of bins
    """
    # 0. Convert 20-dim logits to "occupied" probability [key modification]
    # Method: first apply softmax to get 20-class probabilities, then sum probabilities of classes 1-19 (all occupied classes)
    # For numerical stability, use max subtraction trick
    py_logits = np.array(py_logits)
    py_max = np.max(py_logits, axis=-1, keepdims=True)
    py_exp = np.exp(py_logits - py_max)
    py_probs_all = py_exp / np.sum(py_exp, axis=-1, keepdims=True)  # Shape: (N, 20)
    
    # Compute total "occupied" probability = 1 - "empty" probability = sum of all occupied class probabilities
    py_occupancy_probs = np.sum(py_probs_all[:, 1:], axis=1)  # Shape: (N,)
    
    # 1. Binarize ground truth labels: valid non-empty class (>0 and != ignore_label) -> 1, empty (0) -> 0
    y_binary = np.zeros_like(y_true, dtype=np.int32)
    valid_mask = (y_true > 0) & (y_true != ignore_label)
    y_binary[valid_mask] = 1
    
    # 2. Apply mask, exclude regions with invalid label 255
    eval_mask = (y_true != ignore_label)
    y_binary_valid = y_binary[eval_mask]
    py_valid = py_occupancy_probs[eval_mask]  # Use computed occupancy probability
    
    # 3. Call binary ECE computation
    return ece_score_binary(py_valid, y_binary_valid, n_bins=n_bins)

def ece_score_binary(probs, labels, n_bins=15):
    """
    Standard binary classification ECE computation (vectorized version).
    Args:
        probs: Predicted probability of positive class (occupied), shape (N,)
        labels: Ground truth binary labels (0 or 1), shape (N,)
    """
    # Vectorized binning
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges[1:-1], right=True)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece, total = 0.0, 0
    for m in range(n_bins):
        mask = (bin_indices == m)
        Bm = np.sum(mask)
        if Bm > 0:
            # Binary classification: accuracy = proportion of labels equal to 1 (occupied) in this bin
            acc = np.mean(labels[mask])
            # Confidence = mean of predicted probabilities in this bin
            conf = np.mean(probs[mask])
            ece += Bm * np.abs(acc - conf)
            total += Bm
    return ece / total if total > 0 else np.nan
# Usage example: exclude labels 0 and 255
# ece = ece_score_segmentation(py_pred, y_true, ignore_labels=[0, 255], n_bins=15)
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    # if 'CLASSES' in checkpoint.get('meta', {}):
    #     model.CLASSES = checkpoint['meta']['CLASSES']
    # else:
    #     model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        # assert False
        model = MMDataParallel(model, device_ids=[0])
        outputs = custom_single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs, ece_results_list, ece_targets_list = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            assert False
            #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs, **eval_kwargs))
        print('ece begin')
        # ece_results = np.array(ece_results)
        # ece_targets = np.array(ece_targets)
        ece_results = np.concatenate(ece_results_list, axis=0)  # Shape: (total_batches*256*256*32, 20)
        ece_targets = np.concatenate(ece_targets_list, axis=0)   # Shape: (total_batches*256*256*32,)
        # ece = ece_score(ece_results, ece_targets, n_bins=15)
        print(f"ece_results shape: {ece_results.shape}")      # Should be (num_samples, num_classes)
        print(f"ece_targets shape: {ece_targets.shape}")      # Should be (num_samples,)
        print(f"ece_targets unique values: {np.unique(ece_targets)}")  # Key: view all ground truth label values
        print(f"ece_targets value counts:")                    # View count per label
        unique, counts = np.unique(ece_targets, return_counts=True)
        for val, cnt in zip(unique, counts):
            print(f"  Label {val}: {cnt} samples ({cnt/len(ece_targets):.2%})")
        # ece_sem = ece_score_segmentation(ece_results, ece_targets, ignore_labels=[0,1,9,10,11,12,13,14,16, 255], n_bins=15)
        # ece_geo = ece_score_geometric(ece_results, ece_targets, ignore_label=255, n_bins=15)
        # print(">>> ece_sem metrics: ", ece_sem)
        # print(">>> ece_geo metrics: ", ece_geo)

if __name__ == '__main__':
    main()