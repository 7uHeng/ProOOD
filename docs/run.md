# Run and Eval

## Training

All configs are located under `projects/configs/sgn/`.

### SemanticKITTI (with MobileStereoNet depth)

```shell
./tools/dist_train.sh projects/configs/sgn/proood-semkitti.py 4
```

### SemanticKITTI (with SQL depth)

```shell
./tools/dist_train.sh projects/configs/sgn/proood-sql-semkitti.py 4
```

### KITTI-360 (with MobileStereoNet depth)

```shell
./tools/dist_train.sh projects/configs/sgn/proood-kitti360.py 4
```

### KITTI-360 (with SQL depth)

```shell
./tools/dist_train.sh projects/configs/sgn/proood-sql-kitti360.py 4
```

## Evaluation

Replace `./path/to/ckpts.pth` with your actual checkpoint path.

### SemanticKITTI

```shell
./tools/dist_test.sh projects/configs/sgn/proood-semkitti.py ./path/to/ckpts.pth 4
./tools/dist_test.sh projects/configs/sgn/proood-sql-semkitti.py ./path/to/ckpts.pth 4
```

### KITTI-360

```shell
./tools/dist_test.sh projects/configs/sgn/proood-kitti360.py ./path/to/ckpts.pth 4
./tools/dist_test.sh projects/configs/sgn/proood-sql-kitti360.py ./path/to/ckpts.pth 4
```

## OOD Evaluation

After training, evaluate OOD detection capability using OccOoD benchmarks:

```shell
# Vaakitti (OOD objects in SemanticKITTI)
./tools/dist_test_ood.sh projects/configs/sgn/proood-ood-vaakitti.py ./path/to/ckpts.pth 4

# Vaakitti360 (OOD objects in KITTI-360)
./tools/dist_test_ood.sh projects/configs/sgn/proood-ood-vaakitti360.py ./path/to/ckpts.pth 4

# STU (cross-dataset OOD)
./tools/dist_test_ood.sh projects/configs/sgn/proood-ood-stu.py ./path/to/ckpts.pth 4
```

## Notes

- The `4` at the end of each command specifies the number of GPUs to use. Adjust based on available hardware.
- Config paths use forward slashes and are relative to the project root.
- Before training, update `data_root` in the config file to point to your local dataset location.
- ResNet-50 pretrained weights are expected at `ckpts/resnet50-19c8e357.pth`.
