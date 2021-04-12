# CNN Tutorial

This repository provide CNN tutorials to construct, train and evaluate CNN models based on PyTorch. Also, you can do ablation tests mentioned in each CNN paper using this repository.

## Performances

| Model | Dataset | Top-1 Accuracy | # Params |
|:-|:-:|:-:|:-:|
| MobileNetV2 1.0 | CIFAR-10 | 95.11% | 2.236M |
| MobileNetV2 1.0 | CIFAR-100 | 76.72% | 2.351M |
| MobileNetV2 1.0 | ImageNet (sample) | 38.78% | 3.504M |
| MobileNetV2 1.0 | ImageNet | 71.48% | 3.504M |

## Getting Started

### Prerequsites

- python == 3.7.10
- pytorch == 1.8.0
- torchvision == 0.9.0
- cudatoolkit == 11.0
- matplotlib == 3.3.4

### Training

```shell
# Single GPU
python train.py --use_res_connect --linear_bottleneck --save_model --save_acc --save_loss

# Multi-GPU
python train.py --use_res_connect --linear_bottleneck --save_model --save_acc --save_loss --use_multi_gpu
```

### Evaluation

```shell
# Single GPU
python eval.py --model_path ./trained_models/[trained_model_file_name] --use_res_connect

# Multi-GPU
python eval.py --model_path ./trained_models/[trained_model_file_name] --use_res_connect --use_multi_gpu
```

## Licence

Distributed under the MIT License. See `LICENSE` for more information.
