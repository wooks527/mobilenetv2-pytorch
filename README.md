# MobileNetV2-PyTorch

This repository provide codes to construct, train and evaluate MobileNetV2 for image classification tasks based on PyTorch. Using these codes, you can do ablation tests mentioned in each CNN paper using this repository.

## Performances

| Model | Dataset | Top-1 Accuracy | # Params |
|:-|:-:|:-:|:-:|
| MobileNetV2 1.0 | CIFAR-10 | 95.11% | 2.236M |
| MobileNetV2 1.0 | CIFAR-100 | 76.72% | 2.351M |
| MobileNetV2 1.0 | ImageNet | 71.48% | 3.504M |

## Detailed Experimental Results

Also, you can see all of the details for experiments for image classification tasks using CIFAR-10/100 and ImageNet Dataset in [CNN Tutorials](https://www.notion.so/CNN-Tutorial-ddd78f6c58274959a875f12680758465) Notion page.

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
