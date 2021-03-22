# CNN Tutorial

This repository provide CNN tutorials to construct, train and evaluate CNN models based on PyTorch. Also, you can do ablation tests mentioned in each CNN paper using this repository.


## Getting Started
---

### Prerequsites

- python == 3.7.10
- pytorch == 1.8.0
- torchvision == 0.9.0
- cudatoolkit == 11.0
- matplotlib == 3.3.4

### Training

```shell
# Single GPU
python train.py --use_res_connect --linear_bottleneck --save_model --save_acc

# Multi-GPU
python train.py --use_res_connect --linear_bottleneck --save_model --save_acc --use_multi_gpu
```

### Evaluation

```shell
# Single GPU
python eval.py --model_path ./trained_models/[trained_model_file_name] --use_res_connect

# Multi-GPU
python eval.py --model_path ./trained_models/[trained_model_file_name] --use_res_connect --use_multi_gpu
```


## CNN Models
---

- MobileNetV2

## Licence
---

Distributed under the MIT License. See `LICENSE` for more information.
