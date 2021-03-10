# Baseline
python train.py --use_res_connect --linear_bottleneck --save_model --save_acc --print_to_file

# Inverted bottlenecks with ReLU instead of Linear bottleneck
python train.py --use_res_connect --save_model --save_acc --print_to_file

# Multiplier experiments
python train.py --linear_bottleneck --use_res_connect --width_mult 0.75 --save_model --save_acc --print_to_file
python train.py --linear_bottleneck --use_res_connect --width_mult 0.5 --save_model --save_acc --print_to_file
python train.py --linear_bottleneck --use_res_connect --width_mult 0.3 --save_model --save_acc --print_to_file
python train.py --linear_bottleneck --use_res_connect --width_mult 1.4 --save_model --save_acc --print_to_file

# Without Residual connections
python train.py --linear_bottleneck --save_model --save_acc --print_to_file

# Residual connection between explosions or depthwise layers
python train.py --use_res_connect --res_loc 1 --linear_bottleneck --save_model --save_acc --print_to_file
python train.py --use_res_connect --res_loc 2 --linear_bottleneck --save_model --save_acc --print_to_file

# CIFAR-100 for Linear bottleneck experiments
python train.py --use_res_connect --save_model --save_acc --print_to_file --dataset 1
python train.py --use_res_connect --linear_bottleneck --save_model --save_acc --print_to_file --dataset 1