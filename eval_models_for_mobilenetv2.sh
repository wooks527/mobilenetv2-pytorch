# Multiplier experiments
python eval.py --model_path ./trained_models/mobilenetv2_1614726127_with_mul-1.4 --width_mult 1.4 --use_res_connect
python eval.py --model_path ./trained_models/mobilenetv2_1614688724_baseline --use_res_connect
python eval.py --model_path ./trained_models/mobilenetv2_1614706503_with_mul-0.75 --width_mult 0.75 --use_res_connect
python eval.py --model_path ./trained_models/mobilenetv2_1614711782_with_mul-0.5 --width_mult 0.5 --use_res_connect
python eval.py --model_path ./trained_models/mobilenetv2_1614716193_with_mul-0.3 --width_mult 0.3 --use_res_connect

# Residual connection experiments
python eval.py --model_path ./trained_models/mobilenetv2_1614688724_baseline --use_res_connect
python eval.py --model_path ./trained_models/mobilenetv2_1615198699_with_exp_residuals --use_res_connect --res_loc 1
python eval.py --model_path ./trained_models/mobilenetv2_1615205120_with_dep_residuals --use_res_connect --res_loc 2
python eval.py --model_path ./trained_models/mobilenetv2_1614733487_without_residual
