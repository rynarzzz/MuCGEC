#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：fast-gector 
@File    ：run_train.py
@IDE     ：PyCharm 
@Author  ：rengengchen
@Time    ：2024/5/24 14:58 
"""
import os
import shutil
import subprocess
from datetime import datetime


def main():
    # Define paths
    detect_vocab_path = "./data/output_vocabulary_chinese_char_hsk+lang8_5/d_tags.txt"
    correct_vocab_path = "./data/output_vocabulary_chinese_char_hsk+lang8_5/labels.txt"
    train_path = "../../data/MuCGEC_exp_data/train/train.para"
    valid_path = "../../data/MuCGEC_exp_data/valid/valid.para"
    config_path = "configs/ds_config_zero1_fp16.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format like the Bash script
    save_dir = f"ckpts/ckpt_{timestamp}"
    tensorboard_dir = f"logs/tb/gector_{timestamp}"
    pretrained_transformer_path = "./plm/chinese-struct-bert-large"

    # Create directories
    os.makedirs(save_dir, exist_ok=True)

    # Copy files
    shutil.copy(__file__, save_dir)  # Assumes this script is the executed script
    shutil.copy(config_path, save_dir)

    # Construct the command
    run_cmd = (
        f"deepspeed --include localhost:1 --master_port 49828 train.py "
        f"--deepspeed "
        f"--deepspeed_config {config_path} "
        f"--num_epochs 2 "
        f"--max_num_tokens 200 "
        f"--batch_size 256 "
        f"--cold_step_count 0 "
        f"--warmup 0.1 "
        f"--cold_lr 1e-3 "
        f"--skip_correct 1 "
        f"--skip_complex 0 "
        f"--sub_token_mode average "
        f"--special_tokens_fix 1 "
        f"--unk2keep 0 "
        f"--tp_prob 1 "
        f"--tn_prob 0 "
        f"--detect_vocab_path {detect_vocab_path} "
        f"--correct_vocab_path {correct_vocab_path} "
        f"--do_eval "
        f"--train_path {train_path} "
        f"--valid_path {valid_path} "
        f"--save_dir {save_dir} "
        f"--use_cache 0 "
        f"--log_interval 1 "
        f"--eval_interval 50 "
        f"--save_interval 50 "
        f"--pretrained_transformer_path {pretrained_transformer_path} "
        f"--tensorboard_dir {tensorboard_dir}"
    )

    # Execute the command
    print(run_cmd)
    with open(f"{save_dir}/train-{timestamp}.log", "w") as logfile:
        process = subprocess.run(run_cmd, shell=True, text=True, stdout=logfile, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
