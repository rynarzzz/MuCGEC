#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MuCGEC 
@File    ：pipeline.py
@IDE     ：PyCharm 
@Author  ：rengengchen
@Time    ：2024/5/22 17:01 
"""
import os
import subprocess
import shutil

# Define file paths and directories
base_dir = "../../data"
train_src_file = os.path.join(base_dir, "MuCGEC_exp_data/train/train.para")
train_tgt_file = os.path.join(base_dir, "MuCGEC_exp_data/train/train.para")
label_file = os.path.join(base_dir, "MuCGEC_exp_data/train.label")
dev_set = os.path.join(base_dir, "valid_data/MuCGEC_CGED_Dev.label")
model_dir = "./exps/seq2edit_lang8"
pretrain_weights_dir = "./plm/chinese-struct-bert-large"
vocab_path = "./data/output_vocabulary_chinese_char_hsk+lang8_5"


# Step 1: Data Preprocessing
def download_structbert():
    if not os.path.isfile(os.path.join(pretrain_weights_dir, "pytorch_model.bin")):
        subprocess.run(["wget", "https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/ch_model"])
        shutil.move("ch_model", os.path.join(pretrain_weights_dir, "pytorch_model.bin"))


def tokenize(file):
    char_file = file + ".char"
    if not os.path.isfile(char_file):
        subprocess.run(["python", "../../tools/segment/segment_bert.py"], stdin=open(file, 'r'),
                       stdout=open(char_file, 'w'))


def generate_label_file():
    if not os.path.isfile(label_file):
        subprocess.run([
            "python", "./utils/preprocess_data.py", "-s", train_src_file + ".char", "-t", train_tgt_file + ".char",
            "-o", label_file, "--worker_num", "32"
        ])
        subprocess.run(["shuf", label_file], stdout=open(label_file + ".shuf", 'w'))


# Step 2: Training
def train_model(stage, model_name, tune_bert, batch_size, n_epoch, lr, accumulation_size=None, patience=None):
    args = [
        "python", "train.py", "--tune_bert", str(tune_bert),
        "--train_set", label_file + ".shuf",
        "--dev_set", dev_set,
        "--model_dir", model_dir,
        "--model_name", model_name,
        "--vocab_path", vocab_path,
        "--batch_size", str(batch_size),
        "--n_epoch", str(n_epoch),
        "--lr", str(lr),
        "--weights_name", pretrain_weights_dir,
        "--seed", "1"
    ]
    if stage == 2:
        args.extend(
            ["--accumulation_size", str(accumulation_size), "--patience", str(patience), "--pretrain_folder", model_dir,
             "--pretrain", "Temp_Model"])
    subprocess.run(args)


# Step 3: Inference
def inference():
    input_file = os.path.join(base_dir, "MuCGEC_exp_data/test/MuCGEC.input")
    result_dir = os.path.join(model_dir, "results")
    output_file = os.path.join(result_dir, "MuCGEC_test.output")

    if not os.path.isfile(input_file + ".char"):
        subprocess.run(["python", "../../tools/segment/segment_bert.py"], stdin=open(input_file, 'r'),
                       stdout=open(input_file + ".char", 'w'))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    subprocess.run([
        "python", "predict.py", "--model_path", os.path.join(model_dir, "Best_Model_Stage_2.th"),
        "--weights_name", pretrain_weights_dir,
        "--vocab_path", vocab_path,
        "--input_file", input_file + ".char",
        "--output_file", output_file, "--log"
    ])


# Main execution
if __name__ == "__main__":
    # download_structbert()
    # tokenize(train_src_file)
    # tokenize(train_tgt_file)
    # generate_label_file()
    # train_model(1, "Best_Model_Stage_1", 0, 128, 2, 1e-3)
    # train_model(2, "Best_Model_Stage_2", 1, 32, 20, 1e-5, 4, 3)
    inference()
