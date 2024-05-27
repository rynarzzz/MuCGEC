#!/bin/bash
base_path="/data/rengengchen/workspace/MuCGEC"
code_path=$base_path"/models/seq2edit-based-CGEC"
detect_vocab_path=$code_path"/data/output_vocabulary_chinese_char_hsk+lang8_5/d_tags.txt"
correct_vocab_path=$code_path"/data/output_vocabulary_chinese_char_hsk+lang8_5/labels.txt"
train_path=$base_path"/data/MuCGEC_exp_data/train/train.label.shuf"
valid_path=$base_path"/data/MuCGEC_exp_data/valid/valid.label"
config_path=$code_path"/configs/ds_config_zero1_fp16.json"
timestamp=`date "+%Y%0m%0d_%T"`
save_dir=$code_path"/ckpts/ckpt_$timestamp"
tensorboard_dir=$code_path"/logs/tb/gector_${timestamp}"
pretrained_transformer_path=$code_path"/plm/chinese-struct-bert-large"
mkdir -p $save_dir
cp $0 $save_dir
cp $config_path $save_dir


run_cmd="deepspeed --include localhost:1 --master_port 49828 train.py \
    --deepspeed \
    --deepspeed_config $config_path \
    --num_epochs 2 \
    --max_num_tokens 200 \
    --batch_size 256 \
    --cold_step_count 0 \
    --warmup 0.1 \
    --cold_lr 1e-3 \
    --skip_correct 1 \
    --skip_complex 0 \
    --sub_token_mode average \
    --special_tokens_fix 1 \
    --unk2keep 0 \
    --tp_prob 1 \
    --tn_prob 0 \
    --detect_vocab_path $detect_vocab_path \
    --correct_vocab_path $correct_vocab_path \
    --do_eval \
    --train_path $train_path \
    --valid_path $valid_path \
    --save_dir $save_dir \
    --use_cache 1 \
    --log_interval 1 \
    --eval_interval 50 \
    --save_interval 50 \
    --pretrained_transformer_path $pretrained_transformer_path \
    --tune_bert 0 \
    --tensorboard_dir $tensorboard_dir \
    2>&1 | tee ${save_dir}/train-${timestamp}.log"

echo ${run_cmd}
eval ${run_cmd}
