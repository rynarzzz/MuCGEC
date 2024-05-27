# -*- coding:UTF-8 -*-
from trainer import Trainer
from argparse import ArgumentParser
import deepspeed
import os


def main(args):
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_num_tokens", type=int, default=200)
    parser.add_argument("--max_pieces_per_token", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--tn_prob", type=float, default=0)
    parser.add_argument("--tp_prob", type=float, default=1)
    parser.add_argument("--additional_confidence", type=float, default=0.0)
    parser.add_argument("--cold_lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--predictor_dropout", type=float, default=0.0)
    parser.add_argument("--cold_step_count", type=int, default=0)
    parser.add_argument("--sub_token_mode", type=str, default="average")
    # 标签抽取策略，前者每个位置只保留一个标签，后者保留所有标签
    parser.add_argument("--tag_strategy", choices=['keep_one', 'merge_all'],
                        type=str, default="keep_one")
    parser.add_argument("--unk2keep", type=int, default=0,
                        help="replace oov label with keep")
    parser.add_argument("--special_tokens_fix", type=int, default=0)
    parser.add_argument("--skip_correct", type=int, default=1)
    parser.add_argument("--skip_complex", choices=[0, 1, 2, 3, 4, 5],
                        type=int, default=0)
    parser.add_argument("--detect_vocab_path", type=str, required=True)
    parser.add_argument("--correct_vocab_path", type=str, required=True)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--train_path", type=str, default="/data/rengengchen/workspace/MuCGEC/data/MuCGEC_exp_data/train/train.label.shuf")
    parser.add_argument("--valid_path", type=str, default="/data/rengengchen/workspace/MuCGEC/data/MuCGEC_exp_data/valid/valid.label")
    parser.add_argument("--use_cache", default=1, type=int,
                        help="use processed data cache")
    parser.add_argument("--num_workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--ckpt_id", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--pretrained_transformer_path", type=str, required=True)
    parser.add_argument("--tune_bert", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="path to save tensorboard args")
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    main(args)
