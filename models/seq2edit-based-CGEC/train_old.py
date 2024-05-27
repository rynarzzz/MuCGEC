# -*- coding: utf-8
import argparse
import logging
import os
import time
from random import seed
import torch

from gector.dataset import Seq2EditVocab
from embedder import get_transformer_embedder
from gector.seq2labels_model import Seq2Labels
from utils.data_utils import init_dataloader
from utils.helpers import PAD_LABEL


def fix_seed(s):
    """
    固定随机种子
    """
    torch.manual_seed(s)
    seed(s)



def get_data(args, vocab):
    train_loader = init_dataloader(
        subset="train",
        data_path=args.train_path,
        num_workers=args.num_workers,
        use_cache=args.use_cache,
        tokenizer=self.mismatched_tokenizer,
        vocab=vocab,
        input_pad_id=self.base_tokenizer.pad_token_id,
        detect_pad_id=vocab.detect_vocab["tag2id"][PAD_LABEL],
        correct_pad_id=vocab.correct_vocab["tag2id"][PAD_LABEL],
        max_num_tokens=args.max_num_tokens,
        batch_size=int(args.batch_size // args.accumulation_size // args.n_gpus),
        tag_strategy=args.tag_strategy,
        skip_complex=args.skip_complex,
        skip_correct=args.skip_correct,
        tp_prob=args.tp_prob,
        tn_prob=args.tn_prob)
    valid_loader = init_dataloader(
        subset="valid",
        data_path=args.valid_path,
        use_cache=args.use_cache,
        num_workers=args.num_workers,
        tokenizer=self.mismatched_tokenizer,
        vocab=vocab,
        input_pad_id=self.base_tokenizer.pad_token_id,
        detect_pad_id=vocab.detect_vocab["tag2id"][PAD_LABEL],
        correct_pad_id=vocab.correct_vocab["tag2id"][PAD_LABEL],
        max_num_tokens=args.max_num_tokens,
        batch_size=int(args.batch_size // args.n_gpus),
        tag_strategy=args.tag_strategy,
        skip_complex=args.skip_complex,
        skip_correct=args.skip_correct,
        tp_prob=args.tp_prob,
        tn_prob=args.tn_prob)
    return train_loader, valid_loader


def get_model(encoder_path, vocab, tune_bert=False, predictor_dropout=0,
              label_smoothing=0.0,
              confidence=0,
              model_dir="",
              log=None):
    token_embs = get_transformer_embedder(encoder_path, tune_bert=tune_bert)
    model = Seq2Labels(vocab=vocab,
                       text_field_embedder=token_embs,
                       dropout=predictor_dropout,
                       label_smoothing=label_smoothing,
                       additional_confidence=confidence,
                       model_dir=model_dir,
                       cuda_device=args.cuda_device,
                       dev_file=args.dev_set,
                       logger=log,
                       save_metric=args.save_metric)
    return model


def main(args):
    fix_seed(args.seed)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    logger = logging.getLogger(__file__)
    logger.setLevel(level=logging.INFO)
    start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    handler = logging.FileHandler(args.model_dir + '/logs_{:s}.txt'.format(str(start_time)))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    pretrain_weights_dir = args.pretrain_weights_dir

    vocab = Seq2EditVocab(args.detect_vocab_path,
                          args.correct_vocab_path,
                          unk2keep=bool(args.unk2keep))
    train_loader, valid_loader = get_data(args, vocab)
    model = get_model(pretrain_weights_dir,
                      vocab,
                      tune_bert=args.tune_bert,
                      predictor_dropout=args.predictor_dropout,
                      label_smoothing=args.label_smoothing,
                      model_dir=os.path.join(args.model_dir, args.model_name + '.th'),
                      log=logger)

    device = torch.device("cuda:" + str(args.cuda_device) if int(args.cuda_device) >= 0 else "cpu")
    if args.pretrain:  # 只加载部分预训练模型
        pretrained_dict = torch.load(os.path.join(args.pretrain_folder, args.pretrain + '.th'), map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load pretrained model')
        logger.info('load pretrained model')

    model = model.to(device)
    print("Model is set")
    logger.info("Model is set")

    parameters = [
        (n, p)
        for n, p in model.named_parameters() if p.requires_grad
    ]

    # 使用Adam算法进行SGD
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    self.encoder_requires_grad = True
    global_train_step = 0  # init a global step
    for epoch in range(args.n_epoch):
        if isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        if args.cold_step_count:
            if epoch < self.cold_step_count:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = args.lr
                self.encoder_requires_grad = False
            else:
                if self.encoder_requires_grad == False:
                    torch.clear_autocast_cache()
                    torch.cuda.empty_cache()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = args.lr
                    self.encoder_requires_grad = True
        train_loss, global_train_step = self._train_epoch(global_train_step)



    optimizer = AdamOptimizer(parameters, lr=args.lr, betas=(0.9, 0.999))
    scheduler = ReduceOnPlateauLearningRateScheduler(optimizer)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    tensorboardWriter = TensorboardWriter(args.model_dir)
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=build_data_loaders(train_data, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                       batches_per_epoch=args.updates_per_epoch),
        validation_data_loader=build_data_loaders(dev_data, batch_size=args.batch_size, num_workers=0, shuffle=False),
        num_epochs=args.num_epochs,
        optimizer=optimizer,
        patience=args.patience,
        validation_metric=args.save_metric,
        cuda_device=device,
        num_gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate_scheduler=scheduler,
        tensorboard_writer=tensorboardWriter,
        use_amp=True  # 混合精度训练，如果显卡不支持请设为false
    )
    print("Start training")
    print('\nepoch: 0')
    logger.info("Start training")
    logger.info('epoch: 0')
    trainer.train()


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        help='Path to the train data',
                        required=True)  # 训练集路径（带标签格式）
    parser.add_argument('--valid_path',
                        help='Path to the valid data',
                        required=True)  # 开发集路径（带标签格式）
    parser.add_argument('--model_dir',
                        help='Path to the model dir',
                        required=True)  # 模型保存路径
    parser.add_argument('--model_name',
                        help='The name of saved checkpoint',
                        required=True)  # 模型名称
    parser.add_argument('--vocab_path',
                        help='Path to the model vocabulary directory.'
                             'If not set then build vocab from data',
                        default="./data/output_vocabulary_chinese_char_hsk+lang8_5")  # 词表路径
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of the batch.',
                        default=256)  # batch大小（句子数目）
    parser.add_argument('--max_num_tokens',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=200)  # 最大输入长度，过长句子将被截断
    parser.add_argument('--target_vocab_size',
                        type=int,
                        help='The size of target vocabularies.',
                        default=1000)  # 词表规模（生成词表时才需要）
    parser.add_argument('--num_epochs',
                        type=int,
                        help='The number of epoch for training model.',
                        default=2)  # 训练轮数
    parser.add_argument('--patience',
                        type=int,
                        help='The number of epoch with any improvements'
                             ' on validation set.',
                        default=3)  # 早停轮数
    parser.add_argument('--skip_correct',
                        type=int,
                        help='If set than correct sentences will be skipped '
                             'by data reader.',
                        default=1)  # 是否跳过正确句子
    parser.add_argument('--skip_complex',
                        type=int,
                        help='If set than complex corrections will be skipped '
                             'by data reader.',
                        choices=[0, 1, 2, 3, 4, 5],
                        default=0)  # 是否跳过复杂句子
    parser.add_argument('--tune_bert',
                        type=int,
                        help='If more then 0 then fine tune bert.',
                        default=0)  # 是否微调bert
    parser.add_argument('--tag_strategy',
                        choices=['keep_one', 'merge_all'],
                        help='The type of the data reader behaviour.',
                        default='keep_one')  # 标签抽取策略，前者每个位置只保留一个标签，后者保留所有标签
    parser.add_argument('--cold_lr',
                        type=float,
                        help='Set initial learning rate.',
                        default=1e-3)  # 初始学习率
    parser.add_argument('--predictor_dropout',
                        type=float,
                        help='The value of dropout for predictor.',
                        default=0.0)  # dropout率（除bert以外部分）
    parser.add_argument('--label_smoothing',
                        type=float,
                        help='The value of parameter alpha for label smoothing.',
                        default=0.0)  # 标签平滑
    parser.add_argument('--tn_prob',
                        type=float,
                        help='The probability to take TN from data.',
                        default=0)  # 保留正确句子的比例
    parser.add_argument('--tp_prob',
                        type=float,
                        help='The probability to take TP from data.',
                        default=1)  # 保留错误句子的比例
    parser.add_argument('--pretrain_folder',
                        help='The name of the pretrain folder.',
                        default=None)  # 之前已经训练好的checkpoint的文件夹
    parser.add_argument('--pretrain',
                        help='The name of the pretrain weights in pretrain_folder param.',
                        default=None)  # 之前已经训练好的checkpoint名称
    parser.add_argument('--cuda_device',
                        help='The number of GPU',
                        default=0)  # 使用GPU编号
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        help='How many batches do you want accumulate.',
                        default=1)  # 梯度累积
    parser.add_argument('--pretrain_weights_dir',
                        type=str,
                        default="chinese-struct-bert")  # 预训练语言模型路径
    parser.add_argument('--save_metric',
                        type=str,
                        choices=["+labels_accuracy", "+labels_accuracy_except_keep"],
                        default="+labels_accuracy")  # 模型保存指标
    parser.add_argument('--updates_per_epoch',
                        type=int,
                        default=None)  # 每个epoch更新次数
    parser.add_argument('--seed',
                        type=int,
                        default=1)  # 随机种子
    args = parser.parse_args()
    main(args)
