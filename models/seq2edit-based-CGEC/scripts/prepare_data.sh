#!/bin/bash
## Tokenize
TRAIN_FILE="/data/rengengchen/workspace/MuCGEC/data/MuCGEC_exp_data/train/train.para"
if [ ! -f $TRAIN_FILE".char" ]; then
    python /data/rengengchen/workspace/MuCGEC/tools/segment/segment_bert.py -f $TRAIN_FILE
fi
VALID_FILE="/data/rengengchen/workspace/MuCGEC/data/MuCGEC_exp_data/valid/valid.para"
if [ ! -f $VALID_FILE".char" ]; then
    python /data/rengengchen/workspace/MuCGEC/tools/segment/segment_bert.py -f $VALID_FILE
fi

## Generate label file
TRAIN_LABEL_FILE=/data/rengengchen/workspace/MuCGEC/data/MuCGEC_exp_data/train/train.label  # 训练数据
if [ ! -f $TRAIN_LABEL_FILE ]; then
    python /data/rengengchen/workspace/MuCGEC/models/seq2edit-based-CGEC/utils/preprocess_data.py -i $TRAIN_FILE".char" -o $TRAIN_LABEL_FILE --worker_num 32
    shuf $TRAIN_LABEL_FILE > $TRAIN_LABEL_FILE".shuf"
fi
## Generate label file
VALID_LABEL_FILE=/data/rengengchen/workspace/MuCGEC/data/MuCGEC_exp_data/valid/valid.label  # 训练数据
if [ ! -f $VALID_LABEL_FILE ]; then
    python /data/rengengchen/workspace/MuCGEC/models/seq2edit-based-CGEC/utils/preprocess_data.py -i $VALID_FILE".char" -o $VALID_LABEL_FILE --worker_num 32
    shuf $VALID_LABEL_FILE > $VALID_LABEL_FILE".shuf"
fi
