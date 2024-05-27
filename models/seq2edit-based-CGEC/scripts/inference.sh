#!/bin/bash
# Inference
MODEL_DIR="./exps/seq2edit_lang8+hsk"
if [ ! -d $MODEL_DIR ]; then
  mkdir -p $MODEL_DIR
fi

MODEL_PATH=$MODEL_DIR"/Best_Model_Stage_2.th"
RESULT_DIR=$MODEL_DIR"/results"

INPUT_FILE=/data/rengengchen/workspace/MuCGEC/data/MuCGEC_exp_data/test/MuCGEC.input
if [ ! -f $INPUT_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py -f $INPUT_FILE
fi
if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi
OUTPUT_FILE=$RESULT_DIR"/MuCGEC_test.output"

echo "Generating..."
SECONDS=0
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict.py --model_path $MODEL_PATH\
                  --weights_name $PRETRAIN_WEIGHTS_DIR\
                  --vocab_path $VOCAB_PATH\
                  --input_file $INPUT_FILE".char"\
                  --output_file $OUTPUT_FILE --log

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."
