#!/usr/bin/env bash

MODEL_DIR=../trained_model/masked_lm_only_L-12_H-768_A-12
EVAL_DIR=../evaluation_dataset


# fine-tune and evaluate using Twitter日本語評判分析データセット[Suzuki, 2017]
## hottoSNS-BERT
python run_classifier.py \
  --task_name=PublicTwitterSentiment \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$EVAL_DIR/twitter_sentiment \
  --vocab_file=$MODEL_DIR/tokenizer_spm_32K.vocab.to.bert \
  --spm_file=$MODEL_DIR/tokenizer_spm_32K.model \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/model.ckpt-1000000 \
  --max_seq_length=64 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./eval_hottoSNS/



MODEL_DIR=../trained_model/wikipedia_ja_L-12_H-768_A-12
EVAL_DIR=../evaluation_dataset


## Wikipedia JP
python run_classifier.py \
  --task_name=PublicTwitterSentiment \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$EVAL_DIR/twitter_sentiment \
  --vocab_file=$MODEL_DIR/wiki-ja.vocab.to.bert \
  --spm_file=$MODEL_DIR/wiki-ja.model \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/model.ckpt-1400000 \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./eval_wikija/



## MultiLingual Model
MODEL_DIR=../trained_model/multi_cased_L-12_H-768_A-12
EVAL_DIR=../evaluation_dataset


python run_classifier.py \
  --task_name=PublicTwitterSentiment \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --do_lower_case=false \
  --data_dir=$EVAL_DIR/twitter_sentiment \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./eval_multi/
