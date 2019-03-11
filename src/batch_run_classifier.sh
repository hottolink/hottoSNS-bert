#!/usr/bin/env bash
rm -rf ./eval_all
./batch_run_classifier.py \
--bert_model_root_dir=../trained_model \
--evalset_root_dir=../evaluation_dataset \
--n_eval=2 \
--temporary_dir=./eval_temporary \
--save_dir=./eval_all
