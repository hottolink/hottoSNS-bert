#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, sys, io
from copy import deepcopy
import argparse
import subprocess
from pprint import pprint
import shutil
from sklearn.metrics import accuracy_score, classification_report, f1_score

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)


def evaluate_metrics(path_prediction: str, path_ground_truth: str):
    lst_pred = [s.strip() for s in io.open(path_prediction, mode="r").readlines()]
    lst_gt = [s.strip() for s in io.open(path_ground_truth, mode="r").readlines()]

    acc = accuracy_score(lst_gt, lst_pred)
    fvalue = f1_score(lst_gt, lst_pred, average="macro")

    print(classification_report(lst_gt, lst_pred))

    dict_ret = {
        "accuracy":acc,
        "f-value":fvalue
    }

    return dict_ret

def config_model_args():
    dict_args = {
        "multilingual":{
            "model_dir":"{BERT_MODEL_ROOT_DIR}/multi_cased_L-12_H-768_A-12",
            "bert_config_file":"bert_config.json",
            "vocab_file":"vocab.txt",
            "do_lower_case":"false",
            "normalizer_name":"twitter_normalizer_for_bert_encoder",
            "spm_file":None,
            "init_checkpoint":"bert_model.ckpt",
            "max_seq_length":"128"
        },
        "ja_wikipedia":{
            "model_dir":"{BERT_MODEL_ROOT_DIR}/wikipedia_ja_L-12_H-768_A-12",
            "bert_config_file":"bert_config.json",
            "do_lower_case":"true",
            "vocab_file":"wiki-ja.vocab.to.bert",
            "spm_file":"wiki-ja.model",
            "normalizer_name":"twitter_normalizer_for_bert_encoder",
            "init_checkpoint":"model.ckpt-1400000",
            "max_seq_length":"128"
        },
        "ja_twitter_masked_lm_only":{
            "model_dir":"{BERT_MODEL_ROOT_DIR}/masked_lm_only_L-12_H-768_A-12",
            "bert_config_file":"bert_config.json",
            "do_lower_case":"true",
            "vocab_file":"tokenizer_spm_32K.vocab.to.bert",
            "spm_file":"tokenizer_spm_32K.model",
            "normalizer_name":"twitter_normalizer_for_bert_encoder",
            "init_checkpoint":"model.ckpt-1000000",
            "max_seq_length":"64"
        }
    }
    return dict_args


def config_fine_tune_default_args():

    dict_args = {
        "--task_name":None,
        "--do_train":"true",
        "--do_eval":"true",
        "--do_predict":"true",
        "--do_lower_case":None,
        "--data_dir":None,
        "--vocab_file":None,
        "--spm_file":None,
        "--bert_config_file":None,
        "--init_checkpoint":None,
        "--max_seq_length":None,
        "--train_batch_size":"32",
        "--learning_rate":"2e-5",
        "--num_train_epochs":"3.0",
        "--output_dir":None
    }
    return dict_args


def config_datasets():

    dict_datasets = {
        "twitter_sentiment":{
            "task_name":"PublicTwitterSentiment",
            "data_dir":"{EVALSET_ROOT_DIR}/twitter_sentiment"
        }
    }
    return dict_datasets


def _parse_args():
    parser = argparse.ArgumentParser(description="hoge")
    parser.add_argument("--bert_model_root_dir", required=True, type=str, help="root directory of the BERT pretrained model. each subdirectory has BERT pretrained model.")
    parser.add_argument("--evalset_root_dir", required=True, type=str, help="root directory of the evaluation datasets for each downstream task.")
    parser.add_argument("--n_eval", required=True, type=int, help="number of evaluation for each dataset and model. recommended:10")
    parser.add_argument("--temporary_dir", required=True, type=str, help="temporary directory which is used to save BERT fine-tuning result.")
    parser.add_argument("--save_dir", required=True, type=str, help="directory where prediction result and evaluation summary are stored.")
    parser.add_argument("--fine_tune_script_name", required=False, type=str, default="run_classifier.py", help="filename of the fine-tuning script.")
    parser.add_argument("--testset_prediction_file_name", required=False, type=str, default="test_results_label.tsv", help="filename of predicted labels for testset.")
    parser.add_argument("--testset_ground_truth_file_name", required=False, type=str, default="test_results_ground_truth.tsv", help="filename of ground-truth labels for testset.")
    args = parser.parse_args()

    return args


def main():

    args = _parse_args()

    cfg_datasets = config_datasets()
    cfg_models = config_model_args()
    cfg_fine_tune_args = config_fine_tune_default_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    else:
        raise AssertionError(f"specified directory:{args.save_dir} exists. delete it beforehand.")

    # dataset loop
    for dataset_name, cfg_dataset in cfg_datasets.items():
        task_name = cfg_dataset["task_name"]
        data_dir = cfg_dataset["data_dir"].format(EVALSET_ROOT_DIR=args.evalset_root_dir)

        # model loop
        for model_name, cfg_model in cfg_models.items():

            model_dir = cfg_model["model_dir"].format(BERT_MODEL_ROOT_DIR=args.bert_model_root_dir)

            dict_args = deepcopy(cfg_fine_tune_args)

            dict_args["--task_name"] = task_name
            dict_args["--data_dir"] = data_dir
            dict_args["--vocab_file"] = os.path.join(model_dir, cfg_model["vocab_file"])
            if cfg_model["spm_file"] is not None:
                dict_args["--spm_file"] = os.path.join(model_dir, cfg_model["spm_file"])
            else:
                del dict_args["--spm_file"]
            dict_args["--bert_config_file"] = os.path.join(model_dir, cfg_model["bert_config_file"])
            dict_args["--init_checkpoint"] = os.path.join(model_dir, cfg_model["init_checkpoint"])
            # it seems you must pass `do_lower_case=false` argument, not `do_lower_case` and `false` separately.
            dict_args[f"--do_lower_case={cfg_model['do_lower_case']}"] = None
            del dict_args["--do_lower_case"]
            # dict_args["--do_lower_case"] = cfg_model['do_lower_case']
            dict_args["--normalizer_name"] = cfg_model["normalizer_name"]
            dict_args["--max_seq_length"] = cfg_model["max_seq_length"]
            dict_args["--output_dir"] = args.temporary_dir

            # evaluation loop
            for n_loop in range(1, args.n_eval+1):

                if os.path.exists(args.temporary_dir):
                    print(f"remove temporary directory:{args.temporary_dir}")
                    shutil.rmtree(args.temporary_dir)

                print(f"=== task:{task_name}, model:{model_name}, loop:{n_loop}/{args.n_eval} ===")

                pprint(dict_args)

                print("execute fine-tuning...")
                lst_call = ["python", args.fine_tune_script_name]
                # serialize, omit none
                lst_args = list(filter(bool, sum([p for p in dict_args.items()], ())))
                subprocess.call(lst_call + lst_args)

                # copy predicted file and ground-truth file
                tgt_file_prefix = f"{task_name}_{model_name}_{n_loop}_"
                for testset_file_name in [args.testset_prediction_file_name, args.testset_ground_truth_file_name]:
                    path_src = os.path.join(args.temporary_dir, testset_file_name)
                    path_dest = os.path.join(args.save_dir, tgt_file_prefix + testset_file_name)
                    print(f"copy testset file:{path_src} -> {path_dest}")
                    shutil.copy(path_src, path_dest)

                print(f"evaluate classification metrics...")
                path_pred = os.path.join(args.temporary_dir, args.testset_prediction_file_name)
                path_gt = os.path.join(args.temporary_dir, args.testset_ground_truth_file_name)
                dict_result = evaluate_metrics(path_prediction=path_pred, path_ground_truth=path_gt)

                # save evaluation summary
                dict_result["task_name"] = task_name
                dict_result["model_name"] = model_name
                dict_result["n_loop"] = n_loop

                pprint(dict_result)

                path_summary = os.path.join(args.save_dir, "summary.txt")
                if not os.path.exists(path_summary):
                    s_write = "\t".join(list(dict_result.keys())) + "\n"
                else:
                    s_write = ""
                with io.open(path_summary, mode="a") as ofs:
                    s_write += "\t".join([f"{v}" for v in dict_result.values()]) + "\n"
                    ofs.write(s_write)
                    ofs.close()

                print("done. proceed to next.")




if __name__ == "__main__":
    main()
