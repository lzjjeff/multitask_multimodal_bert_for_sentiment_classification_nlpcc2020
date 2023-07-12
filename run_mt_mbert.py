from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import argparse
import csv
import math
import random
from tqdm import tqdm, trange
from copy import deepcopy

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import L1Loss
from torch.optim import Adam

from model.bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from model.bert.tokenization import BertTokenizer
from model.bert.optimization import BertAdam
from model.mt_model import MultiModalMultiTaskSentimentClassification
from config import *
from util import loss_plot
from dataset import SingleTaskDataset, MultiTaskDataset, MultiTaskSampler, MultiTaskDataloader
from dataset_utils import processors, output_modes, convert_examples_to_features, compute_metrics


if sys.version_info[0] == 2:
    pass
else:
    pass

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
logger = logging.getLogger(__name__)


def train(args, train_dataloader, model, tokenizer, optimizer, device, eval_dataloader=None, predict_dataloader=None):
    output_dir_head = deepcopy(config1.bert_config) if config1.bert_config.endswith('e') else deepcopy(args.output_dir)

    global_step = 0
    # lrs = {"multi": [], "visual": [], "acoustic": []}
    losses = {"multi": [], "visual": [], "acoustic": []}
    num_idx = {"multi": [0], "visual": [0], "acoustic": [0]}
    # num_idx = [idx+i*num_train_samples for i in range(3) for idx in range(0, num_train_samples, config1.train_batch_size)]
    for epoch in trange(int(config0.num_train_epochs), desc="Epoch", disable=config0.local_rank not in [-1, 0]):
        if output_dir_head.endswith('e'):
            new_epoch = int(output_dir_head.split('_')[-1][:-1]) + epoch + 1
            args.output_dir = "%s_%se" % ('_'.join(output_dir_head.split('_')[:-1]), new_epoch)
        else:
            new_epoch = epoch + 1
            args.output_dir = "%s_%se" % (output_dir_head, new_epoch)
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        logs = {}
        for step, batch in enumerate(
                tqdm(train_dataloader, desc="Iteration", disable=config0.local_rank not in [-1, 0])):
            batch["data"] = tuple(t.squeeze().to(device) for t in batch["data"])
            data_type = batch["data_type"][0]

            num_idx[data_type].append(len(batch["data"][0]) + num_idx[data_type][-1])

            outputs = model(batch)
            if outputs is None:
                continue
            else:
                logits, targets = outputs

            loss_fct = L1Loss()
            loss = loss_fct(logits, targets)  # 计算loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if config0.gradient_accumulation_steps > 1:
                loss = loss / config0.gradient_accumulation_steps

            loss.backward()

            # lrs[data_type].append(max(set(optimizer[data_type].get_lr())))
            losses[data_type].append(loss.item())
            tr_loss += loss.item()
            nb_tr_steps += 1

            optimizer[data_type].step()
            optimizer[data_type].zero_grad()
            global_step += 1

            # print("%s | loss: %s | lr: %s" % (data_type, str(round(loss.item(), 4)), lrs[data_type][-1]))
        print("\nmulti loss: %s | visual loss: %s | acoustic loss: %s\n" % (
            np.mean(losses["multi"]), np.mean(losses["visual"]), np.mean(losses["acoustic"])))

        log_loss = tr_loss / nb_tr_steps
        logs['global_step'] = global_step
        logs['log_loss'] = log_loss
        for key in logs.keys():
            logger.info(" Epoch %d | %s = %s", new_epoch, key, str(logs[key]))

        # 保存loss
        loss_save_path = os.path.join(args.output_dir, "linear_losses.json")
        with open(loss_save_path, "w", encoding="utf-8") as f:
            json.dump(losses, f)

        # Save checkpoint
        # if new_epoch % 5 == 0:
        save_checkpoint(args, model, tokenizer)

        # Save config files to json
        save_config_to_json_file(args.output_dir)

        # Evaluation
        if config0.do_eval:
            logger.info("***** Epoch %d | Running evaluation *****", new_epoch)
            logger.info("  Num examples = %d", args.num_eval_examples)
            logger.info("  Batch size = %d", config1.eval_batch_size)
            evaluate(args, eval_dataloader, model, device, new_epoch)

        # Test
        if config0.do_predict:
            logger.info("***** Epoch %d | Running prediction *****", new_epoch)
            logger.info("  Num examples = %d", args.num_predict_examples)
            logger.info("  Batch size = %d", config1.predict_batch_size)
            predict(args, predict_dataloader, model, device, new_epoch)

    save_checkpoint(args, model, tokenizer)

    # 绘制学习率曲线
    # lr_plot(config0.plot_types, num_idx, lrs)
    # 绘制损失值曲线
    # loss_plot(config0.plot_types, num_idx, losses, output_dir=args.output_dir)


def evaluate(args, eval_dataloader, model, device, new_epoch=0):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    eval_preds = []
    eval_labels = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch["data"] = tuple(t.squeeze().to(device) for t in batch["data"])

        with torch.no_grad():
            outputs = model(batch)
            if outputs is None:
                continue
            else:
                logits, targets = outputs

        # create eval loss and other metric required by the task
        loss_fct = L1Loss()
        tmp_eval_loss = loss_fct(logits, targets)  # 计算loss

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(eval_preds) == 0:
            eval_preds.append(logits.detach().cpu().numpy())
            eval_labels = targets.detach().cpu().numpy()
        else:
            eval_preds[0] = np.append(
                eval_preds[0], logits.detach().cpu().numpy(), axis=0)
            eval_labels = np.append(
                eval_labels, targets.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    eval_preds = eval_preds[0]
    eval_preds = np.squeeze(eval_preds)

    result = compute_metrics(config1.task_name, eval_preds, eval_labels)

    result['eval_loss'] = eval_loss

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Epoch %d | Eval results *****", new_epoch)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    output_eval_preds = os.path.join(args.output_dir, "eval_predicts.csv")
    with open(output_eval_preds, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["True sen", "Pred sen"])
        for true_sen, pred_sen in zip(eval_labels.tolist(), eval_preds.tolist()):
            csv_writer.writerow([true_sen, pred_sen])

    logger.info("***** Epoch %d | Eval finish *****", new_epoch)


def predict(args, predict_dataloader, model, device, new_epoch=0):
    test_preds = []
    test_labels = []

    for batch in tqdm(predict_dataloader, desc="Predicting"):
        batch["data"] = tuple(t.squeeze().to(device) for t in batch["data"])

        with torch.no_grad():
            outputs = model(batch)
            if outputs is None:
                continue
            else:
                logits, targets = outputs

            if len(test_preds) == 0:
                test_preds.append(logits.detach().cpu().numpy())
                test_labels = targets.detach().cpu().numpy()
            else:
                test_preds[0] = np.append(
                    test_preds[0], logits.detach().cpu().numpy(), axis=0)
                test_labels = np.append(
                    test_labels, targets.detach().cpu().numpy(), axis=0)

    test_preds = test_preds[0]
    test_preds = np.squeeze(test_preds)

    result = compute_metrics(config1.task_name, test_preds, test_labels)
    output_test_file = os.path.join(args.output_dir, "test_results.txt")
    with open(output_test_file, "a") as writer:
        logger.info("***** Epoch %d | Test results *****", new_epoch)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    output_test_preds = os.path.join(args.output_dir, "test_predicts.csv")
    with open(output_test_preds, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["True sen", "Pred sen"])
        for true_sen, pred_sen in zip(test_labels.tolist(), test_preds.tolist()):
            csv_writer.writerow([true_sen, pred_sen])

    logger.info("***** Epoch %d | Test finish *****", new_epoch)



def main(args):
    if config0.local_rank == -1 or config0.no_cuda:
        device = torch.device("cuda:%s" % config0.device_ids[0] if torch.cuda.is_available() and not config0.no_cuda else "cpu")
        # gpu数量
        # n_gpu = torch.cuda.device_count()
        args.n_gpu = len(config0.device_ids)
    else:
        torch.cuda.set_device(config0.local_rank)
        device = torch.device("cuda", config0.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if config0.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.n_gpu, bool(config0.local_rank != -1), config0.fp16))

    if config0.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            config0.gradient_accumulation_steps))

    random.seed(config0.seed)
    np.random.seed(config0.seed)
    torch.manual_seed(config0.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(config0.seed)

    if not config0.do_train and not config0.do_eval and not config0.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_predict` must be True.")

    config1.task_name = config1.task_name.lower()

    if config1.task_name not in processors:     # 检查任务是否在录
        raise ValueError("Task not found: %s" % (config1.task_name))

    processor = processors[config1.task_name]()
    args.output_mode = output_modes[config1.task_name]   # "classification" or "regression"

    label_list = processor.get_labels()
    args.num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(config1.bert_config, do_lower_case=config1.do_lower_case)
    model = MultiModalMultiTaskSentimentClassification(config1, config2, config3, config4)
    print(model)
    # # save config file
    # model.save_config_to_json_file(args.output_dir)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=config0.device_ids)

    # Prepare data loader
    if config0.do_train:
        # load multi data
        train_multi_examples = processor.get_train_examples(config1.data_dir)
        train_multi_features = convert_examples_to_features(
            train_multi_examples, label_list, config1.max_seq_length, tokenizer, args.output_mode, config1.pad_mode)

        all_input_ids = torch.tensor([f.input_ids for f in train_multi_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_multi_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_multi_features], dtype=torch.long)
        all_input_visual = torch.tensor([f.input_visual for f in train_multi_features], dtype=torch.float)
        all_input_acoustic = torch.tensor([f.input_acoustic for f in train_multi_features], dtype=torch.float)
        all_input_lengths = all_input_mask.sum(1).squeeze()
        all_label_ids = torch.tensor([f.label_id for f in train_multi_features], dtype=torch.float)

        train_multi_data = SingleTaskDataset(TensorDataset(all_input_ids, all_input_visual, all_input_acoustic,
                                                           all_input_lengths, all_input_mask, all_segment_ids,
                                                           all_label_ids),
                                             is_train=config1.is_train,
                                             task_id=0,
                                             task_type=config1.task_type,
                                             batch_size=config1.train_batch_size)
        train_datasets = [train_multi_data]

        if config2.is_train:
            # load visual data
            train_visual_data = SingleTaskDataset(TensorDataset(all_input_visual, all_input_lengths, all_label_ids),
                                                  is_train=config2.is_train,
                                                  task_id=1,
                                                  task_type=config2.task_type,
                                                  batch_size=config2.train_batch_size)
            train_datasets.append(train_visual_data)

        if config3.is_train:
            # load acoustic data
            train_acoustic_data = SingleTaskDataset(TensorDataset(all_input_acoustic, all_input_lengths, all_label_ids),
                                                    is_train=config3.is_train,
                                                    task_id=2,
                                                    task_type=config3.task_type,
                                                    batch_size=config3.train_batch_size)
            train_datasets.append(train_acoustic_data)

        multi_task_train_datasets = MultiTaskDataset(train_datasets)
        multi_task_batch_sampler = MultiTaskSampler(train_datasets, config0.mix_opt, config0.ratio)
        train_dataloader = MultiTaskDataloader(multi_task_train_datasets, batch_sampler=multi_task_batch_sampler)

    if config0.do_eval:
        eval_multi_data, args.num_eval_examples = load_and_collate_data(args, processor, tokenizer, label_list,
                                                                        predix='eval')
        eval_datasets = [eval_multi_data]
        multi_task_eval_datasets = MultiTaskDataset(eval_datasets)
        multi_task_batch_sampler = MultiTaskSampler(eval_datasets, config0.mix_opt, config0.ratio)
        eval_dataloader = DataLoader(multi_task_eval_datasets, batch_sampler=multi_task_batch_sampler)

    if config0.do_predict:
        predict_multi_data, args.num_predict_examples = load_and_collate_data(args, processor, tokenizer, label_list,
                                                                              predix='predict')
        predict_datasets = [predict_multi_data]
        multi_task_predict_datasets = MultiTaskDataset(predict_datasets)
        multi_task_batch_sampler = MultiTaskSampler(predict_datasets, config0.mix_opt, config0.ratio)
        predict_dataloader = DataLoader(multi_task_predict_datasets, batch_sampler=multi_task_batch_sampler)

    # Start training
    if config0.do_train:
        num_train_samples = len(multi_task_train_datasets)
        num_train_optimization_steps = len(train_dataloader) // config0.gradient_accumulation_steps * config0.num_train_epochs
        num_bert_optimization_steps = math.ceil(len(train_multi_data) / config1.train_batch_size) // \
                                      config0.gradient_accumulation_steps * config0.num_train_epochs

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'b_h', 'LayerNorm.bias', 'LayerNorm.weight']
        if config4.msl == "linear":
            mc_module = ['mbert_clf', 'msl_linear']      # multimodal classification module
        elif config4.msl == "rnn":
            mc_module = ['mbert_clf', 'rnn']
        elif config4.msl == "rnn-attn":
            mc_module = ['mbert_clf', 'rnn', 'attn']
        vc_module = ['visual']     # visual classification module
        ac_module = ['acoustic']   # acoustic classification module
        multi_optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and
                        any(mcm in n for mcm in mc_module)], 'weight_decay': config1.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and
                        any(mcm in n for mcm in mc_module)], 'weight_decay': 0.0}
            ]
        multi_optimizer_grouped_parameters_names = [
            {'params': [n for n, p in param_optimizer if not any(nd in n for nd in no_decay) and
                        any(mcm in n for mcm in mc_module)]},
            {'params': [n for n, p in param_optimizer if any(nd in n for nd in no_decay) and
                        any(mcm in n for mcm in mc_module)]}
        ]
        visual_optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if any(vcm in n for vcm in vc_module)]}
        ]
        visual_optimizer_grouped_parameters_names = [
            {'params': [n for n, p in param_optimizer if any(vcm in n for vcm in vc_module)]}
        ]
        acoustic_optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if any(acm in n for acm in ac_module)]}
        ]
        acoustic_optimizer_grouped_parameters_names = [
            {'params': [n for n, p in param_optimizer if any(acm in n for acm in ac_module)]}
        ]

        multi_optimizer = BertAdam(multi_optimizer_grouped_parameters,
                                   lr=config1.lr,
                                   warmup=config1.warmup_proportion,
                                   t_total=num_bert_optimization_steps)

        visual_optimizer = Adam(visual_optimizer_grouped_parameters, lr=config2.lr, weight_decay=config2.weight_decay)
        acoustic_optimizer = Adam(acoustic_optimizer_grouped_parameters, lr=config3.lr, weight_decay=config3.weight_decay)
        optimizer = {"multi": multi_optimizer,
                     "visual": visual_optimizer,
                     "acoustic": acoustic_optimizer
                     }

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_train_samples)
        logger.info("  Batch size = %d", config1.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train(args, train_dataloader, model, tokenizer, optimizer, device, eval_dataloader, predict_dataloader)


def save_checkpoint(args, model, tokenizer):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.mbert_clf.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    output_config0_file = os.path.join(args.output_dir, 'training_config0.bin')
    torch.save(config0, output_config0_file)

    return model, tokenizer


def save_config_to_json_file(output_dir):
    config0.to_json_file(os.path.join(output_dir, "config_global.json"))
    config1.to_json_file(os.path.join(output_dir, "config_mbert_clf.json"))
    config2.to_json_file(os.path.join(output_dir, "config_visual_clf.json"))
    config3.to_json_file(os.path.join(output_dir, "config_acoustic_clf.json"))
    config4.to_json_file(os.path.join(output_dir, "config_multitask.json"))


def load_and_collate_data(args, processor, tokenizer, label_list, predix='train'):
    if predix == "train":
        examples = processor.get_train_examples(config1.data_dir)
    elif predix == 'eval':
        examples = processor.get_dev_examples(config1.data_dir)
    elif predix == 'predict':
        examples = processor.get_test_examples(config1.data_dir)
    else:
        raise ValueError("Wrong predix!")

    features = convert_examples_to_features(
        examples, label_list, config1.max_seq_length, tokenizer, args.output_mode, config1.pad_mode)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_input_visual = torch.tensor([f.input_visual for f in features], dtype=torch.float)
    all_input_acoustic = torch.tensor([f.input_acoustic for f in features], dtype=torch.float)
    all_input_lengths = all_input_mask.sum(1).squeeze()
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    multi_data = SingleTaskDataset(TensorDataset(all_input_ids, all_input_visual, all_input_acoustic,
                                                      all_input_lengths, all_input_mask, all_segment_ids,
                                                      all_label_ids),
                                   is_train=False,
                                   task_id=0,
                                   task_type="multimodal sentiment classification",
                                   batch_size=config1.eval_batch_size)

    return multi_data, len(examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--dataset",
                        type=str,
                        default='mosi',
                        choices=['mosi', 'mosei'],
                        help="dataset name")
    args = parser.parse_args()

    # load configs
    config0 = GlobalConfig()
    config1 = MBertClfConfig()
    config2 = VisualClfConfig()
    config3 = AcousticClfConfig()
    config4 = MultiTaskConfig()

    for config in [config0, config1, config2, config3, config4]:
        config.dataset = args.dataset

    main(args)
