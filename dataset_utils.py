# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import numpy as np
import pickle

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, mean_absolute_error, classification_report

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class MultiInputExample(object):
    """A single training/test example for multi-modal sequence classification."""

    def __init__(self, guid, text, visual, acoustic, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            visual: ndarray.
            acoustic: ndarray.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.visual = visual
        self.acoustic = acoustic
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MultiInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, input_v, input_a, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.input_visual = input_v
        self.input_acoustic = input_a
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


########### self task ###########
class MOSIProcessor(DataProcessor):
    def __init__(self, data_dir='./data/mosi/'):
        super(MOSIProcessor, self).__init__()
        self.train_data, self.dev_data, self.test_data = pickle.load(open(data_dir+'mosi.pkl', 'rb'))

    def get_train_examples(self, data_dir):
        return self._create_examples(self.train_data, set_type='train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(self.dev_data, set_type='dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(self.test_data, set_type='test')

    def get_labels(self):
        return [np.float]

    def _create_examples(self, samples, set_type):
        examples = []
        for index, sample in enumerate(samples):
            guid = '%s-%s' % (set_type, index)
            text = ' '.join(sample[0][0].tolist())     # string
            visual = sample[0][1]
            acoustic = sample[0][2]
            label = sample[1]     # regression
            examples.append(MultiInputExample(guid=guid, text=text, visual=visual, acoustic=acoustic, label=label))
        return examples


class MOSEIBaseProcessor(DataProcessor):
    def __init__(self, data_dir='./data/mosei/'):
        super(MOSEIBaseProcessor, self).__init__()
        self.train_data, self.dev_data, self.test_data = pickle.load(open(data_dir+'mosei.pkl', 'rb'))
        self.print()    # print data scale

    def get_train_examples(self, data_dir):
        return self._create_examples(self.train_data, set_type='train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(self.dev_data, set_type='dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(self.test_data, set_type='test')

    def get_labels(self):
        raise NotImplementedError()

    def _create_examples(self, data, set_type):
        raise NotImplementedError()

    def print(self):
        train = self._create_examples(self.train_data, set_type='train')
        dev = self._create_examples(self.dev_data, set_type='dev')
        test = self._create_examples(self.test_data, set_type='test')
        print("train: %s" % len(train))
        print("dev: %s" % len(dev))
        print("test: %s" % len(test))


class MOSEISentimentProcessor(MOSEIBaseProcessor):
    """ >=0: postive
        <0: negtive
    """
    def __init__(self, data_dir='./data/mosei/'):
        super(MOSEISentimentProcessor, self).__init__()

    def get_labels(self):
        return [np.float]

    def _create_examples(self, samples, set_type):
        examples = []
        for index, sample in enumerate(samples):
            guid = '%s-%s' % (set_type, index)
            text = ' '.join(sample[0][0].tolist())     # string
            visual = sample[0][1]
            acoustic = sample[0][2]
            label = sample[1][0]
            examples.append(MultiInputExample(guid=guid, text=text, visual=visual, acoustic=acoustic, label=label))
        return examples


def wp_align(arr, tokens, words, pad):
    """ anyhow    it was really good
        any ##how it was really good
        0   1     2  3   4      5
    :param arr: array for reshaping as tokens
    :param words: raw words
    :param tokens: bpe tokens of words
    :return: reshaped array
    """
    divisors = []
    for i, token in enumerate(tokens):
        # "it's" and "{lg}" for mosei
        if token.startswith('##'):
            divisors.append(0)
            if pad == 'zero':
                arr = np.insert(arr, i, [0]*arr.shape[-1], axis=0)
            elif pad == 'copy' or pad == 'mean':
                arr = np.insert(arr, i, arr[i-1], axis=0)
        else:
            divisors.append(1)

    if pad == 'mean':
        sp = 0
        ep = 0
        while ep < len(divisors):
            if divisors[ep] == 0:
                ep += 1
            elif divisors[ep] == 1:
                count = ep-sp
                divisors = divisors[:sp] + [count]*len(divisors[sp:ep]) + divisors[ep:]
                sp = ep
                ep += 1
        # 末位mask为0的情况
        if divisors[ep-1] == 0:
            count = ep - sp
            divisors = divisors[:sp] + [count] * len(divisors[sp:ep]) + divisors[ep:]
        divisors = np.array(divisors)
        arr = arr / divisors.reshape(-1, 1)

    assert len(arr) == len(tokens)

    return arr


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, pad_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # print(ex_index)
        tokens = tokenizer.tokenize(example.text)
        input_visual = wp_align(example.visual, tokens, example.text, pad=pad_mode)
        input_acoustic = wp_align(example.acoustic, tokens, example.text, pad=pad_mode)

        assert len(input_visual) == len(input_acoustic)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
            input_visual = input_visual[:max_seq_length-2, :]
            input_acoustic = input_acoustic[:max_seq_length-2, :]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        v_len = len(input_visual)
        v_size = input_visual.shape[1]
        a_len = len(input_acoustic)
        a_size = input_acoustic.shape[1]
        # [CLS] pad
        input_visual = np.insert(input_visual, 0, [0] * v_size, axis=0)
        input_acoustic = np.insert(input_acoustic, 0, [0] * a_size, axis=0)
        # [SEP] pad
        input_visual = np.insert(input_visual, v_len+1, [0] * v_size, axis=0)
        input_acoustic = np.insert(input_acoustic, v_len+1, [0] * a_size, axis=0)
        # other pad
        if v_len+2 < max_seq_length:
            input_visual = np.insert(input_visual, v_len+2, [[0]*v_size]*(max_seq_length-v_len-2), axis=0)
            input_acoustic = np.insert(input_acoustic, a_len+2, [[0]*a_size]*(max_seq_length-a_len-2), axis=0)

        # padding_v = np.zeros((max_seq_length - len(input_visual), input_visual.shape[1]))
        # padding_a = np.zeros((max_seq_length - len(input_acoustic), input_acoustic.shape[1]))
        # input_visual = np.concatenate((input_visual, padding_v), axis=0)
        # input_acoustic = np.concatenate((input_acoustic, padding_a), axis=0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_visual) == max_seq_length
        assert len(input_acoustic) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                MultiInputFeatures(input_ids=input_ids,
                                   input_v=input_visual,
                                   input_a=input_acoustic,
                                   input_mask=input_mask,
                                   segment_ids=segment_ids,
                                   label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "f1": f1
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def mosi_metrics(y_pred, y_true):
    """ Binary accuracy, F1 score, MAE, Correlation"""
    y_true_bin = y_true >= 0
    y_pred_bin = y_pred >= 0

    metrics = acc_and_f1(y_pred_bin, y_true_bin, average='weighted')
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["corr"] = np.corrcoef(y_pred, y_true)[0][1]

    return metrics


def mosi_metrics_noneq(y_pred, y_true):
    """ Binary accuracy, F1 score, MAE, Correlation"""
    y_true_bin = y_true > 0
    y_pred_bin = y_pred > 0

    metrics = acc_and_f1(y_pred_bin, y_true_bin, average='weighted')
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["corr"] = np.corrcoef(y_pred, y_true)[0][1]

    return metrics


def mosi_metrics_nonzero(y_pred, y_true):
    """ Binary accuracy, F1 score, MAE, Correlation"""
    non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])

    y_true_bin = y_true[non_zeros] > 0
    y_pred_bin = y_pred[non_zeros] > 0

    metrics = acc_and_f1(y_pred_bin, y_true_bin, average='weighted')
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["corr"] = np.corrcoef(y_pred, y_true)[0][1]

    return metrics


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "mosi":
        return mosi_metrics(preds, labels)
    elif task_name == "mosei":
        return mosi_metrics(preds, labels)
    else:
        raise KeyError(task_name)

processors = {
    # self task
    "mosi": MOSIProcessor,
    "mosei": MOSEISentimentProcessor,
}

output_modes = {
    "mosi": "regression",
    "mosei": "regression",
}
