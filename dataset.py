import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from functools import reduce
from copy import deepcopy


def enc_collate_fn(batch):
    if batch[0]["data_type"] == "audio":
        data_type = [sample["data_type"] for sample in batch]

        inputs = torch.FloatTensor([sample["data"][0] for sample in batch])
        output_sizes = torch.IntTensor([sample["data"][1] for sample in batch])
        target_sizes = torch.IntTensor([sample["data"][3] for sample in batch])

        # sort to decreasing order
        output_sizes, indices = output_sizes.sort(descending=True)
        inputs = inputs[indices]
        target_sizes = target_sizes[indices]

        targets = [sample["data"][2] for sample in batch]
        targets = [targets[int(i)] for i in indices]
        targets = torch.IntTensor(reduce(lambda x, y: x + y, targets))

        return {"data_type": data_type,
                "data": (inputs, targets, output_sizes, target_sizes)}

    else:
        return default_collate(batch)


class MultiTaskDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MultiTaskDataloader, self).__init__(*args, **kwargs)
        self.collate_fn = enc_collate_fn


class MultiTaskSampler(BatchSampler):
    def __init__(self, datasets, mix_opt=0, extra_task_ratio=0):
        self._datasets = datasets
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        train_data_list = []
        for dataset in datasets:
            if dataset.get_data_name() == "an4":
                index_batches = [self._get_shuffled_index_batches(len(dataset), dataset.get_batch_size())]
                for i in range(15):
                    index_batches.append(deepcopy(index_batches[-1]))
                    random.shuffle(index_batches[-1])
                index_batches = reduce(lambda x,y:x+y, index_batches)
                train_data_list.append(index_batches)
            elif dataset.get_task_type() in ["visual sentiment classification", "acoustic sentiment classification"]:
                index_batches = [self._get_shuffled_index_batches(len(dataset), dataset.get_batch_size())]
                for i in range(0):
                    index_batches.append(deepcopy(index_batches[-1]))
                    random.shuffle(index_batches[-1])
                index_batches = reduce(lambda x,y:x+y, index_batches)
                train_data_list.append(index_batches)
            else:
                train_data_list.append(self._get_shuffled_index_batches(len(dataset), dataset.get_batch_size()))
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [list(range(i, min(i+batch_size, dataset_len))) for i in range(0, dataset_len, batch_size)]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(self._train_data_list, self._mix_opt, self._extra_task_ratio)
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].get_task_id()
            batch = next(all_iters[local_task_idx])
            indices = [(task_id, sample_id) for sample_id in batch]
            random.shuffle(indices)
            yield indices

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices)))
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices


class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, "Duplicate task_id %s" % task_id
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]


class SingleTaskDataset(Dataset):
    def __init__(self,
                 data,
                 is_train=True,
                 data_name=None,
                 task_id=0,
                 task_type="multimodal sentiment classification",
                 factor=1.0,
                 batch_size=None,
                 ):
        self._data = data
        self._data_name = data_name
        self._task_id = task_id
        self._task_type = task_type
        self._factor = factor
        self._batch_size = batch_size

    def get_data_name(self):
        return self._data_name

    def get_task_id(self):
        return self._task_id

    def get_task_type(self):
        return self._task_type

    def get_factor(self):
        return self._factor

    def get_batch_size(self):
        return self._batch_size

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if self._task_type == "multimodal sentiment classification":
            instance = {"data_id": item,
                        "data_type": "multi",
                        "data": self._data[item]}
            return instance
        elif self._task_type == "image caption":
            instance = {"data_id": item,
                        "data_type": "image",
                        "data": self._data[item]}
            return instance
        elif self._task_type == "speech recognition":
            instance = {"data_id": item,
                        "data_type": "audio",
                        "data": self._data[item]}
            return instance
        elif self._task_type == "text sentiment classification":
            instance = {"data_id": item,
                        "data_type": "text",
                        "data": self._data[item]}
            return instance
        elif self._task_type == "visual sentiment classification":
            instance = {"data_id": item,
                        "data_type": "visual",
                        "data": self._data[item]}
            return instance
        elif self._task_type == "acoustic sentiment classification":
            instance = {"data_id": item,
                        "data_type": "acoustic",
                        "data": self._data[item]}
            return instance
        else:
            raise ValueError()

