from collections import Counter, defaultdict
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, List, Set
import json
import torch

from classify.data.dataset import Dataset
from classify.data.loaders.loader import DataLoader
from utils.parsing import SNLIArguments
from classify.data.text import TextField
from classify.data.utils import split_data, text_to_sentences

import jsonlines
from collections import defaultdict
import random


class SNLIDataLoader(DataLoader):
    def __init__(self, args: SNLIArguments):
        """Loads the pubmed dataset."""

        # Determine word to index mapping
        self.small_data = args.small_data

        # Load data
        train_label, train_evidences = self.load_label(
            args.snli_path, "train", args.small_data
        )
        dev_label, dev_evidences = self.load_label(
            args.snli_path, "val", args.small_data
        )
        test_label, test_evidences = self.load_label(
            args.snli_path, "test", args.small_data
        )

        train_id_mapping = {
            k + "_premise": k + "_hypothesis" for k in train_label.keys()
        }
        dev_id_mapping = {k + "_premise": k + "_hypothesis" for k in dev_label.keys()}
        test_id_mapping = {k + "_premise": k + "_hypothesis" for k in test_label.keys()}
        train_ids = set(train_label.keys())
        dev_ids = set(dev_label.keys())
        test_ids = set(test_label.keys())

        if self.small_data:
            allids = (
                list(train_id_mapping.keys())
                + list(train_id_mapping.values())
                + list(dev_id_mapping.keys())
                + list(dev_id_mapping.values())
                + list(test_id_mapping.keys())
                + list(test_id_mapping.values())
            )
            id_to_text = self.load_text(args.snli_path)
            id_to_text = {k: v for k, v in id_to_text.items() if k in allids}
        else:
            id_to_text = self.load_text(args.snli_path)

        texts = list(id_to_text.values())
        self._text_field = TextField()
        self._text_field.build_vocab(texts)

        sampled = {k: id_to_text[k] for k in random.sample(list(id_to_text.keys()), 10)}
        print(sampled)

        # Convert sentences to indices
        id_to_doctoken: Dict[str, List[torch.LongTensor]] = {
            idx: self._text_field.process(text)
            # for id, sentence in tqdm(id_to_doctoken.items())
            for idx, text in tqdm(id_to_text.items())
        }

        print(len(id_to_text))
        print(len(id_to_doctoken))
        sampled = {
            k: id_to_doctoken[k] for k in random.sample(list(id_to_doctoken.keys()), 10)
        }
        print(sampled)
        # import sys; sys.exit()
        # Define train, dev, test datasets
        self._train = Dataset(
            ids=train_ids,
            id_to_document=id_to_doctoken,
            id_mapping=train_id_mapping,
            label_map=train_label,
            evidence=train_evidences,
        )
        self._dev = Dataset(
            ids=dev_ids,
            id_to_document=id_to_doctoken,
            id_mapping=dev_id_mapping,
            label_map=dev_label,
            evidence=dev_evidences,
        )
        self._test = Dataset(
            ids=test_ids,
            id_to_document=id_to_doctoken,
            id_mapping=test_id_mapping,
            label_map=test_label,
            evidence=test_evidences,
        )

        self.print_stats()

    @staticmethod
    def load_text(path: str, small_data: bool = False) -> List[List[List[List[str]]]]:
        data = defaultdict(dict)
        print(f"reading text from {path}")
        reader = jsonlines.Reader(open(path))
        for line in reader:
            # doc, side = line['docid'].split('_')
            # data[doc][side] = line['document']
            data[line["docid"]] = line["document"]
        return data

    @staticmethod
    def load_label(
        path: str, flavor: str, small_data: bool = False
    ) -> List[List[List[List[str]]]]:
        label_path = path.replace("docs", flavor)
        print(f"reading labels from {label_path}")
        labels = defaultdict(dict)
        evidences = defaultdict(list)
        label_toi = {"entailment": 0, "contradiction": 1, "neutral": 2}

        reader = jsonlines.Reader(open(label_path))
        for line in reader:
            label = line["classification"]
            idx = line["annotation_id"]
            labels[idx] = label_toi[label]
            evidences[label + "_hypothesis"] = []
            evidences[label + "_premise"] = []
            for evi in line["evidences"][0]:
                evidences[evi["docid"]].append((evi["start_token"], evi["end_token"]))
            if small_data and len(labels) > 1500:
                break
        return labels, evidences

    @property
    def train(self) -> Dataset:
        return self._train

    @property
    def dev(self) -> Dataset:
        return self._dev

    @property
    def test(self) -> Dataset:
        return self._test

    @property
    def text_field(self) -> TextField:
        return self._text_field
