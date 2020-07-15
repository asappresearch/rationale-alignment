import os
from collections import Counter, defaultdict
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, List, Set
import json
import torch

from classify.data.dataset import Dataset
from classify.data.loaders.loader import DataLoader
from utils.parsing import MultircArguments
from classify.data.text import TextField
from utils.berttokenizer import BTTokenizer
from classify.data.utils import split_data, text_to_sentences

import jsonlines
from collections import defaultdict
import random


class MultircDataLoader(DataLoader):
    def __init__(self, args: MultircArguments):
        """Loads the pubmed dataset."""

        # Determine word to index mapping
        # self.small_data = args.small_data #dataset small enough
        self.args = args
        id_to_document = self.load_text(args.multirc_path)
        self.id_to_document = id_to_document

        # Load data
        train_label, train_evidences = self.load_label(
            args.multirc_path, "train", args.small_data
        )
        dev_label, dev_evidences = self.load_label(
            args.multirc_path, "val", args.small_data
        )
        test_label, test_evidences = self.load_label(
            args.multirc_path, "test", args.small_data
        )

        train_id_mapping = {idx: idx.split(":")[0] for idx in train_label.keys()}
        dev_id_mapping = {idx: idx.split(":")[0] for idx in dev_label.keys()}
        test_id_mapping = {idx: idx.split(":")[0] for idx in test_label.keys()}
        train_ids = set(train_label.keys())
        dev_ids = set(dev_label.keys())
        test_ids = set(test_label.keys())

        if args.bert:
            self._text_field = BTTokenizer(args)
        else:
            if self.args.word_to_word:
                texts = list(id_to_document.values())
            else:
                texts = [x for doc in list(id_to_document.values()) for x in doc]
            self._text_field = TextField()
            self._text_field.build_vocab(texts)

        sampled = {
            k: id_to_document[k] for k in random.sample(list(id_to_document.keys()), 10)
        }
        print(sampled)

        # Convert sentences to indices
        if self.args.word_to_word:
            id_to_doctoken: Dict[str, List[torch.LongTensor]] = {
                id: self._text_field.process(text)[: args.max_sentence_length]
                for id, text in tqdm(id_to_document.items())
            }
            id_to_lengths = {}
        else:
            id_to_doctoken: Dict[str, List[torch.LongTensor]] = {
                id: [self._text_field.process(sentence) for sentence in document]
                for id, document in id_to_document.items()
            }
            id_to_sentlength = {
                id: [len(sentence.split()) for sentence in document]
                for id, document in id_to_document.items()
            }

        print(len(id_to_document))
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
            id_to_sentlength=id_to_sentlength,
        )
        self._dev = Dataset(
            ids=dev_ids,
            id_to_document=id_to_doctoken,
            id_mapping=dev_id_mapping,
            label_map=dev_label,
            evidence=dev_evidences,
            id_to_sentlength=id_to_sentlength,
        )
        self._test = Dataset(
            ids=test_ids,
            id_to_document=id_to_doctoken,
            id_mapping=test_id_mapping,
            label_map=test_label,
            evidence=test_evidences,
            id_to_sentlength=id_to_sentlength,
        )

        self.print_stats()

    def load_text(
        self, path: str, small_data: bool = False
    ) -> List[List[List[List[str]]]]:
        # return mapping from id to text, text list is splited as sentences.
        data = defaultdict(dict)
        print(f"reading text from {path}")
        # Reading the documents
        files = os.listdir(os.path.join(path, "docs"))
        for f in files:
            docid = f
            text = open(os.path.join(path, "docs", f), "r").readlines()
            text = [x.strip() for x in text]
            data[docid] = " ".join(text) if self.args.word_to_word else text
        for flavor in ["train", "test", "val"]:
            label_path = os.path.join(path, flavor + ".jsonl")
            print(f"reading questions from {label_path}")
            reader = jsonlines.Reader(open(label_path))
            for line in reader:
                docid = line["annotation_id"]
                text = [x.strip() for x in line["query"].split("||")]
                assert len(text) == 2
                if len(text[0].split()) == 0 or len(text[1].split()) == 0:
                    print(text)
                    print("bad queries")
                else:
                    data[docid] = " ".join(text) if self.args.word_to_word else text
        return data

    def load_label(
        self, path: str, flavor: str, small_data: bool = False
    ) -> List[List[List[List[str]]]]:
        label_path = os.path.join(path, flavor + ".jsonl")
        print(f"reading labels from {label_path}")
        labels = defaultdict(dict)
        evidences = {"token": defaultdict(list), "sentence": defaultdict(list)}
        label_toi = {"False": 0, "True": 1}
        reader = jsonlines.Reader(open(label_path))
        for line in reader:
            label = line["classification"]
            idx = line["annotation_id"]
            if idx in self.id_to_document:
                docid0 = idx.split(":")[0]
                labels[idx] = label_toi[label]
                assert len(line["evidences"]) == 1
                for evi in line["evidences"][0]:
                    assert evi["docid"] == docid0
                    assert evi["end_sentence"] - evi["start_sentence"] == 1
                    evidences["token"][idx].append(
                        (evi["start_token"], evi["end_token"])
                    )
                    evidences["sentence"][idx].append(
                        (evi["start_sentence"], evi["end_sentence"])
                    )
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

