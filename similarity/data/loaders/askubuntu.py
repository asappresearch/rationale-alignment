from collections import defaultdict
import csv
import random
from typing import Dict, List, Set

import torch
from tqdm import tqdm

from similarity.data.dataset import Dataset
from similarity.data.loaders.loader import DataLoader
from utils.parsing import AskUbuntuArguments
from similarity.data.text import TextField
from utils.berttokenizer import BTTokenizer
from similarity.data.utils import split_data, text_to_sentences


class AskUbuntuDataLoader(DataLoader):
    def __init__(self, args: AskUbuntuArguments):
        """
        Constructs an AskUbuntu document alignment dataset.
        https://github.com/taolei87/askubuntu
        https://github.com/ASAPPinc/QRA_benchmark
        """
        # Load similar and dissimilar train, dev, and test IDs
        self.small_data = args.small_data
        self.args = args
        if args.dataset == "superuser_askubuntu":
            # In this case, we are only given similar pairs
            # and we need to split into train, dev, and test
            assert args.dev_path is None and args.test_path is None

            similar_id_mapping = self._load_similar_id_mapping(
                args.train_path, args.transitive
            )
            similar_ids = list(similar_id_mapping.keys())
            train_ids, dev_ids, test_ids = split_data(similar_ids)
            train_id_mapping = self._filter_id_mapping(similar_id_mapping, train_ids)
            dev_id_mapping = self._filter_id_mapping(similar_id_mapping, dev_ids)
            test_id_mapping = self._filter_id_mapping(similar_id_mapping, test_ids)
        elif args.dataset == "askubuntu":
            # In this case, we are given similar and dissimilar pairs
            # and we are given the train, dev, and test splits
            assert args.dev_path is not None and args.test_path is not None

            train_id_mapping = self._load_id_mapping(
                args.train_path, prune_similar_count=10
            )
            dev_id_mapping = self._load_id_mapping(args.dev_path)
            test_id_mapping = self._load_id_mapping(args.test_path)
        else:
            raise ValueError(f"Ubuntu data type {args.dataset} not supported")

        # Use fewer train questions when debugging
        # if args.small_data:
        #     train_ids = list(train_id_mapping.keys())
        #     random.seed(0)
        #     random.shuffle(train_ids)
        #     train_ids = set(train_ids[:10])
        #     train_id_mapping = {id: id_map for id, id_map in train_id_mapping.items() if id in train_ids}
        #     dev_ids = list(dev_id_mapping.keys())
        #     random.shuffle(dev_ids)
        #     dev_ids = set(dev_ids[:10])
        #     dev_id_mapping = {id: id_map for id, id_map in dev_id_mapping.items() if id in dev_ids}
        #     test_ids = list(test_id_mapping.keys())
        #     random.shuffle(test_ids)
        #     test_ids = set(test_ids[:10])
        #     test_id_mapping = {id: id_map for id, id_map in test_id_mapping.items() if id in test_ids}

        # Get train, dev, test IDs
        train_ids = set(train_id_mapping.keys())
        dev_ids = set(dev_id_mapping.keys())
        test_ids = set(test_id_mapping.keys())

        # Load tokenized text data and map id to title and text
        id_to_question = self._load_text(args.text_path)
        print(f"total document count: {len(id_to_question)}")
        if self.small_data:
            dev_ids = test_ids = train_ids
            dev_id_mapping = test_id_mapping = train_id_mapping
            id_to_question = {
                k: v
                for k, v in id_to_question.items()
                if k in train_ids or k in dev_ids or k in test_ids
            }
            print(f"total small document count: {len(id_to_question)}")
        # if self.args.eval_only:
        #     print("evaluating only, not using train set")
        #     dev_available_ids: Set[str] = set.union(
        #         set(dev_ids),
        #         *[dev_id_mapping[document_id]["similar"] for document_id in dev_ids],
        #     )
        #     test_available_ids: Set[str] = set.union(
        #         set(test_ids),
        #         *[test_id_mapping[document_id]["similar"] for document_id in test_ids],
        #     )

        #     devtest_available_ids = list(dev_available_ids) + list(test_available_ids)
        #     print(f"total document count: {len(devtest_available_ids)}")
        #     id_to_question = {
        #         k: v
        #         for k, v in id_to_question.items()
        #         if k in list(devtest_available_ids)
        #     }
        #     print(f"total document count: {len(id_to_question)}")

        # Get texts
        texts = [question["title"] for question in id_to_question.values()]
        if not args.title_only:
            texts += [question["text"] for question in id_to_question.values()]

        # Determine word to index mapping
        if args.bert:
            self._text_field = BTTokenizer(args)
        else:
            self._text_field = TextField()
            self._text_field.build_vocab(texts)

        print_stats_for_paper = False
        if print_stats_for_paper:
            print("\n\n==count of simiular pairs:")
            # print(cnt_pospair)
            # cnt_list =  [ len(mapping['similar']) for mapping in train_id_mapping.values()] + [ len(mapping['similar']) for mapping in dev_id_mapping.values()] \
            #     + [ len(mapping['similar']) for mapping in test_id_mapping.values()]
            cnt_list = [len(mapping) - 1 for mapping in similar_id_mapping.values()]
            print(sum(cnt_list) / 2)
            print(sum([len(mapping) for mapping in similar_id_mapping.values()]) / 2)

            from similarity.data.utils import tokenize_sentence

            cnt_sent_per_doc = [len(tokenize_sentence(article)) for article in texts]
            print("\n==Average sentence count:")
            print(sum(cnt_sent_per_doc) / len(cnt_sent_per_doc))
            print("==Max sentence count:")
            print(max(cnt_sent_per_doc))

            texts = [
                question["title"] + " " + question["text"]
                for question in id_to_question.values()
            ]
            question = list(id_to_question.values())[0]
            print(question["title"])
            print(question["text"])
            print(texts[0])

            cnt_words_per_doc = [len(article.split()) for article in texts]
            print("==Average words count:")
            print(sum(cnt_words_per_doc) / len(cnt_words_per_doc))

            print("==Max words count:")
            print(max(cnt_words_per_doc))

            print("==count of total documents:")
            print(len(cnt_sent_per_doc))
            print(len(texts))
            print(len(id_to_question))

            # print(f'\n==Vocabulary size = {len(self.text_field.vocabulary):,}')
            # import sys
            # sys.exit()

        # Split text into sentences
        if args.no_sentence_tokenize:
            id_to_document: Dict[str, List[str]] = {
                id: [
                    " ".join(
                        (
                            question["title"].split()
                            + (question["text"].split() if not args.title_only else [])
                        )[: args.max_sentence_length]
                    )
                ]
                for id, question in tqdm(id_to_question.items())
            }
        else:
            id_to_document: Dict[str, List[str]] = {
                id: [" ".join(question["title"].split()[: args.max_sentence_length])]
                + (
                    text_to_sentences(
                        text=question["text"],
                        sentence_tokenize=not args.no_sentence_tokenize,
                        max_num_sentences=args.max_num_sentences,
                    )
                    if not args.title_only
                    else []
                )
                for id, question in tqdm(id_to_question.items())
            }

        # Convert sentences to indices
        id_to_document: Dict[str, List[torch.LongTensor]] = {
            id: [self._text_field.process(sentence) for sentence in document]
            for id, document in tqdm(id_to_document.items())
        }

        # Define train, dev, test datasets
        self._train = Dataset(
            ids=train_ids, id_to_document=id_to_document, id_mapping=train_id_mapping
        )
        self._dev = Dataset(
            ids=dev_ids, id_to_document=id_to_document, id_mapping=dev_id_mapping
        )
        self._test = Dataset(
            ids=test_ids, id_to_document=id_to_document, id_mapping=test_id_mapping
        )

        self.print_stats()

    @staticmethod
    def _load_text(path: str) -> Dict[str, Dict[str, str]]:
        """
        Loads text data as a dictionary mapping question id to text.
        :param path: Path to .txt file containing text data.
        :return: A dictionary mapping question id to a dictionary with the title and text.
        """
        with open(path) as f:
            id_to_question = {
                id: {"title": title, "text": text}
                for id, title, text in csv.reader(f, delimiter="\t")
            }

        return id_to_question

    @staticmethod
    def _load_id_mapping(
        path: str, prune_similar_count: int = None, skip_if_all_same: bool = True
    ) -> Dict[str, Dict[str, Set[str]]]:
        """
        Loads a mapping from question ID to similar and dissimilar question IDs.
        The format is:
        question ID \t similar question IDs (space separated) \t dissimilar question IDs (space separated)
        :param path: Path to a .txt file containing the mapping.
        :param prune_similar_count: Prunes the questions with more than this number of similar documents.
        :param skip_if_all_same: Whether to skip examples with all similar or all dissimilar questions.
        :return: A dictionary mapping from question ID to a dictionary mapping to similar and dissimilar IDs.
        """
        id_mapping = {}

        with open(path) as f:
            for row in csv.reader(f, delimiter="\t"):
                id = row[0]
                similar_ids = set(row[1].split()) - {id}
                other_ids = set(row[2].split()) - {id}
                all_ids = similar_ids | other_ids | {id}
                dissimilar_ids = all_ids - similar_ids - {id}

                if skip_if_all_same and (
                    len(similar_ids) == 0 or len(dissimilar_ids) == 0
                ):
                    continue

                if (
                    prune_similar_count is not None
                    and len(similar_ids) > prune_similar_count
                ):
                    continue

                id_mapping[id] = {"similar": similar_ids, "dissimilar": dissimilar_ids}

        return id_mapping

    # @staticmethod
    def _load_similar_id_mapping(
        self, path: str, transitive: bool = False
    ) -> Dict[str, Set[str]]:
        """
        Loads a mapping from question ID to similar question IDs.
        The format is:
        question_0_ID question_1_ID
        where question_0 and question_1 are similar questions.
        :param path: Path to a .txt file containing the mapping.
        :param transitive: Whether to apply transitivity to the
        similarity markings, i.e. if 1 is similar to 2 and 2 is
        similar to 3, mark 1 as similar to 3.
        :return: A dictionary mapping from question ID to a set
        of similar question IDs.
        """
        similar_id_mapping = defaultdict(set)
        count = 0

        with open(path) as f:
            for row in csv.reader(f, delimiter=" "):
                id_0, id_1 = row
                similar_id_mapping[id_0].add(id_1)
                similar_id_mapping[id_1].add(id_0)

                if transitive:
                    similar_id_mapping[id_0] |= similar_id_mapping[id_1]
                    similar_id_mapping[id_1] |= similar_id_mapping[id_0]

                count += 1
                if count > 100 and self.small_data:
                    print(count)
                    break
        return dict(similar_id_mapping)

    @staticmethod
    def _filter_id_mapping(
        similar_id_mapping: Dict[str, Set[str]], keep_ids: List[str]
    ) -> Dict[str, Dict[str, Set[str]]]:
        """
        Takes a similar id mapping and filters it to only include provided ids.
        Also changes format to match format of id mapping with a set of all ids,
        a set of similar ids, and an empty set of dissimilar ids.
        :param similar_id_mapping: A dictionary mapping from question ID to a set
        of similar question IDs.
        :param keep_ids: A list of ids to keep.
        :return: A dictionary mapping from question ID to a dictionary mapping
        to similar and dissimilar IDs.
        """
        keep_ids = set(keep_ids) & set(similar_id_mapping.keys())

        id_mapping = {}
        for id in keep_ids:
            similar_ids = similar_id_mapping[id] & keep_ids

            if len(similar_ids) > 0:
                id_mapping[id] = {"similar": similar_ids, "dissimilar": set()}

        return id_mapping

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
