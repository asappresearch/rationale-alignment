from collections import Counter
from copy import deepcopy
from tqdm import tqdm
from typing import Dict, List, Set

import torch

from similarity.data.dataset import Dataset
from similarity.data.loaders.loader import DataLoader
from similarity.data.text import TextField
from similarity.data.utils import split_data, text_to_sentences
from utils.parsing import MultiNewsArguments


class MultiNewsDataLoader(DataLoader):
    def __init__(self, args: MultiNewsArguments):
        """Loads the multinews dataset."""
        # Load data
        article_groups = self.load_data(args.news_path, small_data=args.small_data)

        # Create ID mapping
        id_mapping = {}
        for i, articles in enumerate(article_groups):
            similar_ids = {f"{i}_{j}" for j in range(len(articles))}

            for j, article in enumerate(articles):
                id = f"{i}_{j}"
                similar_ids.remove(id)
                id_mapping[id] = {"similar": deepcopy(similar_ids), "dissimilar": set()}
                similar_ids.add(id)

        # Create ID to document
        id_to_document: Dict[str, List[str]] = {
            f"{i}_{j}": text_to_sentences(
                text=article,
                sentence_tokenize=not args.no_sentence_tokenize,
                max_num_sentences=args.max_num_sentences,
                max_sentence_length=args.max_sentence_length,
            )
            for i, articles in enumerate(tqdm(article_groups))
            for j, article in enumerate(articles)
        }

        # Create text field
        self._text_field = TextField()
        self._text_field.build_vocab(
            sentence for document in id_to_document.values() for sentence in document
        )

        print_stats_for_paper = False
        if print_stats_for_paper:
            print("\n\n==count of simiular pairs:")
            # print(cnt_pospair)
            print(sum([len(mapping["similar"]) for mapping in id_mapping.values()]) / 2)

            # Create ID to document
            from similarity.data.utils import tokenize_sentence

            cnt_sent_per_doc = [
                len(tokenize_sentence(article))
                for articles in article_groups
                for article in articles
            ]
            print("\n==Average sentence count:")
            print(sum(cnt_sent_per_doc) / len(cnt_sent_per_doc))
            print("==Max sentence count:")
            print(max(cnt_sent_per_doc))

            cnt_words_per_doc = [
                len(article.split())
                for articles in article_groups
                for article in articles
            ]
            print("==Average words count:")
            print(sum(cnt_words_per_doc) / len(cnt_words_per_doc))

            print("==Max words count:")
            print(max(cnt_words_per_doc))

            print("==count of total documents:")
            print(len(cnt_sent_per_doc))
            print("\n\n")
            print(f"\n==Vocabulary size = {len(self.text_field.vocabulary):,}")
            import sys

            sys.exit()

        # Convert sentences to indices
        id_to_document: Dict[str, List[torch.LongTensor]] = {
            id: [self._text_field.process(sentence) for sentence in document]
            for id, document in tqdm(id_to_document.items())
        }

        # Split data
        train_groups, dev_groups, test_groups = split_data(
            list(range(len(article_groups)))
        )

        train_ids = {
            f"{i}_{j}" for i in train_groups for j in range(len(article_groups[i]))
        }
        dev_ids = {
            f"{i}_{j}" for i in dev_groups for j in range(len(article_groups[i]))
        }
        test_ids = {
            f"{i}_{j}" for i in test_groups for j in range(len(article_groups[i]))
        }

        # Define train, dev, test datasets
        self._train = Dataset(
            ids=train_ids, id_to_document=id_to_document, id_mapping=id_mapping
        )
        self._dev = Dataset(
            ids=dev_ids, id_to_document=id_to_document, id_mapping=id_mapping
        )
        self._test = Dataset(
            ids=test_ids, id_to_document=id_to_document, id_mapping=id_mapping
        )

        self.print_stats()

    @staticmethod
    def load_data(path: str, small_data: bool = False) -> List[List[str]]:
        num_examples = 100 if small_data else float("inf")

        article_groups = []
        with open(path) as f:
            for line in tqdm(f):
                articles = [
                    article.strip()
                    for article in line.replace("NEWLINE_CHAR", "\n").split("|||||")
                ]
                articles = [article for article in articles if article != ""]
                article_groups.append(articles)

                if len(article_groups) >= num_examples:
                    break

        # Try to remove junk by only keeping articles that appear once (i.e. not a common error message)
        article_counts = Counter(
            article for articles in article_groups for article in articles
        )
        article_groups = [
            [article for article in articles if article_counts[article] == 1]
            for articles in article_groups
        ]

        # Require at least two articles per group so that there are similar articles
        article_groups = [articles for articles in article_groups if len(articles) >= 2]

        return article_groups

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
