from collections import OrderedDict
import csv
from itertools import chain
import os
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from tqdm import tqdm


class TextField:
    def __init__(self,
                 skip_oov: bool = False,
                 lower: bool = False,
                 tokenizer: Callable[[str], Iterable[str]] = lambda text: text.split(),
                 pad_token: Optional[str] = '<pad>',
                 unk_token: Optional[str] = '<unk>',
                 sos_token: Optional[str] = None,
                 eos_token: Optional[str] = None,
                 vocabulary: Optional[Dict[str, int]] = None):
        if vocabulary:
            self.vocabulary = OrderedDict((tok, i) for i, tok in enumerate(vocabulary))
        else:
            # Add specials
            self.vocabulary = OrderedDict()

            specials = [pad_token, unk_token, sos_token, eos_token]

            for token in filter(lambda x: x is not None, specials):
                self.vocabulary[token] = len(self.vocabulary)

        self.pad = pad_token
        self.unk = unk_token
        self.sos = sos_token
        self.eos = eos_token

        self.lower = lower
        self.tokenizer = tokenizer

        self._embeddings = None

        self._build_reverse_vocab()
        self.skip_oov = skip_oov
        self.weights = {}
        self.avg_weight = 1.0

    def build_idf_weights(self, *data: Iterable[str]) -> torch.FloatTensor:
        """Build IDF weights.

        Parameters
        ----------
        data : Iterable[str]
            List of input strings.

        """
        data = chain.from_iterable(data)
        if self.lower:
            data = [text.lower() for text in data]

        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), binary=False)
        vectorizer.fit(data)

        self.weights = {
            word: idf
            for word, idf in zip(vectorizer.get_feature_names(), vectorizer.idf_)
            if word in self.vocabulary
        }
        self.avg_weight = np.mean(list(self.weights.values()))

    def _init_vocabulary(self) -> None:
        """Initializes vocabulary with special tokens."""
        # Add specials
        self.vocabulary = OrderedDict()

        specials = [self.pad, self.unk, self.sos, self.eos]

        for token in filter(lambda x: x is not None, specials):
            self.vocabulary[token] = len(self.vocabulary)  # type: ignore

    def load_vocab(self, path: str) -> Dict[str, int]:
        """Loads a vocabulary from a .txt file.

        Returns
        -------
        Dict[str, int]
            A vocabulary dictionary mapping from string to int.

        """
        self._init_vocabulary()

        with open(path) as f:
            words = [word for line in f for word in line.strip().split()]

        for word in words:
            self.vocabulary[word] = len(self.vocabulary)

        return self.vocabulary

    def build_vocab(self, data: Iterable[str], *args) -> Dict[str, int]:
        """Build the vocabulary.
        Parameters
        ----------
        data : Iterable[str]
            List of input strings.
        """
        datasets = [data] + list(args)
        for dataset in datasets:
            for example in tqdm(dataset):
                # Lowercase if requested
                example = example.lower() if self.lower else example
                # Tokenize and add to vocabulary
                for token in self.tokenizer(example):
                    self.vocabulary.setdefault(token, len(self.vocabulary))

        self._build_reverse_vocab()

        return self.vocabulary

    def load_embeddings(self, path: str) -> torch.FloatTensor:
        """Load pretrained word embeddings.

        Parameters
        ----------
        path : str
            The path to the pretrained embeddings
        Returns
        -------
        torch.FloatTensor
            The matrix of pretrained word embeddings

        """
        ext = os.path.splitext(path)[-1]

        if ext == '.bin':  # fasttext
            try:
                import fasttext
            except Exception:
                try:
                    import fastText as fasttext
                except Exception:
                    raise ValueError("fasttext not installed.")
            model = fasttext.load_model(path)
            vectors = [model.get_word_vector(token) * self.weights.get(token, self.avg_weight) for token in tqdm(self.vocabulary)]
        else:
            # Load any .txt or word2vec kind of format
            model = dict()
            data = pd.read_csv(path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
            embedding_size = len(data.columns)
            for word, vector in data.iterrows():
                if word in self.vocabulary:
                    model[word] = np.array(vector.values) * self.weights.get(word, self.avg_weight)

            # Reorder according to self._vocab
            vectors = [model.get(token, np.zeros(embedding_size)) for token in self.vocabulary]

        self.embeddings = torch.FloatTensor(np.array(vectors))
        return self.embeddings

    def process(self, example: str) -> torch.LongTensor:  # type: ignore
        """Process an example, and create a Tensor.
        Parameters
        ----------
        example: str
            The example to process, as a single string
        Returns
        -------
        torch.LongTensor
            The processed example, tokenized and numericalized
        """
        # Lowercase and tokenize
        example = example.lower() if self.lower else example
        tokens = self.tokenizer(example)

        # Add extra tokens
        if self.sos is not None:
            tokens = [self.sos] + list(tokens)
        if self.eos is not None:
            tokens = list(tokens) + [self.eos]

        # Numericalize
        numericals = []
        for token in tokens:
            if token not in self.vocabulary:
                if self.unk is None or self.unk not in self.vocabulary:
                    raise ValueError("Encounterd out-of-vocabulary token \
                                      but the unk_token is either missing \
                                      or not defined in the vocabulary.")
                else:
                    token = self.unk

            numerical = self.vocabulary[token]  # type: ignore
            numericals.append(numerical)

        processed = torch.LongTensor(numericals)
        return processed

    def _build_reverse_vocab(self) -> None:
        """Builds reverse vocabulary."""
        self._reverse_vocab = {index: token for token, index in self.vocabulary.items()}

    def deprocess(self, indices: torch.LongTensor) -> str:
        """Converts indices to string."""
        pad_index = self.vocabulary[self.pad]
        return ' '.join(self._reverse_vocab[index.item()] for index in indices if index != pad_index)

    def pad_index(self) -> int:
        return self.vocabulary[self.pad]
