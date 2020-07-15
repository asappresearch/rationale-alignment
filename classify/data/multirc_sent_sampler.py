import random
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from classify.data.text import TextField
from classify.data.dataset import Dataset

import numpy as np


class MultircSentSampler:
    def __init__(
        self,
        data: Dataset,
        text_field: TextField,
        batch_size: int,
        shuffle: bool = False,
        num_positives: Optional[int] = None,
        num_negatives: Optional[int] = None,
        resample_negatives: bool = False,
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Constructs a SimilarityDataSampler.

        :param data: A Dataset.
        :param text_field: The TextField object initialized with all text data.
        :param batch_size: Batch size.
        :param shuffle: Whether to shuffle the data.
        :param num_positives: Number of positives per example. Defaults to all of them.
        :param num_negatives: Number of negatives per example. Defaults to all of them.
        :param resample_negatives: Whether to resample negatives after each epoch.
        :param seed: Initial random seed.
        :param device: The torch device to broadcast to.
        """
        self.data = data
        self.text_field = text_field
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.resample_negatives = resample_negatives
        self.seed = seed
        self.device = device
        self.id_to_sentlength = data.id_to_sentlength

        # print('initialize special sampler for multirc')

        # self.pad_index = self.text_field.vocabulary[self.text_field.pad]
        self.pad_index = self.text_field.pad_index()

    def sample(
        self,
    ) -> Iterator[
        Tuple[
            torch.LongTensor,
            List[Tuple[torch.LongTensor, torch.LongTensor]],
            List[Dict[str, torch.LongTensor]],
        ]
    ]:
        """
        Samples pairs of similar/dissimilar documents.

        :return: A tuple consisting of:
        1) batch_sentences: A tensor with all the sentences that need to be encoded (num_sentences x sentence_length).
        2) batch_scope: A list of tuples of tensors indicating the indices in batch_sentences
        corresponding to each of the two documents being compared.
        3) batch_targets: A dictionary mapping to the binary targets for each document pair
        and mapping to the indices of all pairs, positive pairs, and negative pairs.
        """
        # Seed
        self.seed += 1
        random.seed(self.seed)

        # Shuffle
        if self.shuffle:
            random.shuffle(self.data.id_list)

        # Iterate through batcches of data
        for i in trange(0, len(self.data), self.batch_size):
            # Get batch ids
            batch_document_ids = self.data.id_list[i : i + self.batch_size]

            # Initialize batch variables
            sentence_index = label_scope = 0
            batch_sentences, batch_scope, batch_targets = [], [], []
            scope, positives, negatives, targets = [], [], [], []
            id_to_scope = {}
            # id_to_lengths = {}

            # print(f'\n the {i} samples')
            # if i == 250:
            #     import pdb; pdb.set_trace()

            for query_id in batch_document_ids:
                docid = query_id.split(":")[0]

                id_to_scope[query_id] = []
                for sentence in self.data.id_to_document[query_id]:
                    batch_sentences.append(sentence)
                    id_to_scope[query_id].append(sentence_index)
                    sentence_index += 1
                if docid not in id_to_scope:
                    id_to_scope[docid] = []
                    # id_to_lengths[docid] = []
                    for sentence in self.data.id_to_document[docid]:
                        batch_sentences.append(sentence)
                        # id_to_lengths[docid].append(len(sentence)) ## Need to change how this is computed.
                        id_to_scope[docid].append(sentence_index)
                        sentence_index += 1

                batch_scope.append(
                    (
                        torch.LongTensor(id_to_scope[query_id]).to(self.device),
                        torch.LongTensor(id_to_scope[docid]).to(self.device),
                    )
                )

                row_r = np.zeros(len(self.data.id_to_document[query_id]))
                # for s,e in self.data.evidence[id+'_premise']:
                #     row_r[s:e] = 1
                column_r = np.zeros(len(self.data.id_to_document[docid]))
                for s, e in self.data.evidence["sentence"][query_id]:
                    column_r[s:e] = 1

                # if self.data.label_map[query_id]==0:
                #     positives.append(label_scope)
                # else:
                #     negatives.append(label_scope)

                lengths = self.id_to_sentlength[docid]
                w_column_r = np.zeros(sum(lengths))
                for s, e in self.data.evidence["token"][query_id]:
                    w_column_r[s:e] = 1

                new_column_r = rationale_sent_to_token(lengths, column_r)
                assert np.array_equal(w_column_r, new_column_r)

                batch_targets.append(
                    {
                        "annotationid": query_id,
                        "docid": docid,
                        "lengths": self.id_to_sentlength[docid],
                        "scope": torch.LongTensor([label_scope]).to(self.device),
                        "row_evidence": torch.LongTensor(row_r).to(self.device),
                        "column_evidence": torch.LongTensor(column_r).to(self.device),
                        # 'positives': torch.LongTensor(positives).to(self.device),
                        # 'negatives': torch.LongTensor(negatives).to(self.device),
                        "targets": torch.LongTensor([self.data.label_map[query_id]]).to(
                            self.device
                        ),
                    }
                )
                label_scope += 1
                # batch_targets.append(self.data.label_map[id])

            # batch_targets = torch.LongTensor(batch_targets).to(self.device)

            # Pad sentences
            batch_sentences = pad_sequence(
                batch_sentences, batch_first=True, padding_value=self.pad_index
            )
            # Convert sentences to tensors
            batch_sentences = torch.LongTensor(batch_sentences).to(self.device)

            assert len(batch_scope) == len(batch_targets)
            # assert len(batch_sentences) ==  2*len(batch_targets)

            yield batch_sentences, batch_scope, batch_targets

    def __len__(self) -> int:
        """Return the number of batches in the sampler."""
        return len(self.data) // self.batch_size

    def __call__(
        self,
    ) -> Iterator[
        Tuple[
            torch.LongTensor,
            List[Tuple[torch.LongTensor, torch.LongTensor]],
            List[Dict[str, torch.LongTensor]],
        ]
    ]:
        return self.sample()


def rationale_sent_to_token(lengths, rationales):
    total_l = sum(lengths)
    r_tk = np.zeros(total_l)
    for i, s in enumerate(list(rationales)):
        if s != 0:
            r_tk[sum(lengths[:i]) : sum(lengths[: i + 1])] = 1
    return r_tk
