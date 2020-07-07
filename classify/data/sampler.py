import random
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from classify.data.text import TextField
from classify.data.dataset import Dataset


class Sampler:
    def __init__(self,
                 data: Dataset,
                 text_field: TextField,
                 batch_size: int,
                 shuffle: bool = False,
                 num_positives: Optional[int] = None,
                 num_negatives: Optional[int] = None,
                 resample_negatives: bool = False,
                 seed: int = 0,
                 device: torch.device = torch.device('cpu')):
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

        # self.pad_index = self.text_field.vocabulary[self.text_field.pad]
        self.pad_index = self.text_field.pad_index()

    def sample_negatives(self,
                         id: str,
                         available_ids: Set[str]) -> Set[str]:
        """Samples negative ids for a document from a list of available of ids.

        :param id: The id of the document for which negatives should be sampled.
        :param available_ids: A list of ids available to sample from.
        :return: A (sorted) list of negative ids sampled from available ids.
        """
        # IDs that can't be negatives
        non_negative_ids = self.data.id_mapping[id]['similar'] | {id}
        valid_negative_ids = self.data.negative_ids - non_negative_ids

        # Determine negative ids
        negative_ids = available_ids & valid_negative_ids
        if len(negative_ids) == 0:
            negative_ids = available_ids
            print('zero size')
            print(non_negative_ids)
            print(valid_negative_ids)
            print(available_ids)

        # Add more negatives from outside of available_ids if necessary
        if len(negative_ids) < self.num_negatives:
            candidate_ids = [candidate_id for candidate_id in self.data.id_list if candidate_id in valid_negative_ids]
            random.shuffle(candidate_ids)

            for candidate_id in candidate_ids:
                negative_ids.add(candidate_id)

                if len(negative_ids) >= self.num_negatives:
                    break

        # Subsample negatives if too many
        if len(negative_ids) > self.num_negatives:
            negative_ids = sorted(negative_ids)
            random.shuffle(negative_ids)
            negative_ids = set(negative_ids[:self.num_negatives])

        return negative_ids

    def sample(self) -> Iterator[Tuple[torch.LongTensor,
                                       List[Tuple[torch.LongTensor, torch.LongTensor]],
                                       List[Dict[str, torch.LongTensor]]]]:
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

        # Iterate through batches of data
        for i in trange(0, len(self.data), self.batch_size):
            # Get batch ids
            batch_document_ids = self.data.id_list[i:i + self.batch_size]

            if len(batch_document_ids) <4:  
                continue

            # Get ids of all documents which will be encoded in this batch
            # (i.e. batch_document_ids plus all similar ids)
            batch_available_ids: Set[str] = set.union(
                set(batch_document_ids),
                *[self.data.id_mapping[document_id]['similar'] for document_id in batch_document_ids]
            )

            # Initialize batch variables
            sentence_index = scope_index = 0
            id_to_scope = {}
            batch_sentences, batch_scope, batch_targets = [], [], []

            # Loop through document ids and add sentences, scope, and targets
            for document_id in batch_document_ids:
                # Add scope and targets
                scope, positives, negatives, targets = [], [], [], []

                # Get similar and dissimilar ids
                similar_ids = self.data.id_mapping[document_id]['similar']
                dissimilar_ids = self.data.id_mapping[document_id]['dissimilar']

                # Sample dissimilar ids if necessary
                if self.resample_negatives or len(dissimilar_ids) == 0:
                    dissimilar_ids = self.sample_negatives(
                        id=document_id,
                        available_ids=batch_available_ids
                    )
                    self.data.id_mapping[document_id]['dissimilar'] = dissimilar_ids

                # Subsample positives if too many
                if self.num_positives is not None and len(similar_ids) > self.num_positives:
                    similar_ids = sorted(similar_ids)
                    random.shuffle(similar_ids)
                    similar_ids = set(similar_ids[:self.num_positives])

                # Subsample negatives if too many
                if self.num_negatives is not None and len(dissimilar_ids) > self.num_negatives:
                    dissimilar_ids = sorted(dissimilar_ids)
                    random.shuffle(dissimilar_ids)
                    dissimilar_ids = set(dissimilar_ids[:self.num_negatives])

                # Add all sentences related to this document
                related_ids = set.union({document_id}, similar_ids, dissimilar_ids)
                new_ids = related_ids - set(id_to_scope.keys())
                for id in new_ids:
                    # Initialize scope for id
                    id_to_scope[id] = []

                    # Add sentences
                    for sentence in self.data.id_to_document[id]:
                        batch_sentences.append(sentence)
                        id_to_scope[id].append(sentence_index)
                        # if len(id_to_scope[id]) > 1: 
                        #     print(f'sentence longer than 1. {id} ')
                        sentence_index += 1

                # Add similar document scope/target
                for similar_id in similar_ids:
                    batch_scope.append((torch.LongTensor(id_to_scope[document_id]).to(self.device),
                                        torch.LongTensor(id_to_scope[similar_id]).to(self.device)))
                    scope.append(scope_index)
                    positives.append(scope_index)
                    scope_index += 1
                    targets.append(1)

                # Add dissimilar document scope/target
                for dissimilar_id in dissimilar_ids:
                    batch_scope.append((torch.LongTensor(id_to_scope[document_id]).to(self.device),
                                        torch.LongTensor(id_to_scope[dissimilar_id]).to(self.device)))
                    scope.append(scope_index)
                    negatives.append(scope_index)
                    scope_index += 1
                    targets.append(0)

                batch_targets.append({
                    'scope': torch.LongTensor(scope).to(self.device),
                    'positives': torch.LongTensor(positives).to(self.device),
                    'negatives': torch.LongTensor(negatives).to(self.device),
                    'targets': torch.LongTensor(targets).to(self.device)
                })

            # Pad sentences
            batch_sentences = pad_sequence(batch_sentences, batch_first=True, padding_value=self.pad_index)

            # Convert sentences to tensors
            batch_sentences = torch.LongTensor(batch_sentences).to(self.device)

            yield batch_sentences, batch_scope, batch_targets

    def __len__(self) -> int:
        """Return the number of batches in the sampler."""
        return len(self.data) // self.batch_size

    def __call__(self) -> Iterator[Tuple[torch.LongTensor,
                                   List[Tuple[torch.LongTensor, torch.LongTensor]],
                                   List[Dict[str, torch.LongTensor]]]]:
        return self.sample()
