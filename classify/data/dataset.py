from typing import Dict, List, Set

import torch


class Dataset:
    def __init__(self,
                 ids: Set[str],
                 id_to_document: Dict[str, List[torch.LongTensor]],
                 id_mapping: Dict[str, Dict[str, Set[str]]],
                 negative_ids: Set[id] = None,
                 label_map: Dict[str, int] = None,
                 evidence: Dict[str,list] = None,
                 id_to_sentlength: Dict[str,list] = None):
        """
        Holds an AskUbuntu alignment dataset.

        :param ids: A set of ids from which to sample during training.
        Note: May not contain all ids since some ids should not be sampled.
        :param id_to_document: A dictionary mapping ids to a dictionary
        which maps "sentences" to the sentences in the document.
        :param id_mapping: A dictionary mapping ids to a dictionary which maps
        "similar" to similar ids and "dissimilar" to dissimilar ids.
        :param negative_ids: The set of ids which can be sampled as negatives.
        If None, any id can be sampled as a negative.
        :param id_to_sentlength: save the length of sentences in document. Only used in multiRC with bert model.
        """
        self.id_set = ids
        self.id_list = sorted(self.id_set)
        self.id_to_document = id_to_document
        self.id_mapping = id_mapping
        self.negative_ids = negative_ids or self.id_set
        self.label_map = label_map
        self.evidence = evidence
        self.id_to_sentlength = id_to_sentlength

    def __len__(self) -> int:
        return len(self.id_set)
