import torch
from transformers import *
from typing import List
from torch.nn.utils.rnn import pad_sequence

class BTTokenizer:    
    """
    Preprocessor that splits text on whitespace into tokens.
    Example:
        >>> preprocessor = SplitPreprocessor()
        >>> preprocessor.process('Hi how may I help you?')
        ['Hi', 'how', 'may', 'I', 'help', you?']
    """
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_type) 
        print('finish loading bert tokenizer')
        self.add_special_tokens = True
        self.max_len = args.max_sentence_length
        # self.pad_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.pad_token = self.tokenizer.pad_token

    def pad_index(self) -> int:
        """Get the padding index.

        Returns
        -------
        int
            The padding index in the vocabulary

        """
        # pad_token = tokenizer._convert_token_to_id(tokenizer.pad_token)
        pad_token = self.tokenizer.pad_token
        return self.tokenizer.convert_tokens_to_ids(pad_token)

    def process(self, text: str) -> List[str]:
        """Split text on whitespace into tokens."""
        tokens = self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens, max_length=self.max_len)
        if len(tokens) > self.max_len:
            print(len(tokens))
        processed = torch.LongTensor(tokens)
        return processed

    def deprocess(self, idx) -> List[str]:
        """Split text on whitespace into tokens."""
        text = self.tokenizer.decode(idx.numpy(),skip_special_tokens=True) 
        # print(text)
        text = text.replace('[CLS]','').replace('[SEP]', '').replace(self.pad_token,'')
        text = text.strip()
        # print(text)
        return text


class BertBatcher():
    def __init__(self, cuda, pad):
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.pad = pad


    def embed(self, arr, dtype=torch.long):
        pad_token = self.pad
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
        return padded, lens #, mask
