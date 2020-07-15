import random
from typing import Any, List, Optional, Tuple



sentence_tokenizer = None


def split_data(data: List[Any],
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Randomly splits data into train, val, and test sets according to the provided sizes.

    :param data: The data to split into train, val, and test.
    :param sizes: The sizes of the train, val, and test sets (as a proportion of total size).
    :param seed: Random seed.
    :return: Train, val, and test sets.
    """
    # Checks
    assert len(sizes) == 3
    assert all(0 <= size <= 1 for size in sizes)
    assert sum(sizes) == 1

    # Shuffle
    random.seed(seed)
    random.shuffle(data)

    # Determine split sizes
    train_size = int(sizes[0] * len(data))
    train_val_size = int((sizes[0] + sizes[1]) * len(data))

    # Split
    train = data[:train_size]
    val = data[train_size:train_val_size]
    test = data[train_val_size:]

    return train, val, test


def tokenize_sentence(text: str) -> List[str]:
    """
    Tokenizes text into sentences.

    :param text: A string.
    :return: A list of sentences.
    """
    global sentence_tokenizer

    if sentence_tokenizer is None:
        import nltk
        sentence_tokenizer = nltk.load('tokenizers/punkt/english.pickle')

    return sentence_tokenizer.tokenize(text)


def text_to_sentences(text: str,
                      tokenizer: str='sentence', 
                      sentence_tokenize: bool = True,
                      max_num_sentences: Optional[int] = None,
                      max_sentence_length: Optional[int] = None) -> List[str]:
    """
    Splits text into sentences (if desired).

    Also enforces a maximum sentence length
    and maximum number of sentences.

    :param text: The text to split.
    :param sentence_tokenize: Whether to split into sentences.
    :param max_num_sentences: Maximum number of sentences.
    :param max_sentence_length: Maximum length of a sentence (in tokens).
    :return: The text split into sentences (if desired)
    or as just a single sentence.
    """
    # Sentence tokenize
    if sentence_tokenize:
        sentences = tokenize_sentence(text)[:max_num_sentences]
    else:
        sentences = [text]

    # Enforce maximum sentence length
    sentences = [' '.join(sentence.split()[:max_sentence_length]) for sentence in sentences]

    return sentences

def pubmed_tokenizer(text: str,
                    tokenizer: str='sentence', 
                    # predictor: Predictor=None, 
                    max_num_sentences: Optional[int] = None,
                    max_sentence_length: Optional[int] = None) -> List[str]:
    pass
    
'''
def pubmed_tokenizer(text: str,
                    tokenizer: str='sentence', 
                    predictor: Predictor=None, 
                    max_num_sentences: Optional[int] = None,
                    max_sentence_length: Optional[int] = None) -> List[str]:
    """
    # from allennlp.predictors import Predictor
    Splits text into sentences (if desired).

    Also enforces a maximum sentence length
    and maximum number of sentences.

    :param text: The text to split.
    :param sentence_tokenize: Whether to split into sentences.
    :param max_num_sentences: Maximum number of sentences.
    :param max_sentence_length: Maximum length of a sentence (in tokens).
    :return: The text split into sentences (if desired)
    or as just a single sentence.
    """
    # Sentence tokenize
    if tokenizer =='sentence':
        sentences = tokenize_sentence(text)[:max_num_sentences]
    elif tokenizer == 'word':
        sentences = [text]
    elif tokenizer == 'phrase':
        print('tokenizing phrase')
        full_sentences = tokenize_sentence(text)[:max_num_sentences]
        sentences = []
        for sent in full_sentences:
            sentences.extend(phrase_tokenizer(sent, predictor, phrase_len=5))
        sentences = sentences[:max_num_sentences*3]
    else: 
        print('unknow tokenizer')
    # Enforce maximum sentence length
    sentences = [' '.join(sentence.split()[:max_sentence_length]) for sentence in sentences]

    return sentences
'''


def process_pubmed_sentences(text: List[List[str]],
                      sentence_tokenize: bool = True,
                      max_num_sentences: Optional[int] = None,
                      max_sentence_length: Optional[int] = None) -> List[str]:
    """
    Splits text into sentences (if desired).
    Also enforces a maximum sentence length
    and maximum number of sentences.
    :param text: The text to split.
    :param sentence_tokenize: Whether to split into sentences.
    :param max_num_sentences: Maximum number of sentences.
    :param max_sentence_length: Maximum length of a sentence (in tokens).
    :return: The text split into sentences (if desired)
    or as just a single sentence.
    """
    # Sentence tokenize
    if sentence_tokenize:
        sentences = text[:max_num_sentences]
    else:
        sentences = [word for sent in text for word in sent]

    # Enforce maximum sentence length
    sentences = [' '.join(sentence[:max_sentence_length]) for sentence in sentences]

    return sentences