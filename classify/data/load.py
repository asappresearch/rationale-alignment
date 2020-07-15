from typing import Tuple

import torch

# from classify.data.loaders import (
#     AskUbuntuDataLoader,
#     MultiNewsDataLoader,
#     SummaryDataLoader,
# )
# from classify.data.loaders import PubmedDataLoader, PubmedSummaryDataLoader
from classify.data.loaders.snli import SNLIDataLoader
from classify.data.loaders.multirc import MultircDataLoader

from classify.data.text import TextField
from utils.parsing import Arguments
from classify.data.sampler import Sampler


def load_data(
    args: Arguments, device: torch.device
) -> Tuple[TextField, Sampler, Sampler, Sampler]:
    """Loads data and returns a TextField and train, dev, and test Samplers."""
    # Default to sampling negatives
    resample_negatives = True

    print("initializing dataloader")
    # Get DataLoader
    if args.dataset == "snli":
        from classify.data.snli_sampler import SNLISampler as Sampler

        data_loader = SNLIDataLoader(args)
    # elif args.dataset == "multirc" and args.word_to_word:
    #     from classify.data.multirc_word_sampler import MultircWSampler as Sampler

    #     data_loader = MultircDataLoader(args)
    elif args.dataset == "multirc":
        from classify.data.multirc_sent_sampler import MultircSentSampler as Sampler

        data_loader = MultircDataLoader(args)
    else:
        raise ValueError(f'Dataset "{args.dataset}" not supported')

    print("initializing sampler")
    # Create Samplers
    train_sampler = Sampler(
        data=data_loader.train,
        text_field=data_loader.text_field,
        batch_size=args.batch_size,
        shuffle=True,
        num_positives=args.num_positives,
        num_negatives=args.num_negatives,
        resample_negatives=resample_negatives,
        device=device,
    )

    dev_sampler = Sampler(
        data=data_loader.dev,
        text_field=data_loader.text_field,
        batch_size=args.batch_size,
        num_positives=args.num_eval_positives,
        num_negatives=args.num_eval_negatives,
        device=device,
    )

    test_sampler = Sampler(
        data=data_loader.test,
        text_field=data_loader.text_field,
        batch_size=args.batch_size,
        num_positives=args.num_eval_positives,
        num_negatives=args.num_eval_negatives,
        device=device,
    )

    return data_loader.text_field, train_sampler, dev_sampler, test_sampler
