from typing import Tuple

import torch

from similarity.data.loaders import (
    AskUbuntuDataLoader,
    MultiNewsDataLoader,
)
from similarity.data.sampler import Sampler
from similarity.data.text import TextField
from rationale_alignment.parsing import Arguments


def load_data(
    args: Arguments, device: torch.device
) -> Tuple[TextField, Sampler, Sampler, Sampler]:
    """Loads data and returns a TextField and train, dev, and test Samplers."""
    # Default to sampling negatives
    resample_negatives = True

    # Get DataLoader
    if args.dataset in ["askubuntu", "superuser_askubuntu"]:
        data_loader = AskUbuntuDataLoader(args)
        assert (args.dev_path is None) == (args.test_path is None)
        resample_negatives = args.dev_path is None
    elif args.dataset == "summary":
        data_loader = SummaryDataLoader(args)
    elif args.dataset == "multinews":
        data_loader = MultiNewsDataLoader(args)
    elif args.dataset == "pubmed":
        data_loader = PubmedDataLoader(args)
    elif args.dataset == "pubmedsummary":
        data_loader = PubmedSummaryDataLoader(args)
    else:
        raise ValueError(f'Dataset "{args.dataset}" not supported')

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
