import os
import pickle

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


from utils.parsing import Arguments, parse_args
from utils.utils import load_weights, makedirs, make_schedular, NoamLR


def train(args: Arguments) -> None:
    """Trains an AlignmentModel to align sets of sentences."""

    if args.task == "classify":
        from classify.compute import AlignmentTrainer
        from classify.data import load_data
        from classify.metric import load_loss_and_metrics

        if args.word_to_word:
            from classify.models.ot_atten import AlignmentModel
        else:
            from classify.models.ot_atten_sent import AlignmentModel

    elif args.task == "similarity":
        from similarity.compute import AlignmentTrainer
        from similarity.data import load_data
        from similarity.metric import load_loss_and_metrics
        from similarity.models import AlignmentModel

    # Determine device
    device = (
        torch.device(args.gpu) if torch.cuda.is_available() else torch.device("cpu")
    )
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Loading data")
    text_field, train_sampler, dev_sampler, test_sampler = load_data(args, device)

    print("Building model")
    model = AlignmentModel(
        args=args, text_field=text_field, domain=args.dataset, device=device,
    )

    saved_step = 0
    if args.checkpoint_path is not None:
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        saved_step = 1 + int(args.checkpoint_path.split("_")[-1].replace(".pt", ""))
        print(f"trainig from step {saved_step}")
        load_weights(model, args.checkpoint_path)

    print(model)
    print(f"Number of parameters = {model.num_parameters(trainable=True):,}")

    print(f"Moving model to device: {device}")
    model.to(device)

    print("Defining loss and metrics")
    (
        loss_fn,
        metric_fn,
        extra_training_metrics,
        extra_validation_metrics,
    ) = load_loss_and_metrics(args)

    print("Creating optimizer and scheduler")
    if args.bert:
        # Prepare optimizer and schedule (linear warmup and decay)
        from transformers import AdamW
        from transformers import get_linear_schedule_with_warmup

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        # optimizer_grouped_parameters = get_params(model)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        # num_batch_per_epoch = min(train_data.num_batches, args.max_batches_per_epoch)
        num_batch_per_epoch = len(train_sampler)
        t_total = int(
            num_batch_per_epoch // args.gradient_accumulation_steps * args.epochs
        )
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(t_total*0.06), t_total=t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(t_total * args.warmup_ratio),
            num_training_steps=t_total,
        )
    else:
        optimizer = Adam(
            model.trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = make_schedular(
            args, optimizer, model.output_size, last_epoch=saved_step - 1
        )

    print("Building Trainer")
    trainer = AlignmentTrainer(
        args=args,
        train_sampler=train_sampler,
        dev_sampler=dev_sampler,
        test_sampler=test_sampler,
        model=model,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        extra_training_metrics=extra_training_metrics,
        extra_validation_metrics=extra_validation_metrics,
        log_dir=args.log_dir,
        log_frequency=args.log_frequency,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        sparsity_thresholds=args.sparsity_thresholds,
        saved_step=saved_step,
    )

    if args.epochs > 0:
        print("Training")
        while not trainer.step():
            pass

    if args.preds_dir is not None or args.viz_dir is not None:
        print("Predicting")
        sentences, preds, targets = trainer.predict(num_predict=args.num_predict)

        # Extract targets
        targets = [target["targets"] for target in targets]
        targets = [t.item() for target in targets for t in target]

        # Convert indices back to tokens
        sentences = [
            (
                [text_field.deprocess(sentence) for sentence in doc_1],
                [text_field.deprocess(sentence) for sentence in doc_2],
            )
            for doc_1, doc_2 in sentences
        ]

        # Save predictions
        if args.preds_dir is not None:
            makedirs(args.preds_dir)
            preds_path = os.path.join(args.preds_dir, "preds.pkl")
            with open(preds_path, "wb") as f:
                sentences, preds, targets = sentences, preds, targets
                pickle.dump((sentences, preds, targets), f)

    elif args.epochs == 0:
        print("Evaluating")
        trainer.eval_step()


if __name__ == "__main__":
    import sys

    from utils.utils import Logger

    # Parse args
    args = parse_args()

    # Set up logging to console and file
    sys.stdout = Logger(
        pipe=sys.stdout, log_path=os.path.join(args.log_dir, "stdout.txt")
    )
    sys.stderr = Logger(
        pipe=sys.stderr, log_path=os.path.join(args.log_dir, "stderr.txt")
    )

    # Print and save args
    print(args)
    args.save(os.path.join(args.log_dir, "args.json"))

    # Train
    train(args)
