import os
import wandb
import torch

from datasets import copy_first_input, denoising
from datasets import sequential_mnist, permuted_mnist, row_mnist

from models_pytorch.models import Model
from models_pytorch.utils import set_seed
from argparse import ArgumentParser


CELLS = ('lstm', 'chrono', 'gru', 'mgu', 'brc', 'nbrc')
BENCHMARKS = ('copy', 'denoising', 'seq_mnist', 'per_mnist', 'row_mnist')


def main(config):

    torch.set_num_threads(4)

    # Set random seed for reproducibility
    args.seed = set_seed(args.seed)

    # Initialize wandb
    tags = [args.wandb_tag] if args.wandb_tag else None
    run = wandb.init(project="warmup-supervised", config=config, tags=tags)

    # Dataset
    if args.benchmark == 'copy':
        train = copy_first_input(
            args.train_samples, args.seq_length, batch_first=False)
        test = copy_first_input(
            args.test_samples, args.seq_length, batch_first=False)
    elif args.benchmark == 'denoising':
        train = denoising(args.train_samples, args.seq_length, args.pad_length)
        test = denoising(args.test_samples, args.seq_length, args.pad_length)
    elif args.benchmark == 'seq_mnist':
        train = sequential_mnist(train=True, black_pixels=args.black_pixels)
        test = sequential_mnist(train=False, black_pixels=args.black_pixels)
    elif args.benchmark == 'per_mnist':
        train = permuted_mnist(train=True, black_pixels=args.black_pixels)
        test = permuted_mnist(train=False, black_pixels=args.black_pixels)
    elif args.benchmark == 'row_mnist':
        train = row_mnist(train=True, black_pixels=args.black_pixels)
        test = row_mnist(train=False, black_pixels=args.black_pixels)
    else:
        raise NotImplementedError

    if args.model_selection:
        train, valid2 = train.train_test_split(test_size=10_000)

    # Model
    model = Model(
        cell=args.cell,
        input_size=train.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=train.output_size,
        regression=train.regression,
        t_max=args.t_max,
        double=args.double,
    )

    # Warmup
    if args.warmup:
        model.warmup(
            inputs=train.inputs,
            num_steps=args.warmup_num_steps,
            learning_rate=args.warmup_learning_rate,
            batch_size=args.warmup_batch_size,
            num_estimates_vaa=args.num_estimates_vaa,
            stabilization_increment=args.stabilization_increment,
            stabilization_max=args.stabilization_max,
            stabilization_measure=args.stabilization_measure,
            epsilon=args.vaa_epsilon,
            vaa_target=args.vaa_target,
            double=args.double,
        )

    # Train
    model.train(
        dataset=train,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        logger=wandb.log,
        num_estimates_vaa=args.num_estimates_vaa,
        stabilization_measure=args.stabilization_measure,
        epsilon=args.vaa_epsilon,
    )

    # Evaluate
    test_loss = model.eval(
        dataset=test,
        batch_size=args.batch_size)
    wandb.summary['test_loss'] = test_loss

    if args.model_selection:
        valid2_loss = model.eval(
            dataset=valid2,
            batch_size=args.batch_size)
        wandb.summary['valid2_loss'] = valid2_loss

    if not test.regression:
        test_accuracy = model.eval(
            dataset=test,
            batch_size=args.batch_size,
            accuracy=True)
        wandb.summary['test_accuracy'] = test_accuracy

        if args.model_selection:
            valid2_accuracy = model.eval(
                dataset=valid2,
                batch_size=args.batch_size,
                accuracy=True)
            wandb.summary['valid2_accuracy'] = valid2_accuracy

    # Save
    os.makedirs("weights", exist_ok=True)
    model.save(f"weights/{run.id}.pth")


if __name__ == '__main__':

    parser_sl = ArgumentParser(description='Launch SL experiment')
    parser_sl.add_argument("--seed", type=int, default=None)

    # Architecture
    parser_sl.add_argument('--cell', type=str, choices=CELLS, default='gru')
    parser_sl.add_argument('--t-max', type=int, default=600)
    parser_sl.add_argument('--hidden-size', type=int, default=256)
    parser_sl.add_argument('--num-layers', type=int, default=2)

    # Training
    parser_sl.add_argument('--num-epochs', type=int, default=50)
    parser_sl.add_argument('--batch-size', type=int, default=32)
    parser_sl.add_argument('--learning-rate', type=float, default=1e-3)
    parser_sl.add_argument('--num-estimates-vaa', type=int, default=10)
    parser_sl.add_argument('--stabilization-measure', type=int, default=10_000)

    # Warmup
    parser_sl.add_argument('--warmup', action='store_true')
    parser_sl.add_argument('--double', action='store_true')
    parser_sl.add_argument('--warmup-num-steps', type=int, default=100)
    parser_sl.add_argument('--warmup-learning-rate', type=float, default=1e-2)
    parser_sl.add_argument('--warmup-batch-size', type=int, default=32)
    parser_sl.add_argument('--stabilization-increment', type=int, default=10)
    parser_sl.add_argument('--stabilization-max', type=int, default=200)
    parser_sl.add_argument('--vaa-epsilon', type=float, default=1e-4)
    parser_sl.add_argument('--vaa-target', type=float, default=0.95)

    # Dataset
    parser_sl.add_argument('--benchmark', choices=BENCHMARKS, default='copy')
    parser_sl.add_argument('--train-samples', type=int, default=60_000)
    parser_sl.add_argument('--test-samples', type=int, default=10_000)

    parser_sl.add_argument('--seq-length', type=int, default=100)
    parser_sl.add_argument('--pad-length', type=int, default=5)
    parser_sl.add_argument('--black-pixels', type=int, default=5)

    # Model selection
    parser_sl.add_argument('--model-selection', action='store_true')

    # Wandb
    parser_sl.add_argument('--wandb-tag', type=str, default=None)

    # Parser arguments
    args = parser_sl.parse_args()
    print('\n'.join(f'\033[90m{k}=\033[0m{v}' for k, v in vars(args).items()))

    main(config=args)
