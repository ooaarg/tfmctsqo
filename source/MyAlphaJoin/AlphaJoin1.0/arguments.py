import argparse


def get_args():
    """
    CLI arguments for AlphaJoin scripts (a wrapper around the old 0.arguments.py).
    """
    parse = argparse.ArgumentParser(description="AlphaJoin training")
    parse.add_argument(
        "--env-name",
        type=str,
        default="postgresql",
        help="training environment name (unused placeholder)",
    )
    parse.add_argument(
        "--save-dir",
        type=str,
        default="saved_models/",
        help="directory to store trained models",
    )
    # Additional parameters used by the patched supervised/pretreatment code:
    parse.add_argument(
        "--data-file",
        type=str,
        default="./data/training.csv",
        help="training data file: queryName,hint,runtime(,...)",
    )
    parse.add_argument(
        "--timeout-value",
        type=float,
        default=1e9,
        help='value to use when runtime is "timeout"',
    )
    parse.add_argument(
        "--train-steps",
        type=int,
        default=300000,
        help="number of SGD steps for supervised training",
    )
    parse.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate",
    )
    parse.add_argument(
        "--test-every",
        type=int,
        default=1000,
        help="run test_network() every N steps",
    )
    parse.add_argument(
        "--save-every",
        type=int,
        default=200000,
        help="save model every N steps",
    )

    return parse.parse_args()
