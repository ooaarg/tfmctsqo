import time

from arguments import get_args
from supervised import supervised

if __name__ == "__main__":
    args = get_args()

    trainer = supervised(args)
    print("Pretreatment running...")
    start = time.perf_counter()
    # Use the path passed via CLI arguments
    trainer.pretreatment(args.data_file)
    elapsed = time.perf_counter() - start
    print("Pretreatment time used:", elapsed)
