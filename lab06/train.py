"""Lab 6 Task 1/7: run the YAML-driven pipeline.

Examples (from lab06/):
  python train.py                                   # config.yaml as is
  python train.py --set train.lr=0.01 run_name=x    # override keys
  python train.py --set run_name=demo --stop-after 3   # simulate a crash after epoch 3
  python train.py --set run_name=demo --resume         # continue from checkpoints/demo/last.pt
"""
import argparse

from pipeline import load_config, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int, default=None)
    p.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE")
    args = p.parse_args()
    run(load_config(args.config, args.overrides), resume=args.resume, stop_after=args.stop_after)


if __name__ == "__main__":
    main()
