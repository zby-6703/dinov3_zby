"""DraftFormer training entry using the OpenOCR-style config/trainer system."""

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from shipdraft_mtl.engine.config import ArgsParser, Config
from shipdraft_mtl.engine.trainer import Trainer


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--eval",
        action="store_true",
        default=True,
        help="Whether to perform evaluation during training.",
    )
    args = parser.parse_args()
    return args


def main():
    flags = parse_args()
    cfg = Config(flags.config)
    flags = vars(flags)
    opt = flags.pop("opt")
    cfg.merge_dict(flags)
    cfg.merge_dict(opt)
    trainer = Trainer(
        cfg,
        mode="train_eval" if flags["eval"] else "train",
        task="multitask",
    )
    trainer.train()


if __name__ == "__main__":
    main()
