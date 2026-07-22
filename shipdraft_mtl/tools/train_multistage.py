r"""Run DraftFormer multi-stage training end-to-end (stage1 -> stage2 -> stage3).

Usage (from project root ``dinov3_zby``)::

    # Full pipeline (uses stage configs as-is)
    python shipdraft_mtl/tools/train_multistage.py

    # Smoke test: 1 epoch per stage
    python shipdraft_mtl/tools/train_multistage.py --smoke-test

    # Only print commands
    python shipdraft_mtl/tools/train_multistage.py --dry-run

    # Resume from stage 2 (assumes stage1 already finished)
    python shipdraft_mtl/tools/train_multistage.py --start-stage 2

    # Extra config overrides applied to every stage
    python shipdraft_mtl/tools/train_multistage.py -o Global.use_amp=False
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_NET = PROJECT_ROOT / "shipdraft_mtl" / "train_net.py"

STAGES = [
    {
        "id": 1,
        "name": "stage1_keypoints",
        "config": PROJECT_ROOT / "shipdraft_mtl" / "configs" / "stage1_keypoints.yml",
        "output_dir": PROJECT_ROOT / "outputs" / "shipdraft_mtl" / "stage1_keypoints",
        # stage1 trains from scratch / backbone pretrained weights only
        "needs_previous": False,
    },
    {
        "id": 2,
        "name": "stage2_depth",
        "config": PROJECT_ROOT / "shipdraft_mtl" / "configs" / "stage2_depth.yml",
        "output_dir": PROJECT_ROOT / "outputs" / "shipdraft_mtl" / "stage2_depth",
        "needs_previous": True,
        "previous_output_dir": PROJECT_ROOT / "outputs" / "shipdraft_mtl" / "stage1_keypoints",
    },
    {
        "id": 3,
        "name": "stage3_joint",
        "config": PROJECT_ROOT / "shipdraft_mtl" / "configs" / "stage3_joint.yml",
        "output_dir": PROJECT_ROOT / "outputs" / "shipdraft_mtl" / "stage3_joint",
        "needs_previous": True,
        "previous_output_dir": PROJECT_ROOT / "outputs" / "shipdraft_mtl" / "stage2_depth",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch train_net.py",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="First stage to run (1/2/3)",
    )
    parser.add_argument(
        "--end-stage",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Last stage to run (inclusive)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not train",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Short run: Global.epoch_num=1 and more frequent eval for pipeline check",
    )
    parser.add_argument(
        "--prefer-latest",
        action="store_true",
        help="Prefer latest.pth over best.pth when wiring stage N-1 -> N",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If a stage fails, continue to the next stage (default: stop)",
    )
    parser.add_argument(
        "-o",
        "--opt",
        nargs="*",
        default=[],
        help="Extra config overrides passed to every stage (same as train_net -o)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable eval during training (passes through by not using default eval)",
    )
    return parser.parse_args()


def resolve_checkpoint(output_dir: Path, prefer_latest: bool = False) -> Path:
    """Pick a checkpoint under a stage output directory."""
    candidates = (
        ["latest.pth", "best.pth"] if prefer_latest else ["best.pth", "latest.pth"]
    )
    for name in candidates:
        path = output_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No checkpoint found in {output_dir} (looked for {', '.join(candidates)}). "
        "Finish the previous stage first, or pass --start-stage to skip completed ones."
    )


def path_for_cli(path: Path) -> str:
    """Use forward slashes so Windows ``\\b`` in best.pth never gets corrupted."""
    return path.resolve().as_posix()


def build_command(
    python_exe: str,
    stage: dict,
    pretrained: Optional[Path],
    extra_opts: Sequence[str],
    smoke_test: bool,
    no_eval: bool,
) -> List[str]:
    cmd = [
        python_exe,
        str(TRAIN_NET),
        "-c",
        path_for_cli(stage["config"]),
    ]
    opts: List[str] = list(extra_opts)
    # Always pin output_dir to the stage default (absolute) for reliable chaining.
    opts.append(f"Global.output_dir={path_for_cli(stage['output_dir'])}")
    if pretrained is not None:
        opts.append(f"Global.pretrained_model={path_for_cli(pretrained)}")
        opts.append("Global.checkpoints=")
    if smoke_test:
        # Tiny run: a few iters per stage to verify launch + checkpoint wiring.
        opts.extend(
            [
                "Global.epoch_num=1",
                "Global.max_iter=3",
                "Global.eval_epoch_step=[0,1]",
                "Global.save_epoch_step=[0,1]",
                "Global.print_batch_step=1",
                "LRScheduler.warmup_epoch=0",
                "LRScheduler.milestones=[1]",
                "Train.loader.num_workers=0",
                "Eval.loader.num_workers=0",
            ]
        )
    if opts:
        cmd.append("-o")
        cmd.extend(opts)
    # train_net defaults --eval True; keep unless disabled.
    # (No flag needed for eval=True.)
    if no_eval:
        cmd.append("--no-eval")
    return cmd


def run_stage(
    python_exe: str,
    stage: dict,
    pretrained: Optional[Path],
    extra_opts: Sequence[str],
    smoke_test: bool,
    dry_run: bool,
    no_eval: bool,
) -> int:
    stage["output_dir"].mkdir(parents=True, exist_ok=True)
    cmd = build_command(python_exe, stage, pretrained, extra_opts, smoke_test, no_eval)
    banner = f"========== {stage['name']} (stage {stage['id']}) =========="
    print("\n" + banner)
    print("Command:")
    print("  " + " ".join(cmd))
    if pretrained is not None:
        print(f"Init weights: {pretrained}")
    print(f"Output dir  : {stage['output_dir']}")
    print(banner + "\n", flush=True)

    if dry_run:
        print("[dry-run] skipped execution")
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    elapsed = time.time() - started
    print(
        f"\n[{stage['name']}] finished with code={proc.returncode} "
        f"in {elapsed / 60.0:.1f} min",
        flush=True,
    )
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    if args.start_stage > args.end_stage:
        print("error: --start-stage must be <= --end-stage", file=sys.stderr)
        return 2
    if not TRAIN_NET.is_file():
        print(f"error: train_net not found: {TRAIN_NET}", file=sys.stderr)
        return 2

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {args.python}")
    print(f"Stages       : {args.start_stage} -> {args.end_stage}")
    print(f"Smoke test   : {args.smoke_test}")
    print(f"Dry run      : {args.dry_run}")

    selected = [s for s in STAGES if args.start_stage <= s["id"] <= args.end_stage]
    results = []

    for stage in selected:
        pretrained = None
        if stage.get("needs_previous"):
            prev_dir = Path(stage["previous_output_dir"])
            try:
                pretrained = resolve_checkpoint(prev_dir, prefer_latest=args.prefer_latest)
            except FileNotFoundError as exc:
                print(f"error: {exc}", file=sys.stderr)
                if args.dry_run:
                    # Still show what would run, with a placeholder path.
                    pretrained = prev_dir / ("latest.pth" if args.prefer_latest else "best.pth")
                    print(f"[dry-run] would require: {pretrained}")
                else:
                    return 1

        code = run_stage(
            python_exe=args.python,
            stage=stage,
            pretrained=pretrained,
            extra_opts=args.opt or [],
            smoke_test=args.smoke_test,
            dry_run=args.dry_run,
            no_eval=args.no_eval,
        )
        results.append((stage["name"], code))
        if code != 0 and not args.continue_on_error:
            print(f"error: {stage['name']} failed (code={code}); stopping pipeline", file=sys.stderr)
            break

    print("\n========== multi-stage summary ==========")
    for name, code in results:
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"  {name}: {status}")
    print("=========================================\n")

    failed = [name for name, code in results if code != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
