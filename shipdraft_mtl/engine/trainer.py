"""DraftFormer trainer following the ShipNameRecognition/OpenOCR control flow."""

from __future__ import annotations

import datetime
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from shipdraft_mtl.data import build_dataloader
from shipdraft_mtl.metrics import build_metric
from shipdraft_mtl.modeling import build_model
from shipdraft_mtl.optimizer import build_optimizer
from shipdraft_mtl.utils.ckpt import load_ckpt, save_ckpt
from shipdraft_mtl.utils.logging import get_logger
from shipdraft_mtl.utils.task_train import apply_task_train_policy, uses_partial_training

__all__ = ["Trainer"]


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class SmoothedStats:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.data = {}

    def update(self, values):
        for key, value in values.items():
            self.data.setdefault(key, []).append(float(value))
            if len(self.data[key]) > self.window_size:
                self.data[key].pop(0)

    def get(self):
        return {key: sum(values) / len(values) for key, values in self.data.items() if values}

    def log(self):
        return ", ".join(f"{key}: {value:.6f}" for key, value in self.get().items())


def _resolve_train_limits(global_cfg, step_each_epoch=1):
    raw_epoch_num = global_cfg.get("epoch_num")
    epoch_num = int(raw_epoch_num) if raw_epoch_num not in ("", None) else 0

    raw_max_iter = global_cfg.get("max_iter")
    max_iter = None if raw_max_iter in ("", None) else int(raw_max_iter)

    if epoch_num <= 0:
        if max_iter is None:
            raise ValueError("Global.epoch_num or Global.max_iter must be set to a positive value")
        step_each_epoch = max(1, int(step_each_epoch))
        epoch_num = max(1, math.ceil(max_iter / step_each_epoch))

    return epoch_num, max_iter


class Trainer:
    def __init__(self, cfg, mode="train", task="multitask"):
        self.cfg = cfg.cfg
        self.task = task
        self.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        self.set_device(self.cfg["Global"].get("device", "gpu"))
        mode = mode.lower()
        if mode not in ["train_eval", "train", "eval", "test"]:
            raise ValueError("mode should be train, train_eval, eval or test")

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1 and "train" in mode:
            if self.cfg["Global"].get("device", "gpu") != "gpu":
                raise RuntimeError("Distributed training currently requires Global.device=gpu")
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.device)
            self.cfg["Global"]["distributed"] = True
        else:
            self.cfg["Global"]["distributed"] = False
            self.local_rank = 0

        self.cfg["Global"]["output_dir"] = self.cfg["Global"].get(
            "output_dir", "output/shipdraft_mtl"
        )
        os.makedirs(self.cfg["Global"]["output_dir"], exist_ok=True)

        self.writer = None
        if is_main_process() and self.cfg["Global"].get("use_tensorboard", False) and "train" in mode:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.cfg["Global"]["output_dir"])

        self.logger = get_logger(
            "shipdraft_mtl",
            os.path.join(self.cfg["Global"]["output_dir"], "train.log")
            if "train" in mode and is_main_process()
            else None,
        )
        cfg.print_cfg(self.logger.info)
        self.set_random_seed(self.cfg["Global"].get("seed", 48))

        self.train_dataloader = None
        if "train" in mode:
            self.train_dataloader = build_dataloader(self.cfg, "Train", self.logger, task=task)
            if is_main_process():
                cfg.save(os.path.join(self.cfg["Global"]["output_dir"], "config.yml"), self.cfg)
            self.logger.info(f"train dataloader has {len(self.train_dataloader)} iters")

        self.valid_dataloader = None
        if "eval" in mode and self.cfg.get("Eval"):
            self.valid_dataloader = build_dataloader(self.cfg, "Eval", self.logger, task=task)
            self.logger.info(f"valid dataloader has {len(self.valid_dataloader)} iters")

        self.model = build_model(self.cfg).to(self.device)
        # Apply multi-stage TaskTrain freezes before optimizer construction.
        self.task_train = apply_task_train_policy(self.model, self.cfg, logger=self.logger)
        self.eval_class = build_metric(self.cfg.get("Metric", {}))
        self.logger.info(get_parameter_number(model=self.model))

        if self.cfg["Global"].get("use_sync_bn", False):
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.logger.info("convert_sync_batchnorm")

        self.accumulation_steps = int(self.cfg["Global"].get("accumulation_steps", 1))
        if self.accumulation_steps <= 0:
            raise ValueError("Global.accumulation_steps must be a positive integer")
        self.optimizer, self.lr_scheduler = None, None
        raw_steps_per_epoch = len(self.train_dataloader) if self.train_dataloader is not None else 1
        optimizer_steps_per_epoch = max(1, math.ceil(raw_steps_per_epoch / self.accumulation_steps))
        epochs, _ = _resolve_train_limits(self.cfg["Global"], step_each_epoch=raw_steps_per_epoch)
        if self.train_dataloader is not None:
            self.optimizer, self.lr_scheduler = build_optimizer(
                self.cfg["Optimizer"],
                self.cfg["LRScheduler"],
                epochs=epochs,
                step_each_epoch=optimizer_steps_per_epoch,
                model=self.model,
            )
        self.grad_clip_val = self.cfg["Global"].get("grad_clip_val", 0)
        self.status = load_ckpt(self.model, self.cfg, self.optimizer, self.lr_scheduler, logger=self.logger)
        # Re-apply freezes after checkpoint load.
        self.task_train = apply_task_train_policy(self.model, self.cfg, logger=self.logger)

        if self.cfg["Global"]["distributed"]:
            find_unused = uses_partial_training(self.cfg) or bool(
                self.cfg["Global"].get("find_unused_parameters", False)
            )
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, [self.local_rank], find_unused_parameters=find_unused
            )
            self.logger.info(f"DDP find_unused_parameters={find_unused}")

        self.scaler = torch.amp.GradScaler() if self.cfg["Global"].get("use_amp", False) else None
        self.logger.info(f"run with torch {torch.__version__} and device {self.device}")

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    def set_device(self, device):
        if device == "gpu" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        elif device == "gpu":
            raise RuntimeError(
                "Global.device is 'gpu', but CUDA is not available in this process. "
                "Check that the command has GPU access and the draftformer env can see CUDA."
            )
        else:
            self.device = torch.device("cpu")

    def train(self):
        step_each_epoch = len(self.train_dataloader) if self.train_dataloader is not None else 1
        epoch_num, max_iter = _resolve_train_limits(self.cfg["Global"], step_each_epoch=step_each_epoch)
        print_batch_step = self.cfg["Global"]["print_batch_step"]
        log_smooth_window = self.cfg["Global"].get("log_smooth_window", 20)
        eval_epoch_step = self.cfg["Global"].get("eval_epoch_step", 1)
        eval_batch_step = self.cfg["Global"].get("eval_batch_step", [10**12, 1])

        start_eval_epoch = 0
        if self.valid_dataloader is not None and isinstance(eval_epoch_step, list):
            start_eval_epoch, eval_epoch_step = eval_epoch_step[:2]
        elif self.valid_dataloader is None:
            start_eval_epoch = 10**12

        start_eval_step = 0
        if isinstance(eval_batch_step, list) and len(eval_batch_step) >= 2:
            start_eval_step, eval_batch_step = eval_batch_step[:2]

        save_epoch_step = self.cfg["Global"].get("save_epoch_step", [0, 1])
        start_save_epoch, save_epoch_interval = save_epoch_step[:2]
        save_iter_step = self.cfg["Global"].get("save_iter_step", [10**12, 2000])
        start_save_iter, save_iter_interval = save_iter_step[:2]

        global_step = self.status.get("global_step", 0)
        start_epoch = self.status.get("epoch", 1)
        self.best_metric = self.status.get("metrics", {})
        self.best_metric.setdefault(self.eval_class.main_indicator, 0.0)
        self.logger.info(
            "best-checkpoint criterion: main_indicator=%s (higher is better); "
            "evaluate_character=%s evaluate_waterline=%s evaluate_depth=%s",
            self.eval_class.main_indicator,
            getattr(self.eval_class, "evaluate_character", True),
            getattr(self.eval_class, "evaluate_waterline", True),
            getattr(self.eval_class, "evaluate_depth", True),
        )

        if max_iter is not None and global_step >= max_iter:
            self.logger.info(
                f"skip training because global_step {global_step} already reached max_iter {max_iter}"
            )
            self.logger.info(
                "best metric, "
                + ", ".join(f"{key}: {value}" for key, value in self.best_metric.items())
            )
            if self.writer is not None:
                self.writer.close()
            if dist.is_available() and dist.is_initialized():
                torch.distributed.barrier()
            return

        stats = SmoothedStats(log_smooth_window)
        eta_meter = AverageMeter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        should_stop = False
        for epoch in range(start_epoch, epoch_num + 1):
            if self.cfg["Global"].get("distributed", False) and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)

            reader_start = time.time()
            train_reader_cost = 0.0
            train_batch_cost = 0.0
            total_samples = 0

            for idx, batch in enumerate(self.train_dataloader):
                train_reader_cost += time.time() - reader_start
                loss_dict = self._forward_loss(batch)
                group_offset = idx % self.accumulation_steps
                group_start = idx - group_offset
                group_size = min(self.accumulation_steps, len(self.train_dataloader) - group_start)
                if max_iter is not None:
                    group_size = min(group_size, max_iter - (global_step - group_offset))
                group_size = max(1, group_size)
                loss = self._loss_for_backward(loss_dict) / group_size
                reaches_limit = max_iter is not None and global_step + 1 >= max_iter
                should_update = (
                    group_offset + 1 >= group_size
                    or idx + 1 >= len(self.train_dataloader)
                    or reaches_limit
                )
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if should_update:
                        if self.grad_clip_val > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_val)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if should_update:
                        if self.grad_clip_val > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_val)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if self.lr_scheduler is not None and should_update:
                    self.lr_scheduler.step()

                batch_time = time.time() - reader_start
                train_batch_cost += batch_time
                eta_meter.update(batch_time)
                total_samples += len(batch)

                scalar_losses = self._scalarize_loss_dict(loss_dict)
                scalar_losses["loss"] = self._loss_value(loss_dict)
                scalar_losses["lr"] = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 0.0
                stats.update(scalar_losses)

                if self.writer is not None:
                    for key, value in stats.get().items():
                        self.writer.add_scalar(f"TRAIN/{key}", value, global_step)

                if is_main_process() and (
                    (global_step > 0 and global_step % print_batch_step == 0)
                    or idx >= len(self.train_dataloader) - 1
                ):
                    eta_sec = ((epoch_num + 1 - epoch) * len(self.train_dataloader) - idx - 1) * eta_meter.avg
                    self.logger.info(
                        f"epoch: [{epoch}/{epoch_num}], global_step: {global_step}, {stats.log()}, "
                        f"avg_reader_cost: {train_reader_cost / max(1, print_batch_step):.5f} s, "
                        f"avg_batch_cost: {train_batch_cost / max(1, print_batch_step):.5f} s, "
                        f"avg_samples: {total_samples / max(1, print_batch_step):.2f}, "
                        f"ips: {total_samples / max(train_batch_cost, 1e-6):.5f} samples/s, "
                        f"eta: {str(datetime.timedelta(seconds=int(eta_sec)))}"
                    )
                    train_reader_cost = 0.0
                    train_batch_cost = 0.0
                    total_samples = 0

                if (
                    self.valid_dataloader is not None
                    and should_update
                    and global_step > start_eval_step
                    and (global_step - start_eval_step) % eval_batch_step == 0
                ):
                    self.eval_step(global_step, epoch)

                if is_main_process() and should_update and global_step > start_save_iter and global_step % save_iter_interval == 0:
                    save_ckpt(
                        self.model,
                        self.cfg,
                        self.optimizer,
                        self.lr_scheduler,
                        epoch,
                        global_step,
                        self.best_metric,
                        is_best=False,
                        prefix=f"iter_{global_step}",
                    )

                if max_iter is not None and global_step >= max_iter:
                    should_stop = True
                    break
                reader_start = time.time()

            if (
                self.valid_dataloader is not None
                and epoch > start_eval_epoch
                and (epoch - start_eval_epoch) % eval_epoch_step == 0
            ):
                self.eval_step(global_step, epoch)

            if is_main_process():
                save_ckpt(
                    self.model,
                    self.cfg,
                    self.optimizer,
                    self.lr_scheduler,
                    epoch,
                    global_step,
                    self.best_metric,
                    is_best=False,
                    prefix=None,
                )
                if epoch > start_save_epoch and (epoch - start_save_epoch) % save_epoch_interval == 0:
                    save_ckpt(
                        self.model,
                        self.cfg,
                        self.optimizer,
                        self.lr_scheduler,
                        epoch,
                        global_step,
                        self.best_metric,
                        is_best=False,
                        prefix="epoch_" + str(epoch),
                    )

            if should_stop:
                break

        self.logger.info(
            "best metric, "
            + ", ".join(f"{key}: {value}" for key, value in self.best_metric.items())
        )
        if self.writer is not None:
            self.writer.close()
        if dist.is_available() and dist.is_initialized():
            torch.distributed.barrier()

    def _forward_loss(self, batch):
        if self.scaler is not None:
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                return self.model(batch)
        return self.model(batch)

    def _loss_for_backward(self, loss_dict):
        if "loss" in loss_dict:
            return loss_dict["loss"]
        loss = None
        for key, value in loss_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.startswith("weight_") or key.startswith("sigma_") or key.startswith("stat_"):
                continue
            if loss is None:
                loss = value
            else:
                loss = loss + value
        if loss is None:
            raise ValueError("No differentiable loss found in loss_dict")
        return loss

    def _loss_value(self, loss_dict):
        return float(self._loss_for_backward(loss_dict).detach().float().mean().item())

    def _scalarize_loss_dict(self, loss_dict):
        scalar_losses = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                scalar_losses[key] = value.detach().float().mean().item()
        return scalar_losses

    def eval_step(self, global_step, epoch):
        cur_metric = self.eval()
        if not is_main_process():
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return
        # Prefer a compact, stage-relevant log line for selection.
        indicator = self.eval_class.main_indicator
        highlight_keys = [
            indicator,
            "selection_score",
            "keypoint_score",
            "char_score",
            "PCK",
            "PCK_class_aware",
            "point_f1",
            "waterline_score",
            "waterline_y_L1",
            "waterline_y_L1_median",
            "waterline_y_score",
            "waterline_y_median_score",
            "waterline_curve_L1",
            "waterline_curve_score",
            "waterline_curve_PCK",
            "MADDE",
            "draft_score",
            "hybrid_score",
            "fps",
        ]
        summary = []
        for key in highlight_keys:
            if key in cur_metric and isinstance(cur_metric[key], (float, int)):
                summary.append(f"{key}: {cur_metric[key]}")
        self.logger.info("cur metric, " + ", ".join(summary))
        # Full dump (skip bulky nested dicts).
        full = []
        for key, value in cur_metric.items():
            if isinstance(value, (float, int, str, bool)):
                full.append(f"{key}: {value}")
        self.logger.info("cur metric(full), " + ", ".join(full))

        if self.writer is not None:
            for key, value in cur_metric.items():
                if isinstance(value, (float, int)):
                    self.writer.add_scalar(f"EVAL/{key}", value, global_step)

        cur_score = float(cur_metric.get(indicator, cur_metric.get("selection_score", 0.0)) or 0.0)
        best_score = float(self.best_metric.get(indicator, 0.0) or 0.0)
        selection_valid = bool(cur_metric.get("selection_valid", True))
        if not selection_valid:
            self.logger.warning("skip best-checkpoint update: validation has no valid target samples")
        if selection_valid and cur_score >= best_score:
            self.best_metric = {
                k: v for k, v in cur_metric.items() if isinstance(v, (float, int, str, bool))
            }
            self.best_metric["best_epoch"] = epoch
            self.best_metric["best_global_step"] = global_step
            self.best_metric[indicator] = cur_score
            save_ckpt(
                self.model,
                self.cfg,
                self.optimizer,
                self.lr_scheduler,
                epoch,
                global_step,
                self.best_metric,
                is_best=True,
                prefix=None,
            )
            self.logger.info(
                "new best checkpoint saved: %s=%.6f (epoch=%s step=%s)",
                indicator,
                cur_score,
                epoch,
                global_step,
            )
        self.logger.info(
            "best metric, "
            + ", ".join(
                f"{key}: {value}"
                for key, value in self.best_metric.items()
                if isinstance(value, (float, int, str, bool))
            )
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def eval(self):
        self.model.eval()
        model_module = self.model.module if hasattr(self.model, "module") else self.model
        previous_auxiliary = getattr(model_module, "return_auxiliary_outputs", False)
        model_module.return_auxiliary_outputs = bool(
            self.cfg.get("Eval", {}).get("return_auxiliary_outputs", previous_auxiliary)
        )
        total_frame = 0
        total_time = 0.0
        try:
            with torch.no_grad():
                pbar = tqdm(
                    total=len(self.valid_dataloader),
                    desc="eval model:",
                    position=0,
                    leave=True,
                    disable=not is_main_process(),
                )
                for batch in self.valid_dataloader:
                    start = time.time()
                    outputs = model_module(batch)
                    total_time += time.time() - start
                    self.eval_class(outputs, batch)
                    total_frame += len(batch)
                    pbar.update(1)
                pbar.close()
            metric = self.eval_class.get_metric()
            metric["fps"] = total_frame / max(total_time, 1e-9)
        finally:
            model_module.return_auxiliary_outputs = previous_auxiliary
            self.model.train()
        return metric
