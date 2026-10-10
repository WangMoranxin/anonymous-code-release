import argparse
import inspect
import os
import shutil
import time

import torch
from torch.utils.data import SequentialSampler
from transformers import LlamaConfig, Trainer, TrainingArguments

from apd_gr.cache_dataset import (
    DraftCacheCollator,
    DraftCacheDataset,
    MANIFEST_NAME,
    ShardLocalSampler,
)
from apd_gr.loss import compute_parallel_draft_loss
from apd_gr.modeling_parallel_draft import ParallelDraftModel
from apd_gr.target_adapters import (apply_adapter_item_token_num, make_target_adapter,
                             resolve_target_layer_ids)
from apd_gr.utils import (
    add_common_args,
    build_position_token_ids,
    load_item_indices,
    parse_int_list,
    set_seed,
)

class ParallelDraftTrainer(Trainer):
    def __init__(
        self,
        *args,
        shuffle_cache=False,
        ce_alpha=1.0,
        ce_decay_gamma=0.0,
        kl_alpha=0.0,
        kl_temperature=2.0,
        kl_decay_gamma=4.0,
        projection_stats_steps=0,
        projection_delta_stats=True,
        save_resume_state=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shuffle_cache = bool(shuffle_cache)
        self.ce_alpha = float(ce_alpha)
        self.ce_decay_gamma = float(ce_decay_gamma)
        self.kl_alpha = float(kl_alpha)
        self.kl_temperature = float(kl_temperature)
        self.kl_decay_gamma = float(kl_decay_gamma)
        self.projection_stats_steps = int(projection_stats_steps or 0)
        self.projection_delta_stats = bool(projection_delta_stats)
        self.save_resume_state = bool(save_resume_state)
        self._last_metrics_step_logged = None
        self._last_projection_step_logged = None
        self._last_fc_weight_cpu = None
        self._last_hidden_norm_weight_cpu = None

    def _save_checkpoint(self, model, trial, metrics=None):
        # Transformers <=4.45 accepted ``metrics`` here, while >=4.51 removed
        # it.  MiniOneRec checkpoints may use newer Transformers releases, so dispatch against
        # the installed Trainer signature without hiding unrelated TypeErrors.
        parent_save = super()._save_checkpoint
        if "metrics" in inspect.signature(parent_save).parameters:
            parent_save(model, trial, metrics=metrics)
        else:
            parent_save(model, trial)
        if self.save_resume_state or not self.args.should_save:
            return

        checkpoint_dir = os.path.join(
            self._get_output_dir(trial),
            f"checkpoint-{self.state.global_step}",
        )
        removed_bytes = 0
        if os.path.isdir(checkpoint_dir):
            for name in os.listdir(checkpoint_dir):
                path = os.path.join(checkpoint_dir, name)
                if name.startswith("global_step") and os.path.isdir(path):
                    for root, _, files in os.walk(path):
                        for filename in files:
                            try:
                                removed_bytes += os.path.getsize(os.path.join(root, filename))
                            except OSError:
                                pass
                    shutil.rmtree(path)
            for name in os.listdir(checkpoint_dir):
                path = os.path.join(checkpoint_dir, name)
                if name in {"latest", "zero_to_fp32.py", "optimizer.pt", "scheduler.pt", "scaler.pt"} or name.startswith("rng_state"):
                    if os.path.isfile(path):
                        try:
                            removed_bytes += os.path.getsize(path)
                        except OSError:
                            pass
                        os.remove(path)
        print(
            f"Pruned resume state from {checkpoint_dir} "
            f"({removed_bytes / (1024 ** 3):.2f} GiB freed)",
            flush=True,
        )

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if isinstance(dataset, DraftCacheDataset):
            # Accelerate shards the prepared DataLoader across processes. Keep this
            # sampler global; pre-sharding it by rank would divide the data twice.
            return ShardLocalSampler(
                dataset,
                shuffle_shards=self.shuffle_cache,
                shuffle_within_shard=self.shuffle_cache,
                seed=self.args.seed,
                num_replicas=1,
                rank=0,
                drop_last=False,
            )
        if self.shuffle_cache:
            try:
                return super()._get_train_sampler(dataset)
            except TypeError:
                return super()._get_train_sampler()
        return SequentialSampler(dataset)

    def _unwrap_model(self, model):
        module = model
        while hasattr(module, "module"):
            module = module.module
        return module

    def _projection_metrics(self, model):
        module = self._unwrap_model(model)
        if not hasattr(module, "fc") or not hasattr(module, "hidden_norm"):
            return {}
        metrics = {}
        fc_weight = module.fc.weight.detach()
        fc_norm = fc_weight.float().norm().clamp_min(1e-12)
        metrics["fc_weight_norm"] = float(fc_norm.detach().cpu().item())
        hidden_weight = module.hidden_norm.weight.detach()
        hidden_norm = hidden_weight.float().norm().clamp_min(1e-12)
        metrics["hidden_norm_weight_norm"] = float(hidden_norm.detach().cpu().item())
        if self.projection_delta_stats:
            fc_cpu = fc_weight.to(device="cpu", dtype=torch.float32).contiguous()
            if self._last_fc_weight_cpu is not None:
                delta = (fc_cpu - self._last_fc_weight_cpu).norm() / self._last_fc_weight_cpu.norm().clamp_min(1e-12)
                metrics["fc_delta_rel"] = float(delta.item())
            self._last_fc_weight_cpu = fc_cpu
            hidden_cpu = hidden_weight.to(device="cpu", dtype=torch.float32).contiguous()
            if self._last_hidden_norm_weight_cpu is not None:
                delta = (hidden_cpu - self._last_hidden_norm_weight_cpu).norm() / self._last_hidden_norm_weight_cpu.norm().clamp_min(1e-12)
                metrics["hidden_norm_delta_rel"] = float(delta.item())
            self._last_hidden_norm_weight_cpu = hidden_cpu
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "target_hidden_states": inputs["target_hidden_states"],
            "target_ids": inputs["target_ids"],
            "root_hidden": inputs["root_hidden"],
        }
        if "context_is_projected" in inputs:
            model_kwargs["context_is_projected"] = inputs["context_is_projected"]
        outputs = model(**model_kwargs)
        logging_steps = max(int(self.args.logging_steps), 1)
        should_loss_log = (
            self.state.global_step % logging_steps == 0
            and self._last_metrics_step_logged != self.state.global_step
        )
        should_projection_log = (
            self.projection_stats_steps > 0
            and self.state.global_step % self.projection_stats_steps == 0
            and self._last_projection_step_logged != self.state.global_step
        )
        teacher_logits = None
        if "teacher_legal_logits" in inputs:
            legal_sizes = inputs["teacher_legal_sizes"]
            if not torch.equal(legal_sizes, legal_sizes[:1].expand_as(legal_sizes)):
                raise ValueError("teacher_legal_sizes must be identical within a batch")
            teacher_logits = [
                inputs["teacher_legal_logits"][:, pos, : int(size)]
                for pos, size in enumerate(legal_sizes[0].tolist())
            ]
        loss, metrics = compute_parallel_draft_loss(
            logits=outputs.logits,
            target_local_ids=outputs.target_local_ids,
            eval_mask=outputs.eval_mask,
            teacher_logits=teacher_logits,
            ce_alpha=self.ce_alpha,
            ce_decay_gamma=self.ce_decay_gamma,
            kl_alpha=self.kl_alpha,
            kl_temperature=self.kl_temperature,
            kl_decay_gamma=self.kl_decay_gamma,
            compute_metrics=should_loss_log,
        )
        if should_projection_log:
            metrics.update(self._projection_metrics(model))
        if (should_loss_log or should_projection_log) and metrics:
            self.log(metrics)
            if should_loss_log:
                self._last_metrics_step_logged = self.state.global_step
            if should_projection_log:
                self._last_projection_step_logged = self.state.global_step
        return (loss, outputs) if return_outputs else loss

def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train parallel_draft draft model.")
    parser = add_common_args(parser)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--local_cache_dir", type=str, default=None)
    parser.add_argument("--force_stage_cache", action="store_true", default=False)
    parser.add_argument("--cache_mmap", action="store_true", default=False)
    parser.add_argument("--log_shard_loads", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--draft_init_ckpt", type=str, default=None)
    parser.add_argument("--freeze_context_projection", type=str2bool, default=False)
    parser.add_argument("--num_draft_layers", type=int, default=None)
    parser.add_argument("--draft_hidden_size", type=int, default=None)
    parser.add_argument("--draft_intermediate_size", type=int, default=None)
    parser.add_argument("--draft_num_attention_heads", type=int, default=None)
    parser.add_argument("--mask_token_id", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.04)
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--save_final_model", type=str2bool, default=True)
    parser.add_argument("--save_resume_state", type=str2bool, default=False)
    parser.add_argument("--records_per_shard", type=int, default=512)
    parser.add_argument("--shuffle_cache", action="store_true", default=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    parser.add_argument("--dataloader_persistent_workers", type=str2bool, default=True)
    parser.add_argument("--ce_alpha", type=float, default=0.7)
    parser.add_argument("--ce_decay_gamma", type=float, default=0.0,
                        help="CE position-decay gamma; <=0 keeps all positions equally weighted")
    parser.add_argument("--kl_alpha", type=float, default=0.3)
    parser.add_argument("--kl_temperature", type=float, default=2.0)
    parser.add_argument("--kl_decay_gamma", type=float, default=4.0)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--projection_stats_steps", type=int, default=0)
    parser.add_argument("--projection_delta_stats", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    return parser.parse_args()

def build_draft_config(args, tokenizer, adapter):
    profile = adapter.profile
    num_layers = args.num_draft_layers or profile.layers
    hidden_size = args.draft_hidden_size or profile.hidden_size
    intermediate_size = args.draft_intermediate_size or profile.intermediate_size
    attention_heads = args.draft_num_attention_heads or profile.attention_heads
    if hidden_size % attention_heads != 0:
        raise ValueError("draft_hidden_size must be divisible by draft_num_attention_heads")
    if int(hidden_size) != int(adapter.hidden_size):
        raise ValueError(
            "The current ParallelDraft head requires draft_hidden_size == target hidden size "
            f"for root_hidden compatibility ({hidden_size} != {adapter.hidden_size})"
        )
    draft_config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_hidden_layers=int(num_layers),
        num_attention_heads=int(attention_heads),
        num_key_value_heads=int(attention_heads),
        max_position_embeddings=int(args.model_max_length) + int(args.item_token_num) + 8,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
    )
    draft_config.architectures = ["ParallelDraftModel"]
    draft_config.target_model_type = adapter.kind
    draft_config.target_hidden_size = int(adapter.hidden_size)
    draft_config.num_target_layers = int(adapter.num_layers)
    draft_config.target_layer_ids = parse_int_list(args.target_layer_ids)
    draft_config.item_token_num = int(args.item_token_num)
    if adapter.uses_integer_codes:
        draft_config.position_token_ids = adapter.position_token_ids()
    else:
        draft_config.position_token_ids = build_position_token_ids(
            load_item_indices(args.data_path, args.dataset, args.index_file),
            tokenizer,
            args.item_token_num,
        )
    if args.mask_token_id is None:
        draft_config.mask_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
    else:
        draft_config.mask_token_id = int(args.mask_token_id)
    return draft_config

def _stage_cache_dir(cache_dir: str, local_cache_dir: str | None, force: bool = False) -> str:
    if not local_cache_dir:
        return cache_dir
    src = os.path.abspath(cache_dir)
    dst = os.path.abspath(local_cache_dir)
    if src == dst:
        return cache_dir
    manifest_path = os.path.join(dst, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not force:
        print(f"Using staged cache at {dst}", flush=True)
        return dst

    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok=True)
    lock_dir = f"{dst}.copylock"
    while True:
        try:
            os.mkdir(lock_dir)
            owns_lock = True
        except FileExistsError:
            owns_lock = False
        if owns_lock:
            tmp_dst = f"{dst}.tmp.{os.getpid()}"
            try:
                if force and os.path.exists(dst):
                    shutil.rmtree(dst)
                if os.path.exists(tmp_dst):
                    shutil.rmtree(tmp_dst)
                print(f"Staging cache from {src} to {dst}", flush=True)
                shutil.copytree(src, tmp_dst, symlinks=True)
                if not os.path.exists(dst):
                    os.replace(tmp_dst, dst)
                else:
                    shutil.rmtree(tmp_dst)
            finally:
                try:
                    os.rmdir(lock_dir)
                except OSError:
                    pass
            break
        if os.path.exists(manifest_path):
            break
        print(f"Waiting for staged cache at {dst}", flush=True)
        time.sleep(5)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Staged cache manifest was not found at {manifest_path}")
    return dst

def _training_arguments(args):
    kwargs = dict(
        output_dir=args.output_dir,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_strategy="steps",
        disable_tqdm=False,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        dataloader_pin_memory=True,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if (
        "dataloader_persistent_workers" in parameters
        and int(args.dataloader_num_workers) > 0
    ):
        kwargs["dataloader_persistent_workers"] = bool(args.dataloader_persistent_workers)
    if (
        "dataloader_prefetch_factor" in parameters
        and int(args.dataloader_num_workers) > 0
        and int(args.dataloader_prefetch_factor) > 0
    ):
        kwargs["dataloader_prefetch_factor"] = int(args.dataloader_prefetch_factor)
    return TrainingArguments(**kwargs)

def _load_or_init_model(args, draft_config):
    if args.draft_init_ckpt:
        return ParallelDraftModel.from_pretrained(args.draft_init_ckpt)
    return ParallelDraftModel(draft_config)

def _freeze_context_projection(model):
    for module in (model.fc, model.hidden_norm):
        for param in module.parameters():
            param.requires_grad_(False)


def _validate_cache_and_model_identity(args, dataset, model):
    """Fail early when a cache, checkpoint, and CLI describe different experiments."""
    manifest = dataset.manifest
    arg_layers = parse_int_list(args.target_layer_ids)
    cache_layers = manifest.get("target_layer_ids")
    model_layers = getattr(model.config, "target_layer_ids", None)

    if cache_layers is not None and [int(x) for x in cache_layers] != arg_layers:
        raise ValueError(
            "Layer mismatch: cache manifest target_layer_ids="
            f"{cache_layers}, but --target_layer_ids={arg_layers}."
        )
    if model_layers is not None and [int(x) for x in model_layers] != arg_layers:
        raise ValueError(
            "Layer mismatch: draft checkpoint/config target_layer_ids="
            f"{model_layers}, but --target_layer_ids={arg_layers}."
        )

    expected_dim = len(arg_layers) * int(model.config.target_hidden_size)
    raw_dim = manifest.get("raw_target_context_dim")
    if raw_dim is not None and int(raw_dim) != expected_dim:
        raise ValueError(
            "Context dimension mismatch: cache raw_target_context_dim="
            f"{raw_dim}, expected {expected_dim} for layers {arg_layers}."
        )

    if dataset.context_is_projected:
        projection_ckpt = manifest.get("projection_ckpt")
        if not projection_ckpt:
            raise ValueError("Projected cache manifest is missing projection_ckpt.")
        if not args.draft_init_ckpt:
            raise ValueError("Projected cache requires --draft_init_ckpt.")
        if os.path.realpath(projection_ckpt) != os.path.realpath(args.draft_init_ckpt):
            raise ValueError(
                "Checkpoint mismatch: projected cache was generated with "
                f"{projection_ckpt}, but --draft_init_ckpt={args.draft_init_ckpt}."
            )

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    adapter = make_target_adapter(args)
    apply_adapter_item_token_num(args, adapter)
    resolved_layers = resolve_target_layer_ids(args.target_layer_ids, adapter.num_layers)
    args.target_layer_ids = ",".join(map(str, resolved_layers))
    if args.num_draft_layers is None:
        args.num_draft_layers = len(resolved_layers)
    print(
        f"Resolved architecture: target_layers={resolved_layers}, "
        f"draft_layers={args.num_draft_layers}",
        flush=True,
    )
    tokenizer = adapter.load_tokenizer(padding_side="right")
    cache_dir = _stage_cache_dir(args.cache_dir, args.local_cache_dir, args.force_stage_cache)
    dataset = DraftCacheDataset(
        cache_dir,
        records_per_shard=args.records_per_shard,
        use_mmap=args.cache_mmap,
        log_shard_loads=args.log_shard_loads,
    )
    print(
        f"Loaded cache samples: {len(dataset)} from {len(dataset.shards)} shards "
        f"at {cache_dir}",
        flush=True,
    )
    if dataset.manifest.get("field_dtypes"):
        print(f"Cache field dtypes: {dataset.manifest['field_dtypes']}", flush=True)
    if dataset.manifest.get("field_bytes"):
        print(f"Cache field bytes: {dataset.manifest['field_bytes']}", flush=True)

    draft_config = build_draft_config(args, tokenizer, adapter)
    teacher_cache = dataset.manifest.get("teacher_cache", "none")
    if float(args.kl_alpha) > 0 and teacher_cache != "full_legal":
        raise ValueError(
            f"KL loss requires teacher_cache=full_legal, but this cache has "
            f"teacher_cache={teacher_cache!r}. Regenerate the projected cache."
        )
    model = _load_or_init_model(args, draft_config)
    draft_config = model.config
    _validate_cache_and_model_identity(args, dataset, model)
    if dataset.context_is_projected:
        if not args.draft_init_ckpt:
            raise ValueError(
                "Projected-context cache requires --draft_init_ckpt pointing to the same "
                "draft checkpoint used by prepare_cache.py --draft_ckpt_for_projection. "
                "This keeps fc/hidden_norm available for normal raw-context evaluation."
            )
        if args.freeze_context_projection:
            _freeze_context_projection(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer.save_pretrained(args.output_dir)
    draft_config.save_pretrained(args.output_dir)

    training_args = _training_arguments(args)
    trainer = ParallelDraftTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DraftCacheCollator(),
        shuffle_cache=args.shuffle_cache,
        ce_alpha=args.ce_alpha,
        ce_decay_gamma=args.ce_decay_gamma,
        kl_alpha=args.kl_alpha,
        kl_temperature=args.kl_temperature,
        kl_decay_gamma=args.kl_decay_gamma,
        projection_stats_steps=args.projection_stats_steps,
        projection_delta_stats=args.projection_delta_stats,
        save_resume_state=args.save_resume_state,
    )
    print(
        "Start parallel_draft training... "
        f"shuffle_cache={args.shuffle_cache} cache_mmap={args.cache_mmap} "
        f"teacher_cache={teacher_cache} context_is_projected={dataset.context_is_projected} "
        f"save_resume_state={args.save_resume_state}",
        flush=True,
    )
    trainer.train()
    if args.save_resume_state:
        trainer.save_state()
    if args.save_final_model:
        trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
