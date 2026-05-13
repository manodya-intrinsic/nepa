#!/usr/bin/env python3
"""
3D NEPA video fine-tuning script for HMDB51 action classification.

This script loads a pretrained 3D ViT-NEPA encoder and trains a linear
classification head on top of pooled tubelet embeddings.
"""

import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from PIL import Image as PILImage
from datasets import Video, load_dataset, Dataset, Features, ClassLabel, Value
from torchvision.transforms import CenterCrop, Compose, Lambda, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils.import_utils import is_sagemaker_mp_enabled

from models.vit_nepa.configuration_vit_nepa import ViTNepaConfig
from models.vit_nepa.modeling_vit_nepa_3d import ViTNepaVideoModel

logger = logging.getLogger(__name__)

try:
    from decord import VideoReader, cpu as decord_cpu
except ImportError:
    logger.warning("Decord is not installed. Video decoding will fail.")
    VideoReader = None
    decord_cpu = None


def _configure_quiet_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _configure_quiet_logging():
    for logger_name in ["datasets.builder", "datasets.download", "datasets.utils", "urllib3.connectionpool"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _resolve_video_path(video_entry) -> str:
    if hasattr(video_entry, "path") and video_entry.path is not None:
        return video_entry.path
    if hasattr(video_entry, "local_path") and video_entry.local_path is not None:
        return video_entry.local_path
    if isinstance(video_entry, dict):
        if video_entry.get("path") is not None:
            return video_entry["path"]
        if video_entry.get("local_path") is not None:
            return video_entry["local_path"]
    return str(video_entry)


def _sample_frame_indices(total_frames: int, num_frames: int, train: bool) -> torch.Tensor:
    if total_frames <= 0:
        raise ValueError("Video contains no frames.")
    if total_frames >= num_frames:
        if train:
            max_start = total_frames - num_frames
            start = random.randint(0, max_start) if max_start > 0 else 0
            return torch.arange(start, start + num_frames)
        return torch.linspace(0, total_frames - 1, num_frames).round().long()

    base = torch.arange(total_frames)
    pad = torch.full((num_frames - total_frames,), total_frames - 1, dtype=torch.long)
    return torch.cat([base, pad], dim=0)


def _video_to_clip_tensor(video_path: str, num_frames: int, spatial_size: int, train: bool) -> Optional[torch.Tensor]:
    if VideoReader is None or decord_cpu is None:
        logger.warning("Decord is not available. Returning None for video decoding.")
        return None

    try:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            logger.warning(f"Video {video_path} has 0 frames. Skipping.")
            return None
        frame_indices = _sample_frame_indices(total_frames, num_frames, train=train)
        frames_np = vr.get_batch(frame_indices.cpu().numpy()).asnumpy()  # (T, H, W, C)
        video = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)
    except Exception as exc:
        logger.warning(f"Decord failed for {video_path}: {exc}. Returning None.")
        return None

    if video.ndim != 4 or video.shape[0] == 0:
        logger.warning(f"Decoded video not valid for {video_path}. Skipping.")
        return None

    if train:
        frame_transform = Compose(
            [
                RandomResizedCrop(spatial_size),
                RandomHorizontalFlip(),
                ToTensor(),
                Lambda(lambda x: x),
            ]
        )
    else:
        frame_transform = Compose(
            [
                Resize(spatial_size),
                CenterCrop(spatial_size),
                ToTensor(),
                Lambda(lambda x: x),
            ]
        )

    processed_frames = []
    for frame in video:
        pil_frame = PILImage.fromarray(frame.permute(1, 2, 0).cpu().numpy())
        processed_frames.append(frame_transform(pil_frame))

    return torch.stack(processed_frames, dim=1)  # [C, T, H, W]


@dataclass
class DataTrainingArguments:
    train_dir: Optional[str] = field(default="/content/hmdb51_local/train")
    validation_dir: Optional[str] = field(default="/content/hmdb51_local/val")
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    video_column_name: str = field(default="video")
    resize_size: int = field(default=224)
    num_frames: int = field(default=16)
    tubelet_size: int = field(default=2)

    def __post_init__(self):
        if self.train_dir is None:
            raise ValueError("You must specify a training directory.")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    pretrained_model_name_or_path: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    token: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=False)
    num_labels: int = field(default=51)


class ViTNepaVideoForActionClassification(nn.Module):
    def __init__(self, config: ViTNepaConfig, num_labels: int = 51):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.vit_nepa = ViTNepaVideoModel(config)
        self.fc_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        assert pixel_values.ndim == 5, f"Expected [B, C, T, H, W], got {tuple(pixel_values.shape)}"
        if labels is None:
            assert not self.training, "labels=None while model is in training mode; eval() may not be set correctly."
        if self.training and labels is not None and not hasattr(self, "_printed_input_shape"):
            print(f"pixel_values.shape before encoder: {tuple(pixel_values.shape)}")
            self._printed_input_shape = True
        outputs = self.vit_nepa(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state
        token_embeddings = sequence_output[:, 1:, :]
        pooled_output = self.fc_norm(token_embeddings.mean(dim=1))
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


class VideoCollator:
    def __init__(self, num_frames: int, resize_size: int, train: bool, num_labels: int):
        self.num_frames = num_frames
        self.resize_size = resize_size
        self.train = train
        self.num_labels = num_labels

    def __call__(self, examples):
        pixel_values = []
        labels = []

        for example in examples:
            video_entry = example["video"]
            video_path = _resolve_video_path(video_entry)
            clip = _video_to_clip_tensor(video_path, self.num_frames, self.resize_size, train=self.train)
            if clip is None:
                continue
            pixel_values.append(clip)
            labels.append(int(example["label"]))

        if not pixel_values:
            raise ValueError("All videos in this batch failed to decode; skip this batch rather than injecting zeros.")

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        assert labels_tensor.numel() > 0, "Empty label batch should never be emitted."
        assert labels_tensor.min().item() >= 0, f"Found negative label ids: {labels_tensor.tolist()}"
        assert labels_tensor.max().item() < self.num_labels, (
            f"Label id out of range for num_labels={self.num_labels}: {labels_tensor.tolist()}"
        )

        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": labels_tensor,
        }


class VideoActionClassificationTrainer(Trainer):
    def __init__(self, *args, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator

    def get_decay_parameter_names(self, model) -> list[str]:
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"layer_scale", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        return get_parameter_names(model, [torch.nn.LayerNorm], forbidden_name_patterns)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        backbone_lr = self.args.learning_rate * 0.1
        head_lr = self.args.learning_rate
        llrd = 0.85
        weight_decay = self.args.weight_decay

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        assert opt_model is not None, "Optimizer creation requires a non-None model."
        decay_parameters = set(self.get_decay_parameter_names(opt_model))

        # CRITICAL: Identify head parameters so they DON'T get layer decay
        head_param_ids = set()
        if hasattr(self.model, "classifier"):
            head_param_ids.update(id(p) for p in self.model.classifier.parameters())
        if hasattr(self.model, "fc_norm"):
            head_param_ids.update(id(p) for p in self.model.fc_norm.parameters())

        encoder_layers = []
        if (hasattr(self.model, "vit_nepa") and 
            hasattr(self.model.vit_nepa, "encoder") and 
            hasattr(self.model.vit_nepa.encoder, "layer")):
            encoder_layers = list(self.model.vit_nepa.encoder.layer)
        num_layers = len(encoder_layers)

        grouped = {}
        for full_name, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue

            # HEAD gets fixed high LR, NO decay
            if id(p) in head_param_ids:
                lr = head_lr  # Fixed at 1e-4
                wd = 0.0 if p.ndim <= 1 else weight_decay
            
            # BACKBONE gets decayed LR based on layer depth
            else:
                lr = backbone_lr
                wd = 0.0 if p.ndim <= 1 else (weight_decay if full_name in decay_parameters else 0.0)
                
                # Apply layer decay only to backbone layers
                for layer_idx, layer in enumerate(encoder_layers):
                    if any(id(p) == id(param) for param in layer.parameters()):
                        scale = num_layers - 1 - layer_idx
                        lr = backbone_lr * (llrd ** scale)
                        break

            grouped.setdefault((lr, wd), []).append(p)

        optimizer_grouped_parameters = [
            {"params": params, "lr": lr, "weight_decay": wd}
            for (lr, wd), params in grouped.items()
        ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_collator is None:
            return super().get_eval_dataloader(eval_dataset)

        old_collator = self.data_collator
        self.data_collator = self.eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = old_collator

    def log(self, logs, start_time=None):
        if logs and self.is_world_process_zero():
            if self.state.global_step % 50 == 0 and hasattr(self.model, "classifier"):
                print(f"Classifier weight norm: {self.model.classifier.weight.norm().item():.4f}")
        return super().log(logs, start_time=start_time)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if predictions.ndim != 2:
        raise ValueError(f"Expected predictions to have shape [batch, num_labels], got {predictions.shape}")

    labels = labels.reshape(-1)
    if labels.ndim != 1:
        raise ValueError(f"Expected labels to flatten to [batch], got shape {labels.shape}")
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Prediction batch size {predictions.shape[0]} does not match label batch size {labels.shape[0]}"
        )

    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = (predicted_labels == labels).mean()
    return {"accuracy": float(accuracy)}


def _assert_config_match(source_config: ViTNepaConfig, target_config: ViTNepaConfig):
    # Validate key architectural fields to prevent partial/silent mismatches.
    fields = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "num_frames",
        "tubelet_size",
        "patch_size",
        "image_size",
        "num_channels",
    ]
    mismatches = []
    for name in fields:
        if getattr(source_config, name, None) != getattr(target_config, name, None):
            mismatches.append(
                f"{name}: source={getattr(source_config, name, None)} target={getattr(target_config, name, None)}"
            )

    if mismatches:
        raise AssertionError(
            "Pretrained encoder config does not match finetune encoder config. "
            + "; ".join(mismatches)
        )


def _assert_3d_checkpoint_shapes(model: ViTNepaVideoForActionClassification, state_dict: dict):
    model_state = model.vit_nepa.state_dict()
    key = "embeddings.patch_embeddings.projection.weight"

    if key in state_dict:
        src = state_dict[key]
        dst = model_state[key]
        if src.ndim != 5 or dst.ndim != 5:
            raise AssertionError(
                f"Expected 3D Conv projection weights to be rank-5, got src.ndim={src.ndim}, dst.ndim={dst.ndim}"
            )
        if src.shape != dst.shape:
            raise AssertionError(
                f"3D patch projection shape mismatch: source={tuple(src.shape)} target={tuple(dst.shape)}"
            )


def _unwrap_checkpoint_state_dict(checkpoint, trace_key: str = "vit_nepa.encoder.layer.0.attention.query.weight"):
    selected_container = "root"
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "ema_model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                selected_container = key
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    logger.info("Checkpoint container selected: %s", selected_container)
    if trace_key in checkpoint:
        logger.info("Trace key present before prefix cleanup: %s", trace_key)
    else:
        logger.info("Trace key not present before prefix cleanup: %s", trace_key)

    for prefix in ("module.", "model."):
        if any(key.startswith(prefix) for key in checkpoint.keys()):
            logger.info("Stripping checkpoint prefix: %s", prefix)
            checkpoint = {
                key[len(prefix):] if key.startswith(prefix) else key: value
                for key, value in checkpoint.items()
            }

    if trace_key in checkpoint:
        logger.info("Trace key present after prefix cleanup: %s", trace_key)
    else:
        if any(key.endswith("encoder.layer.0.attention.query.weight") for key in checkpoint.keys()):
            example = next(key for key in checkpoint.keys() if key.endswith("encoder.layer.0.attention.query.weight"))
            logger.info("Trace fallback key after cleanup: %s", example)

    return checkpoint


def _load_pretrained_weights(model: ViTNepaVideoForActionClassification, pretrained_path: str):
    if os.path.isdir(pretrained_path):
        pretrained_model = ViTNepaVideoModel.from_pretrained(pretrained_path)
        _assert_config_match(pretrained_model.config, model.vit_nepa.config)
        state_dict = pretrained_model.state_dict()
    else:
        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Backward compatibility for older torch versions without weights_only.
            checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = _unwrap_checkpoint_state_dict(checkpoint)
        if any(key.startswith("vit_nepa.") for key in state_dict.keys()):
            state_dict = {
                key[len("vit_nepa."):]: value
                for key, value in state_dict.items()
                if key.startswith("vit_nepa.")
            }

    _assert_3d_checkpoint_shapes(model, state_dict)

    model_state = model.vit_nepa.state_dict()
    loaded_tensor_keys = [
        key
        for key, value in model_state.items()
        if key in state_dict and state_dict[key].shape == value.shape
    ]
    loaded_params = sum(model_state[key].numel() for key in loaded_tensor_keys)
    total_params = sum(value.numel() for value in model_state.values())

    missing_keys, unexpected_keys = model.vit_nepa.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded encoder tensors: %d/%d | parameters: %d/%d (%.2f%%)",
        len(loaded_tensor_keys),
        len(model_state),
        loaded_params,
        total_params,
        (100.0 * loaded_params / max(total_params, 1)),
    )
    if loaded_params == 0:
        raise AssertionError("Loaded 0 encoder parameters from pretrained checkpoint.")

    if missing_keys:
        logger.info("Missing keys when loading pretrained weights: %s", missing_keys[:10])
        if any("patch_embeddings" in key for key in missing_keys):
            raise AssertionError(
                "patch_embeddings keys are missing after load; encoder would be partially/randomly initialized."
            )
    if unexpected_keys:
        logger.info("Unexpected keys when loading pretrained weights: %s", unexpected_keys[:10])

    print(
        f"[OK] Pretrained encoder loaded: {loaded_params}/{total_params} parameters "
        f"({100.0 * loaded_params / max(total_params, 1):.2f}%)"
    )


class EpochAccuracyCallback(TrainerCallback):
    """Callback to log learning behavior at each step and epoch."""
    
    def __init__(self):
        self.last_logged_epoch = -1
        self.last_logged_step = -1
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log training progress every N steps (early in training)."""
        current_step = state.global_step
        
        # Log every 25 steps in first 100 steps to see early behavior
        if current_step <= 100 and current_step % 25 == 0:
            loss = state.log_history[-1].get("loss", None) if state.log_history else None
            lr = state.log_history[-1].get("learning_rate", None) if state.log_history else None
            lr_str = f" | LR: {lr:.2e}" if lr is not None else ""
            loss_str = f" | Loss: {loss:.4f}" if loss is not None else ""
            print(f"  Step {current_step:5d}{loss_str}{lr_str}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training progress at each epoch boundary."""
        if logs is None:
            return
        
        current_epoch = state.epoch
        # Log only at epoch boundaries (when epoch changes)
        if current_epoch > self.last_logged_epoch and int(current_epoch) > 0:
            self.last_logged_epoch = int(current_epoch)
            loss = logs.get("loss", None)
            learning_rate = logs.get("learning_rate", None)
            
            lr_str = f" | LR: {learning_rate:.2e}" if learning_rate is not None else ""
            loss_str = f" | Train Loss: {loss:.4f}" if loss is not None else ""
            print(f"[Epoch {int(current_epoch):2d}/30]{loss_str}{lr_str}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics after validation."""
        if metrics is not None:
            epoch = state.epoch
            accuracy = metrics.get("eval_accuracy", 0.0)
            eval_loss = metrics.get("eval_loss", 0.0)
            print(f"  ↳ Eval @ Epoch {epoch:.1f} | Accuracy: {accuracy:.4f} | Eval Loss: {eval_loss:.4f}")


def _run_sanity_check(model: ViTNepaVideoForActionClassification, collator: VideoCollator, train_dataset):
    if train_dataset is None or len(train_dataset) == 0:
        return

    batch_examples = [train_dataset[i] for i in range(min(1, len(train_dataset)))]
    batch = collator(batch_examples)
    
    # Move batch to the same device as the model
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])

    logits = outputs["logits"]
    loss = outputs["loss"]
    assert logits.shape[0] == batch["pixel_values"].shape[0], f"Logits batch mismatch: {tuple(logits.shape)}"
    assert logits.shape[1] == model.num_labels, f"Logits class mismatch: {tuple(logits.shape)}"
    assert loss is not None and torch.isfinite(loss), f"Sanity-check loss is invalid: {loss}"


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    _configure_quiet_warnings()
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    training_args.disable_tqdm = True
    training_args.report_to = []
    _configure_quiet_logging()
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_default_handler()

    training_args.remove_unused_columns = False

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    set_seed(training_args.seed)

    config_source = model_args.config_name or model_args.model_name_or_path
    if config_source is None:
        raise ValueError("You must pass --config_name or --model_name_or_path.")

    # Resolve config path: convert relative paths to absolute
    if config_source and not os.path.isabs(config_source) and not config_source.startswith(("http", "s3")):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_source = os.path.join(script_dir, config_source)
        logger.info("Resolved config path to: %s", config_source)

    config = ViTNepaConfig.from_pretrained(
        config_source,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
    )
    config.num_labels = model_args.num_labels

    # Load train and validation datasets by scanning directory structure directly
    # This avoids torchcodec dependency issues
    from pathlib import Path
    
    def build_dataset_from_directory(root_dir: str, max_samples: Optional[int] = None):
        """Build dataset by scanning directory structure for videos.
        
        Uses Value("string") to store video paths as strings, avoiding torchcodec issues.
        VideoCollator handles actual decoding with Decord.
        """
        root_path = Path(root_dir)
        videos = []
        labels = []
        class_to_idx = {}
        
        for class_idx, class_dir in enumerate(sorted(root_path.iterdir())):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            class_to_idx[class_name] = class_idx
            
            for video_path in sorted(class_dir.glob("*.avi")):
                if max_samples is not None and len(videos) >= max_samples:
                    break
                videos.append(str(video_path))
                labels.append(class_idx)
            
            if max_samples is not None and len(videos) >= max_samples:
                break
        
        if not videos:
            raise ValueError(f"No .avi files found in {root_dir}")
        
        # Create dataset dict with string paths (not Video objects)
        # This avoids torchcodec import which is incompatible with Torch 2.4.1
        data_dict = {
            "video": videos,
            "label": labels,
        }
        
        # Use Value("string") instead of Video() to store paths as strings
        # VideoCollator will decode with Decord
        features = Features({
            "video": Value("string"),
            "label": ClassLabel(num_classes=len(class_to_idx), names=sorted(class_to_idx.keys())),
        })
        
        dataset = Dataset.from_dict(data_dict, features=features)
        logger.info(f"Built dataset from {root_dir}: {len(dataset)} videos, {len(class_to_idx)} classes")
        return dataset
    
    logger.info("Loading dataset from %s (scanning directory structure)", data_args.train_dir)
    train_dataset = build_dataset_from_directory(data_args.train_dir, data_args.max_train_samples)
    
    eval_dataset = None
    if training_args.do_eval and data_args.validation_dir is not None:
        logger.info("Loading validation dataset from %s", data_args.validation_dir)
        eval_dataset = build_dataset_from_directory(data_args.validation_dir, data_args.max_eval_samples)
    
    logger.info("Train dataset size: %s", len(train_dataset))
    if eval_dataset is not None:
        logger.info("Validation dataset size: %s", len(eval_dataset))

    model = ViTNepaVideoForActionClassification(config, num_labels=model_args.num_labels)
    trainable_backbone = sum(p.numel() for p in model.vit_nepa.parameters() if p.requires_grad)
    total_backbone = sum(p.numel() for p in model.vit_nepa.parameters())
    logger.info("Trainable backbone params: %d/%d", trainable_backbone, total_backbone)
    if model_args.pretrained_model_name_or_path:
        logger.info("Loading pretrained weights from %s", model_args.pretrained_model_name_or_path)
        _load_pretrained_weights(model, model_args.pretrained_model_name_or_path)

    train_collator = VideoCollator(data_args.num_frames, data_args.resize_size, train=True, num_labels=model_args.num_labels)
    eval_collator = VideoCollator(data_args.num_frames, data_args.resize_size, train=False, num_labels=model_args.num_labels)

    trainer = VideoActionClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_collator,
        eval_collator=eval_collator,
        compute_metrics=compute_metrics,
        callbacks=[EpochAccuracyCallback()],
    )

    _run_sanity_check(model, train_collator, train_dataset)

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval and eval_dataset is not None:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        logger.info("Final evaluation accuracy: %.4f", metrics.get("eval_accuracy", 0.0))

    logger.info("Training complete. Output directory: %s", training_args.output_dir)


if __name__ == "__main__":
    main()
