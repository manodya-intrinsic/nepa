# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training entrypoint for 3D NEPA on video clips.

This runner keeps the existing NEPA objective (next-embedding prediction) but swaps the input pipeline
from 2D images to 3D video clips with tubelet embedding.
"""

import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
from datasets import ClassLabel, Video, load_dataset
from PIL import Image
try:
    from decord import VideoReader, cpu as decord_cpu
except Exception:
    VideoReader = None
    decord_cpu = None
from torchvision.transforms import CenterCrop, Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor

import transformers
from transformers import HfArgumentParser, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from models.vit_nepa.modeling_vit_nepa_3d import ViTNepaVideoForPreTraining
from models.vit_nepa.configuration_vit_nepa import ViTNepaConfig
from run_nepa import EnhancedTrainer


logger = logging.getLogger(__name__)


def _configure_quiet_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The video decoding and encoding capabilities of torchvision are deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Parameter 'transform'.*couldn't be hashed properly.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*libtorchcodec.*",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def _configure_quiet_logging():
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("torchvision").setLevel(logging.ERROR)
    logging.getLogger("wandb").setLevel(logging.ERROR)
    logger.setLevel(logging.ERROR)


class LossOnlyCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.is_world_process_zero is False:
            return control

        loss = logs.get("loss")
        if loss is None:
            return control

        parts = [f"[step {state.global_step}]", f"loss={loss:.4f}"]
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if "grad_norm" in logs:
            parts.append(f"grad_norm={logs['grad_norm']:.2f}")
        print(" | ".join(parts))
        return control


@dataclass
class DataTrainingArguments:
    train_dir: Optional[str] = field(
        default="/content/hmdb51_local",
        metadata={"help": "A folder containing the training videos. Defaults to /content/hmdb51_local."},
    )
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing validation videos."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Truncate train set for debugging."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Truncate eval set for debugging."})
    video_column_name: str = field(
        default="video",
        metadata={"help": "The name of the dataset column containing the video data. Defaults to 'video'."},
    )
    resize_size: int = field(default=224, metadata={"help": "Spatial size used for video preprocessing."})
    num_frames: int = field(default=16, metadata={"help": "Number of frames sampled per clip."})
    tubelet_size: int = field(default=2, metadata={"help": "Temporal tubelet size used by the 3D embedding."})
    load_from_disk: bool = field(default=False, metadata={"help": "Load from disk."})
    keep_in_memory: bool = field(default=False, metadata={"help": "Keep dataset in memory."})

    def __post_init__(self):
        if self.train_dir is None:
            raise ValueError("You must specify a training directory.")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store downloaded models."})
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use."})
    token: str = field(default=None, metadata={"help": "HF token for private files."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code from the Hub."})
    ignore_mismatched_sizes: bool = field(default=False, metadata={"help": "Allow mismatched weights."})
    embed_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate for embeddings."})


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
        raise ValueError(f"Video entry does not contain a path: {video_entry}")
    return str(video_entry)


def _get_video_entries(example_batch, column_name: str):
    if column_name in example_batch:
        return example_batch[column_name]
    for fallback_name in ("video", "path", "file", "video_file"):
        if fallback_name in example_batch:
            logger.warning(
                f"--video_column_name {column_name!r} was not present in the batch; using {fallback_name!r} instead."
            )
            return example_batch[fallback_name]
    raise KeyError(f"Could not find a video column in batch keys: {list(example_batch.keys())}")


def _sample_frame_indices(total_frames: int, num_frames: int, train: bool) -> torch.Tensor:
    if total_frames <= 0:
        raise ValueError("Video contains no frames.")
    if total_frames >= num_frames:
        if train:
            max_start = total_frames - num_frames
            start = random.randint(0, max_start) if max_start > 0 else 0
            return torch.arange(start, start + num_frames)
        return torch.linspace(0, total_frames - 1, num_frames).round().long()

    # Repeat the last frame to reach the target length.
    base = torch.arange(total_frames)
    pad = torch.full((num_frames - total_frames,), total_frames - 1, dtype=torch.long)
    return torch.cat([base, pad], dim=0)


def _video_to_clip_tensor(video_path: str, num_frames: int, spatial_size: int, train: bool) -> torch.Tensor:
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
    except Exception as e:
        logger.warning(f"Decord failed for {video_path}: {e}. Returning None.")
        return None

    # At this point `video` is a torch tensor shaped (T, C, H, W)
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
        pil_frame = Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
        processed_frames.append(frame_transform(pil_frame))

    # Stack to [C, T, H, W] for the 3D NEPA model.
    clip = torch.stack(processed_frames, dim=1)
    return clip


def _classwise_train_validation_split(dataset, split_ratio: float, seed: int):
    if "label" in dataset.features and isinstance(dataset.features["label"], ClassLabel):
        return dataset.train_test_split(split_ratio, seed=seed, stratify_by_column="label")
    logger.warning(
        "Dataset does not expose a ClassLabel 'label' column; falling back to a non-stratified split."
    )
    return dataset.train_test_split(split_ratio, seed=seed)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    _configure_quiet_warnings()

    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    training_args.disable_tqdm = True
    training_args.report_to = []
    _configure_quiet_logging()
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_default_handler()
    if hasattr(transformers.utils.logging, "disable_explicit_format"):
        transformers.utils.logging.disable_explicit_format()

    # Prevent Trainer from removing video/label columns before dataset transforms can access them
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

    data_files = {"train": os.path.join(data_args.train_dir, "**")}
    if data_args.validation_dir is not None:
        data_files["validation"] = os.path.join(data_args.validation_dir, "**")
    dataset = load_dataset("videofolder", data_files=data_files, cache_dir=model_args.cache_dir)

    # Keep the dataset from auto-decoding videos with TorchCodec; Decord handles decoding in the transform.
    dataset = dataset.cast_column(data_args.video_column_name, Video(decode=False))

    if data_args.video_column_name not in (dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names):
        raise ValueError(f"--video_column_name {data_args.video_column_name} not found in the dataset columns.")

    def collate_fn(examples):
        # Filter out examples with None pixel_values (corrupted videos).
        valid_examples = [ex for ex in examples if ex["pixel_values"] is not None]
        if not valid_examples:
            # Keep training moving by emitting a single zero clip instead of a size-0 batch.
            pixel_values = torch.zeros(
                (1, 3, data_args.num_frames, data_args.resize_size, data_args.resize_size),
                dtype=torch.float32,
            )
        else:
            pixel_values = torch.stack([example["pixel_values"] for example in valid_examples])

        batch_size = pixel_values.shape[0]
        num_tokens = 2048
        num_masked = 1843
        bool_masked_pos = torch.zeros((batch_size, num_tokens), dtype=torch.bool)

        for i in range(batch_size):
            masked_indices = torch.randperm(num_tokens)[:num_masked]
            bool_masked_pos[i, masked_indices] = True

        return {"pixel_values": pixel_values, "bool_masked_pos": bool_masked_pos}

    data_args.train_val_split = None if "validation" in dataset else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = _classwise_train_validation_split(dataset["train"], data_args.train_val_split, training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    config = ViTNepaConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.num_frames = data_args.num_frames
    config.tubelet_size = data_args.tubelet_size
    config.image_size = data_args.resize_size

    if model_args.model_name_or_path:
        model = ViTNepaVideoForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

        # Ensure mask token stays enabled even when loading older checkpoints.
        backbone_state = model.vit_nepa.state_dict()
        model.vit_nepa = model.vit_nepa.__class__(config, use_mask_token=True)
        model.post_init()
        missing_keys, unexpected_keys = model.vit_nepa.load_state_dict(backbone_state, strict=False)
        if missing_keys or unexpected_keys:
            logger.warning(
                f"Mask-token backbone reload had missing keys={missing_keys}, unexpected keys={unexpected_keys}"
            )
    else:
        logger.info("Training new 3D NEPA model from scratch")
        model = ViTNepaVideoForPreTraining(config)

    def train_transforms(example_batch):
        video_entries = _get_video_entries(example_batch, data_args.video_column_name)
        pixel_values = [
            _video_to_clip_tensor(_resolve_video_path(video_item), data_args.num_frames, data_args.resize_size, True)
            for video_item in video_entries
        ]
        example_batch["pixel_values"] = pixel_values
        return example_batch

    def val_transforms(example_batch):
        video_entries = _get_video_entries(example_batch, data_args.video_column_name)
        pixel_values = [
            _video_to_clip_tensor(_resolve_video_path(video_item), data_args.num_frames, data_args.resize_size, False)
            for video_item in video_entries
        ]
        example_batch["pixel_values"] = pixel_values
        return example_batch

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
        dataset["validation"].set_transform(val_transforms)

    trainer = EnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=None,
        data_collator=collate_fn,
        embed_lr=model_args.embed_lr,
        callbacks=[LossOnlyCallback()],
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()

    if training_args.push_to_hub:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "embedded-prediction",
            "dataset": data_args.train_dir,
            "tags": ["embedded-prediction", "video", "3d"],
        }
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()