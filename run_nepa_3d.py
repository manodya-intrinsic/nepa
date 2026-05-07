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
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import ClassLabel, Video, load_dataset, load_from_disk
from PIL import Image
from torchvision.io import read_video
from torchvision.transforms import CenterCrop, Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor

import transformers
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from models.vit_nepa.modeling_vit_nepa_3d import ViTNepaVideoForPreTraining
from models.vit_nepa.configuration_vit_nepa import ViTNepaConfig
from run_nepa import EnhancedTrainer


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of a dataset from the hub, or a local dataset path to load."},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training videos."})
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
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError(
                "You must specify either a dataset name from the hub or a train and/or validation directory."
            )


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
    if isinstance(video_entry, dict):
        if video_entry.get("path") is not None:
            return video_entry["path"]
        raise ValueError(f"Video entry does not contain a path: {video_entry}")
    if hasattr(video_entry, "path") and video_entry.path is not None:
        return video_entry.path
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
    try:
        video, _, _ = read_video(video_path, pts_unit="sec")
    except Exception as e:
        logger.warning(f"Failed to read video {video_path}: {e}. Returning None.")
        return None
    
    if video.ndim != 4:
        logger.warning(f"Expected video tensor with 4 dims [T,H,W,C], got shape {tuple(video.shape)} for {video_path}. Returning None.")
        return None

    if video.shape[0] == 0:
        logger.warning(f"Video {video_path} has 0 frames. Skipping.")
        return None

    # torchvision returns [T, H, W, C]. Convert to [T, C, H, W].
    if video.shape[-1] == 3:
        video = video.permute(0, 3, 1, 2)

    frame_indices = _sample_frame_indices(video.shape[0], num_frames, train=train)
    video = video[frame_indices]

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

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

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

    if data_args.dataset_name is not None:
        if data_args.load_from_disk:
            dataset = load_from_disk(data_args.dataset_name, keep_in_memory=data_args.keep_in_memory)
        else:
            dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset("videofolder", data_files=data_files, cache_dir=model_args.cache_dir)

    if data_args.video_column_name not in (dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names):
        raise ValueError(f"--video_column_name {data_args.video_column_name} not found in the dataset columns.")

    # Normalize video feature to a path/bytes struct for easy local decoding.
    dataset = dataset.cast_column(data_args.video_column_name, Video(decode=False))

    def collate_fn(examples):
        # Filter out examples with None pixel_values (corrupted videos).
        valid_examples = [ex for ex in examples if ex["pixel_values"] is not None]
        if not valid_examples:
            # Keep training moving by emitting a single zero clip instead of a size-0 batch.
            logger.warning("All examples in batch are corrupted. Returning a zero clip fallback.")
            pixel_values = torch.zeros(
                (1, 3, data_args.num_frames, data_args.resize_size, data_args.resize_size),
                dtype=torch.float32,
            )
        else:
            pixel_values = torch.stack([example["pixel_values"] for example in valid_examples])

        batch_size = pixel_values.shape[0]
        num_tokens = 1568
        num_masked = 1411
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
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "embedded-prediction",
        "dataset": data_args.dataset_name,
        "tags": ["embedded-prediction", "video", "3d"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()