# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Training entrypoint for 3D NEPA on video clips.

# This runner keeps the existing NEPA objective (next-embedding prediction) but swaps the input pipeline
# from 2D images to 3D video clips with tubelet embedding.
# """

# import logging
# from collections import deque
# import os
# import random
# import sys
# import warnings
# from dataclasses import dataclass, field
# from typing import Optional

# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# os.environ.setdefault("WANDB_DISABLED", "true")
# os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# import torch
# import torch.nn.functional as F
# from datasets import ClassLabel, Video, load_dataset
# from PIL import Image
# try:
#     from decord import VideoReader, cpu as decord_cpu
# except Exception:
#     VideoReader = None
#     decord_cpu = None
# from torchvision.transforms import CenterCrop, Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor

# import transformers
# from transformers import HfArgumentParser, TrainerCallback, TrainingArguments, set_seed
# from transformers.trainer_utils import get_last_checkpoint

# from models.vit_nepa.modeling_vit_nepa_3d import ViTNepaVideoForPreTraining
# from models.vit_nepa.modeling_vit_nepa import prediction_loss
# from models.vit_nepa.configuration_vit_nepa import ViTNepaConfig
# from run_nepa import EnhancedTrainer


# logger = logging.getLogger(__name__)


# def _configure_quiet_warnings():
#     warnings.filterwarnings(
#         "ignore",
#         message="The video decoding and encoding capabilities of torchvision are deprecated.*",
#         category=UserWarning,
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message="Parameter 'transform'.*couldn't be hashed properly.*",
#         category=UserWarning,
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message=".*libtorchcodec.*",
#         category=UserWarning,
#     )
#     warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


# def _configure_quiet_logging():
#     logging.getLogger().setLevel(logging.WARNING)
#     logging.getLogger("transformers").setLevel(logging.ERROR)
#     logging.getLogger("datasets").setLevel(logging.ERROR)
#     logging.getLogger("torchvision").setLevel(logging.ERROR)
#     logging.getLogger("wandb").setLevel(logging.ERROR)
#     logger.setLevel(logging.INFO)  # Keep our logger at INFO to see debug messages


# class VideoPretrainTrainer(EnhancedTrainer):
#     def log(self, logs, start_time=None):
#         if logs and self.is_world_process_zero():
#             loss = logs.get("loss")
#             if loss is not None:
#                 epoch = self.state.epoch if self.state.epoch is not None else 0.0
#                 lr = logs.get("learning_rate", logs.get("lr", float("nan")))
#                 grad_norm = logs.get("grad_norm", float("nan"))
#                 # Embedding variance indicates feature diversity (collapse = very low variance)
#                 emb_var = getattr(self.model, "_last_embedding_variance", float("nan"))
#                 print(
#                     f"Step {self.state.global_step} | Epoch {epoch:.2f} | Loss {float(loss):.4f} | "
#                     f"Sim {-float(loss):.4f} | LR {float(lr):.2e} | Grad {float(grad_norm):.2f} | Var {float(emb_var):.4f}"
#                 )

#         return super().log(logs, start_time=start_time)

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         pixel_values = inputs["pixel_values"]
#         bool_masked_pos = inputs["bool_masked_pos"]

#         outputs = model.vit_nepa(
#             pixel_values=pixel_values,
#             bool_masked_pos=bool_masked_pos,
#         )

#         sequence_input = outputs.input_embedding
#         sequence_output = outputs.last_hidden_state

#         # Mirror prediction_loss(shift=True) tensors and guard against accidental leakage.
#         pred = F.normalize(sequence_output[:, :-1, :], dim=-1)
#         target = F.normalize(sequence_input[:, 1:, :].detach(), dim=-1)
#         if torch.allclose(pred, target, atol=1e-3):
#             raise AssertionError(
#                 "pred and target are allclose(atol=1e-3) before loss; check for data leakage or wiring bugs."
#             )

#         loss = prediction_loss(sequence_input, sequence_output).float()

#         with torch.no_grad():
#             token_states = sequence_output[:, 1:, :].float() if sequence_output.shape[1] > 1 else sequence_output.float()
#             embedding_variance = token_states.var(dim=0, unbiased=False).mean()
#             model._last_embedding_variance = float(embedding_variance.detach().cpu())

#         return (loss, outputs) if return_outputs else loss


# @dataclass
# class DataTrainingArguments:
#     train_dir: Optional[str] = field(
#         default="/content/hmdb51_local",
#         metadata={"help": "A folder containing the training videos. Defaults to /content/hmdb51_local."},
#     )
#     validation_dir: Optional[str] = field(
#         default=None,
#         metadata={"help": "A folder containing validation videos."},
#     )
#     train_val_split: Optional[float] = field(
#         default=0.15,
#         metadata={"help": "Percent to split off of train for validation."},
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={"help": "Truncate train set for debugging."},
#     )
#     max_eval_samples: Optional[int] = field(
#         default=None,
#         metadata={"help": "Truncate eval set for debugging."},
#     )
#     video_column_name: str = field(
#         default="video",
#         metadata={"help": "The name of the dataset column containing the video data. Defaults to 'video'."},
#     )
#     resize_size: int = field(
#         default=224,
#         metadata={"help": "Spatial size used for video preprocessing."},
#     )
#     num_frames: int = field(
#         default=16,
#         metadata={"help": "Number of frames sampled per clip."},
#     )
#     tubelet_size: int = field(
#         default=2,
#         metadata={"help": "Temporal tubelet size used by the 3D embedding."},
#     )

#     # NEW: configurable masking
#     mask_ratio: float = field(
#         default=0.75,
#         metadata={
#             "help": "Masking ratio for 3D NEPA pretraining. Try 0.50, 0.75, or 0.90."
#         },
#     )
#     mask_type: str = field(
#         default="tube",
#         metadata={
#             "help": "Masking strategy. Use 'tube' for VideoMAE-style tube masking or 'random' for random 3D token masking."
#         },
#     )

#     load_from_disk: bool = field(
#         default=False,
#         metadata={"help": "Load from disk."},
#     )
#     keep_in_memory: bool = field(
#         default=False,
#         metadata={"help": "Keep dataset in memory."},
#     )

#     def __post_init__(self):
#         if self.train_dir is None:
#             raise ValueError("You must specify a training directory.")

#         if not 0.0 <= self.mask_ratio <= 1.0:
#             raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {self.mask_ratio}")

#         if self.mask_type not in {"tube", "random"}:
#             raise ValueError(f"mask_type must be 'tube' or 'random', got {self.mask_type}")


# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(
#         default=None,
#         metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
#     )
#     config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path."})
#     cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store downloaded models."})
#     model_revision: str = field(default="main", metadata={"help": "The specific model version to use."})
#     token: str = field(default=None, metadata={"help": "HF token for private files."})
#     trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code from the Hub."})
#     ignore_mismatched_sizes: bool = field(default=False, metadata={"help": "Allow mismatched weights."})
#     embed_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate for embeddings."})


# def _resolve_video_path(video_entry) -> str:
#     if hasattr(video_entry, "path") and video_entry.path is not None:
#         return video_entry.path
#     if hasattr(video_entry, "local_path") and video_entry.local_path is not None:
#         return video_entry.local_path
#     if isinstance(video_entry, dict):
#         if video_entry.get("path") is not None:
#             return video_entry["path"]
#         if video_entry.get("local_path") is not None:
#             return video_entry["local_path"]
#         raise ValueError(f"Video entry does not contain a path: {video_entry}")
#     return str(video_entry)


# def _get_video_entries(example_batch, column_name: str):
#     if column_name in example_batch:
#         return example_batch[column_name]
#     for fallback_name in ("video", "path", "file", "video_file"):
#         if fallback_name in example_batch:
#             logger.warning(
#                 f"--video_column_name {column_name!r} was not present in the batch; using {fallback_name!r} instead."
#             )
#             return example_batch[fallback_name]
#     raise KeyError(f"Could not find a video column in batch keys: {list(example_batch.keys())}")


# def _sample_frame_indices(total_frames: int, num_frames: int, train: bool) -> torch.Tensor:
#     if total_frames <= 0:
#         raise ValueError("Video contains no frames.")
#     if total_frames >= num_frames:
#         if train:
#             max_start = total_frames - num_frames
#             start = random.randint(0, max_start) if max_start > 0 else 0
#             return torch.arange(start, start + num_frames)
#         return torch.linspace(0, total_frames - 1, num_frames).round().long()

#     # Repeat the last frame to reach the target length.
#     base = torch.arange(total_frames)
#     pad = torch.full((num_frames - total_frames,), total_frames - 1, dtype=torch.long)
#     return torch.cat([base, pad], dim=0)


# def _video_to_clip_tensor(video_path: str, num_frames: int, spatial_size: int, train: bool, debug: bool = False) -> torch.Tensor:
#     if VideoReader is None or decord_cpu is None:
#         logger.warning("Decord is not available. Returning None for video decoding.")
#         return None

#     try:
#         vr = VideoReader(video_path, ctx=decord_cpu(0))
#         total_frames = len(vr)
#         if total_frames == 0:
#             logger.warning(f"Video {video_path} has 0 frames. Skipping.")
#             return None
#         frame_indices = _sample_frame_indices(total_frames, num_frames, train=train)
#         frames_np = vr.get_batch(frame_indices.cpu().numpy()).asnumpy()  # (T, H, W, C)
#         video = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)
        
#         if debug:
#             logger.info(
#                 f"[DECODE] {video_path}: {total_frames} total frames, sampled {len(frame_indices)}, "
#                 f"shape={video.shape}, dtype={video.dtype}, range=[{video.min():.1f}, {video.max():.1f}]"
#             )
        
#     except Exception as e:
#         logger.warning(f"Decord failed for {video_path}: {e}. Returning None.")
#         return None

#     # At this point `video` is a torch tensor shaped (T, C, H, W)
#     if video.ndim != 4 or video.shape[0] == 0:
#         logger.warning(f"Decoded video not valid for {video_path}. Skipping.")
#         return None

#     if train:
#         frame_transform = Compose(
#             [
#                 RandomResizedCrop(spatial_size),
#                 RandomHorizontalFlip(),
#                 ToTensor(),
#                 Lambda(lambda x: x),
#             ]
#         )
#     else:
#         frame_transform = Compose(
#             [
#                 Resize(spatial_size),
#                 CenterCrop(spatial_size),
#                 ToTensor(),
#                 Lambda(lambda x: x),
#             ]
#         )

#     processed_frames = []
#     for frame in video:
#         pil_frame = Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
#         processed_frames.append(frame_transform(pil_frame))

#     # Stack to [C, T, H, W] for the 3D NEPA model.
#     clip = torch.stack(processed_frames, dim=1)
#     return clip


# def _classwise_train_validation_split(dataset, split_ratio: float, seed: int):
#     if "label" in dataset.features and isinstance(dataset.features["label"], ClassLabel):
#         return dataset.train_test_split(split_ratio, seed=seed, stratify_by_column="label")
#     logger.warning(
#         "Dataset does not expose a ClassLabel 'label' column; falling back to a non-stratified split."
#     )
#     return dataset.train_test_split(split_ratio, seed=seed)


# def make_tube_mask(batch_size, num_frames, image_size, patch_size, tubelet_size, mask_ratio):
#     t_tokens = num_frames // tubelet_size
#     h_tokens = image_size // patch_size
#     w_tokens = image_size // patch_size

#     spatial_tokens = h_tokens * w_tokens
#     num_masked_spatial = int(spatial_tokens * mask_ratio)

#     masks = torch.zeros((batch_size, t_tokens, h_tokens, w_tokens), dtype=torch.bool)

#     for b in range(batch_size):
#         spatial_mask = torch.zeros(spatial_tokens, dtype=torch.bool)
#         masked_idx = torch.randperm(spatial_tokens)[:num_masked_spatial]
#         spatial_mask[masked_idx] = True

#         spatial_mask = spatial_mask.view(h_tokens, w_tokens)

#         # same H/W mask repeated through time = tube masking
#         masks[b] = spatial_mask.unsqueeze(0).expand(t_tokens, h_tokens, w_tokens)

#     return masks.flatten(1)

# def main():
#     parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     _configure_quiet_warnings()

#     logging.basicConfig(
#         format="%(message)s",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )

#     training_args.disable_tqdm = True
#     training_args.report_to = []
#     _configure_quiet_logging()
#     transformers.utils.logging.set_verbosity_error()
#     transformers.utils.logging.disable_default_handler()
#     if hasattr(transformers.utils.logging, "disable_explicit_format"):
#         transformers.utils.logging.disable_explicit_format()

#     # Prevent Trainer from removing video/label columns before dataset transforms can access them
#     training_args.remove_unused_columns = False

#     last_checkpoint = None
#     if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
#         last_checkpoint = get_last_checkpoint(training_args.output_dir)
#         if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
#             raise ValueError(
#                 f"Output directory ({training_args.output_dir}) already exists and is not empty. "
#                 "Use --overwrite_output_dir to overcome."
#             )

#     set_seed(training_args.seed)

#     data_files = {"train": os.path.join(data_args.train_dir, "**")}
#     if data_args.validation_dir is not None:
#         data_files["validation"] = os.path.join(data_args.validation_dir, "**")
#     dataset = load_dataset("videofolder", data_files=data_files, cache_dir=model_args.cache_dir)

#     # Keep the dataset from auto-decoding videos with TorchCodec; Decord handles decoding in the transform.
#     dataset = dataset.cast_column(data_args.video_column_name, Video(decode=False))

#     if data_args.video_column_name not in (dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names):
#         raise ValueError(f"--video_column_name {data_args.video_column_name} not found in the dataset columns.")

#     # ========================================================================
#     # DEBUG: Dataset structure and first-batch inspection
#     # ========================================================================
#     if training_args.do_train and "train" in dataset:
#         train_dataset = dataset["train"]
#         print("\n" + "="*70)
#         print("[DEBUG] DATASET STRUCTURE")
#         print("="*70)
#         print(f"Train dataset size: {len(train_dataset)}")
#         print(f"Train dataset columns: {train_dataset.column_names}")
#         print(f"Train dataset features: {train_dataset.features}")
        
#         # Peek at first 5 video paths
#         print("\n[DEBUG] First 5 video entries (raw):")
#         for i in range(min(5, len(train_dataset))):
#             entry = train_dataset[i]
#             video_item = entry.get(data_args.video_column_name)
#             print(f"  [{i}] type={type(video_item).__name__}, value={video_item}")
#             try:
#                 resolved_path = _resolve_video_path(video_item)
#                 print(f"       -> resolved to: {resolved_path}")
#             except Exception as e:
#                 print(f"       -> FAILED to resolve: {e}")
        
#         if training_args.do_eval and "validation" in dataset:
#             val_dataset = dataset["validation"]
#             print(f"\nValidation dataset size: {len(val_dataset)}")

#     debug_preview_limit = 3
#     debug_preview_state = {"train": 0, "val": 0, "collate": 0}

#     def collate_fn(examples):
#         # Filter out examples with None pixel_values (corrupted videos).
#         valid_examples = [ex for ex in examples if ex["pixel_values"] is not None]
#         if not valid_examples:
#             # Keep training moving by emitting a single zero clip instead of a size-0 batch.
#             pixel_values = torch.zeros(
#                 (1, 3, data_args.num_frames, data_args.resize_size, data_args.resize_size),
#                 dtype=torch.float32,
#             )
#         else:
#             pixel_values = torch.stack([example["pixel_values"] for example in valid_examples])

#         debug_this_batch = debug_preview_state["collate"] < debug_preview_limit
#         debug_preview_state["collate"] += 1

#         if debug_this_batch:
#             print(f"\n[DEBUG COLLATE] Batch shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
#             print(f"[DEBUG COLLATE] pixel_values range: [{pixel_values.min():.4f}, {pixel_values.max():.4f}]")
#             print(f"[DEBUG COLLATE] pixel_values mean: {pixel_values.mean():.4f}, std: {pixel_values.std():.4f}")

#             if torch.isnan(pixel_values).any():
#                 print("[DEBUG COLLATE] ⚠️  WARNING: pixel_values contains NaN!")
#             if torch.isinf(pixel_values).any():
#                 print("[DEBUG COLLATE] ⚠️  WARNING: pixel_values contains Inf!")
#             if (pixel_values == 0).all():
#                 print("[DEBUG COLLATE] ⚠️  WARNING: pixel_values is ALL ZEROS!")
        
#         batch_size = pixel_values.shape[0]
#         num_tokens = 2048
#         num_masked = 1843
#         bool_masked_pos = torch.zeros((batch_size, num_tokens), dtype=torch.bool)

#         for i in range(batch_size):
#             masked_indices = torch.randperm(num_tokens)[:num_masked]
#             bool_masked_pos[i, masked_indices] = True

#             num_masked_per_sample = bool_masked_pos[i].sum().item()
#             print(f"[DEBUG COLLATE] bool_masked_pos shape: {bool_masked_pos.shape}, num_true: {num_masked_per_sample} per sample")
#             print("="*70)

#         return {"pixel_values": pixel_values, "bool_masked_pos": bool_masked_pos}

#     data_args.train_val_split = None if "validation" in dataset else data_args.train_val_split
#     if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
#         split = _classwise_train_validation_split(dataset["train"], data_args.train_val_split, training_args.seed)
#         dataset["train"] = split["train"]
#         dataset["validation"] = split["test"]

#     config = ViTNepaConfig.from_pretrained(
#         model_args.config_name or model_args.model_name_or_path,
#         cache_dir=model_args.cache_dir,
#         revision=model_args.model_revision,
#         token=model_args.token,
#         trust_remote_code=model_args.trust_remote_code,
#     )
#     config.num_frames = data_args.num_frames
#     config.tubelet_size = data_args.tubelet_size
#     config.image_size = data_args.resize_size

#     if model_args.model_name_or_path:
#         model = ViTNepaVideoForPreTraining.from_pretrained(
#             model_args.model_name_or_path,
#             from_tf=bool(".ckpt" in model_args.model_name_or_path),
#             config=config,
#             cache_dir=model_args.cache_dir,
#             revision=model_args.model_revision,
#             token=model_args.token,
#             trust_remote_code=model_args.trust_remote_code,
#             ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
#         )

#         # Ensure mask token stays enabled even when loading older checkpoints.
#         backbone_state = model.vit_nepa.state_dict()
#         model.vit_nepa = model.vit_nepa.__class__(config, use_mask_token=True)
#         model.post_init()
#         missing_keys, unexpected_keys = model.vit_nepa.load_state_dict(backbone_state, strict=False)
#         if missing_keys or unexpected_keys:
#             logger.warning(
#                 f"Mask-token backbone reload had missing keys={missing_keys}, unexpected keys={unexpected_keys}"
#             )
#     else:
#         logger.info("Training new 3D NEPA model from scratch")
#         model = ViTNepaVideoForPreTraining(config)

#     def train_transforms(example_batch):
#         video_entries = _get_video_entries(example_batch, data_args.video_column_name)
#         debug_this_batch = debug_preview_state["train"] < debug_preview_limit
#         debug_preview_state["train"] += 1
#         pixel_values = [
#             _video_to_clip_tensor(
#                 _resolve_video_path(video_item),
#                 data_args.num_frames,
#                 data_args.resize_size,
#                 True,
#                 debug=debug_this_batch,
#             )
#             for video_item in video_entries
#         ]
        
#         example_batch["pixel_values"] = pixel_values
#         return example_batch

#     def val_transforms(example_batch):
#         video_entries = _get_video_entries(example_batch, data_args.video_column_name)
#         debug_this_batch = debug_preview_state["val"] < debug_preview_limit
#         debug_preview_state["val"] += 1
#         pixel_values = [
#             _video_to_clip_tensor(
#                 _resolve_video_path(video_item),
#                 data_args.num_frames,
#                 data_args.resize_size,
#                 False,
#                 debug=debug_this_batch,
#             )
#             for video_item in video_entries
#         ]
        
#         example_batch["pixel_values"] = pixel_values
#         return example_batch

#     if training_args.do_train:
#         if "train" not in dataset:
#             raise ValueError("--do_train requires a train dataset")
#         if data_args.max_train_samples is not None:
#             dataset["train"] = dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
#         dataset["train"].set_transform(train_transforms)

#     if training_args.do_eval:
#         if "validation" not in dataset:
#             raise ValueError("--do_eval requires a validation dataset")
#         if data_args.max_eval_samples is not None:
#             dataset["validation"] = dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
#         dataset["validation"].set_transform(val_transforms)

#     trainer = VideoPretrainTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset["train"] if training_args.do_train else None,
#         eval_dataset=dataset["validation"] if training_args.do_eval else None,
#         processing_class=None,
#         data_collator=collate_fn,
#         embed_lr=model_args.embed_lr,
#     )

#     if training_args.do_train:
#         checkpoint = None
#         if training_args.resume_from_checkpoint is not None:
#             checkpoint = training_args.resume_from_checkpoint
#         elif last_checkpoint is not None:
#             checkpoint = last_checkpoint
#         train_result = trainer.train(resume_from_checkpoint=checkpoint)
#         trainer.save_model()
#         trainer.save_state()

#     if training_args.push_to_hub:
#         kwargs = {
#             "finetuned_from": model_args.model_name_or_path,
#             "tasks": "embedded-prediction",
#             "dataset": data_args.train_dir,
#             "tags": ["embedded-prediction", "video", "3d"],
#         }
#         trainer.push_to_hub(**kwargs)


# if __name__ == "__main__":
#     main()

# Licensed under the Apache License, Version 2.0

"""Training entrypoint for 3D NEPA on video clips."""

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
import torch.nn.functional as F
from datasets import ClassLabel, Video, load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

try:
    from decord import VideoReader, cpu as decord_cpu
except Exception:
    VideoReader = None
    decord_cpu = None

import transformers
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from models.vit_nepa.configuration_vit_nepa import ViTNepaConfig
from models.vit_nepa.modeling_vit_nepa import prediction_loss
from models.vit_nepa.modeling_vit_nepa_3d import ViTNepaVideoForPreTraining
from run_nepa import EnhancedTrainer


logger = logging.getLogger(__name__)


def _configure_quiet_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


def _configure_quiet_logging():
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)


class VideoPretrainTrainer(EnhancedTrainer):
    def log(self, logs, start_time=None):
        if logs and self.is_world_process_zero():
            loss = logs.get("loss")
            if loss is not None:
                epoch = self.state.epoch if self.state.epoch is not None else 0.0
                lr = logs.get("learning_rate", float("nan"))
                grad_norm = logs.get("grad_norm", float("nan"))
                emb_var = getattr(self.model, "_last_embedding_variance", float("nan"))

                print(
                    f"Step {self.state.global_step} | Epoch {epoch:.2f} | "
                    f"Loss {float(loss):.4f} | Sim {-float(loss):.4f} | "
                    f"LR {float(lr):.2e} | Grad {float(grad_norm):.2f} | "
                    f"Var {float(emb_var):.4f}"
                )

        return super().log(logs, start_time=start_time)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pixel_values = inputs["pixel_values"]
        bool_masked_pos = inputs["bool_masked_pos"]

        outputs = model.vit_nepa(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
        )

        sequence_input = outputs.input_embedding
        sequence_output = outputs.last_hidden_state

        pred = F.normalize(sequence_output[:, :-1, :], dim=-1)
        target = F.normalize(sequence_input[:, 1:, :].detach(), dim=-1)

        if torch.allclose(pred, target, atol=1e-3):
            raise AssertionError(
                "pred and target are too similar before loss. Check leakage/wiring."
            )

        loss = prediction_loss(sequence_input, sequence_output).float()

        with torch.no_grad():
            token_states = sequence_output[:, 1:, :].float()
            model._last_embedding_variance = float(
                token_states.var(dim=0, unbiased=False).mean().detach().cpu()
            )

        return (loss, outputs) if return_outputs else loss


@dataclass
class DataTrainingArguments:
    train_dir: Optional[str] = field(
        default="/content/hmdb51_local/train",
        metadata={"help": "Folder containing training videos."},
    )
    validation_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Folder containing validation videos."},
    )
    train_val_split: Optional[float] = field(
        default=0.15,
        metadata={"help": "Percent to split off train if no validation dir is given."},
    )
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    video_column_name: str = field(default="video")
    resize_size: int = field(default=224)
    num_frames: int = field(default=16)
    tubelet_size: int = field(default=2)

    mask_ratio: float = field(
        default=0.75,
        metadata={"help": "Masking ratio. Try 0.50, 0.75, 0.90."},
    )
    mask_type: str = field(
        default="tube",
        metadata={"help": "Mask type: 'tube' or 'random'."},
    )

    def __post_init__(self):
        if self.train_dir is None:
            raise ValueError("You must specify --train_dir")

        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0,1], got {self.mask_ratio}")

        if self.mask_type not in {"tube", "random"}:
            raise ValueError(f"mask_type must be 'tube' or 'random', got {self.mask_type}")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    token: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)
    embed_lr: Optional[float] = field(default=None)


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


def _get_video_entries(example_batch, column_name: str):
    if column_name in example_batch:
        return example_batch[column_name]

    for fallback_name in ("video", "path", "file", "video_file"):
        if fallback_name in example_batch:
            return example_batch[fallback_name]

    raise KeyError(f"Could not find video column in batch keys: {list(example_batch.keys())}")


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


def _video_to_clip_tensor(
    video_path: str,
    num_frames: int,
    spatial_size: int,
    train: bool,
    debug: bool = False,
):
    if VideoReader is None or decord_cpu is None:
        logger.warning("Decord is not installed.")
        return None

    try:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        total_frames = len(vr)

        if total_frames <= 0:
            return None

        frame_indices = _sample_frame_indices(total_frames, num_frames, train=train)
        frames_np = vr.get_batch(frame_indices.cpu().numpy()).asnumpy()
        video = torch.from_numpy(frames_np).permute(0, 3, 1, 2)

    except Exception as e:
        logger.warning(f"Decord failed for {video_path}: {e}")
        return None

    if video.ndim != 4 or video.shape[0] == 0:
        return None

    normalize = Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        frame_transform = Compose(
            [
                RandomResizedCrop(spatial_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
    else:
        frame_transform = Compose(
            [
                Resize(spatial_size),
                CenterCrop(spatial_size),
                ToTensor(),
                normalize,
            ]
        )

    processed_frames = []
    for frame in video:
        pil_frame = Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
        processed_frames.append(frame_transform(pil_frame))

    clip = torch.stack(processed_frames, dim=1)

    if debug:
        print(
            f"[DECODE] {video_path} -> clip={tuple(clip.shape)}, "
            f"range=[{clip.min():.4f}, {clip.max():.4f}], "
            f"mean={clip.mean():.4f}, std={clip.std():.4f}"
        )

    return clip


def make_bool_masked_pos(
    batch_size: int,
    num_frames: int,
    image_size: int,
    patch_size: int,
    tubelet_size: int,
    mask_ratio: float,
    mask_type: str = "tube",
) -> torch.BoolTensor:
    t_tokens = num_frames // tubelet_size
    h_tokens = image_size // patch_size
    w_tokens = image_size // patch_size
    num_tokens = t_tokens * h_tokens * w_tokens

    if mask_ratio <= 0.0:
        return torch.zeros((batch_size, num_tokens), dtype=torch.bool)

    if mask_type == "random":
        num_masked = int(num_tokens * mask_ratio)
        mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool)

        for b in range(batch_size):
            idx = torch.randperm(num_tokens)[:num_masked]
            mask[b, idx] = True

        return mask

    if mask_type == "tube":
        spatial_tokens = h_tokens * w_tokens
        num_masked_spatial = int(spatial_tokens * mask_ratio)

        mask = torch.zeros(
            (batch_size, t_tokens, h_tokens, w_tokens),
            dtype=torch.bool,
        )

        for b in range(batch_size):
            spatial_mask = torch.zeros(spatial_tokens, dtype=torch.bool)
            idx = torch.randperm(spatial_tokens)[:num_masked_spatial]
            spatial_mask[idx] = True
            spatial_mask = spatial_mask.view(h_tokens, w_tokens)

            mask[b] = spatial_mask.unsqueeze(0).expand(t_tokens, h_tokens, w_tokens)

        return mask.flatten(1)

    raise ValueError(f"Unknown mask_type: {mask_type}")


def _classwise_train_validation_split(dataset, split_ratio: float, seed: int):
    if "label" in dataset.features and isinstance(dataset.features["label"], ClassLabel):
        return dataset.train_test_split(
            test_size=split_ratio,
            seed=seed,
            stratify_by_column="label",
        )

    return dataset.train_test_split(test_size=split_ratio, seed=seed)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    _configure_quiet_warnings()

    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    training_args.disable_tqdm = True
    training_args.report_to = []
    training_args.remove_unused_columns = False

    _configure_quiet_logging()
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_default_handler()

    last_checkpoint = None

    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory {training_args.output_dir} already exists. "
                "Use --overwrite_output_dir."
            )

    set_seed(training_args.seed)

    data_files = {"train": os.path.join(data_args.train_dir, "**")}

    if data_args.validation_dir is not None:
        data_files["validation"] = os.path.join(data_args.validation_dir, "**")

    dataset = load_dataset(
        "videofolder",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    dataset = dataset.cast_column(data_args.video_column_name, Video(decode=False))

    if "validation" not in dataset and data_args.train_val_split:
        split = _classwise_train_validation_split(
            dataset["train"],
            data_args.train_val_split,
            training_args.seed,
        )
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

    print("\n" + "=" * 70)
    print("[3D NEPA CONFIG]")
    print(f"num_frames     = {data_args.num_frames}")
    print(f"tubelet_size   = {data_args.tubelet_size}")
    print(f"image_size     = {data_args.resize_size}")
    print(f"patch_size     = {config.patch_size}")
    print(f"mask_ratio     = {data_args.mask_ratio}")
    print(f"mask_type      = {data_args.mask_type}")
    print("=" * 70 + "\n")

    if model_args.model_name_or_path:
        model = ViTNepaVideoForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

        backbone_state = model.vit_nepa.state_dict()
        model.vit_nepa = model.vit_nepa.__class__(config, use_mask_token=True)
        model.post_init()
        model.vit_nepa.load_state_dict(backbone_state, strict=False)

    else:
        model = ViTNepaVideoForPreTraining(config)

    debug_limit = 3
    debug_state = {"train": 0, "val": 0, "collate": 0}

    def train_transforms(example_batch):
        video_entries = _get_video_entries(example_batch, data_args.video_column_name)
        debug = debug_state["train"] < debug_limit
        debug_state["train"] += 1

        example_batch["pixel_values"] = [
            _video_to_clip_tensor(
                _resolve_video_path(video_item),
                data_args.num_frames,
                data_args.resize_size,
                train=True,
                debug=debug,
            )
            for video_item in video_entries
        ]

        return example_batch

    def val_transforms(example_batch):
        video_entries = _get_video_entries(example_batch, data_args.video_column_name)
        debug = debug_state["val"] < debug_limit
        debug_state["val"] += 1

        example_batch["pixel_values"] = [
            _video_to_clip_tensor(
                _resolve_video_path(video_item),
                data_args.num_frames,
                data_args.resize_size,
                train=False,
                debug=debug,
            )
            for video_item in video_entries
        ]

        return example_batch

    def collate_fn(examples):
        valid_examples = [ex for ex in examples if ex["pixel_values"] is not None]

        if not valid_examples:
            pixel_values = torch.zeros(
                (1, 3, data_args.num_frames, data_args.resize_size, data_args.resize_size),
                dtype=torch.float32,
            )
        else:
            pixel_values = torch.stack([example["pixel_values"] for example in valid_examples])

        batch_size = pixel_values.shape[0]

        bool_masked_pos = make_bool_masked_pos(
            batch_size=batch_size,
            num_frames=data_args.num_frames,
            image_size=data_args.resize_size,
            patch_size=config.patch_size,
            tubelet_size=data_args.tubelet_size,
            mask_ratio=data_args.mask_ratio,
            mask_type=data_args.mask_type,
        )

        debug = debug_state["collate"] < debug_limit
        debug_state["collate"] += 1

        if debug:
            print("\n[DEBUG COLLATE]")
            print(f"pixel_values shape = {tuple(pixel_values.shape)}")
            print(f"pixel range        = [{pixel_values.min():.4f}, {pixel_values.max():.4f}]")
            print(f"pixel mean/std     = {pixel_values.mean():.4f} / {pixel_values.std():.4f}")
            print(f"mask shape         = {tuple(bool_masked_pos.shape)}")
            print(f"masked per sample  = {bool_masked_pos.sum(dim=1).tolist()}")
            print("=" * 70)

        return {
            "pixel_values": pixel_values,
            "bool_masked_pos": bool_masked_pos,
        }

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_train_samples))
            )

        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval and "validation" in dataset:
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_eval_samples))
            )

        dataset["validation"].set_transform(val_transforms)

    trainer = VideoPretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval and "validation" in dataset else None,
        processing_class=None,
        data_collator=collate_fn,
        embed_lr=model_args.embed_lr,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()
        trainer.save_state()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)


if __name__ == "__main__":
    main()