import collections.abc
from typing import Optional, cast

import torch
from torch import nn

from transformers.modeling_outputs import BaseModelOutput
from transformers.utils.generic import can_return_tuple, check_model_inputs

from .configuration_vit_nepa import ViTNepaConfig
from .modeling_vit_nepa import (
	BaseModelOutputWithEmbedding,
	EmbeddedModelingOutput,
	ViTNepaEncoder,
	ViTNepaPreTrainedModel,
	prediction_loss,
)


class ViTNepaVideoPatchEmbeddings(nn.Module):
	"""Video to tubelet token embeddings using Conv3d."""

	def __init__(self, config: ViTNepaConfig):
		super().__init__()
		image_size = config.image_size
		patch_size = config.patch_size
		num_channels = config.num_channels
		hidden_size = config.hidden_size
		tubelet_size = config.tubelet_size

		image_size_tuple = tuple(image_size) if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
		patch_size_tuple = tuple(patch_size) if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
		image_height, image_width = int(image_size_tuple[0]), int(image_size_tuple[1])
		patch_height, patch_width = int(patch_size_tuple[0]), int(patch_size_tuple[1])

		self.image_size = (image_height, image_width)
		self.patch_size = (patch_height, patch_width)
		self.num_channels = num_channels
		self.num_frames = config.num_frames
		self.tubelet_size = tubelet_size

		self.projection = nn.Conv3d(
			num_channels,
			hidden_size,
			kernel_size=(tubelet_size, patch_height, patch_width),
			stride=(tubelet_size, patch_height, patch_width),
		)

	def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
		batch_size, num_channels, num_frames, height, width = pixel_values.shape
		if num_channels != self.num_channels:
			raise ValueError(
				f"Input channel dimension ({num_channels}) doesn't match model ({self.num_channels})."
			)
		if height != self.image_size[0] or width != self.image_size[1]:
			raise ValueError(
				f"Input video spatial size ({height}*{width}) doesn't match model"
				f" ({self.image_size[0]}*{self.image_size[1]})."
			)
		if num_frames % self.tubelet_size != 0:
			raise ValueError(
				f"Input frame length ({num_frames}) must be divisible by tubelet_size ({self.tubelet_size})."
			)

		# Conv3d output is [B, D, T, H, W]. Keep time-major order when flattening to tokens.
		x = self.projection(pixel_values)
		tokens = x.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, x.shape[1])
		return tokens


class ViTNepaVideoEmbeddings(nn.Module):
	"""Construct CLS token and tubelet embeddings."""

	def __init__(self, config: ViTNepaConfig, use_mask_token: bool = False):
		super().__init__()
		self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
		self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
		self.patch_embeddings = ViTNepaVideoPatchEmbeddings(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(
		self,
		pixel_values: torch.Tensor,
		bool_masked_pos: Optional[torch.BoolTensor] = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		batch_size = pixel_values.shape[0]
		embeddings = self.patch_embeddings(pixel_values)
		embeddings_clean = embeddings

		if bool_masked_pos is not None:
			if self.mask_token is None:
				raise ValueError("bool_masked_pos was provided but use_mask_token is disabled.")
			seq_length = embeddings.shape[1]
			mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
			mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
			embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

		cls_tokens = self.cls_token.expand(batch_size, -1, -1)
		embeddings = torch.cat((cls_tokens, embeddings), dim=1)
		embeddings_clean = torch.cat((cls_tokens, embeddings_clean), dim=1)

		embeddings = self.dropout(embeddings)
		return embeddings, embeddings_clean


class ViTNepaVideoRopePositionEmbedding(nn.Module):
	"""Axis-aware 3D RoPE for video patch tokens over (time, height, width)."""

	def __init__(self, config: ViTNepaConfig):
		super().__init__()
		self.base = config.rope_theta
		self.head_dim = config.hidden_size // config.num_attention_heads
		self.patch_size = (
			config.patch_size if isinstance(config.patch_size, tuple) else (config.patch_size, config.patch_size)
		)
		self.tubelet_size = config.tubelet_size

		# Split RoPE channels across temporal, vertical, and horizontal axes.
		base_chunk = self.head_dim // 3
		self.axis_dims = [base_chunk, base_chunk, self.head_dim - (2 * base_chunk)]

		for idx, axis_dim in enumerate(self.axis_dims):
			even_dim = axis_dim if axis_dim % 2 == 0 else axis_dim - 1
			if even_dim <= 0:
				inv_freq = torch.empty(0, dtype=torch.float32)
			else:
				inv_freq = 1.0 / (
					self.base ** (torch.arange(0, even_dim, 2, dtype=torch.float32) / max(even_dim, 1))
				)
			self.register_buffer(f"inv_freq_axis_{idx}", inv_freq, persistent=False)

	def _axis_angles(self, coords: torch.Tensor, axis_dim: int, inv_freq: torch.Tensor) -> torch.Tensor:
		if axis_dim <= 0:
			return torch.empty(coords.shape[0], 0, device=coords.device, dtype=torch.float32)

		even_dim = axis_dim if axis_dim % 2 == 0 else axis_dim - 1
		if even_dim <= 0:
			return torch.zeros(coords.shape[0], axis_dim, device=coords.device, dtype=torch.float32)

		angles = coords.unsqueeze(1) * inv_freq.unsqueeze(0)
		angles = torch.cat((angles, angles), dim=-1)

		if even_dim != axis_dim:
			pad = torch.zeros(coords.shape[0], 1, device=coords.device, dtype=angles.dtype)
			angles = torch.cat((angles, pad), dim=-1)

		return angles

	def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		_, _, t, h, w = pixel_values.shape
		t_tokens = t // self.tubelet_size
		h_tokens = h // self.patch_size[0]
		w_tokens = w // self.patch_size[1]

		device = pixel_values.device
		dtype = pixel_values.dtype

		# Use normalized patch-center coordinates in [-1, 1] for each axis.
		t_coords = (2.0 * ((torch.arange(t_tokens, device=device, dtype=torch.float32) + 0.5) / t_tokens)) - 1.0
		h_coords = (2.0 * ((torch.arange(h_tokens, device=device, dtype=torch.float32) + 0.5) / h_tokens)) - 1.0
		w_coords = (2.0 * ((torch.arange(w_tokens, device=device, dtype=torch.float32) + 0.5) / w_tokens)) - 1.0

		t_grid, h_grid, w_grid = torch.meshgrid(t_coords, h_coords, w_coords, indexing="ij")
		flat_t = t_grid.reshape(-1)
		flat_h = h_grid.reshape(-1)
		flat_w = w_grid.reshape(-1)

		angles_t = self._axis_angles(flat_t, self.axis_dims[0], cast(torch.Tensor, self.inv_freq_axis_0))
		angles_h = self._axis_angles(flat_h, self.axis_dims[1], cast(torch.Tensor, self.inv_freq_axis_1))
		angles_w = self._axis_angles(flat_w, self.axis_dims[2], cast(torch.Tensor, self.inv_freq_axis_2))
		angles = torch.cat((angles_t, angles_h, angles_w), dim=-1)

		cos = torch.cos(angles).to(dtype=dtype)
		sin = torch.sin(angles).to(dtype=dtype)
		return cos, sin


class ViTNepaVideoModel(ViTNepaPreTrainedModel):
	def __init__(self, config: ViTNepaConfig, use_mask_token: bool = False):
		super().__init__(config)
		self.embeddings = ViTNepaVideoEmbeddings(config, use_mask_token=use_mask_token)
		self.rope_embeddings = ViTNepaVideoRopePositionEmbedding(config)
		self.encoder = ViTNepaEncoder(config)
		self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.post_init()

	@check_model_inputs
	def forward(
		self,
		pixel_values: Optional[torch.Tensor] = None,
		bool_masked_pos: Optional[torch.BoolTensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		output_attentions: Optional[bool] = None,
		**kwargs,
	) -> BaseModelOutputWithEmbedding:
		if pixel_values is None:
			raise ValueError("You have to specify pixel_values")

		expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
		if pixel_values.dtype != expected_dtype:
			pixel_values = pixel_values.to(expected_dtype)

		head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
		embedding_input, embedding_clean = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
		position_embeds = self.rope_embeddings(pixel_values)

		encoder_outputs: BaseModelOutput = self.encoder(
			embedding_input,
			head_mask=head_mask,
			output_attentions=output_attentions,
			position_embeddings=position_embeds,
		)
		sequence_output = self.layernorm(encoder_outputs.last_hidden_state)

		return BaseModelOutputWithEmbedding(
			last_hidden_state=sequence_output,
			input_embedding=embedding_clean,
			attentions=encoder_outputs.attentions,
			hidden_states=encoder_outputs.hidden_states,
		)


class ViTNepaVideoForPreTraining(ViTNepaPreTrainedModel):
	def __init__(self, config: ViTNepaConfig):
		super().__init__(config)
		self.vit_nepa = ViTNepaVideoModel(config, use_mask_token=True)
		self.post_init()

	@can_return_tuple
	def forward(
		self,
		pixel_values: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		output_attentions: Optional[bool] = None,
		**kwargs,
	) -> EmbeddedModelingOutput:
		outputs: BaseModelOutputWithEmbedding = self.vit_nepa(
			pixel_values,
			head_mask=head_mask,
			output_attentions=output_attentions,
			**kwargs,
		)

		sequence_input = outputs.input_embedding
		sequence_output = outputs.last_hidden_state
		embedded_loss = cast(torch.FloatTensor, prediction_loss(sequence_input, sequence_output).float())

		return EmbeddedModelingOutput(
			loss=embedded_loss,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)


__all__ = ["ViTNepaVideoForPreTraining", "ViTNepaVideoModel"]
