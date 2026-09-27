"""
Runtime compatibility layer for mPLUG-Owl3 on newer Transformers versions.

mPLUG-Owl3's remote modeling code imports the legacy classes:

    Qwen2Attention
    Qwen2SdpaAttention
    Qwen2FlashAttention2

Transformers 4.57 removed the specialized SDPA and FlashAttention2 classes
and changed Qwen2Attention's forward interface. This module temporarily
provides compatible class names while the trusted remote model code is
imported and instantiated.

No installed package or Hugging Face cache file is modified.

The attention implementation below is adapted from Hugging Face
Transformers v4.40.2, licensed under the Apache License 2.0:
https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/qwen2/modeling_qwen2.py
"""

from __future__ import annotations

import math
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
import torch.nn.functional as F
from torch import nn
import inspect
from types import MethodType
from typing import Any


_PATCH_LOCK = threading.RLock()
_MISSING = object()


class _LegacyQwen2RotaryEmbedding(nn.Module):
    """Qwen2 rotary embedding implementation compatible with Transformers 4.40."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(
                    0,
                    self.dim,
                    2,
                    dtype=torch.int64,
                    device=device,
                ).float()
                / self.dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.max_seq_len_cached = seq_len

        positions = torch.arange(
            self.max_seq_len_cached,
            device=device,
            dtype=torch.int64,
        ).type_as(self.inv_freq)

        frequencies = torch.outer(positions, self.inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)

        self.register_buffer(
            "cos_cached",
            embeddings.cos().to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            embeddings.sin().to(dtype),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len,
                device=x.device,
                dtype=x.dtype,
            )

        cos = self.cos_cached[:seq_len].to(
            device=x.device,
            dtype=x.dtype,
        )
        sin = self.sin_cached[:seq_len].to(
            device=x.device,
            dtype=x.dtype,
        )

        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def install_siglip_hidden_states_compat(owl3_model: Any) -> Any:


    vision_model = getattr(owl3_model, "vision_model", None)

    if vision_model is None:
        raise AttributeError(
            "OWL3 model has no attribute 'vision_model'; "
        )


    if getattr(
        vision_model,
        "_ddm_siglip_hidden_states_compat_installed",
        False,
    ):
        return owl3_model

    encoder = getattr(vision_model, "encoder", None)

    if encoder is None:
        raise AttributeError(
            "OWL3 vision_model has no attribute 'encoder'."
        )


    try:
        parameters = inspect.signature(encoder.forward).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "output_hidden_states" in parameters:
        return owl3_model

    required_attributes = (
        "embeddings",
        "encoder",
        "post_layernorm",
        "use_head",
    )

    missing_attributes = [
        name
        for name in required_attributes
        if not hasattr(vision_model, name)
    ]

    if missing_attributes:
        raise TypeError(
            "Unsupported SigLIP vision implementation; missing attributes: "
            + ", ".join(missing_attributes)
        )

    from transformers.modeling_outputs import BaseModelOutputWithPooling

    original_forward = vision_model.forward

    def forward_with_hidden_states(
        self,
        pixel_values,
        interpolate_pos_encoding=False,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        **kwargs,
    ):


        if output_hidden_states is None:
            output_hidden_states = bool(
                getattr(self.config, "output_hidden_states", False)
            )


        if not output_hidden_states:
            return original_forward(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
                **kwargs,
            )


        if output_attentions:
            raise NotImplementedError(
                "The OWL3 SigLIP compatibility path supports "
                "output_hidden_states=True, but not output_attentions=True."
            )

        attention_mask = kwargs.pop("attention_mask", None)

        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        all_hidden_states = [hidden_states]

        # �ȼ����°� SiglipEncoder.forward��ͬʱ�ռ��м�����
        for encoder_layer in self.encoder.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                **kwargs,
            )
            all_hidden_states.append(hidden_states)


        last_hidden_state = self.post_layernorm(hidden_states)

        pooler_output = (
            self.head(last_hidden_state)
            if self.use_head
            else None
        )

        outputs = BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=tuple(all_hidden_states),
            attentions=None,
        )

        if return_dict is False:
            return outputs.to_tuple()

        return outputs

    # ֻ�滻��ǰģ��ʵ���� forward��
    vision_model.forward = MethodType(
        forward_with_hidden_states,
        vision_model,
    )

    vision_model._ddm_siglip_hidden_states_compat_installed = True
    return owl3_model

def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)

    return query, key

def ensure_generation_mixin(model: nn.Module) -> nn.Module:
    """
    Add Transformers generation support to legacy remote model classes.

    Transformers >= 4.50 no longer makes PreTrainedModel inherit from
    GenerationMixin automatically.
    """
    from transformers import GenerationConfig
    from transformers.generation import GenerationMixin

    if isinstance(model, GenerationMixin):
        return model

    original_class = model.__class__

    compatible_class = type(
        f"{original_class.__name__}WithGenerationMixin",
        (original_class, GenerationMixin),
        {
            "__module__": original_class.__module__,
            "__doc__": original_class.__doc__,
        },
    )

    model.__class__ = compatible_class

    if getattr(model, "generation_config", None) is None:
        model.generation_config = (
            GenerationConfig.from_model_config(model.config)
        )

    return model

def _repeat_kv(
    hidden_states: torch.Tensor,
    repetitions: int,
) -> torch.Tensor:
    """
    Expand KV heads to match the number of query heads.

    Input:
        [batch, num_key_value_heads, seq_len, head_dim]

    Output:
        [batch, num_attention_heads, seq_len, head_dim]
    """
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape

    if repetitions == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        repetitions,
        seq_len,
        head_dim,
    )

    return hidden_states.reshape(
        batch_size,
        num_key_value_heads * repetitions,
        seq_len,
        head_dim,
    )


class LegacyQwen2Attention(nn.Module):
    """
    Qwen2 attention with the Transformers 4.40 calling convention.

    This class is used by mPLUG-Owl3 remote code, which expects attention
    layers to return:

        attention_output, attention_weights, past_key_value
    """

    def __init__(
        self,
        config: Any,
        layer_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = (
            self.num_heads // self.num_key_value_heads
        )

        self.max_position_embeddings = getattr(
            config,
            "max_position_embeddings",
            32768,
        )
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = True

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"hidden_size={self.hidden_size}, "
                f"num_attention_heads={self.num_heads}"
            )

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        self.rotary_emb = _LegacyQwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        past_key_value: Optional[Any],
        **kwargs: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        batch_size, query_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        key_states = key_states.view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        value_states = value_states.view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        past_length = 0

        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    "layer_idx must be set when KV caching is enabled."
                )

            past_length = past_key_value.get_usable_length(
                query_length,
                self.layer_idx,
            )

        kv_seq_len = query_length + past_length

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + query_length,
                device=hidden_states.device,
                dtype=torch.long,
            ).unsqueeze(0)

            position_ids = position_ids.expand(batch_size, -1)
        else:
            position_ids = position_ids.to(
                device=hidden_states.device,
                dtype=torch.long,
            )

        # Protect against position IDs that are greater than kv_seq_len,
        # for example when custom padding or offsets are used.
        rotary_seq_len = max(
            kv_seq_len,
            int(position_ids.max().item()) + 1,
        )

        cos, sin = self.rotary_emb(
            value_states,
            seq_len=rotary_seq_len,
        )

        query_states, key_states = _apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids,
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
            }

            # StaticCache in newer Transformers versions may use this.
            cache_position = kwargs.get("cache_position")
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position

            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        kv_seq_len = key_states.shape[-2]

        key_states = _repeat_kv(
            key_states,
            self.num_key_value_groups,
        )
        value_states = _repeat_kv(
            value_states,
            self.num_key_value_groups,
        )

        return (
            query_states,
            key_states,
            value_states,
            kv_seq_len,
        )

    @staticmethod
    def _validate_attention_mask(
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        query_length: int,
        kv_seq_len: int,
    ) -> None:
        if attention_mask is None:
            return

        expected_shape = (
            batch_size,
            1,
            query_length,
            kv_seq_len,
        )

        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                "Incorrect attention mask shape: "
                f"received={tuple(attention_mask.shape)}, "
                f"expected={expected_shape}"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Any],
    ]:
        del use_cache  # Cache mutation is controlled by past_key_value.

        if "padding_mask" in kwargs:
            warnings.warn(
                "padding_mask is deprecated; use attention_mask instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            if attention_mask is None:
                attention_mask = kwargs.pop("padding_mask")

        batch_size, query_length, _ = hidden_states.size()

        (
            query_states,
            key_states,
            value_states,
            kv_seq_len,
        ) = self._prepare_qkv(
            hidden_states,
            position_ids,
            past_key_value,
            **kwargs,
        )

        self._validate_attention_mask(
            attention_mask,
            batch_size,
            query_length,
            kv_seq_len,
        )

        attention_weights = torch.matmul(
            query_states,
            key_states.transpose(2, 3),
        ) / math.sqrt(self.head_dim)

        expected_weights_shape = (
            batch_size,
            self.num_heads,
            query_length,
            kv_seq_len,
        )

        if tuple(attention_weights.shape) != expected_weights_shape:
            raise ValueError(
                "Incorrect attention weights shape: "
                f"received={tuple(attention_weights.shape)}, "
                f"expected={expected_weights_shape}"
            )

        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask

        attention_weights = F.softmax(
            attention_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(query_states.dtype)

        attention_weights = F.dropout(
            attention_weights,
            p=self.attention_dropout,
            training=self.training,
        )

        attention_output = torch.matmul(
            attention_weights,
            value_states,
        )

        expected_output_shape = (
            batch_size,
            self.num_heads,
            query_length,
            self.head_dim,
        )

        if tuple(attention_output.shape) != expected_output_shape:
            raise ValueError(
                "Incorrect attention output shape: "
                f"received={tuple(attention_output.shape)}, "
                f"expected={expected_output_shape}"
            )

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous()

        attention_output = attention_output.reshape(
            batch_size,
            query_length,
            self.hidden_size,
        )

        attention_output = self.o_proj(attention_output)

        if not output_attentions:
            attention_weights = None

        return (
            attention_output,
            attention_weights,
            past_key_value,
        )


class LegacyQwen2SdpaAttention(LegacyQwen2Attention):
    """
    Qwen2 SDPA attention using the legacy mPLUG-Owl3 interface.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Any],
    ]:
        if output_attentions:
            # PyTorch SDPA does not return attention weights, so use the
            # manual implementation when they are requested.
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=True,
                use_cache=use_cache,
                **kwargs,
            )

        if "padding_mask" in kwargs:
            warnings.warn(
                "padding_mask is deprecated; use attention_mask instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            if attention_mask is None:
                attention_mask = kwargs.pop("padding_mask")

        batch_size, query_length, _ = hidden_states.size()

        (
            query_states,
            key_states,
            value_states,
            kv_seq_len,
        ) = self._prepare_qkv(
            hidden_states,
            position_ids,
            past_key_value,
            **kwargs,
        )

        self._validate_attention_mask(
            attention_mask,
            batch_size,
            query_length,
            kv_seq_len,
        )

        # CUDA memory-efficient SDPA requires contiguous tensors when a
        # custom attention mask is supplied.
        if (
            query_states.device.type == "cuda"
            and attention_mask is not None
        ):
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=(
                self.attention_dropout if self.training else 0.0
            ),
            is_causal=(
                self.is_causal
                and attention_mask is None
                and query_length > 1
            ),
        )

        attention_output = attention_output.transpose(
            1,
            2,
        ).contiguous()

        attention_output = attention_output.reshape(
            batch_size,
            query_length,
            self.hidden_size,
        )

        attention_output = self.o_proj(attention_output)

        return (
            attention_output,
            None,
            past_key_value,
        )


class UnsupportedLegacyQwen2FlashAttention2(LegacyQwen2Attention):
    """
    Placeholder required because mPLUG-Owl3 imports this symbol eagerly.

    The current DDM-VLM OWL3 wrapper explicitly selects SDPA, so this class
    should never be instantiated.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

        raise RuntimeError(
            "The DDM-VLM OWL3 compatibility layer only supports "
            "attn_implementation='sdpa'."
        )


@contextmanager
def mplug_owl3_qwen2_compat() -> Iterator[None]:
    """
    Temporarily expose the Transformers 4.40 Qwen2 attention API.

    Usage:

        with mplug_owl3_qwen2_compat():
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation="sdpa",
            )

    Installed Transformers files are never modified. The module attributes
    are restored after model loading, while the imported OWL3 module keeps
    references to the compatibility classes.
    """
    from transformers.models.qwen2 import modeling_qwen2

    required_legacy_names = (
        "Qwen2SdpaAttention",
        "Qwen2FlashAttention2",
    )

    # Transformers 4.40 and similar versions already expose the classes.
    if all(
        hasattr(modeling_qwen2, name)
        for name in required_legacy_names
    ):
        yield
        return

    replacements = {
        "Qwen2Attention": LegacyQwen2Attention,
        "Qwen2SdpaAttention": LegacyQwen2SdpaAttention,
        "Qwen2FlashAttention2": (
            UnsupportedLegacyQwen2FlashAttention2
        ),
    }

    # Hold the lock for the full import/initialization window. This prevents
    # two simultaneous OWL3 initializations from restoring the symbols in
    # the wrong order.
    with _PATCH_LOCK:
        originals = {
            name: getattr(modeling_qwen2, name, _MISSING)
            for name in replacements
        }

        for name, replacement in replacements.items():
            setattr(modeling_qwen2, name, replacement)

        try:
            yield
        finally:
            for name, original in originals.items():
                if original is _MISSING:
                    try:
                        delattr(modeling_qwen2, name)
                    except AttributeError:
                        pass
                else:
                    setattr(modeling_qwen2, name, original)


__all__ = [
    "LegacyQwen2Attention",
    "LegacyQwen2SdpaAttention",
    "ensure_generation_mixin",
    "mplug_owl3_qwen2_compat",
]
