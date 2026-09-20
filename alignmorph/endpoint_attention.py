"""Endpoint attention processors derived from FreeMorph; see NOTICE.md."""

from typing import Optional

import torch
import torch.nn.functional as F

from .sampling import append_dims


class EndpointInterpolatedAttention:
    """Mix attention outputs from the two endpoint frames."""

    def __init__(
        self,
        t: Optional[torch.Tensor] = None,
        is_fused: bool = False,
    ):
        self.coef = t
        self.is_fused = is_fused

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Specify the first and last key and value
        key_begin = key[0:1]
        key_end = key[-1:]
        value_begin = value[0:1]
        value_end = value[-1:]

        key_begin = torch.repeat_interleave(key_begin, batch_size, dim=0)
        key_end = torch.repeat_interleave(key_end, batch_size, dim=0)
        value_begin = torch.repeat_interleave(value_begin, batch_size, dim=0)
        value_end = torch.repeat_interleave(value_end, batch_size, dim=0)

        # Fused with self-attention
        if self.is_fused:
            key_end = torch.cat([key, key_end], dim=-2)
            value_end = torch.cat([value, value_end], dim=-2)
            key_begin = torch.cat([key, key_begin], dim=-2)
            value_begin = torch.cat([value, value_begin], dim=-2)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key_begin = key_begin.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_begin = value_begin.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        key_end = key_end.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_end = value_end.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states_begin = F.scaled_dot_product_attention(
            query,
            key_begin,
            value_begin,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states_begin = hidden_states_begin.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states_begin = hidden_states_begin.to(query.dtype)

        hidden_states_end = F.scaled_dot_product_attention(
            query,
            key_end,
            value_end,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states_end = hidden_states_end.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states_end = hidden_states_end.to(query.dtype)

        # Apply outer interpolation on attention
        coef = append_dims(self.coef, hidden_states_begin.ndim).to(
            query.device, query.dtype
        )
        hidden_states = (1 - coef) * hidden_states_begin + coef * hidden_states_end

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class FrameMeanAttention:
    def __init__(self):
        pass

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        hidden_states = hidden_states.mean(dim=0, keepdim=True).repeat_interleave(
            batch_size, dim=0
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class EndpointMeanAttention:
    def __init__(
        self,
        coef: torch.Tensor = None,
        is_fused: bool = False,
    ):
        self.is_fused = is_fused
        if coef is not None:
            self.coef = coef
        else:
            self.coef = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        key_begin = key[0:1].repeat_interleave(batch_size, dim=0)
        key_end = key[-1:].repeat_interleave(batch_size, dim=0)
        value_begin = value[0:1].repeat_interleave(batch_size, dim=0)
        value_end = value[-1:].repeat_interleave(batch_size, dim=0)

        # Fused with self-attention
        if self.is_fused:
            key_end = torch.cat([key, key_end], dim=-2)
            value_end = torch.cat([value, value_end], dim=-2)
            key_begin = torch.cat([key, key_begin], dim=-2)
            value_begin = torch.cat([value, value_begin], dim=-2)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_begin = key_begin.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_end = key_end.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_begin = value_begin.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        value_end = value_end.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states_begin = F.scaled_dot_product_attention(
            query,
            key_begin,
            value_begin,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states_end = F.scaled_dot_product_attention(
            query,
            key_end,
            value_end,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        if self.coef is not None:
            coef = append_dims(self.coef, hidden_states_begin.ndim).to(
                query.device, query.dtype
            )
            hidden_states = (1 - coef) * hidden_states_begin + coef * hidden_states_end
        else:
            hidden_states = (hidden_states_begin + hidden_states_end) / 2

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
