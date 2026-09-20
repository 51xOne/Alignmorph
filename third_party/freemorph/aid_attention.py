from typing import Optional

import torch
import torch.nn.functional as F

from aid_utils import append_dims, generate_beta_tensor


class OuterInterpolatedAttnProcessor_SDPA:
    r"""
    Personalized processor for performing outer attention interpolation.

    The attention output of interpolated image is obtained by:
    (1 - t) * Q_t * K_1 * V_1 + t * Q_t * K_m * V_m;
    If fused with self-attention:
    (1 - t) * Q_t * [K_1, K_t] * [V_1, V_t] + t * Q_t * [K_m, K_t] * [V_m, V_t];
    """

    def __init__(
        self,
        t: Optional[torch.Tensor] = None,
        is_fused: bool = False,
    ):
        """
        t: float, interpolation point between 0 and 1, if not specified, size is set to 3
        """
        self.coef = t
        self.is_fused = is_fused

    def __call__(
        self,
        attn,
        hidden_states: torch.torch.Tensor,
        encoder_hidden_states: Optional[torch.torch.Tensor] = None,
        attention_mask: Optional[torch.torch.Tensor] = None,
        temb: Optional[torch.torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

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
        value_begin = value_begin.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
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

        hidden_states_begin = hidden_states_begin.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_begin = hidden_states_begin.to(query.dtype)

        hidden_states_end = F.scaled_dot_product_attention(
            query,
            key_end,
            value_end,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states_end = hidden_states_end.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_end = hidden_states_end.to(query.dtype)

        # Apply outer interpolation on attention
        coef = append_dims(self.coef, hidden_states_begin.ndim).to(query.device, query.dtype)
        hidden_states = (1 - coef) * hidden_states_begin + coef * hidden_states_end

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class InnerInterpolatedAttnProcessor_SDPA:
    r"""
    Personalized processor for performing inner attention interpolation.

    The attention output of interpolated image is obtained by:
    (1 - t) * Q_t * K_1 * V_1 + t * Q_t * K_m * V_m;
    If fused with self-attention:
    (1 - t) * Q_t * [K_1, K_t] * [V_1, V_t] + t * Q_t * [K_m, K_t] * [V_m, V_t];
    """

    def __init__(
        self,
        t: Optional[torch.Tensor] = None,
        size: int = 7,
        is_fused: bool = False,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        """
        t: float, interpolation point between 0 and 1, if not specified, size is set to 3
        """
        if t is None:
            ts = generate_beta_tensor(size, alpha=alpha, beta=beta)
            ts[0], ts[-1] = 0, 1
        else:
            if t.ndim == 0:
                t = t.unsqueeze(0)
            ts = t
        self.coef = ts
        self.is_fused = is_fused

    def __call__(
        self,
        attn,
        hidden_states: torch.torch.Tensor,
        encoder_hidden_states: Optional[torch.torch.Tensor] = None,
        attention_mask: Optional[torch.torch.Tensor] = None,
        temb: Optional[torch.torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

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

        coef = append_dims(self.coef, key_begin.ndim).to(key_begin.device, key_begin.dtype)
        key_cross = (1 - coef) * key_begin + coef * key_end
        value_cross = (1 - coef) * value_begin + coef * value_end

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key_cross = key_cross.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_cross = value_cross.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key_cross,
            value_cross,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class InnerConvergedAttnProcessor_SDPA:
    def __init__(
        self,
        is_fused: bool = False,
    ):
        self.is_fused = is_fused

    def __call__(
        self,
        attn,
        hidden_states: torch.torch.Tensor,
        encoder_hidden_states: Optional[torch.torch.Tensor] = None,
        attention_mask: Optional[torch.torch.Tensor] = None,
        temb: Optional[torch.torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        key_mean = key.mean(dim=0, keepdim=True)
        value_mean = value.mean(dim=0, keepdim=True)

        key_mean = key_mean.repeat_interleave(batch_size, dim=0)
        value_mean = value_mean.repeat_interleave(batch_size, dim=0)

        # Fused with self-attention
        if self.is_fused:
            key_mean = torch.cat([key, key_mean], dim=-2)
            value_mean = torch.cat([value, value_mean], dim=-2)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key_mean = key_mean.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_mean = value_mean.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key_mean,
            value_mean,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class InnerConvergedAttnProcessor_SDPA2:
    def __init__(
        self,
        is_fused: bool = False,
    ):
        self.is_fused = is_fused

    def __call__(
        self,
        attn,
        hidden_states: torch.torch.Tensor,
        encoder_hidden_states: Optional[torch.torch.Tensor] = None,
        attention_mask: Optional[torch.torch.Tensor] = None,
        temb: Optional[torch.torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        key_mean = torch.cat([key[0:1], ((key[0:1] + key[-1:]) / 2).repeat_interleave(batch_size - 2, 0), key[-1:]])
        value_mean = torch.cat(
            [value[0:1], ((value[0:1] + value[-1:]) / 2).repeat_interleave(batch_size - 2, 0), value[-1:]]
        )

        # Fused with self-attention
        if self.is_fused:
            key_mean = torch.cat([key, key_mean], dim=-2)
            value_mean = torch.cat([value, value_mean], dim=-2)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key_mean = key_mean.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_mean = value_mean.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key_mean,
            value_mean,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class OuterConvergedAttnProcessor_SDPA:
    def __init__(self):
        pass

    def __call__(
        self,
        attn,
        hidden_states: torch.torch.Tensor,
        encoder_hidden_states: Optional[torch.torch.Tensor] = None,
        attention_mask: Optional[torch.torch.Tensor] = None,
        temb: Optional[torch.torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

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

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = hidden_states.mean(dim=0, keepdim=True).repeat_interleave(batch_size, dim=0)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class OuterConvergedAttnProcessor_SDPA2:
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
        hidden_states: torch.torch.Tensor,
        encoder_hidden_states: Optional[torch.torch.Tensor] = None,
        attention_mask: Optional[torch.torch.Tensor] = None,
        temb: Optional[torch.torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

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
        value_begin = value_begin.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
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
            coef = append_dims(self.coef, hidden_states_begin.ndim).to(query.device, query.dtype)
            hidden_states = (1 - coef) * hidden_states_begin + coef * hidden_states_end
        else:
            hidden_states = (hidden_states_begin + hidden_states_end) / 2

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
