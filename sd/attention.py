import torch
from comfy.ldm.modules.attention import CrossAttention, default, optimized_attention, optimized_attention_masked
from ..utils import nag


class NAGCrossAttention(CrossAttention):
    def __init__(
            self,
            *args,
            nag_scale: float = 1,
            nag_tau: float = 2.5,
            nag_alpha: float = 0.25,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
    
    def forward(
            self,
            x,
            context=None,
            value=None,
            mask=None,
            transformer_options=None,
            **kwargs,
    ):
        if x.shape[0] == 0:
            return self.to_out(x)
        context = default(context, x)
        origin_bsz = context.shape[0] - x.shape[0]
        if origin_bsz <= 0:
            # nothing to guide
            return super().forward(
                x,
                context=context,
                value=value,
                mask=mask,
                transformer_options=transformer_options,
                **kwargs,
            )

        q = self.to_q(x)
        q = torch.cat([q, q[-origin_bsz:]], dim=0)
        added = origin_bsz

        if mask is not None and added > 0 and mask.shape[0] == x.shape[0]:
            mask = torch.cat([mask, mask[-added:].clone()], dim=0)

        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        if mask is None:
            out = optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision)
        else:
            out = optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision)

        # NAG
        # if there aren't 2*origin_bsz rows available, skip NAG for this layer
        if out.shape[0] < 2 * origin_bsz:
            if added > 0:
                out = out[:-added]
            return self.to_out(out)
        out_negative, out_positive = out[-origin_bsz:], out[-origin_bsz * 2:-origin_bsz]
        out_guidance = nag(out_positive, out_negative, self.nag_scale, self.nag_tau, self.nag_alpha)
        out = torch.cat([out[:-origin_bsz * 2], out_guidance], dim=0)

        return self.to_out(out)
