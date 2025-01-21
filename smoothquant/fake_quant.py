import torch
from torch import nn
from functools import partial


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_token_affine(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])

    xmax = t.max(dim=-1, keepdim=True)[0]
    xmin = t.min(dim=-1, keepdim=True)[0]
    q_max = 2**n_bits - 1

    scales = xmax - xmin
    scales.clamp_(min=1e-5).div_(q_max)
    zeroes = torch.round(-xmin / scales)

    q = torch.clamp(torch.round(t / scales) + zeroes, 0, q_max)
    return scales * (q - zeroes)


@torch.no_grad()
def quantize_activation_per_token_groupwise_affine(t, n_bits, groupsize):
    init_shape = t.shape
    reshaped_t = t.reshape(-1, t.shape[-2], t.shape[-1] // groupsize, groupsize)

    xmax = torch.amax(reshaped_t, dim=3, keepdim=True)
    xmin = torch.amin(reshaped_t, dim=3, keepdim=True)

    q_max = 2**n_bits - 1
    tmp = (xmin == 0) & (xmax == 0)
    xmin[tmp] = -1
    xmax[tmp] = +1
    scales = (xmax - xmin) / q_max
    zeroes = torch.round(-xmin / scales)

    scales = scales.repeat(1, 1, 1, groupsize).reshape(init_shape)
    zeroes = zeroes.repeat(1, 1, 1, groupsize).reshape(init_shape)

    q = torch.clamp(torch.round(t / scales) + zeroes, 0, q_max)
    return scales * (q - zeroes)


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class WQAQLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        a_prec=8,
        w_prec=8,
        o_prec=8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a_prec = a_prec
        self.w_prec = w_prec
        self.o_prec = o_prec

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=self.a_prec
            )
        elif act_quant == "per_token_affine":
            self.act_quant_name = "per_token_affine"
            self.act_quant = partial(
                quantize_activation_per_token_affine, n_bits=self.a_prec
            )
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=self.a_prec
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:  # This is only for K and V, Q is excluded
            self.output_quant_name = "per_token_groupwise_affine"
            self.output_quant = partial(
                quantize_activation_per_token_groupwise_affine,
                n_bits=self.o_prec,
                groupsize=128,
            )
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(WQAQLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_output=False,
        a_prec=8,
        w_prec=8,
        o_prec=8,  # this is only important when quantize_output is True, like in KV quantization
    ):
        assert isinstance(module, (torch.nn.Linear, WQAQLinear))
        new_module = WQAQLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
            a_prec=a_prec,
            w_prec=w_prec,
            o_prec=o_prec,
        )
        if weight_quant == "per_channel":
            new_module.weight = module.weight
            # new_module.weight = quantize_weight_per_channel_absmax(
            #     module.weight, n_bits=w_prec
            # )
        elif weight_quant == "per_tensor":
            new_module.weight = module.weight
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"WQAQLinear(a_prec={self.a_prec}, w_prec={self.w_prec}, {self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


def quantize_opt(
    model,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=True,
    a_prec=8,
    w_prec=8,
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for _, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = WQAQLinear.from_float(
                m.fc1,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.fc2 = WQAQLinear.from_float(
                m.fc2,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = WQAQLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.k_proj = WQAQLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.v_proj = WQAQLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.out_proj = WQAQLinear.from_float(
                m.out_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
    return model


def quantize_llama_like(
    model,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=False,
    a_prec=8,
    w_prec=8,
    o_prec=8,
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2MLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, Qwen2MLP, MistralMLP)):
            m.gate_proj = WQAQLinear.from_float(
                m.gate_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
                o_prec=o_prec,
            )
            m.up_proj = WQAQLinear.from_float(
                m.up_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
                o_prec=o_prec,
            )
            m.down_proj = WQAQLinear.from_float(
                m.down_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
                o_prec=o_prec,
            )
        elif isinstance(m, (LlamaAttention, Qwen2Attention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = WQAQLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=False,  # q_proj output is not quantized
                a_prec=a_prec,
                w_prec=w_prec,
                o_prec=o_prec,
            )
            m.k_proj = WQAQLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                a_prec=a_prec,
                w_prec=w_prec,
                o_prec=o_prec,
            )
            m.v_proj = WQAQLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                a_prec=a_prec,
                w_prec=w_prec,
                o_prec=o_prec,
            )
            m.o_proj = WQAQLinear.from_float(
                m.o_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
                o_prec=o_prec,
            )
    return model


def quantize_mixtral(
    model,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=False,
    a_prec=8,
    w_prec=8,
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = WQAQLinear.from_float(
                m.w1,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.w2 = WQAQLinear.from_float(
                m.w2,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.w3 = WQAQLinear.from_float(
                m.w3,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = WQAQLinear.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.k_proj = WQAQLinear.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.v_proj = WQAQLinear.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.o_proj = WQAQLinear.from_float(
                m.o_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = WQAQLinear.from_float(
                m.gate,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
    return model


def quantize_falcon(
    model,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=True,
    a_prec=8,
    w_prec=8,
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = WQAQLinear.from_float(
                m.dense_h_to_4h,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.dense_4h_to_h = WQAQLinear.from_float(
                m.dense_4h_to_h,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = WQAQLinear.from_float(
                m.query_key_value,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
                a_prec=a_prec,
                w_prec=w_prec,
            )
            m.dense = WQAQLinear.from_float(
                m.dense,
                weight_quant=weight_quant,
                act_quant=act_quant,
                a_prec=a_prec,
                w_prec=w_prec,
            )
    return model


def quantize_model(
    model,
    weight_quant="per_channel",
    act_quant="per_token",
    quantize_bmm_input=False,
    a_prec=8,
    w_prec=8,
):
    from transformers.models.opt.modeling_opt import OPTPreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
    from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel
    from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel
    from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
    from transformers.models.falcon.modeling_falcon import FalconPreTrainedModel

    if isinstance(model, OPTPreTrainedModel):
        return quantize_opt(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            a_prec=a_prec,
            w_prec=w_prec,
        )
    elif isinstance(model, (LlamaPreTrainedModel, Qwen2PreTrainedModel, MistralPreTrainedModel)):
        return quantize_llama_like(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            a_prec=a_prec,
            w_prec=w_prec,
        )
    elif isinstance(model, MixtralPreTrainedModel):
        return quantize_mixtral(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            a_prec=a_prec,
            w_prec=w_prec,
        )
    elif isinstance(model, FalconPreTrainedModel):
        return quantize_falcon(
            model,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_bmm_input=quantize_bmm_input,
            a_prec=a_prec,
            w_prec=w_prec,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
