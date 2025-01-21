import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from smoothquant.calibration import get_act_scales
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_llama_like
import gptq
from gptq import get_wikitext2
import tqdm
from argparse import ArgumentParser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE="cpu"

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples if n_samples else self.dataset.size(1) // 2048

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm.tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-name", default="meta-llama/Llama-2-7b-hf", help="model name"
    )
    parser.add_argument(
        "--a-prec",
        type=int,
        default=8,
        help="activation precision",
    )
    parser.add_argument(
        "--w-prec",
        type=int,
        default=8,
        help="weight precision",
    )
    parser.add_argument(
        "--kv-prec",
        type=int,
        default=8,
        help="KV cache precision",
    )
    parser.add_argument(
        "--rtn",
        action="store_true",
        default=False,
        help="use round to nearest",
    )
    parser.add_argument(
        "--smoothquant",
        action="store_true",
        default=False,
        help="use round to nearest",
    )
    args = parser.parse_args()

    # torch_dtype=torch.float16
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, config=model_fp16.config)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, DEVICE)

    ppl_fp16 = evaluator.evaluate(model_fp16)
    print(f"Original model (fp16) perplexity: {ppl_fp16}")

    # smoothquant
    if args.smoothquant:
        act_scales = get_act_scales(
            model_fp16, tokenizer, "dummy", num_samples=512, seq_len=512
        )
        smooth_lm(model_fp16, act_scales, 0.85)
        model_q = model_fp16
        # model_q = quantize_llama_like(
        #     model_fp16, a_prec=args.a_prec, w_prec=args.w_prec
        # )
    else:
        model_q = model_fp16

    GPTQ = not args.rtn

    if GPTQ:
        trainloader = get_wikitext2(
            nsamples=128,
            seed=0,
            model=args.model_name,
            seqlen=2048,
            eval_mode=False,
        )

        gptq.gptq_fwrd(
            model_q,
            trainloader,
            DEVICE,
            nsamples=128,
            w_bits=args.w_prec,
            w_per_channel=True,
            w_asym=False,
            w_clip=True,
            w_groupsize=-1,
            percdamp=0.01,
            act_order=False,
            export_to_et=False,
        )
    else:  # RTN
        gptq.rtn_fwrd(
            model_q,
            DEVICE,
            w_bits=args.w_prec,
            w_per_channel=True,
            w_asym=False,
            w_clip=False,
            w_groupsize=-1,
            export_to_et=False,
            custom_layers=None,
        )

    model_q = quantize_llama_like(
        model_q,
        act_quant="per_token_affine",
        quantize_bmm_input=True,  # this is for KV cache
        a_prec=args.a_prec,
        w_prec=args.w_prec,
        o_prec=args.kv_prec,
    )

    print(model_q)
    ppl_q = evaluator.evaluate(model_q)
    print(
        f"{args.model_name} W{args.w_prec}/A{args.a_prec}/KV{args.kv_prec}, GPTQ:{not args.rtn}, SQ:{args.smoothquant}, quantized model perplexity: {ppl_q}"
    )

    import lm_eval

    TASK_LIST = [
        "mmlu_other",
        "mmlu_stem",
        "mmlu_social_sciences",
        "mmlu_humanities",
        "arc_easy",
        "arc_challenge",
        "piqa",
        "winogrande",
        "boolq",
        "social_iqa",
        "openbookqa",
        "hellaswag",
    ]
    EVAL_BS = 16

    # This is just to create lm, its model will be updated by quantized model later
    model_args = f"pretrained=meta-llama/Llama-3.2-1b,device={DEVICE},dtype=float,trust_remote_code={True}"
    lm = lm_eval.api.registry.get_model("hf").create_from_arg_string(
        model_args, {"batch_size": EVAL_BS}
    )
    lm._model = model_q
    task_dict = lm_eval.tasks.get_task_dict(TASK_LIST)
    metrics = lm_eval.evaluate(lm, task_dict)["results"]
    print(
        f"{args.model_name} W{args.w_prec}/A{args.a_prec}/KV{args.kv_prec}, GPTQ:{not args.rtn}, SQ:{args.smoothquant}, eval_res: {metrics}"
    )


if __name__ == "__main__":
    main()
