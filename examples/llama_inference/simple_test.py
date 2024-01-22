# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import readline  # type: ignore # noqa
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import fire
import model as fast
import mp_utils
import sample_utils
import torch
from stats import Stats
from tokenizer import Tokenizer

from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)

# from xformers.ops.fmha.attn_bias import (
#     BlockDiagonalMask as AttnBias,
# )


@dataclass
class GenArgs:
    gen_length: int = 3000

    use_sampling: bool = True
    temperature: float = 0.6
    top_p: float = 0.9


class FastGen:
    GRAPH_WARMUPS: int = 3
    tokenizer: Tokenizer

    @staticmethod
    def build(
        ckpt_dir: str,
        gen_args: GenArgs,
        device: Union[torch.device, str],
        tokenizer_path: Optional[str] = None,
    ) -> "FastGen":
        """
        Load a Llama or Code Llama checkpoint and return a new
        generator for this model.
        """
        start_time = time.time()
        world_size = mp_utils.get_world_size()

        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

        assert len(checkpoints) > 0, f"no checkpoint files in {ckpt_dir}"
        assert world_size == len(checkpoints), (
            f"checkpoint for model parallelism {len(checkpoints)}"
            f" but world size is {world_size}"
        )

        ckpt_path = checkpoints[mp_utils.get_rank()]
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args = fast.ModelArgs(**params)

        if tokenizer_path is None:
            tokenizer_path = str(Path(ckpt_dir) / "tokenizer.model")
        if not os.path.isfile(tokenizer_path):
            tokenizer_path = str(Path(ckpt_dir) / ".." / "tokenizer.model")
        if not os.path.isfile(tokenizer_path):
            raise RuntimeError("could not find the tokenizer model")
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words

        torch.set_default_device(device)
        torch.set_default_dtype(torch.bfloat16)

        model = fast.Transformer(model_args)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        print(f"loaded model in {time.time() - start_time:.2f} seconds")

        return FastGen(gen_args, model_args, model, tokenizer)

    def __init__(
        self,
        args: GenArgs,
        model_args: fast.ModelArgs,
        model: fast.Transformer,
        tokenizer: Tokenizer,
    ):
        self.gen_args = args
        self.model_args = model_args
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate_all(
        self, prompts: list[list[int]], use_cuda_graphs: bool
    ) -> Tuple[Stats, list[list[int]]]:
        
        for i in range(2048-18):
            prompts[0].append(i+666)
        bs = len(prompts)
        prompt_lens = [len(p) for p in prompts]
        max_prompt_length = max(prompt_lens)
        gen_length = self.gen_args.gen_length
        max_seq_length = max_prompt_length + gen_length

        cache = fast.make_cache(
            args=self.model_args,
            length=bs * max_seq_length,
        )

        bias = AttnBias.from_seqlens(
            q_seqlen=prompt_lens,
            kv_seqlen=prompt_lens,
            kv_padding=max_seq_length,
        )
        # print(bias.q_seqinfo.seqstart_py[-1])
        # print(bias.k_seqinfo.seqstart_py[-1])
        # mask = bias.materialize(shape=[18, 118])
        # print(mask)
        # print(mask.shape)
        
        bias.q_seqinfo.to("cuda")
        bias.k_seqinfo.to("cuda")

        graph = torch.cuda.CUDAGraph()

        # Input tensors to the cuda graph
        q_seqstart = bias.q_seqinfo.seqstart
        kv_seqlen = bias.k_seqinfo.seqlen
        print("q_seqstart: ", q_seqstart)
        print("kv_seqlen: ", kv_seqlen)
        # print("prompts: ", prompts)
        tokens = torch.IntTensor(sum(prompts, [])).cuda()
        # print("tokens: ", tokens)
        out_tokens = torch.zeros((max_seq_length, bs), dtype=torch.int)

        stats = Stats()
        stats.phase("warmup" if use_cuda_graphs else "total")

        for niter in range(2):

            if niter == 0:
                output = self.model.forward_with_attn_bias(
                    token_values=tokens,
                    attn_bias=bias,
                    cache=cache,
                )
                recording_kwargs = {}
                if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
                    # In PyTorch 2.1+ and nightlies from late Aug 2023,
                    # we can do this to maybe avoid watchdog-related crashes
                    recording_kwargs["capture_error_mode"] = "thread_local"
                with torch.cuda.graph(graph, **recording_kwargs):
                    output = self.model.forward_with_attn_bias(
                        token_values=tokens,
                        attn_bias=bias,
                        cache=cache,
                    )
                for i in range(10):
                    graph.replay()
                # synchronize to get accurate timings
                torch.cuda.synchronize()
                stats.phase("graph", tokens=(niter + 1) * bs)

            # output: (sum(token_lengths), vocab_size)
            # print(output.shape)
            logits = output[q_seqstart[1:] - 1, :]
            # print(logits.shape)
            # print(q_seqstart[1:] - 1)
            logits = logits.view(bs, self.model_args.vocab_size)

            next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(bs)
            out_tokens[niter, :] = next_token
            # print(out_tokens.shape)

            # Update attention bias state for decoding rounds
            if niter == 0:
                stats.phase("prefill", tokens=18)
                for i in range(10):
                    graph.replay()
                q_seqstart.copy_(torch.arange(bs + 1, dtype=torch.int))
                # print("bias.q_seqinfo.min_seqlen: ", bias.q_seqinfo.min_seqlen)
                # print("bias.q_seqinfo.max_seqlen: ", bias.q_seqinfo.max_seqlen)
                bias.q_seqinfo.min_seqlen = 1
                bias.q_seqinfo.max_seqlen = 1
                bias.q_seqinfo.seqstart_py = q_seqstart.tolist()
                # print(bias.q_seqinfo)
                # print(bias.k_seqinfo)
                # mask = bias.materialize(shape=[1, 118])
                # print(mask)
                # print("q_seqstart: ", q_seqstart)
                # print("tokens-1: ", tokens)
                tokens = tokens[:bs]
                # print("tokens-2: ", tokens)

            if niter == 1:
                # print('++++++++++')
                # print(bias.q_seqinfo.seqstart)
                # print(bias.k_seqinfo.seqstart)
                # print("kv_seqlen-2: ", bias.k_seqinfo.seqlen)
                # mask = bias.materialize(shape=[1, 118])
                # print(mask)
                stats.phase("decoding", tokens=100)
                for i in range(100):
                    graph.replay()


            kv_seqlen.add_(kv_seqlen < max_seq_length)
            # print("kv_seqlen-2: ", bias.k_seqinfo.seqlen)

            tokens.copy_(next_token)
            # print("tokens: ", tokens)

        stats.end_phase(tokens=gen_length * bs)

        def trim_answer(prompt, tokens):
            """Trim the answer to end it on an eos token."""
            tokens = tokens[: max_seq_length - len(prompt)]
            eos_id = self.tokenizer.eos_id
            if eos_id in tokens:
                return tokens[: tokens.index(eos_id) + 1]
            else:
                return tokens

        answers = [
            trim_answer(prompt, answer)
            for prompt, answer in zip(prompts, out_tokens.t().tolist())
        ]
        return stats, answers


def get_prompts(interactive: bool) -> Iterable[list[str]]:
    if interactive:
        while True:
            try:
                prompts = input("enter prompt: ").split("\n")
            except EOFError:
                print("exiting")
                sys.exit(0)
            yield prompts
    else:
        yield [
            "can you write a hello world program in C#",
        ]


def main(ckpt_dir: str, interactive: bool = False, add_instruction_tags: bool = True):
    if "WORLD_SIZE" in os.environ:
        mp_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        mp_size = 1
        local_rank = 0

    device = mp_utils.initialize(mp_size, local_rank)

    g = FastGen.build(ckpt_dir, GenArgs(), device)

    for prompts in get_prompts(interactive):
        if add_instruction_tags:
            prompts = [f"[INST]{prompt}[/INST]" for prompt in prompts]

        tokens = [g.tokenizer.encode(x) for x in prompts]
        print(tokens)
        stats, out_tokens = g.generate_all(
            tokens, use_cuda_graphs="NO_CUDA_GRAPHS" not in os.environ
        )

        if mp_utils.get_rank() == 0:
            for i, prompt in enumerate(prompts):
                # print(f"> {prompt}")
                # answer = g.tokenizer.decode(out_tokens[i])
                # print(answer)
                print("---------------")

            for phase_stats in stats.phases:
                print(phase_stats.show())


if __name__ == "__main__":
    fire.Fire(main)
