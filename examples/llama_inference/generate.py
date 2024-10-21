# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import readline  # type: ignore # noqa
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import model as fast
import mp_utils
import sample_utils
import torch
from stats import Stats
from tokenizer import Tokenizer

from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask as AttnBias,
)


@dataclass
class GenArgs:
    gen_length: int = 1000

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

        # checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        checkpoints = sorted(Path(ckpt_dir).glob("*.bin"))
        checkpoints = [checkpoints[0]]

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
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
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
        print("AAA, in generate_all ==========================================================================================")
        bs = len(prompts)
        print(f"prompts is {prompts}, bs is {bs}")
        prompt_lens = [len(p) for p in prompts]
        print("prompt_lens is ", prompt_lens)
        max_prompt_length = max(prompt_lens)
        gen_length = self.gen_args.gen_length
        gen_length = 10
        max_seq_length = max_prompt_length + gen_length
        print(f"max_prompt_length is {max_prompt_length}, gen_length is {gen_length}, max_seq_length is {max_seq_length}")

        cache = fast.make_cache(
            args=self.model_args,
            length=bs * max_seq_length,
        )
        # 检查 cache 中每个元素的类型和内容 , 尤其是 tuple 中的张量
        # for i, value in enumerate(cache):
        #     print(f"Element {i}: Type = {type(value)}")
            
        #     if isinstance(value, tuple):
        #         # 遍历 tuple 中的每个元素
        #         for j, item in enumerate(value):
        #             if isinstance(item, torch.Tensor):
        #                 print(f"  Tuple element {j}: Tensor dtype = {item.dtype}, shape = {item.shape}")
        #             else:
        #                 print(f"  Tuple element {j}: Not a tensor, type = {type(item)}")
        #     else:
        #         # 如果不是 tuple , 尝试直接打印其类型和内容
        #         print(f"Element {i} content: {value}\n")

        # for i, value in enumerate(cache):
        #     if isinstance(value, tuple):
        #         # 创建一个新的 tuple 以存储转换后的张量
        #         new_tuple = []
        #         for item in value:
        #             if isinstance(item, torch.Tensor):
        #                 # 将 bfloat16 转换为 float32
        #                 item = item.to(torch.float32)
        #                 print(f"Converted tensor to dtype = {item.dtype}")
        #             new_tuple.append(item)
        #         # 更新 cache 中的 tuple
        #         cache[i] = tuple(new_tuple)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  ", cache)

        # 假设 cache 是一个列表 , 列表元素可能是张量也可能是元组
        # print("cache contains", len(cache), "items")
        # for idx, item in enumerate(cache):
        #     if isinstance(item, torch.Tensor):
        #         print(f"Shape of tensor {idx}: {item.shape}")
        #     elif isinstance(item, tuple):
        #         print(f"Item {idx} is a tuple with length {len(item)}")
        #         # 如果你想进一步查看元组中的张量形状 , 可以继续这里的检查
        #         for sub_idx, sub_item in enumerate(item):
        #             if isinstance(sub_item, torch.Tensor):
        #                 print(f"  Shape of tensor in tuple[{sub_idx}]: {sub_item.shape}")
        #             else:
        #                 print(f"  Non-tensor item in tuple[{sub_idx}]: {type(sub_item)}")
        #     else:
        #         print(f"Item {idx} is not a tensor or a tuple, it is a {type(item)}")
        """
        cache contains 32 items
        Item 0 is a tuple with length 2
            Shape of tensor in tuple[0]: torch.Size([1, 1009, 32, 1, 128])
            Shape of tensor in tuple[1]: torch.Size([1, 1009, 32, 1, 128])
        ...
        32 层, 每个 Item 包含两个 tuple, 分别对应 K cache 和 V cache
        对于每个 cache, [1, 1009, 32, 1, 128] 对应于 b, s, head_num, not konw, h
        """

        bias = AttnBias.from_seqlens(
            q_seqlen=prompt_lens,
            kv_seqlen=prompt_lens,
            kv_padding=max_seq_length,
        )
        bias.q_seqinfo.to("cuda")
        bias.k_seqinfo.to("cuda")

        print("bias is ", bias)
        """
        这种 mask 类型特别适用于处理具有因果约束 (causal constraints )的 attention , 并且带有偏移填充 (offset padding )的 key 序列。这表明: 

        BlockDiagonal: mask 以块对角线的方式安排 , 适用于处理多个序列 (batch )。
        Causal: 代表因果 attention , 即只允许模型在一个时间步上看到当前时间步及其之前的信息。
        OffsetPaddedKeys: 表明 key 序列可能经过填充并有一定的偏移处理。
        """

        graph = torch.cuda.CUDAGraph()

        # Input tensors to the cuda graph
        q_seqstart = bias.q_seqinfo.seqstart
        kv_seqlen = bias.k_seqinfo.seqlen
        tokens = torch.IntTensor(sum(prompts, [])).cuda()
        out_tokens = torch.zeros((max_seq_length, bs), dtype=torch.int)

        print(f"q_seqstart is {q_seqstart}, kv_seqlen is {kv_seqlen}")
        print(f"tokens is {tokens}")
        print(f"tokens.shape is {tokens.shape}")
        print(f"out_tokens.shape is {out_tokens.shape}")
        print("======================================================================================================================")

        stats = Stats()
        stats.phase("warmup" if use_cuda_graphs else "total")

        for niter in range(gen_length):
            # print("!!!!!!!!!!!!!!!!!!!!!!! cache shape is ")
            # print("Cache length:", len(cache))
            # # 遍历每个元素并检查其是否为 tuple
            # for i, item in enumerate(cache):
            #     if isinstance(item, tuple):
            #         print(f"cache[{i}] is a tuple with length {len(item)}")
            #         for j, sub_item in enumerate(item):
            #             print(f" - Shape of cache[{i}][{j}]: {sub_item.shape if hasattr(sub_item, 'shape') else type(sub_item)}")
            #     else:
            #         print(f"Shape of cache[{i}]: {item.shape if hasattr(item, 'shape') else type(item)}")
            """
            Cache length: 32
            cache[0] is a tuple with length 2
            - Shape of cache[0][0]: torch.Size([1, 19, 32, 1, 128])
            - Shape of cache[0][1]: torch.Size([1, 19, 32, 1, 128])
            """

            for i, item in enumerate(cache):
                for j, sub_item in enumerate(item):
                    if i == 1 and j == 0:
                        torch.set_printoptions(profile="full")
                        print("9th token in layer 1 K-cache, head = 2 i.e. cache[4][0][][3][3] = ", sub_item[:, 9, 2, ...])
                        torch.set_printoptions(profile="default")

            if niter <= self.GRAPH_WARMUPS or not use_cuda_graphs:
                # print("1 niter = ", niter)
                # Keep the first iteration out of the
                # warmup, it processes prompts while all
                # other iterations process sequences of 0
                # or 1 token only
                output = self.model.forward_with_attn_bias(
                    token_values=tokens,
                    attn_bias=bias,
                    cache=cache,
                )
            elif niter == self.GRAPH_WARMUPS + 1:
                # print("2 niter = ", niter)
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
                graph.replay()
                # synchronize to get accurate timings
                torch.cuda.synchronize()
                stats.phase("graph", tokens=(niter + 1) * bs)
            else:
                # print("3 niter = ", niter)
                graph.replay()

            # output: (sum(token_lengths), vocab_size)
            print("output is ", output)
            print("output shape is:", output.shape)
            logits = output.view(bs, self.model_args.vocab_size)
            print("logits is ", logits)
            print("logits shape is ", logits.shape)
            print("---------------------------------------------------------------------------------------------------------")

            if self.gen_args.use_sampling:
                temp = self.gen_args.temperature
                top_p = self.gen_args.top_p
                probs = torch.softmax(logits / temp, dim=-1)
                next_token = sample_utils.top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(bs)
            print("&&&&&&&&&&&&&&&&&& next_token is ", next_token)
            out_tokens[niter, :] = next_token

            # Update attention bias state for decoding rounds
            if niter == 0:
                q_seqstart.copy_(torch.arange(bs + 1, dtype=torch.int))
                bias.q_seqinfo.min_seqlen = 1
                bias.q_seqinfo.max_seqlen = 1
                bias.q_seqinfo.seqstart_py = q_seqstart.tolist()
                tokens = tokens[:bs]

            kv_seqlen.add_(kv_seqlen < max_seq_length)

            tokens.copy_(next_token)
            print("+++++++++++++++++++++++++++++++++++ tokens is ", tokens)

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
            "abc",
            # "can you write a hello world program in C#",
            # "peux tu resoudre le probleme des tours de Hanoi en ocaml",
        ]


def main(ckpt_dir: str, interactive: bool, add_instruction_tags: bool):
    if "WORLD_SIZE" in os.environ:
        mp_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        mp_size = 1
        local_rank = 0

    print(f"mp_size = {mp_size}, local_rank = {local_rank}")
    device = mp_utils.initialize(mp_size, local_rank)

    g = FastGen.build(ckpt_dir, GenArgs(), device)

    for prompts in get_prompts(interactive):
        if add_instruction_tags:
            prompts = [f"[INST]{prompt}[/INST]" for prompt in prompts]

        print("XXX, prompts is : ", prompts)
        tokens = [g.tokenizer.encode(x) for x in prompts]
        print("XXXX, tokens is : ", tokens)
        for i, token_list in enumerate(tokens):
            token_list.append(12428)
            print(f"Input {i}: {token_list}, Length: {len(token_list)}")

        stats, out_tokens = g.generate_all(
            tokens, use_cuda_graphs="NO_CUDA_GRAPHS" not in os.environ
        )

        for i, prompt in enumerate(prompts):
            print("out_tokens is : ", out_tokens)

        if mp_utils.get_rank() == 0:
            for i, prompt in enumerate(prompts):
                print(f"> {prompt}")
                answer = g.tokenizer.decode(out_tokens[i])
                print(answer)
                print("---------------")

            for phase_stats in stats.phases:
                print(phase_stats.show())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Llama inference")
    parser.add_argument("ckpt_dir")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="ask for prompts"
    )
    parser.add_argument(
        "--no-instruction-tags", action="store_true", help="do not add instruction tags"
    )

    args = parser.parse_args()
    main(
        ckpt_dir=args.ckpt_dir,
        interactive=args.interactive,
        add_instruction_tags=not args.no_instruction_tags,
    )
