import torch

import os
import sys
sys.path.append("../../../repos/x-transformers")

from vit.model import vision_transformer
from llm.model import palme

def sequential_L_exec(llm):

    text_tokens = torch.randint(0, 20000, (1, 1024)).to("cuda")
    context = torch.randn(1, 64, 512).to("cuda")
    llm.wrapped_decoder.generate(text_tokens, seq_len=256, context=context)

if __name__ == "__main__":

    print("Start RT2 inference...")

    vit = vision_transformer().to("cuda")
    llm = palme().to("cuda")
    models = [vit, llm]

    sequential_L_exec(llm)
