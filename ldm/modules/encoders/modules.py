
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from ldm.models.clip_zh.simple_tokenizer import WordpieceTokenizer
from .text_encoder import TextEncoder


class FrozenCLIPEmbedder_ZH(nn.Cell):
    def __init__(self, max_length=77, use_fp16=False):
        super(FrozenCLIPEmbedder_ZH, self).__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.max_length = max_length
        self.tokenizer = WordpieceTokenizer()
        self.transformer = TextEncoder(context_length=77, vocab_size=49408, output_dim=768, width=768, layers=12, heads=12, dtype=self.dtype)

    def tokenize(self, texts):
        SOT_TEXT = "[CLS]"
        EOT_TEXT = "[SEP]"
        CONTEXT_LEN = 77

        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = ops.Zeros()((len(all_tokens), CONTEXT_LEN), ms.int64)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > CONTEXT_LEN:
                tokens = tokens[:CONTEXT_LEN - 1] + [eot_token]

            result[i, : len(tokens)] = Tensor(tokens)

        return result

    def encode(self, text):
        batch_encoding = self.tokenize(text)
        outputs = self.transformer(batch_encoding)
        return outputs

    def construct(self, c):
        outputs = self.transformer(c)
        return outputs
