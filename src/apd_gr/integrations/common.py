from __future__ import annotations

import json
import os


class IntegerCodeTokenizer:
    """Tokenizer-compatible metadata facade for integer-code integrations."""

    def __init__(self, vocab_size, pad_token_id=0, eos_token_id=0, tokenizer_type="integer_code"):
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id)
        self.tokenizer_type = str(tokenizer_type)

    def __len__(self):
        return self.vocab_size

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "diger_code_tokenizer.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "type": self.tokenizer_type,
                    "vocab_size": self.vocab_size,
                    "pad_token_id": self.pad_token_id,
                    "eos_token_id": self.eos_token_id,
                },
                handle,
                indent=2,
            )
        return (path,)
