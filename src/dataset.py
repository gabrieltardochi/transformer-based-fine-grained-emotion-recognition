from typing import Tuple, Dict, Iterable, List
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class GoEmotionsDataset(Dataset):
    def __init__(
        self,
        model_name: str,
        texts: Iterable[str],
        labels: Iterable[Iterable[int]],
        additional_special_tokens: List[str] = [],
        padding: str = 'max_length',
        truncation: bool = True,
        max_length: int = 50,
        return_tensors: str = "pt",
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_added_tokens = 0
        self.additional_special_tokens = additional_special_tokens
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.return_tensors = return_tensors
        if len(additional_special_tokens) > 0:
            self.num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': self.additional_special_tokens})

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: float) -> Tuple[Dict, int]:
        txt = str(self.texts[index])
        label = self.labels[index].tolist()

        # Tokenize sentences to get token ids, attention masks and token type ids
        tokenizer_out = self.tokenizer(
            txt,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        tokenizer_out = {k: v.squeeze(0) for k, v in tokenizer_out.items()}
        return tokenizer_out, torch.FloatTensor(label)
