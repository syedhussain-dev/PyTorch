import torch
from torch.utils.data import Dataset

class TextCustom(Dataset):
    def __init__(self, tokens, context_length):
        super().__init__()
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        return len(self.tokens) - self.context_length - 1

    def __getitem__(self, index):
        input_token = self.tokens[index : index + self.context_length]
        output_token = self.tokens[index +1 : index + self.context_length +1]
        return torch.tensor(input_token, dtype=torch.int64), torch.tensor(output_token, dtype=torch.int64)