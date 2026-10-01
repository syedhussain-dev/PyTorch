import tiktoken
import torch

class NanoGPTTokenizer:
    def __init__(self, encoder_name:str="gpt2"):
        self.encoder = tiktoken.get_encoding(encoding_name=encoder_name)
        
    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, ids):
        return self.encoder.decode(ids)
