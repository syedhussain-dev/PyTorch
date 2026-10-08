import torch
from torch import nn
import yaml
import os
from torch.utils.checkpoint import checkpoint
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir,"Parameters.yaml")

with open(yaml_path,"r") as f:
	config = yaml.safe_load(f)["NanoGPT"]

class MultiHeadCasualSelfAttention(nn.Module):
  """
  Applies multi-head self-attention to the input tokens.

  Each token is projected into Query, Key and Value vectors. Attention
  scores are calculated between every pair of tokens, allowing information
  to be exchanged across the entire image. The outputs from all attention
  heads are concatenated and projected back to the original embedding
  dimension. We mask the tokens after the token we are focusing on to
  prevent cheating. This is done my creating a tensor of ones and
  covering part of it. The attention score is made negative infinity if
  a particular token is covered. Softmax and matrix multiplication with the 
  value tensor. The final output matrix is transposed, reshaped and projected.  

  Args:
      embed_dim: Dimension of each token embedding.
      heads: Number of parallel attention heads.
      attn_dropout: Dropout probability applied to the attention weights.

  Returns:
      Tensor of shape (batch_size, num_tokens, embed_dim).

  Example:
      Input:                    (1, 197, 768)
      Query, Key, Value:        (1, 197, 768)
      Split into heads:         (1, 12, 197, 64)
      Attention scores:         (1, 12, 197, 197)
      Attention output:         (1, 12, 197, 64)
      Concatenated output:      (1, 197, 768)
      Final output:             (1, 197, 768)
  """
  def __init__(self, embed_dim, heads, attn_dropout):
      super().__init__()
      self.heads = heads
      self.embed_dim = embed_dim
      self.query_key_value = nn.Linear(embed_dim, embed_dim * 3)
      self.projection = nn.Linear(embed_dim, embed_dim)
      assert embed_dim % heads == 0
      self.head_dim = embed_dim // heads
      self.attention_dropout = nn.Dropout(attn_dropout)

  def forward(self, x):
      batch, tokens, _ = x.shape
      qkv = self.query_key_value(x)
      query, key, value = qkv.chunk(3, dim=-1)

      query = query.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
      key = key.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
      value = value.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

      scores = torch.matmul(query, key.transpose(-2, -1))
      scores = scores / (self.head_dim ** 0.5)
      mask = torch.tril(torch.ones(tokens,tokens,device=x.device))
      scores = scores.masked_fill(mask == 0, float("-inf"))
      attention = self.attention_dropout(torch.softmax(scores, dim=-1))
      out = torch.matmul(attention, value)

      out = out.transpose(1, 2).reshape(batch, tokens, self.embed_dim)
      out = self.projection(out)
      return out

class FeedForward(nn.Module):
  """
  Implements the feed-forward network used within a Transformer encoder.

  Args:
      embed_dim: Dimension of each token embedding.
      mlp_dim: Hidden dimension of the feed-forward network.
      mlp_dropout: Dropout probability applied after each linear layer.

  Returns:
      Tensor of shape (batch_size, num_tokens, embed_dim).

  Example:
      Input:                    (1, 197, 768)
      First Linear Layer:       (1, 197, 3072)
      GELU + Dropout:           (1, 197, 3072)
      Second Linear Layer:      (1, 197, 768)
      Output:                   (1, 197, 768)
  """
  def __init__(self, embed_dim, mlp_dim, mlp_dropout):
      super().__init__()
      self.layer = nn.Sequential(
        nn.Conv1d(in_channels=embed_dim,out_channels=mlp_dim,kernel_size=1),
        nn.GELU(),
        nn.Dropout(mlp_dropout),
        nn.Conv1d(in_channels=mlp_dim,out_channels=embed_dim,kernel_size=1),
        nn.Dropout(mlp_dropout)
      )

  def forward(self, x):
      x = x.transpose(1,2)
      x = self.layer(x)
      x = x.transpose(1,2)
      return x

class GPTTransformBlock(nn.Module):
  """
  Implements a single Vision Transformer encoder block.

  Args:
      embed_dim: Dimension of each token embedding.
      heads: Number of attention heads.
      mlp_dim: Hidden dimension of the feed-forward network.
      mlp_dropout: Dropout probability within the feed-forward network.
      attn_dropout: Dropout probability applied to the attention weights.

  Returns:
      Tensor of shape (batch_size, num_tokens, embed_dim).

  Example:
      Input:                    (1, 197, 768)
      LayerNorm:                (1, 197, 768)
      Self-Attention:           (1, 197, 768)
      First Residual Add:       (1, 197, 768)
      LayerNorm:                (1, 197, 768)
      Feed Forward Network:     (1, 197, 768)
      Second Residual Add:      (1, 197, 768)
      Output:                   (1, 197, 768)
  """
  def __init__(self, embed_dim, heads, mlp_dim, mlp_dropout, attn_dropout):
    super().__init__()

    self.norm1 = nn.LayerNorm(embed_dim)
    self.attention = MultiHeadCasualSelfAttention(
      embed_dim=embed_dim,
      heads=heads,
      attn_dropout=attn_dropout
    )
    self.norm2 = nn.LayerNorm(embed_dim)
    self.mlp = FeedForward(
      embed_dim=embed_dim,
      mlp_dim=mlp_dim,
      mlp_dropout=mlp_dropout
    )

  def forward(self, x):
    x = x + self.attention(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x


class GPT(nn.Module):
  """
  Final GPT class for text generation

  Input tokens from the tokenizer are passed into the model where they are
  then transformered into 1D vectors used embedding. Positional embeddings
  and dropouts areapplied before the tokens are inputted into the encoder blocks.
  Once thetokens have established relationships with each other they are passed
  through a final normalisation and projected to generate text.

  Args:
    vocab_size: The maximum number of tokens
    embed_dim: The embedding dimension used for turning tokens into vectors
    heads: The number of heads in the attention step
    mlp_dim: The expansion of features dimension for adding non-linearity
    mlp_dropout: The dropout probability in the feedforward class
    num_encoder_layers: The number of encoder layers in the model
    context_length: The maximum number of tokens processed in a sequence
    gradient_checkpointing: A boolean to activate gradient checkpointing

  Returns:
    A sequence of generate tokens which the tokenizer decodes
  """
  def __init__(self, vocab_size:int,
              embed_dim:int=config["embed_dim"],
              heads:int=config["heads"],
              mlp_dim:int=config["mlp_dim"],
              mlp_dropout:float=config["mlp_dropout"],
              attn_dropout:float=config["attn_dropout"],
              num_encoder_layers:int=config["num_encoder_layers"],
              context_length:int=config["context_length"],
              gradient_checkpointing:bool=config["gradient_checkpointing"]):
    
    super().__init__()
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.mlp_dim = mlp_dim
    self.attn_dropout = attn_dropout
    self.mlp_dropout = mlp_dropout
    self.num_encoder_layers = num_encoder_layers
    self.context_length = context_length
    self.heads = heads
    self.gradient_checkpointing = gradient_checkpointing

    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
    self.positional_embedding = nn.Parameter(torch.randn((1,self.context_length,self.embed_dim)))
    self.dropout = nn.Dropout(p=0.1)
    self.norm = nn.LayerNorm(self.embed_dim)
    self.encoder_blocks = nn.ModuleList(
      [
        GPTTransformBlock(
          embed_dim=self.embed_dim, heads=self.heads, mlp_dim=self.mlp_dim,
          mlp_dropout=self.mlp_dropout,attn_dropout=self.attn_dropout
        )
        for _ in range(self.num_encoder_layers)
      ]
    )
    self.LLM_head = nn.Linear(
      in_features=self.embed_dim, out_features=self.vocab_size
    )

  def forward(self, x):
    self.embedding = self.embedding.to(x.device)
    x = self.embedding(x)
    x = x + self.positional_embedding
    x = self.dropout(x)
    for block in self.encoder_blocks:
      if self.gradient_checkpointing:
        x = checkpoint(block, x, use_reentrant=False)
      else:
        x = block(x)
    x = self.norm(x)
    x = self.LLM_head(x)
    return x

  def load_pretrained(self):
    from transformers import GPT2LMHeadModel
    gpt = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    with torch.no_grad():
      self.embedding.weight.copy_(
        gpt.transformer.wte.weight
      )
      self.norm.weight.copy_(
        gpt.transformer.ln_f.weight
      )
      self.norm.weight.copy_(
        gpt.transformer.ln_f.weight
      )
      for i in range(self.num_encoder_layers):
        self.encoder_blocks[i].norm1.weight.copy_(
          gpt.transformer.h[i].ln_1.weight
        )
        self.encoder_blocks[i].norm1.bias.copy_(
          gpt.transformer.h[i].ln_1.bias
        )
        self.encoder_blocks[i].norm2.weight.copy_(
          gpt.transformer.h[i].ln_2.weight
        )
        self.encoder_blocks[i].norm2.bias.copy_(
          gpt.transformer.h[i].ln_2.bias
        )
        self.encoder_blocks[i].mlp.layer[0].weight.copy_(
          gpt.transformer.h[i].mlp.c_fc.weight.T.unsqueeze(dim=2)
        )
        self.encoder_blocks[i].mlp.layer[0].bias.copy_(
          gpt.transformer.h[i].mlp.c_fc.bias
        )
        self.encoder_blocks[i].mlp.layer[3].weight.copy_(
          gpt.transformer.h[i].mlp.c_proj.weight.T.unsqueeze(dim=2)
        )
        self.encoder_blocks[i].mlp.layer[3].bias.copy_(
          gpt.transformer.h[i].mlp.c_proj.bias
        )
        self.encoder_blocks[i].attention.query_key_value.weight.copy_(
          gpt.transformer.h[i].attn.c_attn.weight.T
        )
        self.encoder_blocks[i].attention.query_key_value.bias.copy_(
          gpt.transformer.h[i].attn.c_attn.bias
        )
        self.encoder_blocks[i].attention.projection.weight.copy_(
          gpt.transformer.h[i].attn.c_proj.weight
        )
        self.encoder_blocks[i].attention.projection.bias.copy_(
          gpt.transformer.h[i].attn.c_proj.bias
        )
    return self

