"""
Cole aqui a arquitetura real do ZylunCPT 0.1 vinda do notebook.

Este arquivo precisa ter:
- cfg
- class ZylunCPT
- método generate()
"""

# Cole aqui os imports do modelo do notebook:
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Cole aqui a config REAL usada no treino.
# Os valores precisam ser iguais ao checkpoint treinado.
@dataclass
class GPTConfig:
    vocab_size: int = 32000
    seq_len: int = 1024
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    dropout: float = 0.0


cfg = GPTConfig()


# ABAIXO, COLE AS CLASSES REAIS DO NOTEBOOK:
# RMSNorm
# CausalSelfAttention / Attention
# MLP
# Block
# ZylunCPT
