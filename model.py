#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import inspect

from dataclasses import dataclass
@dataclass
class Model_args:
    block_size: int = 1024    #the num of max input
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True        #whether to use bias in Linear and LayerNorm layers
                             #True:bias in Linears and LayerNorms,like GPT-2. False:faster

class flash_att(nn.Module):  #same as NanoGPT
    def __init__(self, args):
        super().__init__()
            #combined q,k,v into one linear
        self.qkv_atten = nn.Linear(args.n_embed, 3*args.n_embed, bias=args.bias)
            
        self.n_head = args.n_head
        self.n_embed = args.n_embed
        assert args.n_embed % args.n_head == 0       #embedding size is divisible by the number of heads
        self.head_size = args.n_embed // args.n_head
        self.dropout = args.dropout
        self.att_dropout = nn.Dropout(self.dropout)
        self.c_proj = nn.Linear(self.n_embed, self.n_embed, bias=args.bias)

    def forward(self, x):
        B, T, C = x.shape               #batch size (B), sequence length (T), and embedding size (C)
        q, k, v = self.qkv_atten(x).split(self.n_embed, dim=2)    #compute query, key, and value vectors
            #Reshape for multi-head attention and transpose dimensions for matrix multiplication
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
            #use flash attention of pytorch
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                       dropout_p=self.dropout if self.training else 0,
                                                       is_causal=True)
            #transpose back and reshape to original dimensions    
        y = y.transpose(1, 2)
        y = y.contiguous().view(B, T, C)
        
        return self.att_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.up_proj = nn.Linear(args.n_embed, 4*args.n_embed, bias=args.bias)
        self.down_c_proj = nn.Linear(4*args.n_embed, args.n_embed, bias=args.bias)
        self.act_func = nn.GELU()    #Gelu may be better
        self.gate = nn.Linear(args.n_embed, 4*args.n_embed, bias=args.bias)

    def forward(self, x):
        gate_proj = self.gate(x)
        
        x = self.up_proj(x)
        x = self.act_func(gate_proj) * x
        x = self.down_c_proj(x)
        return self.dropout(x)
    
class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm = nn.LayerNorm(args.n_embed, eps=1e-5)
        self.attn = flash_att(args)        #Flash attention layer
        self.mlp = MLP(args)               #MLP layer

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x + self.mlp(self.norm(x))

class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(args.vocab_size, args.n_embed),  #Token embedding layer
            wpe=nn.Embedding(args.block_size, args.n_embed),  #Position embedding layer
            drop=nn.Dropout(args.dropout),
            h=nn.ModuleList([Block(args) for _ in range(args.n_layer)]),  #Transformer blocks
            norm=nn.LayerNorm(args.n_embed, eps=1e-5)
        ))

        self.lm_head = nn.Linear(args.n_embed, args.vocab_size, bias=False)   #Output layer
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
            #Calculate total number of parameters
        n_sum = 0     
        for pname, p in self.named_parameters():
            n_sum = n_sum + p.numel()
            if pname.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*args.n_layer))

        print(f"parameters of all：{n_sum}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()          #batch size and sequence length
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        token_embed = self.transformer.wte(idx)
        pos_embed = self.transformer.wpe(pos)
        
        x = self.transformer.drop(token_embed + pos_embed)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.norm(x)
            #calculate loss
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}  #get parameters that require gradients
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]   #parameters for weight decay
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]  #parameters without weight decay
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"parameters using weight decay:{num_decay},parameters without using weight decay:{num_nodecay}")

        fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and device_type == 'cuda'
        if use_fused:
            print("AdamW optimiser use fused!")
        extra_args = {'fused': True} if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        return optimizer

    def generate(self, idx, max_generate_tokens, tempreture=1.0, top_k=None):
        for _ in range(max_generate_tokens):
            idx = idx if idx.shape[1] <= self.args.block_size else idx[:, -self.args.block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / tempreture

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   #sample next token from probabilities
            idx = torch.cat((idx, idx_next), dim=1)              #append next token to the sequence

        return idx

