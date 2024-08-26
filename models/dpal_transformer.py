import timm
import torch
import torch.nn as nn
import math, json
from functools import reduce
from operator import mul


class DPAL_Transformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.num_prompt_tokens = args.num_prompt_tokens
        prompt_dim = self.embed_dim
        self.num_layers = len(self.blocks)
        patch_size = self.patch_embed.patch_size
        
        self.prompt_deep = args.prompt_deep
        self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_prompt_tokens, prompt_dim))
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        if self.prompt_deep: 
            print("Using deep prompt!")
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                self.num_layers-1, self.num_prompt_tokens, prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        self.predictors = nn.ModuleList([Predictor(prompt_dim*self.num_prompt_tokens, prompt_dim, prompt_dim, drop=0.) for i in range(self.num_layers)])
        for module in self.predictors:
            for param in module.parameters():
                param.data.fill_(0)


    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
            
        return self.pos_drop(x)
    
    def forward_deep_prompt(self, x):
        noises_output = []
        B= x.shape[0]
        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.blocks[i](x)  # 64*198*768
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.deep_prompt_embeddings[i-1].expand(B, -1, -1)
                    hidden_states = torch.cat((deep_prompt_emb, hidden_states[:, self.num_prompt_tokens:, :]), dim=1)
                hidden_states = self.blocks[i](hidden_states)
            if self.num_prompt_tokens == 1:
                noise_input = hidden_states[:, 0, :].clone()
            else:
                tensors_to_concatenate = [hidden_states[:, i, :].clone() for i in range(self.num_prompt_tokens)]
                noise_input = torch.cat(tensors_to_concatenate, dim=1)
            noise = self.predictors[i](noise_input)
            noises_output.append(noise)
            hidden_states[:,self.num_prompt_tokens,:] -= noise

        return hidden_states, noises_output


    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        if self.prompt_embeddings is not None:
            x = torch.cat((self.prompt_embeddings.expand(x.shape[0], -1, -1), x), dim=1)
        if self.prompt_deep:
            x, noises_output = self.forward_deep_prompt(x)
        else:
            x = self.blocks(x)
        x = self.norm(x)

        return x, noises_output
    
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, self.num_prompt_tokens]   
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x, noises_output = self.forward_features(x)
        x = self.forward_head(x)
        return x, noises_output



class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop2(x)
        return x
    