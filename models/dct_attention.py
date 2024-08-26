import timm
import torch
import torch.nn as nn
 

class DCT_Attention(timm.models.vision_transformer.Attention):
    def __init__(self, dim=768, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)

        self.domain_conditioner_generator = nn.Linear(dim, dim * 3, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q_postfix, k_postfix, v_postfix = self.domain_conditioner_generator(x[:,0,:]).reshape(B, 1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        q_postfix, k_postfix, v_postfix = q_postfix.expand(B, -1, -1, -1), k_postfix.expand(B, -1, -1, -1), v_postfix.expand(B, -1, -1, -1)
        q, k, v = torch.cat((q, q_postfix), dim=2), torch.cat((k, k_postfix), dim=2), torch.cat((v, v_postfix), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N+1, C)[:,:-1,:]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
