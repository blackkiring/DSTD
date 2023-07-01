import torch
import torch.nn as nn
from torch.nn import Module, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F
from timm.models.layers import DropPath
from tokenizer import Tokenizer
import math
class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1,patch_size=12):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 4, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)
        self.se=SELayer(dim)
        self.sa=SALayer(int(patch_size))
    def forward(self, x,y):
        B, N, C = x.shape
        qkv  = self.qkv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qxo,qx, kx, vx = qkv[0], qkv[1], qkv[2],qkv[3]
        qyo,qy, ky, vy = qkv2[0],qkv2[1],qkv2[2],qkv[3]
        qx2=self.se(qx)
        qy2=self.sa(qy)
        attn1 = (qx @ kx.transpose(-2, -1)) * self.scale+(qx2 @ kx.transpose(-2, -1)) * self.scale
        attn2 = (qy @ ky.transpose(-2, -1)) * self.scale+(qy2 @ ky.transpose(-2, -1)) * self.scale
        attnx=(qyo @ kx.transpose(-2, -1)) * self.scale
        attny=(qxo @ ky.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        
        attn2= attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        attnx=attnx.softmax(dim=-1)
        attnx = self.attn_drop(attnx)
        attny=attny.softmax(dim=-1)
        attny=self.attn_drop(attny)
        
        x = (attn1 @ vx).transpose(1, 2).reshape(B, N, C)
        y = (attn2 @ vy).transpose(1, 2).reshape(B, N, C)
        xo=(attnx @ vx).transpose(1, 2).reshape(B, N, C)
        yo=(attnx @ vy).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        y= self.proj(y)
        xo=self.proj(x)
        yo=self.proj(y)
        x = self.proj_drop(x)
        y = self.proj_drop(y)
        xo=self.proj_drop(xo)
        yo=self.proj_drop(yo)
        return x,y,xo,yo
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool=nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, head,_,c= x.size()
        x=x.transpose(2,3).reshape(b,head*c,_)
        y = self.avg_pool(x).view(b,head*c)
        z=self.max_pool(x).view(b,head*c)
        y = self.fc(y).view(b, head*c, 1)+self.fc(z).view(b, head*c, 1)
        x=x* y.expand_as(x)
        return x.reshape(b,head,c,_).transpose(2,3)
class SALayer(nn.Module):
    def __init__(self,patch):
        super(SALayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.patch=patch
    def forward(self, x):
        b, head,n,c= x.size()
        x=x.transpose(2,3).reshape(b,head*c,self.patch,self.patch)
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        x=x*out
        return x.reshape(b,head,c,n).transpose(2,3)
class DifAttention(Module):

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 4, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)
    def forward(self, x,y):
        B, N, C = x.shape
        qkv  = self.qkv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B, N, 4, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qxo,qx, kx, vx = qkv[0], qkv[1], qkv[2],qkv[3]
        qyo,qy, ky, vy = qkv2[0],qkv2[1],qkv2[2],qkv[3]
        attn1 = (qx @ kx.transpose(-2, -1)) * self.scale
        attn2 = (qy @ ky.transpose(-2, -1)) * self.scale
        attnx=-(qyo @ kx.transpose(-2, -1)) * self.scale
        attny=-(qxo @ ky.transpose(-2, -1)) * self.scale
#        attnx=(torch.sum(qyo.unsqueeze(4)-kx.unsqueeze(3),dim=4)) * self.scale
#        attny=(torch.sum(qxo.unsqueeze(4)-ky.unsqueeze(3),dim=4)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        attn2= attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        attnx=attnx.softmax(dim=-1)
        attnx = self.attn_drop(attnx)
        attny=attny.softmax(dim=-1)
        attny=self.attn_drop(attny)
        
        x = (attn1 @ vx).transpose(1, 2).reshape(B, N, C)
        y = (attn2 @ vy).transpose(1, 2).reshape(B, N, C)
        x1=(attnx @ vx).transpose(1, 2).reshape(B, N, C)
        y1=(attny @ vy).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x+x1)
        y= self.proj(y+y1)
        x = self.proj_drop(x)
        y = self.proj_drop(y)
        return x,y
class cross_attention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x,y):
        B, N, C = x.shape
        qkv  = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qx, kx, vx = qkv[0], qkv[1], qkv[2]
        qy, ky, vy = qkv2[0],qkv2[1],qkv2[2]
        attnx=(qy @ kx.transpose(-2, -1)) * self.scale
        attny=(qx @ ky.transpose(-2, -1)) * self.scale

        attnx=attnx.softmax(dim=-1)
        attnx = self.attn_drop(attnx)
        attny=attny.softmax(dim=-1)
        attny=self.attn_drop(attny)
        
        xo=(attnx @ vx).transpose(1, 2).reshape(B, N, C)
        yo=(attnx @ vy).transpose(1, 2).reshape(B, N, C)

        xo=self.proj(x)
        yo=self.proj(y)
        xo=self.proj_drop(xo)
        yo=self.proj_drop(yo)
        return xo,yo

class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, image_size,dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout,patch_size=image_size/2)
        self.c_attn=DifAttention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor,src1: torch.Tensor,*args, **kwargs) -> torch.Tensor:
        src0,src01=src,src1
        src,src1,srcx,srcy=self.self_attn(self.pre_norm(src),self.pre_norm(src1))
        src = src0 + self.drop_path(src)
        srcx=src0 + self.drop_path(srcx)
        src1=src01 + self.drop_path(src1)
        srcy=src01 + self.drop_path(srcy)
        
        src,src1,srcx,srcy = self.norm1(src),self.norm1(src1),self.norm1(srcx),self.norm1(srcy)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src3  = self.linear2(self.dropout1(self.activation(self.linear1(src1))))
        src2x=self.linear2(self.dropout1(self.activation(self.linear1(srcx))))
        src2y=self.linear2(self.dropout1(self.activation(self.linear1(srcy))))
        
        src  =  src + self.drop_path(self.dropout2(src2))
        src1 =  src1 + self.drop_path(self.dropout2(src3))
        srcx =  srcx + self.drop_path(self.dropout2(src2x))
        srcy =  srcy + self.drop_path(self.dropout2(src2y))
        src,src1=self.c_attn(self.pre_norm(src),self.pre_norm(src1))
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src3  = self.linear2(self.dropout1(self.activation(self.linear1(src1))))
        src  =  src + self.drop_path(self.dropout2(src2))
        src1 =  src1 + self.drop_path(self.dropout2(src3))
        return src,src1,srcx,srcy
class TransformerDecoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead,image_size, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor,src1: torch.Tensor,mask_fu,mask_pan,mask_ms,*args, **kwargs) -> torch.Tensor:
        src0,src01=src,src1
        src,src1,srcx,srcy=self.self_attn(self.pre_norm(src),self.pre_norm(src1),mask_fu,mask_pan,mask_ms)
        src = src0 + self.drop_path(src)
        srcx=src0 + self.drop_path(srcx)
        src1=src01 + self.drop_path(src1)
        srcy=src01 + self.drop_path(srcy)
        
        src,src1,srcx,srcy = self.norm1(src),self.norm1(src1),self.norm1(srcx),self.norm1(srcy)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src3  = self.linear2(self.dropout1(self.activation(self.linear1(src1))))
        src2x=self.linear2(self.dropout1(self.activation(self.linear1(srcx))))
        src2y=self.linear2(self.dropout1(self.activation(self.linear1(srcy))))
        
        src  =  src + self.drop_path(self.dropout2(src2))
        src1 =  src1 + self.drop_path(self.dropout2(src3))
        srcx =  srcx + self.drop_path(self.dropout2(src2x))
        srcy =  srcy + self.drop_path(self.dropout2(src2y))
        return src,src1,srcx,srcy
class gatefusion(Module):
    def __init__(self,dim):
        super().__init__()
        self.fuse1=Linear(dim*2,dim)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x1,x2):
        w=self.sigmoid(self.fuse1(torch.cat((x1,x2),2)))
        out=w*x1+(1-w)*x2
        return out
class gatefusion2(Module):
    def __init__(self,dim):
        super().__init__()
        self.fuse1=Linear(dim*2,dim)
        self.fuse2=Linear(dim*2,dim)
        self.sigmoid=nn.Sigmoid()
        self.norm1 = LayerNorm(dim)
        self.attn=cross_attention(dim,num_heads=2)
    def forward(self,x,x1,x2):
        w=self.sigmoid(self.fuse1(torch.cat((x,x1),2)))
        out1=w*x+(1-w)*x1
        w=self.sigmoid(self.fuse2(torch.cat((x,x2),2)))
        out2=w*x+(1-w)*x2
        x1,x2=self.attn(out1,out2)
        out1,out2=x1+out1,x2+out2
        w=self.sigmoid(self.fuse2(torch.cat((self.norm1(out1),self.norm1(out2)),2)))
        out=w*self.norm1(out1)+(1-w)*self.norm1(out2)
        out=x+self.norm1(out)
        return out



class TransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 image_size=32,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.2,
                 attention_dropout=0.2,
                 stochastic_depth=0.2,
                 positional_embedding='learnable',
                 sequence_length1=None,
                 sequence_length2=None):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length1 = sequence_length1
        self.sequence_length2 = sequence_length2
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert sequence_length1 is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length1 += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb1 = Parameter(torch.zeros(1, sequence_length1, embedding_dim),
                                                requires_grad=True)
                self.positional_emb2 = Parameter(torch.zeros(1, sequence_length2, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb1, std=0.2)
                init.trunc_normal_(self.positional_emb2, std=0.2)
            else:
                self.positional_emb1 = Parameter(self.sinusoidal_embedding(sequence_length1, embedding_dim),
                                                requires_grad=False)
                self.positional_emb2 =  Parameter(self.sinusoidal_embedding(sequence_length2, embedding_dim),
                                                requires_grad=False)
            self.positional_emb1 = None
            self.positional_emb2 = None
        
        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blk1 = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[0],image_size=image_size)
        self.blk2=TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[1],image_size=image_size)
        self.blk3=TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[2],image_size=image_size)
        self.fuse=gatefusion(embedding_dim)
        self.fuse2=gatefusion2(embedding_dim)
        self.fuse3=gatefusion2(embedding_dim)
        self.fuse4=gatefusion2(embedding_dim)
        self.norm = LayerNorm(embedding_dim)
        self.norm2=LayerNorm(embedding_dim)
        self.fc = Linear(embedding_dim, num_classes)
        self.fc2=Linear(embedding_dim, 10)
        self.apply(self.init_weight)
    def forward(self, x,y):
        if self.positional_emb1 is None and self.positional_emb1 is None and  x.size(1) < self.sequence_length1 and y.size(1)<self.sequence_length2:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)
            y = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)
        if not self.seq_pool:
            cls_token1 = self.class_emb.expand(x.shape[0], -1, -1)
            cls_token2 = self.class_emb.expand(y.shape[0], -1, -1)
            x = torch.cat((cls_token1, x), dim=1)
            y= torch.cat((cls_token2, y), dim=1)
            
        if self.positional_emb1 is not None and self.positional_emb2 is not None:
            x += self.positional_emb1
            y+=self.positional_emb2
        x = self.dropout(x)
        y = self.dropout(y)
        x0,y0,xo,yo=self.blk1(x,y)
        x1,y1,xo,yo=self.blk2(xo,yo)
        x2,y2,xo,yo=self.blk3(xo,yo)
        same=self.fuse(xo,yo)
        x=self.norm2(same)
        x=self.fuse2(x,x2,y2)
        x=self.norm2(x)
        x=self.fuse3(x,x1,y1)
        x=self.norm2(x)
        x=self.fuse4(x,x0,y0)
        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
            xo = torch.matmul(F.softmax(self.attention_pool(xo), dim=1).transpose(-1, -2), xo).squeeze(-2)
            yo= torch.matmul(F.softmax(self.attention_pool(yo), dim=1).transpose(-1, -2), yo).squeeze(-2)
            x2 = torch.matmul(F.softmax(self.attention_pool(x2), dim=1).transpose(-1, -2), x2).squeeze(-2)
            y2= torch.matmul(F.softmax(self.attention_pool(y2), dim=1).transpose(-1, -2), y2).squeeze(-2)
        else:
            x = x[:, 0]
        x = self.fc(x)
        return x,self.fc(xo),self.fc(yo),self.fc(x2),self.fc(y2)

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
