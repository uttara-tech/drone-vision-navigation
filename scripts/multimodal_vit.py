import torch
import torch.nn as nn
import math


"""
MultiModalViT: Vision Transformer for UZH-FPV Trajectory Prediction
===================================================================
    Contains:
        - PatchEmbed: 256X256 → 256 patches (16X16)
        - MultiModalEmbed: image + IMU token fusion
        - TransformerEncoderBlock: 8-head self-attention + MLP (depth = 6)
        - MultiModalViT: Img+IMU fusion → pred_horizon-step (x, y, z) position trajectory head

    Architecture: 256 image tokens + 16 IMU tokens + 1 [CLS] → 273 tokens total.
    Exposes:
        - forward_features(img, imu): CLS embedding (latent state)
        - forward(img, imu): flattened trajectory [pred_horizon * 7].
"""


def extract_patches(image,patch_size=16):
    print('[INFO] Extracting patches ...')
    B, C, H, W = image.shape
    patches = image.unfold(2,patch_size,patch_size).unfold(3,patch_size,patch_size)
    patches = patches.contiguous().view(B,-1, patch_size,patch_size)       #Original (after contiguous): [1, 1, 16, 16, 16, 16], After View: [1, (1 * 16 * 16), 16, 16] --> [1,256,16,16]
    return patches  



class PatchEmbed(nn.Module):
    def __init__(self, img_size=(256,256),patch_size=16,embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size)
        patch_dim = self.patch_size * self.patch_size
        self.proj = nn.Linear(patch_dim,embed_dim)

    def forward(self,x):
        patches = extract_patches(x,self.patch_size)
        x = patches.flatten(2)              #Result: [1,256,256] --> merging H and W into 1D sequence
        x = self.proj(x)                    #Output: [1,256,out_dim] --> Learned projection of features
        print('[INFO] Number of patches: ',self.num_patches)
        return x
    
    def forward_o(self,x):
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, -1, self.patch_size*self.patch_size*C)
        x = self.proj(x)                    #[B, 256, 256]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        #Ensure that the model dimesion (d_model) is divisible by num_heads
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        #initialize dimesions
        self.d_model = d_model                          
        self.num_heads = num_heads                      #number of heads
        self.d_k = d_model // num_heads                 #number of dimensions per head

        #Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        #Calculate the attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        probs = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(probs, V)
    
    def forward(self,x):
        B,N,C = x.shape
        Q = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(x).view(B, N, self.num_heads, self.d_k).transpose(1,2)

        attention = self.scaled_dot_product_attention(Q,K,V)
        attention = attention.transpose(1,2).contiguous().view(B,N,C)
        return self.W_o(attention)
    

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model,mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,d_model)
            )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        attention = self.attention(x)
        x = self.norm1(x + self.dropout(attention))
        mlp = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp))
        return x
        

class MultiModalEmbed(nn.Module):
    def __init__(self, img_size=(256,256),patch_size=16,embed_dim=256,imu_seq_len=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.imu_seq_len = imu_seq_len
        self.img_embed = PatchEmbed(self.img_size,self.patch_size,self.embed_dim)
        self.num_img_patches = self.img_embed.num_patches

        self.imu_proj = nn.Linear(6,embed_dim)                      #IMU embeddings: 7 values (acc+gyro+ori) --> embed_dim
        self.imu_pose_embed = nn.Parameter(torch.randn(1,self.imu_seq_len,self.embed_dim))

    def forward(self,img,imu):
        img_tokens = self.img_embed(img)                            #IMU patches [B,256,256]
        imu_tokens = self.imu_proj(imu)
        tokens = torch.cat([img_tokens,imu_tokens],dim=1)           #Concatenating image tokens + imu tokens
        return tokens
    

class MultiModalViT(nn.Module):
    def __init__(self,
                 img_size=(256,256),
                 patch_size=16,
                 imu_seq_len=16,
                 embed_dim=256,
                 depth=6,
                 n_heads=8,
                 mlp_dim=512,
                 dropout=0.1,
                 pred_horizon=16
                 ):
        super().__init__()
        self.imu_seq_len = imu_seq_len
        self.embed_dim = embed_dim
        self.pred_horizon = pred_horizon
        self.n_heads = n_heads
        self.embed = MultiModalEmbed(img_size,patch_size,embed_dim=embed_dim)
        self.num_img_patches = self.embed.num_img_patches
        self.cls_token = nn.Parameter(torch.randn(1,1,self.embed_dim))
        self.position_embed = nn.Parameter(torch.randn(1,1 + self.num_img_patches+self.imu_seq_len, self.embed_dim))

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(self.embed_dim, n_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim,self.pred_horizon*3)


    def forward_features(self,x,imu):
        B = x.shape[0]
        print('[INFO] Input tensor shape [B,C,H,W]: ',x.shape)
        x = self.embed(x,imu)

        #Add class token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls,x],dim = 1)
        x = x + self.position_embed
        
        #Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:,0]
        print('[INFO] Number of heads: ', self.n_heads)
        print('[INFO] CLS feature shape: ', cls_out.shape)
        return cls_out
    
    def forward(self, x, imu):
        cls_out = self.forward_features(x, imu)
        return self.head(cls_out)
    

