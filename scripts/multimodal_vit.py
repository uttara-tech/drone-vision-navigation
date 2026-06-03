import torch
import torch.nn as nn
import math
import numpy as np


"""
MultiModalViT: Vision Transformer for UZH-FPV Trajectory Prediction
===================================================================
    Contains:
        - PatchEmbed: 256X256 -> 256 patches (16X16)
        - MultiModalEmbed: image + IMU token fusion
        - TransformerEncoderBlock: 8-head self-attention + MLP (depth = 6)
        - MultiModalViT: Img+IMU fusion -> pred_horizon-step (x, y, z) position trajectory head

    Architecture: 256 image tokens + 16 IMU tokens + 1 [CLS] -> 273 tokens total.
    Exposes:
        - forward_features(img, imu): CLS embedding (latent state)
        - forward(img, imu): flattened trajectory [pred_horizon * 3].
"""


def extract_patches(image:torch.Tensor,patch_size:int=16) -> torch.Tensor:
    B:int = image.shape[0]
    patches:torch.Tensor = image.unfold(2,patch_size,patch_size).unfold(3,patch_size,patch_size)
    patches = patches.contiguous().view(B,-1, patch_size,patch_size)       #Original (after contiguous): [1, 1, 16, 16, 16, 16], After View: [1, (1 * 16 * 16), 16, 16] --> [1,256,16,16]
    return patches  



class PatchEmbed(nn.Module):

    """
        Class to handle patch embeddings. It performs below operations:
        1. slices incoming 256X256 image into 256 distinct patches of size 16X16
        2. flattens the spatial pixel grid each patch [1,256,16,16] into a 1D vector of 256 elements [1,256,256]
        3. passes each 1D patch vector through a linear NN to project its features into the embedding space
        4. initializes a trainable position_embed parameter to track sequence order during back propagation. 
            -> it is a randomly initialized 3D weight vector. 
            -> during training the optimizer detects this parameter and updates its value during training.
            -> it also enables gradient tracking i.e. it sels requires_grad=true on the tensor under the hood allowing backpropagation to 
               calculate how to adjust these 256 numbers to minimize regreesion loss.
    """

    def __init__(self, img_size:tuple,patch_size:int,embed_dim:int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches:int = (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size)
        patch_dim:int = self.patch_size * self.patch_size
        
        #Single input and single output layer - no hidden layers.
        self.proj:nn.Linear = nn.Linear(patch_dim,self.embed_dim)
        

    def forward(self,x:np.ndarray) -> torch.Tensor:
        patches:torch.Tensor  = extract_patches(x,self.patch_size)
        x:torch.Tensor = patches.flatten(2)              #Result: [1,256,256] --> merging H and W into 1D sequence
        x = self.proj(x)                    #Output: [1,256,out_dim] --> Learned projection of features
        return x


class MultiHeadAttention(nn.Module):

    """
        Class to define multi-head self-attention. It handles the below operations:
        1. creates a Query, Key, Value and Output linear layers, each with 256 inputs and 256 outputs.
        2. splits the 256-dimension Q,K,V outputs into 8 parallel attention heads of 32 features each.
        2. calculates attention scores using scaled dot product using classic Transformer Attention formula, 
            -> applies softmax over the rows (last dimension),
               and multiplies by V to get a blended tensor of shape [1, 8, 273, 32] 
               (how to read this: 1 batch, e.g. 1 image + IMU sample pair, where 8 distinct attention heads are running in parallel with 273 tokens overall and 32 context-blended features inside each head)
        3. next, concatenates all attention heads into a unified sequence matrix of shape [1, 273, 256]
        4. Finally, W_o linear layer mixes the results of all 8 attention heads together.
    """

    def __init__(self, d_model:int, num_heads:int):
        super(MultiHeadAttention, self).__init__()
        #Ensure that the model dimesion (d_model) is divisible by num_heads
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        #initialize dimesions
        self.d_model = d_model                          
        self.num_heads = num_heads                      #number of heads
        self.d_k:int = d_model // num_heads                 #number of dimensions per head

        #Linear layers for transforming inputs
        self.W_q:nn.Linear = nn.Linear(d_model,d_model)
        self.W_k:nn.Linear = nn.Linear(d_model,d_model)
        self.W_v:nn.Linear = nn.Linear(d_model,d_model)
        self.W_o:nn.Linear = nn.Linear(d_model,d_model)
    
    def scaled_dot_product_attention(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask=None) -> torch.Tensor:

        """
        Calculating the attention scores. For example, a 273X273 grid for a single batch and single attention head will look like:
                          
                            [cls] [256 Image Patches] [16 IMU Steps]
                            ┌─────┬───────────────────┬──────────────┐
                      [cls] │  ★  │         ★         │      ★       │
                            ├─────┼───────────────────┼──────────────┤
        TOKEN IS:   [Image] │  ★  │         ★         │      ★       │
                            ├─────┼───────────────────┼──────────────┤
                      [IMU] │  ★  │         ★         │      ★       │
                            └─────┴───────────────────┴──────────────┘
        """

        attention_scores:torch.Tensor = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)    #shape: [1, 8, 273, 273]
        probs:torch.Tensor = torch.softmax(attention_scores, dim=-1)        #probs shape: [1, 8, 273, 273], V shape: [1, 8, 273, 32]      
        return torch.matmul(probs, V)               #[273, 273]  X [273, 32] => shape: [1, 8, 273, 32]
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:

        B,N,C = x.shape
        Q:torch.Tensor = self.W_q(x).view(B, N, self.num_heads, self.d_k).transpose(1,2)
        K:torch.Tensor = self.W_k(x).view(B, N, self.num_heads, self.d_k).transpose(1,2)
        V:torch.tensor = self.W_v(x).view(B, N, self.num_heads, self.d_k).transpose(1,2)

        attention:torch.tensor = self.scaled_dot_product_attention(Q,K,V)
        attention = attention.transpose(1,2).contiguous().view(B,N,C)   #Combines the heads -> [1, 273, 256] -> 32*8
        return self.W_o(attention)          #nn.Linear only looks at the very last dimension of a multi-dimensional tensor
    

class TransformerEncoderBlock(nn.Module):

    """
        Class to define Transformer encoder block. It performs below operations:
        1. Passes an input token through a multi-head self-attention block (shape [1,273,256])
            -> every input token (of shape [1,273,256]) computes relationship with every other token, outputting a context-aware tensor of the same shape ([1,273,256])
        2. Applies a Residual connection, Dropout and Layer Normalization (LayerNorm)
            -> normalisation layer standardizes numerical values across 256 features to have a mean of 0 and variance of 1, prevents exploding gradoents and stabilises training.
            -> dropout prevents overfitting, "x +" adds the original input "x" to the processed attention output (prevents vanishing gradients during back propagation)
        3. Appends a sequential MLP layer: Linear -> GELU -> Dropout -> Linear
            -> MLP operates on each token isolated from others (unlike attention which looks across different tokens)
            -> it takes 256 features of a single token, blows them up to a higher dimension (512), applies a non-linear activation and shrinks them back down to 256.
            -> it acts as a localized reasoning step, allowing each token to think deeply about its own newly acquired context
        4. Applies another Residual connection, Dropout and Layer Normalization (LayerNorm) exactly like in step 2.

    """
    def __init__(self, d_model:int, num_heads:int, mlp_dim:int, dropout:float=0.1):
        super().__init__()
        self.attention:torch.Tensor = MultiHeadAttention(d_model, num_heads)
        self.mlp:nn.Sequential = nn.Sequential(
            nn.Linear(d_model,mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,d_model)
            )
        self.norm1:nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2:nn.LayerNorm = nn.LayerNorm(d_model)
        self.dropout:nn.Dropout = nn.Dropout(dropout)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        attention:torch.Tensor = self.attention(x)
        x = self.norm1(x + self.dropout(attention))
        mlp:torch.Tensor = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp))
        return x
        

class MultiModalEmbed(nn.Module):

    """
        Class to concatenate the visual input image tokens with its corresponding IMU reading tokens. 
        1. Invokes patch embedding in 256 dimensional space
        2. Prepends a trainable class token to the front of the sequence
        3. Adds a trainable, spatial-temporal positional embedding matrix "position_embed" to preserve order during self-attention
    """
    def __init__(self, img_size:tuple,patch_size:int,embed_dim:int,imu_seq_len:int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.imu_seq_len = imu_seq_len
        self.img_embed:PatchEmbed = PatchEmbed(self.img_size,self.patch_size,self.embed_dim)
        #[1 cls_token + 256 image patches + 16 IMU steps]
        self.cls_token:nn.Parameter = nn.Parameter(torch.randn(1,1,self.embed_dim))
        self.position_embed:nn.Parameter = nn.Parameter(torch.randn(1,1 + self.img_embed.num_patches+self.imu_seq_len, self.embed_dim))
        self.imu_proj:nn.Linear = nn.Linear(6,embed_dim)                        #IMU embeddings: 7 values (acc+gyro+ori) --> embed_dim
        self.imu_pose_embed:nn.Parameter = nn.Parameter(torch.randn(1,self.imu_seq_len,self.embed_dim))

    def forward(self,img:torch.Tensor,imu:torch.Tensor) -> torch.Tensor:
        img_tokens:torch.Tensor = self.img_embed(img)                            #Image patches: [1,256,256]
        imu_tokens:torch.Tensor = self.imu_proj(imu)                             #IMU token: [1, 16, 256]
        imu_tokens = imu_tokens.unsqueeze(1)                                       
        tokens:torch.Tensor = torch.cat([self.cls_token, img_tokens,imu_tokens],dim=1)           #Concatenating image tokens + imu tokens: [1, 272, 256]
        # shapes: [1, 1, 256], [1, 256, 256], [1, 1, 256]
        # result: [1, 1 + 256 + 1, 256] = [1, 258, 256]
        return tokens
    

class MultiModalViT(nn.Module):

    """
        This class creates a multi-modal ViT transformer model. 
        Predicts 3D spatial locations for 16 consecutive future time steps (or tracking 16 distict trajectory keypoints).
        Each of these 16 points is a 3D coordinate (x,y,z), providing a total regression output of 48 coordinates total. 
    """
    def __init__(self,
                 img_size:tuple=(256,256),
                 patch_size:int=16,
                 imu_seq_len:int=1,
                 embed_dim:int=256,
                 depth:int=6,
                 n_heads:int=8,
                 mlp_dim:int=512,
                 dropout:float=0.1,
                 pred_horizon:int=16
                 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.imu_seq_len = imu_seq_len
        self.embed_dim = embed_dim
        self.pred_horizon = pred_horizon
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.depth = depth
        
        self.embed:MultiModalEmbed = MultiModalEmbed(self.img_size,self.patch_size,embed_dim=self.embed_dim,imu_seq_len=self.imu_seq_len)
        self.position_embed:nn.Parameter = self.embed.position_embed
        


        self.blocks:nn.ModuleList = nn.ModuleList([                                    #stacking transformer blocks
            TransformerEncoderBlock(self.embed_dim, self.n_heads, self.mlp_dim, self.dropout)
            for _ in range(self.depth)
        ])
        self.norm:nn.LayerNorm = nn.LayerNorm(self.embed_dim)

        self.head:nn.Linear = nn.Linear(self.embed_dim,self.pred_horizon*3)            #output regression layer mapping 256 hidden features to 48 future 3D coordinates (16 steps X 3 axes)


    def forward_features(self,x:torch.Tensor,imu:torch.Tensor):

        """
            Function call to prepare tokens, apply self-attention and extract core information.
            1. Embedding raw image and IMU readings data and projecting them into shared 256-dimensional space
            2. Copy single "cls_token" vector across the batch size B to pair it with every sample in the batch
            3. Appends cls_token to the front of feature tokens
            4. Adds spatial-temporal coordinates to each token
            5. Runs the data through a transformer encoder bloxk of depth 6, letting all tokens cross-examine each other via multi-head attention.
            6. Extracts only the first token i.e. cls_token from the sequence, because it has successfully absorbed all critical information from both images and IMU tokens.
        """
        B = x.shape[0]
        x = self.embed(x,imu)
        x = x + self.position_embed
        
        #Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:,0]
        return cls_out
    
    def forward(self, x, imu):
        """
            function call to pass raw images and IMU readings and get a context-rich cls_out token and passing it to regreesion layer (nn.Linear) that maps the 256 features to
            final target predictions (16 continuous tracking coordinates)
        """

        cls_out = self.forward_features(x, imu)
        return self.head(cls_out)
    

