import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_, Mlp
from xformers.ops import memory_efficient_attention, unbind
from .weather_embedding import WeatherEmbedding  # Pre-existing module

class VariableAwareEmbedding(nn.Module):
    """
    Physics-Informed Variable Embedding Module
    - Separates channels based on the physical characteristics of variables:
      (Group 1: Dynamics, Group 2: Thermodynamics).
    - Passes them through independent embedding layers (d1, d2). [cite: 59, 61]
    """
    def __init__(self, 
        group1_vars,  # List of Dynamics variables (e.g., U, V, Z, P) [cite: 59]
        group2_vars,  # List of Thermodynamics variables (e.g., T, Q) [cite: 59]
        img_size, 
        patch_size, 
        d1,           # Dimension size for Group 1 (e.g., 512) [cite: 61]
        d2,           # Dimension size for Group 2 (e.g., 512) [cite: 64]
        num_heads     # Number of attention heads
    ):
        super().__init__()
        self.group1_vars = group1_vars
        self.group2_vars = group2_vars
        
        # [Core] Create two independent embedding layers
        
        # 1. Dynamics Embedder (Group 1 -> d1)
        # These variables serve as the core backbone processed from Step 1 (Deep Layer). [cite: 61, 62]
        self.embed1 = WeatherEmbedding(
            variables=group1_vars, 
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=d1,
            num_heads=num_heads
        )
        
        # 2. Thermodynamics Embedder (Group 2 -> d2)
        # These variables are information integrated during Step 2 (Shallow Layer). [cite: 64, 65]
        self.embed2 = WeatherEmbedding(
            variables=group2_vars, 
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=d2,
            num_heads=num_heads
        )
        
        # (Optional) VA-MoE style: Add learnable vectors to identify variable types.
        # In VA-MoE, index embeddings allow experts to identify variables.
        # Implemented here by adding group-specific identifiers (uncomment if needed).
        # self.group_token1 = nn.Parameter(torch.zeros(1, 1, d1))
        # self.group_token2 = nn.Parameter(torch.zeros(1, 1, d2))
        # trunc_normal_(self.group_token1, std=0.02)
        # trunc_normal_(self.group_token2, std=0.02)
    
    def forward(self, x, variables):
        """
        x: (Batch, Total_Vars, H, W)
        variables: List of all variable names in the current batch
        """
        
        # 1. Dynamic Indexing
        # Locate Group 1 and Group 2 positions within the current input x.
        # (If overhead is an issue, fix the order during training and use pre-calculated indices.)
        idx1 = [variables.index(v) for v in self.group1_vars if v in variables]
        idx2 = [variables.index(v) for v in self.group2_vars if v in variables]
        
        # Validation: Ensure no variables are missing
        if len(idx1) != len(self.group1_vars) or len(idx2) != len(self.group2_vars):
             # Since input variables are usually fixed during training/inference, 
             # failure here likely indicates an initial setup issue.
             pass 

        # 2. Input Slicing
        x1_input = x[:, idx1, :, :]  # (B, N1, H, W)
        x2_input = x[:, idx2, :, :]  # (B, N2, H, W)
        
        # 3. Individual Embedding
        # embed1 output: (B, L, d1)
        emb1 = self.embed1(x1_input, self.group1_vars)
        
        # embed2 output: (B, L, d2)
        emb2 = self.embed2(x2_input, self.group2_vars)
        
        # (Optional) Add group tokens (borrowed from VA-MoE)
        # emb1 = emb1 + self.group_token1
        # emb2 = emb2 + self.group_token2
        
        # 4. Concatenate along the channel dimension
        # Result: (B, L, d1 + d2)
        # Ensures the first d1 channels are Group 1 and the last d2 are Group 2. [cite: 65]
        return torch.cat([emb1, emb2], dim=2)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations. [cite: 68]
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Linear(1, hidden_size)

    def forward(self, t):
        return self.mlp(t.unsqueeze(-1))

class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """
    A Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning. [cite: 68]
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

def _classify_variables(variables):
    """
    Classifies variables into two groups based on physical characteristics. [cite: 59, 60]
    - Group 1 (Dynamics): U, V, Z (wind components, geopotential), MSLP, Surface Pressure [cite: 59]
    - Group 2 (Thermodynamics): T (temperature), Q (specific humidity) [cite: 59]
    """
    group1_vars = []  # Dynamics
    group2_vars = []  # Thermodynamics
    
    for var in variables:
        var_lower = var.lower()
        # Dynamics: wind components (u, v), geopotential (z), pressure [cite: 59, 60]
        if any(keyword in var_lower for keyword in ['u_component', 'v_component', 'geopotential', 'pressure', 'mslp']):
            group1_vars.append(var)
        # Thermodynamics: temperature, specific humidity [cite: 59, 60]
        elif any(keyword in var_lower for keyword in ['temperature', 'specific_humidity', 'humidity']):
            group2_vars.append(var)
        else:
            # Default: Classify as Dynamics (conservative approach)
            group1_vars.append(var)
    
    return group1_vars, group2_vars

class Sonny(nn.Module):
    """
    Sonny — StepsNet architecture with variable-aware embedding. [cite: 2, 26, 58]
    Two stages: 
    Step 1 (Slow Path, width d1) on dynamics-heavy channels, [cite: 13, 61]
    Step 2 (Fast Path, full width) after fusion with thermodynamics channels. [cite: 13, 65]
    """
    def __init__(self, 
        in_img_size,
        variables,
        patch_size=2,
        hidden_size=384,  # ViT-S: 384 [cite: 76]
        depth=12,         # Total Transformer depth
        num_heads=6, 
        mlp_ratio=4.0,
        step_ratio=0.5,   # Channel ratio allocated to Step 1 (usually half)
        depth_step1=None, # Number of Step 1 blocks; defaults to depth // 2
        group1_vars=None, # List of Dynamics variables (Auto-classified if None)
        group2_vars=None, # List of Thermodynamics variables (Auto-classified if None)
        use_cnn_head=False, 
        cnn_head_channels=256, 
        **kwargs, 
    ):
        super().__init__()
        
        if in_img_size[0] % patch_size != 0:
            pad_size = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + pad_size, in_img_size[1])
        self.in_img_size = in_img_size
        self.variables = variables
        self.patch_size = patch_size
        
        # --- Core StepsNet Implementation ---
        
        # Calculate Step 1 and Step 2 Dimensions (Width)
        self.d1 = int(hidden_size * step_ratio)  # Step 1 width [cite: 61]
        self.d2 = hidden_size - self.d1          # Remaining width [cite: 65]
        
        # Classify variable groups
        if group1_vars is None or group2_vars is None:
            self.group1_vars, self.group2_vars = _classify_variables(variables)
        else:
            self.group1_vars = group1_vars
            self.group2_vars = group2_vars
        
        # 1. Variable-Aware Embedding [cite: 59]
        self.embedding = VariableAwareEmbedding(
            group1_vars=self.group1_vars,
            group2_vars=self.group2_vars,
            img_size=in_img_size,
            patch_size=patch_size,
            d1=self.d1,
            d2=self.d2,
            num_heads=num_heads,
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size)
        
        # Interval embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Adjust head count to match width (maintain dimension per head)
        head_dim = hidden_size // num_heads
        self.num_heads_1 = max(1, self.d1 // head_dim)  # Minimum 1 head
        self.num_heads_2 = num_heads  # Step 2 uses full width
        
        # Depth allocation: Sweeping Step 1 Transformer depth (ablation)
        if depth_step1 is None:
            depth_1 = depth // 2
        else:
            depth_1 = int(depth_step1)
        depth_2 = depth - depth_1
        
        if depth_1 < 1 or depth_2 < 1:
            raise ValueError(f"Invalid depth allocation: depth_1={depth_1}, depth_2={depth_2}")
        
        # Step 1 Blocks (Reduced width d1) [cite: 61, 62]
        self.step1_blocks = nn.ModuleList([
            Block(self.d1, self.num_heads_1, mlp_ratio=mlp_ratio) 
            for _ in range(depth_1)
        ])
        
        # Step 2 Blocks (Return to full hidden_size) [cite: 65]
        self.step2_blocks = nn.ModuleList([
            Block(hidden_size, self.num_heads_2, mlp_ratio=mlp_ratio) 
            for _ in range(depth_2)
        ])
        
        # **Critical**: Step 1 blocks receive d1-sized inputs, but TimestepEmbedder 
        # outputs hidden_size. Projection is required to match dimensions. [cite: 69]
        self.time_proj_step1 = nn.Linear(hidden_size, self.d1)
        
        # Prediction Head [cite: 71, 90]
        self.head = FinalLayer(hidden_size, patch_size, len(variables))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize Transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)
        
        # Initialize adaLN for each step
        for block in self.step1_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.step2_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)
        
        # Initialize time projection
        trunc_normal_(self.time_proj_step1.weight, std=0.02)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        returns imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = len(self.variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def forward(self, x, variables, time_interval):
        # 1. Embedding (B, L, hidden_size) [cite: 59]
        x = self.embedding(x, variables) 
        x = self.embed_norm_layer(x)
        
        # Time Embedding (B, hidden_size) [cite: 68]
        time_interval_emb = self.t_embedder(time_interval)
        
        # --- StepsNet Forward ---
        
        # 2. Split along the channel dimension (dim=2) into d1 and d2 [cite: 61, 64]
        x1 = x[:, :, :self.d1]  # (B, L, d1)
        x2 = x[:, :, self.d1:]  # (B, L, d2)
        
        # 3. Step 1 (Slow Path)
        # Process x1. Time embedding dimension is adjusted for Step 1. [cite: 61, 69]
        time_emb_1 = self.time_proj_step1(time_interval_emb)  # (B, d1)
        
        y1 = x1
        for block in self.step1_blocks:
            y1 = block(y1, time_emb_1)
            
        # 4. Step 2 Preparation (Concatenate)
        # Fuse processed y1 with raw x2. Result shape returns to (B, L, hidden_size). [cite: 64, 65]
        x_step2 = torch.cat([y1, x2], dim=2) 
        
        # 5. Step 2 (Fast Path) [cite: 65, 66]
        y2 = x_step2
        for block in self.step2_blocks:
            y2 = block(y2, time_interval_emb)  # Uses original time_emb
            
        # ------------------------
        
        # Final reconstruction [cite: 71]
        x = self.head(y2, time_interval_emb)
        x = self.unpatchify(x)
        
        return x

    def forward_encode_with_intermediates(
        self,
        x,
        variables,
        time_interval,
        step1_fracs=(0.25, 0.5),
    ):
        """
        Encoder path only: Extracts Step 1 hidden states at fractional depths (Slow path, width d1)
        plus final y2 latent before the prediction head. Used for frozen feature extraction 
        (e.g., Cross-Attention KV) and optional residual decoding.
        """
        x = self.embedding(x, variables)
        x = self.embed_norm_layer(x)
        time_interval_emb = self.t_embedder(time_interval)
        x1 = x[:, :, : self.d1]
        x2 = x[:, :, self.d1 :]
        time_emb_1 = self.time_proj_step1(time_interval_emb)
        depth_1 = len(self.step1_blocks)

        needed = set()
        for frac in step1_fracs:
            k = int(round(frac * depth_1)) - 1
            k = max(0, min(depth_1 - 1, k))
            needed.add(k)

        y1 = x1
        step1_hidden = {}
        for i, block in enumerate(self.step1_blocks):
            y1 = block(y1, time_emb_1)
            if i in needed:
                step1_hidden[i] = y1.clone()

        x_step2 = torch.cat([y1, x2], dim=2)
        y2 = x_step2
        for block in self.step2_blocks:
            y2 = block(y2, time_interval_emb)

        feats_ordered = []
        for frac in step1_fracs:
            k = int(round(frac * depth_1)) - 1
            k = max(0, min(depth_1 - 1, k))
            feats_ordered.append(step1_hidden[k])

        return {
            "step1_feats": feats_ordered,
            "y2": y2,
            "time_interval_emb": time_interval_emb,
        }

    def decode_from_y2(self, y2, time_interval=None, time_interval_emb=None):
        """Tail of forward: Run head + unpatchify on latent y2."""
        if time_interval_emb is None:
            if time_interval is None:
                raise ValueError("Provide time_interval or time_interval_emb")
            time_interval_emb = self.t_embedder(time_interval)
        x = self.head(y2, time_interval_emb)
        return self.unpatchify(x)
