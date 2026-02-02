from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class GenerativeRecursiveReasoningModelInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class GenerativeRecursiveReasoningModelCarry:
    inner_carry: GenerativeRecursiveReasoningModelInnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class GenerativeRecursiveReasoningModelConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # Variational parameters
    latent_dim: int = 256
    kl_coef: float = 0.1
    use_mlp_sampler: bool = False  # Use MLP for prior/posterior instead of linear projection

    # Variational sampling control (for ablation)
    fix_mu: bool = False  # If True, force mu = 0 (both prior and posterior)
    fix_std: bool = False  # If True, force std = 0, deterministic (both prior and posterior)

    # Decoder parameters
    use_mlp_lm_head: bool = False  # Use MLP decoder (SwiGLU) instead of linear projection

    # Value head parameters
    use_value_head: bool = False  # Add value head to predict current state accuracy

    # Patchify parameters (for image data)
    use_patchify: bool = False  # Enable patchify for image inputs
    patch_size: int = 2  # Patch size (2 for MNIST)
    in_channels: int = 1  # Number of input channels (1 for MNIST)
    image_size: int = 28  # Image size (28 for MNIST)


class VariationalLayer(nn.Module):
    """Variational layer with optional MLP encoders"""
    def __init__(self, hidden_size: int, latent_dim: int, vocab_size: int, use_mlp: bool = False, expansion: float = 2.0,
                 fix_mu: bool = False, fix_std: bool = False, posterior_y_dim: int = None):
        super().__init__()
        self.use_mlp = use_mlp
        self.fix_mu = fix_mu
        self.fix_std = fix_std
        # posterior_y_dim: dimension of y input for posterior (default: vocab_size, or hidden_size for patchify)
        self.posterior_y_dim = posterior_y_dim if posterior_y_dim is not None else vocab_size

        if use_mlp:
            # Prior: p(z|x) - MLP encoder outputs 2*latent_dim (mu and logvar)
            self.prior_mlp = SwiGLU(hidden_size, expansion, out_dim=2 * latent_dim)

            # Posterior: q(z|x, y) - MLP encoder outputs 2*latent_dim (mu and logvar)
            self.post_mlp = SwiGLU(hidden_size + self.posterior_y_dim, expansion, out_dim=2 * latent_dim)

            # Project z back to hidden_size
            self.z_proj = nn.Linear(latent_dim, hidden_size)
        else:
            # Prior: p(z|x) - Linear projection
            self.prior_mu = nn.Linear(hidden_size, latent_dim)
            self.prior_logvar = nn.Linear(hidden_size, latent_dim)

            # Posterior: q(z|x, y) - Linear projection
            self.post_mu = nn.Linear(hidden_size + self.posterior_y_dim, latent_dim)
            self.post_logvar = nn.Linear(hidden_size + self.posterior_y_dim, latent_dim)

            # Project z back to hidden_size
            self.z_proj = nn.Linear(latent_dim, hidden_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample_prior(self, x):
        B, N, D = x.shape
        h = x.reshape(B * N, D).to(torch.float32)  # (B*N, D)

        if self.use_mlp:
            mlp_out = self.prior_mlp(h)  # (B*N, 2*latent_dim)
            mu, logvar = mlp_out.chunk(2, dim=-1)  # Each (B*N, latent_dim)
        else:
            mu = self.prior_mu(h)  # (B*N, latent_dim)
            logvar = self.prior_logvar(h)  # (B*N, latent_dim)

        # Apply ablation flags
        if self.fix_mu:
            mu = torch.zeros_like(mu)
        if self.fix_std:
            logvar = torch.full_like(logvar, -1e10)  # std ≈ 0

        z = self.reparameterize(mu, logvar)  # (B*N, latent_dim)

        # Reshape back to (B, N, latent_dim)
        mu = mu.reshape(B, N, -1)
        logvar = logvar.reshape(B, N, -1)
        z = z.reshape(B, N, -1)
        return z, mu, logvar

    def sample_posterior(self, x, y):
        B, N, D1 = x.shape
        _, _, D2 = y.shape
        assert N == y.shape[1], "Sequence length mismatch between x and y"
        h = x.reshape(B * N, D1).to(torch.float32)  # (B*N, D)
        y_flat = y.reshape(B * N, D2).to(torch.float32)  # (B*N, D)
        combined = torch.cat([h, y_flat], dim=-1)  # (B*N, D+vocab_size)

        if self.use_mlp:
            mlp_out = self.post_mlp(combined)  # (B*N, 2*latent_dim)
            mu, logvar = mlp_out.chunk(2, dim=-1)  # Each (B*N, latent_dim)
        else:
            mu = self.post_mu(combined)  # (B*N, latent_dim)
            logvar = self.post_logvar(combined)  # (B*N, latent_dim)

        # Apply ablation flags
        if self.fix_mu:
            mu = torch.zeros_like(mu)
        if self.fix_std:
            logvar = torch.full_like(logvar, -1e10)  # std ≈ 0

        z = self.reparameterize(mu, logvar)  # (B*N, latent_dim)

        # Reshape back to (B, N, latent_dim)
        mu = mu.reshape(B, N, -1)
        logvar = logvar.reshape(B, N, -1)
        z = z.reshape(B, N, -1)
        return z, mu, logvar


class GenerativeRecursiveReasoningModelBlock(nn.Module):
    def __init__(self, config: GenerativeRecursiveReasoningModelConfig) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            effective_seq_len = (self.config.image_size // self.config.patch_size) ** 2 if self.config.use_patchify else self.config.seq_len
            self.mlp_t = SwiGLU(
                hidden_size=effective_seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class GenerativeRecursiveReasoningModelReasoningModule(nn.Module):
    def __init__(self, layers: List[GenerativeRecursiveReasoningModelBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class GenerativeRecursiveReasoningModelInner(nn.Module):
    def __init__(self, config: GenerativeRecursiveReasoningModelConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        # LM head: linear or MLP decoder
        if self.config.use_mlp_lm_head:
            self.lm_head = SwiGLU(self.config.hidden_size, self.config.expansion, out_dim=self.config.vocab_size)
        else:
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # Value head (optional)
        if self.config.use_value_head:
            self.value_head = CastedLinear(self.config.hidden_size, 1, bias=True)

        # Patchify layers (for image data, following dit.py)
        if self.config.use_patchify:
            p, c, h = self.config.patch_size, self.config.in_channels, self.config.hidden_size
            self.init_conv_seq = nn.Sequential(
                nn.Conv2d(c, h // 2, kernel_size=5, padding=2, stride=1),
                nn.SiLU(),
                nn.GroupNorm(32, h // 2),
                nn.Conv2d(h // 2, h // 2, kernel_size=5, padding=2, stride=1),
                nn.SiLU(),
                nn.GroupNorm(32, h // 2),
            )
            self.patch_embedder = nn.Linear(p * p * h // 2, h, bias=True)
            nn.init.constant_(self.patch_embedder.bias, 0)
            # For output: project back to patch logits
            self.patch_out = nn.Linear(h, p * p * c * self.config.vocab_size, bias=True)
            # Compute patched seq_len for position embeddings
            self.num_patches = (self.config.image_size // p) ** 2

        # Variational layer
        posterior_y_dim = self.config.hidden_size if self.config.use_patchify else self.config.vocab_size
        self.variational = VariationalLayer(
            self.config.hidden_size,
            self.config.latent_dim,
            self.config.vocab_size,
            use_mlp=self.config.use_mlp_sampler,
            fix_mu=self.config.fix_mu,
            fix_std=self.config.fix_std,
            posterior_y_dim=posterior_y_dim
        )

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # Compute effective seq_len for position embeddings (patchified or original)
        effective_seq_len = self.num_patches if self.config.use_patchify else self.config.seq_len

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=effective_seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(effective_seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = GenerativeRecursiveReasoningModelReasoningModule(layers=[GenerativeRecursiveReasoningModelBlock(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify image: (B, C, H, W) -> (B, num_patches, patch_size*patch_size*C)"""
        B, C, H, W = x.size()
        p = self.config.patch_size
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x  # (B, num_patches, p*p*C)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Unpatchify: (B, num_patches, p*p*C*vocab) -> (B, C, H, W, vocab)"""
        p = self.config.patch_size
        c = self.config.in_channels
        v = self.config.vocab_size
        h = w = self.config.image_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c, v)
        x = x.permute(0, 5, 1, 3, 2, 4, 6)  # (B, C, h, p, w, p, vocab)
        x = x.reshape(x.shape[0], c, h * p, w * p, v)  # (B, C, H, W, vocab)
        return x

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Patchify branch for image data
        if self.config.use_patchify:
            B = input.shape[0]
            c, s = self.config.in_channels, self.config.image_size
            # Reshape flat input to image: (B, seq) -> (B, C, H, W)
            # Convert token values (1 to vocab_size-1) to pixel values (0 to vocab_size-2)
            x_img = (input - 1).view(B, c, s, s).float()
            # Normalize to [-1, 1] and apply conv (vocab_size-2 = num_levels - 1)
            num_levels = self.config.vocab_size - 1  # Exclude PAD token
            x_img = (2 * x_img / max(num_levels - 1, 1)) - 1.0
            x_conv = self.init_conv_seq(x_img)  # (B, h//2, H, W)
            # Patchify and embed
            x_patch = self._patchify(x_conv)  # (B, num_patches, p*p*h//2)
            embedding = self.patch_embedder(x_patch)  # (B, num_patches, hidden_size)
        else:
            # Token embedding
            embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def _get_label_embedding(self, labels: torch.Tensor) -> torch.Tensor:
        """Get label embedding for posterior in patchify mode.
        
        Args:
            labels: (B, seq_len) flat labels
        Returns:
            (B, num_patches, hidden_size) patch embeddings of labels
        """
        B = labels.shape[0]
        c, s = self.config.in_channels, self.config.image_size
        # Reshape flat labels to image: (B, seq) -> (B, C, H, W)
        # Convert token values (1 to vocab_size-1) to pixel values (0 to vocab_size-2)
        y_img = (labels - 1).view(B, c, s, s).float()
        # Normalize to [-1, 1] and apply conv (same as input)
        num_levels = self.config.vocab_size - 1  # Exclude PAD token
        y_img = (2 * y_img / max(num_levels - 1, 1)) - 1.0
        y_conv = self.init_conv_seq(y_img)  # (B, h//2, H, W)
        # Patchify and embed
        y_patch = self._patchify(y_conv)  # (B, num_patches, p*p*h//2)
        y_emb = self.patch_embedder(y_patch)  # (B, num_patches, hidden_size)
        # Detach to prevent gradient flow through input encoder when processing posterior labels
        return y_emb.to(self.forward_dtype)

    def empty_carry(self, batch_size: int):
        seq_len = self.num_patches if self.config.use_patchify else self.config.seq_len
        return GenerativeRecursiveReasoningModelInnerCarry(
            z_H=torch.empty(batch_size, seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: GenerativeRecursiveReasoningModelInnerCarry):
        return GenerativeRecursiveReasoningModelInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: GenerativeRecursiveReasoningModelInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[GenerativeRecursiveReasoningModelInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

                # Variational in the middle of cycles (posterior during training, prior during eval)
                puzzle_emb = z_H[:, :self.puzzle_emb_len]
                h_t = z_H[:, self.puzzle_emb_len:]

                if self.training:
                    if self.config.use_patchify:
                        y_tar = self._get_label_embedding(batch["labels"])
                    else:
                        y_tar = F.one_hot(batch["labels"].long(), num_classes=self.config.vocab_size).to(torch.float32)
                    z, _, _ = self.variational.sample_posterior(h_t, y_tar)
                else:
                    z, _, _ = self.variational.sample_prior(h_t)
                # Decode: condition h_t on z (per-token z)
                B, N, _ = z.shape
                z_flat = z.reshape(B * N, -1)  # (B*N, latent_dim)
                z_proj_flat = self.variational.z_proj(z_flat)  # (B*N, hidden_size)
                z = z_proj_flat.reshape(B, N, -1).to(self.forward_dtype)  # (B, N, D)
                h_t_conditioned = h_t + z
                z_H = torch.cat([puzzle_emb, h_t_conditioned], dim=1)

        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        
        puzzle_emb = z_H[:, :self.puzzle_emb_len]
        h_T = z_H[:, self.puzzle_emb_len:]
        sample_in = h_T
        # Variational sampling
        vae_outputs = {}
        z_pred, prior_mu, prior_logvar = self.variational.sample_prior(sample_in)
        vae_outputs['prior_mu'] = prior_mu
        vae_outputs['prior_logvar'] = prior_logvar

        if self.training:
            # Sample from posterior q(z|h_T, y_tar)
            if self.config.use_patchify:
                y_tar = self._get_label_embedding(batch["labels"])
            else:
                y_tar = F.one_hot(batch["labels"].long(), num_classes=self.config.vocab_size).to(torch.float32)
            z_post, post_mu, post_logvar = self.variational.sample_posterior(sample_in, y_tar)
            vae_outputs['post_mu'] = post_mu
            vae_outputs['post_logvar'] = post_logvar
            z = z_post
        else:
            z = z_pred

        # Decode: condition h_T on z (per-token z)
        B, N, _ = z.shape
        z_flat = z.reshape(B * N, -1)  # (B*N, latent_dim)
        z_proj_flat = self.variational.z_proj(z_flat)  # (B*N, hidden_size)
        z = z_proj_flat.reshape(B, N, -1).to(self.forward_dtype)  # (B, N, D)
        h_T_conditioned = h_T + z  # (B, N, D)

        # LM Outputs
        z_H_conditioned = torch.cat([puzzle_emb, h_T_conditioned], dim=1)
        new_carry = GenerativeRecursiveReasoningModelInnerCarry(z_H=z_H_conditioned.detach(), z_L=z_L.detach())
        if self.config.use_patchify:
            # (B, num_patches, hidden) -> (B, num_patches, p*p*C*vocab) -> (B, C, H, W, vocab) -> (B, seq, vocab)
            patch_logits = self.patch_out(h_T_conditioned)
            img_logits = self._unpatchify(patch_logits)  # (B, C, H, W, vocab)
            output = img_logits.reshape(img_logits.shape[0], -1, self.config.vocab_size)  # (B, seq, vocab)
        else:
            output = self.lm_head(h_T_conditioned)
        q_logits = self.q_head(puzzle_emb[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position

        # Value head (optional): predict current state accuracy from mean-pooled h_T
        value_pred = None
        if self.config.use_value_head:
            h_T_mean = h_T_conditioned.mean(dim=1)  # (B, hidden_size)
            value_pred = torch.sigmoid(self.value_head(h_T_mean).to(torch.float32)).squeeze(-1)  # (B,) in [0, 1]

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), vae_outputs, value_pred


class GenerativeRecursiveReasoningModel(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = GenerativeRecursiveReasoningModelConfig(**config_dict)
        print("Initializing GRAM with config:", self.config)
        self.inner = GenerativeRecursiveReasoningModelInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return GenerativeRecursiveReasoningModelCarry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: GenerativeRecursiveReasoningModelCarry, batch: Dict[str, torch.Tensor]) -> Tuple[GenerativeRecursiveReasoningModelCarry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), vae_outputs, value_pred = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            **vae_outputs  # Add VAE outputs for loss computation
        }

        # Add value prediction if available
        if value_pred is not None:
            outputs["value_pred"] = value_pred

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return GenerativeRecursiveReasoningModelCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
