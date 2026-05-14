from typing import Tuple, List, Dict
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.recursive_reasoning.gram import VariationalLayer


@dataclass
class LoopedTransformerInnerCarry:
    z: torch.Tensor


@dataclass
class LoopedTransformerCarry:
    inner_carry: LoopedTransformerInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class LoopedTransformerConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    total_cycles: int  # T — number of weight-tied TB applications per inner forward
    supervision_cycles: int = 0  # K — last K cycles: grad tape + per-cycle readout + per-cycle CE loss
                                  # (Yang multi-iter loss, arXiv:2311.12424). Final loss = mean of K CEs.
                                  # First (T-K) cycles run under no_grad for memory saving.
                                  # 0 or >=T → K = T (fully supervised).

    L_layers: int      # depth of the shared TB (stack of transformer blocks)

    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    mlp_t: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

    temperature: float = 0.0
    use_mlp_lm_head: bool = False

    # Stochastic guidance (GRAM-style periodic z injection on top of looped backbone).
    # When False, model is identical to vanilla looped TF.
    use_stochastic_guidance: bool = False
    inject_period: int = 5        # P — TB calls per block (default = L_cycles+1 = 1 H-cycle)
    supervised_blocks: int = 3    # K_b — last K_b blocks run with grad + readout + KL
    latent_dim: int = 256
    use_mlp_sampler: bool = False

    # ACT: gradient only at the predicted-halt step (TRM/GRAM-style q_halt halting).
    # When True (requires halt_max_steps > 1 and training), the full ACT chain runs inside
    # ONE forward() call: outer steps are rolled out under no_grad, q_halt (> 0) decides
    # per-sample halting, and a SINGLE grad pass on each sample's halt-step input produces
    # the supervised last-K-cycle readouts. backward() fires once per puzzle.
    # q_halt is supervised at every rolled-out step (the loss head reads the *_steps tensors).
    # Eval path and the vanilla (halt_max_steps=1) path are unaffected.
    act_grad_at_halt_only: bool = False


class LoopedTransformerBlock(nn.Module):
    def __init__(self, config: LoopedTransformerConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
            )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = rms_norm(
                hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
                variance_epsilon=self.norm_eps,
            )
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class LoopedTransformerBackbone(nn.Module):
    """Stack of L_layers blocks applied once — the shared TB that gets looped T times."""

    def __init__(self, layers: List[LoopedTransformerBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class LoopedTransformerInner(nn.Module):
    def __init__(self, config: LoopedTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size,
            init_std=embed_init_std, cast_to=self.forward_dtype,
        )

        if self.config.use_mlp_lm_head:
            self.lm_head = SwiGLU(self.config.hidden_size, self.config.expansion, out_dim=self.config.vocab_size)
        else:
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = (
            -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        )
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size,
                init_std=embed_init_std, cast_to=self.forward_dtype,
            )

        self.backbone = LoopedTransformerBackbone(
            layers=[LoopedTransformerBlock(self.config) for _ in range(self.config.L_layers)]
        )

        if self.config.use_stochastic_guidance:
            self.variational = VariationalLayer(
                hidden_size=self.config.hidden_size,
                latent_dim=self.config.latent_dim,
                vocab_size=self.config.vocab_size,
                use_mlp=self.config.use_mlp_sampler,
                expansion=self.config.expansion,
            )

        self.z_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return LoopedTransformerInnerCarry(
            z=torch.empty(
                batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: LoopedTransformerInnerCarry):
        return LoopedTransformerInnerCarry(
            z=torch.where(reset_flag.view(-1, 1, 1), self.z_init, carry.z),
        )

    def _inject(self, z: torch.Tensor, batch: Dict[str, torch.Tensor], capture_kl: bool):
        """Periodic z injection (GRAM-style). Conditions token slots only; preserves puzzle_emb."""
        puzzle_part = z[:, :self.puzzle_emb_len]
        h = z[:, self.puzzle_emb_len:]

        prior_mu = prior_logvar = post_mu = post_logvar = None
        if self.training:
            # Clamp ignored positions (IGNORE_LABEL_ID = -100) to 0 — they're masked out of
            # the CE loss anyway; here we just need a valid one-hot for the posterior encoder.
            labels_safe = batch["labels"].clamp(min=0).long()
            y_oh = F.one_hot(labels_safe, num_classes=self.config.vocab_size).to(torch.float32)
            z_var, post_mu_, post_logvar_ = self.variational.sample_posterior(h, y_oh)
            if capture_kl:
                _, prior_mu_, prior_logvar_ = self.variational.sample_prior(h)
                prior_mu, prior_logvar = prior_mu_, prior_logvar_
                post_mu, post_logvar = post_mu_, post_logvar_
        else:
            z_var, _, _ = self.variational.sample_prior(h)

        B, N, _ = z_var.shape
        z_proj_flat = self.variational.z_proj(z_var.reshape(B * N, -1))
        h_cond = h + z_proj_flat.reshape(B, N, -1).to(self.forward_dtype)
        new_z = torch.cat([puzzle_part, h_cond], dim=1)
        return new_z, (prior_mu, prior_logvar, post_mu, post_logvar)

    def forward(
        self,
        carry: LoopedTransformerInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[LoopedTransformerInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)

        # Yang-style Looped Transformer (arXiv:2311.12424).
        #   z_{t+1} = TB(z_t + x) — weight-tied recurrence, input re-injected every cycle.
        #   Training: Yang multi-iter loss — last K cycles each produce a readout; loss = mean of K CEs.
        #   First (T-K) cycles run under no_grad (memory saver, Yang n_loop_start convention).
        # When use_stochastic_guidance=True: T total cycles are grouped into blocks of size P,
        #   and a GRAM-style variational z is injected at every block boundary. The last K_b
        #   blocks run with grad and produce readouts + (prior,posterior) for KL.
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        z = carry.z

        vae_outputs: Dict[str, torch.Tensor] = {}

        if self.config.use_stochastic_guidance:
            P = self.config.inject_period
            K_b = self.config.supervised_blocks
            T = self.config.total_cycles
            H_b = max(T // P, 1)
            if K_b <= 0 or K_b > H_b:
                K_b = H_b
            warmup_b = H_b - K_b

            with torch.no_grad():
                for _ in range(warmup_b):
                    for _ in range(P):
                        z = self.backbone(z + input_embeddings, **seq_info)
                    z, _ = self._inject(z, batch, capture_kl=False)

            intermediate_logits = []
            prior_mus, prior_logvars, post_mus, post_logvars = [], [], [], []
            for _ in range(K_b):
                for _ in range(P):
                    z = self.backbone(z + input_embeddings, **seq_info)
                z, (pmu, plv, qmu, qlv) = self._inject(z, batch, capture_kl=self.training)
                intermediate_logits.append(self.lm_head(z)[:, self.puzzle_emb_len:])
                if self.training:
                    prior_mus.append(pmu); prior_logvars.append(plv)
                    post_mus.append(qmu); post_logvars.append(qlv)

            if self.training and prior_mus:
                vae_outputs = {
                    "prior_mu":     torch.stack(prior_mus, dim=0),
                    "prior_logvar": torch.stack(prior_logvars, dim=0),
                    "post_mu":      torch.stack(post_mus, dim=0),
                    "post_logvar":  torch.stack(post_logvars, dim=0),
                    # Explicit divisor so the loss head averages KL over K_b grad blocks
                    # (mirrors lm_loss being averaged over K readouts in Yang multi-iter loss).
                    "kl_n_blocks":  torch.tensor(len(prior_mus), device=z.device),
                }
        else:
            K = self.config.supervision_cycles
            if K <= 0 or K >= self.config.total_cycles:
                K = self.config.total_cycles
            warmup = self.config.total_cycles - K

            with torch.no_grad():
                for _ in range(warmup):
                    z = self.backbone(z + input_embeddings, **seq_info)

            intermediate_logits = []
            for _ in range(K):
                z = self.backbone(z + input_embeddings, **seq_info)
                intermediate_logits.append(self.lm_head(z)[:, self.puzzle_emb_len:])

        logits = intermediate_logits[-1]
        # (K, B, seq, vocab) — ACTLossHead averages CE across K (multi-iter loss).
        stacked_logits = torch.stack(intermediate_logits, dim=0)

        new_carry = LoopedTransformerInnerCarry(z=z.detach())

        if not self.training and self.config.temperature > 0.0:
            logits = logits / self.config.temperature

        q_logits = self.q_head(z[:, 0]).to(torch.float32)
        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1]), stacked_logits, vae_outputs


class LoopedTransformer(nn.Module):
    """ACT wrapper around the shared-weight recursive backbone."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LoopedTransformerConfig(**config_dict)
        self.inner = LoopedTransformerInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return LoopedTransformerCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def _act_rollout(self, z: torch.Tensor, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """act_grad_at_halt_only — phase 1: roll the ACT chain out under no_grad.

        q_halt (> 0) decides per-sample halting; for each sample we snapshot the z that
        *enters* its halt step. q_halt is recomputed outside no_grad so gradient still
        reaches q_head (the backbone stays detached). Returns the gathered halt-step inputs,
        the per-sample halt step, and the per-rolled-out-step q_halt tensors.
        """
        B = data["inputs"].shape[0]
        device = data["inputs"].device
        halt_max = self.config.halt_max_steps

        halt_input_z = z.clone()                                       # per-sample z entering the halt step
        halted = torch.zeros(B, dtype=torch.bool, device=device)
        halt_step = torch.zeros(B, dtype=torch.int32, device=device)

        # Exploration: force a per-sample minimum number of steps before halting is allowed.
        min_halt = (
            (torch.rand(B, device=device) < self.config.halt_exploration_prob)
            * torch.randint(2, halt_max + 1, (B,), device=device, dtype=torch.int32)
        )

        qh_steps, pred_steps, active_steps = [], [], []
        for step in range(1, halt_max + 1):
            active = ~halted
            with torch.no_grad():
                rolled, logits_s, _, _, _ = self.inner(
                    LoopedTransformerInnerCarry(z=z), data
                )
            # q_halt recomputed outside no_grad so gradient reaches q_head (input stays detached).
            qh = self.inner.q_head(rolled.z[:, 0]).to(torch.float32)[..., 0]
            qh_steps.append(qh)
            pred_steps.append(torch.argmax(logits_s, dim=-1))
            active_steps.append(active)

            if step >= halt_max:
                newly = active                                         # last step: everyone halts
            else:
                newly = active & (qh.detach() > 0) & (step >= min_halt)
            # INVARIANT: snapshot `z` (the *input* to this step) BEFORE `z = rolled.z` below,
            # so halt_input_z holds the input to each sample's halt step, not its output.
            halt_input_z = torch.where(newly.view(-1, 1, 1), z, halt_input_z)
            halt_step = torch.where(newly, torch.full_like(halt_step, step), halt_step)
            halted = halted | newly
            z = rolled.z
            if bool(halted.all()):
                break

        return {
            "halt_input_z": halt_input_z,
            "halt_step": halt_step,
            # Per rolled-out step (n_steps, B[, seq]) — loss head does masked multi-step q_halt BCE.
            "q_halt_logits_steps": torch.stack(qh_steps, dim=0),
            "q_halt_preds_steps": torch.stack(pred_steps, dim=0),
            "q_halt_active_steps": torch.stack(active_steps, dim=0),
        }

    def _act_grad_pass(
        self,
        roll: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
    ) -> Tuple[LoopedTransformerCarry, Dict[str, torch.Tensor]]:
        """act_grad_at_halt_only — phase 2: single grad pass at each sample's halt step.

        Each sample resumes from the z that entered its own halt step (gathered by
        _act_rollout), so backbone gradient flows only from the last K cycles of that step.
        Returns the carry (halted=all True → next batch resets) and the loss-head outputs.
        """
        B = data["inputs"].shape[0]
        device = data["inputs"].device

        grad_carry = LoopedTransformerInnerCarry(z=roll["halt_input_z"])
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), stacked_logits, vae_outputs = self.inner(grad_carry, data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "intermediate_logits": stacked_logits,
            "q_halt_logits_steps": roll["q_halt_logits_steps"],
            "q_halt_preds_steps": roll["q_halt_preds_steps"],
            "q_halt_active_steps": roll["q_halt_active_steps"],
            **vae_outputs,
        }
        new_carry = LoopedTransformerCarry(
            inner_carry=LoopedTransformerInnerCarry(z=new_inner_carry.z),
            steps=roll["halt_step"],
            halted=torch.ones(B, dtype=torch.bool, device=device),     # chain done → next batch resets
            current_data=data,
        )
        return new_carry, outputs

    def forward(
        self,
        carry: LoopedTransformerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[LoopedTransformerCarry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        if self.config.act_grad_at_halt_only and self.training and self.config.halt_max_steps > 1:
            return self._act_grad_pass(self._act_rollout(new_inner_carry.z, new_current_data), new_current_data)

        new_inner_carry, logits, (q_halt_logits, q_continue_logits), stacked_logits, vae_outputs = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "intermediate_logits": stacked_logits,  # always emit — Yang multi-iter loss is mandatory
            **vae_outputs,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return LoopedTransformerCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
