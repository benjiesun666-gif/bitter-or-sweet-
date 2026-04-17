# models/actor_critic_jax.py
import flax.linen as nn
from .loop_lm_jax import LoopLMJax

class OfflineIRLActorCriticJax(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, board_x, global_x, train: bool = False):
        model_cfg = self.config['model']
        d_model = model_cfg['d_model']

        # 实例化 Backbone
        backbone = LoopLMJax(
            num_bin_channels=model_cfg['num_bin_channels'],
            num_global_channels=model_cfg['num_global_channels'],
            d_model=d_model,
            n_loops=model_cfg['n_loops']
        )

        latent_state = backbone(board_x, global_x, train=train)

        # Actor 头
        actor_h = nn.Dense(d_model // 2)(latent_state)
        actor_h = nn.gelu(actor_h)
        policy_logits = nn.Dense(1)(actor_h)

        # Critic 头
        critic_h = nn.Dense(d_model // 2)(latent_state)
        critic_h = nn.gelu(critic_h)
        q_values = nn.Dense(1)(critic_h)

        return policy_logits.squeeze(-1), q_values.squeeze(-1)
