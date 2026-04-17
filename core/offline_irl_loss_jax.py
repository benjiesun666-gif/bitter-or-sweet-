# core/offline_irl_loss_jax.py
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax.scipy.special import logsumexp

def compute_offline_irl_loss(policy_logits, q_values, expert_actions, config):
    """
    JAX 版：基于能量的最大熵离线 IRL 联合损失。
    纯函数无状态实现。
    """
    temperature = config.get('training', {}).get('entropy_temp', 0.1)

    # -----------------------------------------
    # 1. Critic Loss (基于 InfoNCE)
    # -----------------------------------------
    # 使用高级索引提取 expert Q (等价于 PyTorch 的 gather)
    batch_indices = jnp.arange(q_values.shape[0])
    expert_q = q_values[batch_indices, expert_actions]

    # 配分函数的对数
    q_logsumexp = logsumexp(q_values, axis=-1)

    # InfoNCE 损失
    base_irl_loss = q_logsumexp - expert_q
    loss_critic = jnp.mean(base_irl_loss)

    # -----------------------------------------
    # 2. Actor Loss (策略提炼)
    # -----------------------------------------
    policy_log_probs = jnn.log_softmax(policy_logits, axis=-1)
    policy_probs = jnp.exp(policy_log_probs)

    # 【核心严谨】：切断梯度，Actor 绝对不能破坏 Critic 能量场
    q_target = jax.lax.stop_gradient(q_values)

    # KL 散度
    actor_loss_per_state = jnp.sum(
        policy_probs * (temperature * policy_log_probs - q_target),
        axis=-1
    )
    loss_actor = jnp.mean(actor_loss_per_state)

    total_loss = loss_actor + loss_critic

    return total_loss, loss_actor, loss_critic
