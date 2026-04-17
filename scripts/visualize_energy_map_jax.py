# scripts/visualize_energy_map_jax.py
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax.serialization import from_bytes
from models.actor_critic_jax import OfflineIRLActorCriticJax
from scripts.visualize_energy_map import load_single_state # 复用原版的数据提取逻辑

def visualize_energy_jax(config_path, checkpoint_path, npz_path, frame_idx=100, save_path="energy_map_jax.png"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = OfflineIRLActorCriticJax(config)
    rng = jax.random.PRNGKey(0)
    dummy_b = jnp.ones((1, config['model']['num_bin_channels'], 19, 19), jnp.float32)
    dummy_g = jnp.ones((1, config['model']['num_global_channels']), jnp.float32)
    variables = model.init(rng, dummy_b, dummy_g, train=False)

    with open(checkpoint_path, "rb") as f:
        state_dict = from_bytes(None, f.read())
    params = state_dict['params']
    batch_stats = state_dict.get('batch_stats', {})

    board_x, global_x, expert_action, my_stones, opp_stones = load_single_state(npz_path, frame_idx)
    board_x, global_x = jnp.array(board_x.numpy()), jnp.array(global_x.numpy())

    @jax.jit
    def forward_fn(b_x, g_x):
        (logits, q_vals), _ = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            b_x, g_x, train=False, mutable=False
        )
        return logits, q_vals

    policy_logits, q_values = forward_fn(board_x, global_x)
    q_values = np.array(q_values[0])
    
    q_board = q_values[:361].reshape(19, 19)
    q_pass = q_values[361]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#DCB35C')
    for i in range(19):
        ax.plot([0, 18], [i, i], color='black', linewidth=1)
        ax.plot([i, i], [0, 18], color='black', linewidth=1)

    star_points = [3, 9, 15]
    for x in star_points:
        for y in star_points:
            ax.plot(x, y, 'o', markersize=5, color='black')

    im = ax.imshow(q_board, cmap='RdYlGn', alpha=0.65, origin='lower', extent=[-0.5, 18.5, -0.5, 18.5])

    for y in range(19):
        for x in range(19):
            if my_stones[y, x] > 0:
                ax.plot(x, y, 'o', markersize=15, color='black', markeredgecolor='white')
            elif opp_stones[y, x] > 0:
                ax.plot(x, y, 'o', markersize=15, color='white', markeredgecolor='black')

    model_top_action = np.argmax(q_values[:361])
    model_y, model_x = divmod(model_top_action, 19)
    expert_y, expert_x = divmod(expert_action, 19) if expert_action < 361 else (None, None)

    if expert_action < 361:
        ax.plot(expert_x, expert_y, '*', markersize=20, color='blue', label='Expert Move')
    ax.plot(model_x, model_y, 'X', markersize=15, color='red', label='Peak Energy')

    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 19)
    ax.set_xticks(range(19))
    ax.set_yticks(range(19))
    ax.invert_yaxis()

    plt.colorbar(im, fraction=0.046, pad=0.04, label='Critic Q-Value (Energy)')
    plt.title(f"Latent Energy Map Q(s, a)\nPass Q-Value: {q_pass:.3f}", fontsize=14, pad=15)
    plt.legend(loc='upper right')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
