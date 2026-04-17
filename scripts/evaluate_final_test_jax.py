# scripts/evaluate_final_test_jax.py
import yaml
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes
from data.npz_loader import get_dataloaders
from models.actor_critic_jax import OfflineIRLActorCriticJax
from scripts.train_single_jax import calculate_metrics_jax

def final_judgment_jax(config_path, checkpoint_path, data_dir):
    print("\n" + "=" * 50)
    print("⚠️ UNLOCKING THE TEST SET VAULT (JAX ENGINE)")
    print("=" * 50 + "\n")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    _, _, test_loader = get_dataloaders(data_dir, config)
    if test_loader is None:
        raise ValueError("Test dataset is empty!")

    # 1. 初始化模型形状
    model = OfflineIRLActorCriticJax(config)
    rng = jax.random.PRNGKey(0)
    dummy_b = jnp.ones((1, config['model']['num_bin_channels'], 19, 19), jnp.float32)
    dummy_g = jnp.ones((1, config['model']['num_global_channels']), jnp.float32)
    variables = model.init(rng, dummy_b, dummy_g, train=False)

    # 2. 读取 msgpack 检查点
    with open(checkpoint_path, "rb") as f:
        state_dict = from_bytes(None, f.read())
    
    params = state_dict['params']
    batch_stats = state_dict.get('batch_stats', {})

    # 3. 编译高速推理核
    @jax.jit
    def infer_step(board_x, global_x, expert_a):
        (logits, _), _ = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            board_x, global_x, train=False, mutable=False
        )
        return calculate_metrics_jax(logits, expert_a)

    total_acc, total_margin, batches = 0.0, 0.0, 0
    for batch in test_loader:
        b_x, g_x, e_a, _ = [jnp.array(np.array(x)) for x in batch]
        acc, margin = infer_step(b_x, g_x, e_a)
        total_acc += acc.item()
        total_margin += margin.item()
        batches += 1

    print(f"\n[FINAL BLIND TEST RESULT - JAX]")
    print(f"Absolute Generalization Accuracy : {total_acc / batches * 100:.2f}%")
    print(f"Final Decision Margin (Confidence) : {total_margin / batches:.3f}\n")

if __name__ == "__main__":
    # final_judgment_jax("configs/grid_search_params.yaml", "logs/YOUR_BEST_CKPT.msgpack", "data/kata_games")
    pass
