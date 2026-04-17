# scripts/train_single_jax.py

import os
import csv
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax.training import train_state
from flax import struct
from flax.serialization import to_bytes, from_bytes

from models.actor_critic_jax import OfflineIRLActorCriticJax
from core.offline_irl_loss_jax import compute_offline_irl_loss
from optim.sgld_jax import sgld_transform

# ----------------- 【JAX 版：严谨的 Checkpoint 管家】 -----------------
class JaxCheckpointManager:
    def __init__(self, save_dir: str, max_keep: int = 3):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.best_checkpoints = []
        os.makedirs(self.save_dir, exist_ok=True)

    def save_latest(self, state, step, run_id):
        path = os.path.join(self.save_dir, f"{run_id}_latest.msgpack")
        with open(path, "wb") as f:
            f.write(to_bytes(state))

    def save_if_best(self, state, step, val_acc, run_id):
        if len(self.best_checkpoints) < self.max_keep or val_acc > self.best_checkpoints[0][0]:
            ckpt_name = f"{run_id}_best_valAcc_{val_acc:.4f}_step_{step}.msgpack"
            ckpt_path = os.path.join(self.save_dir, ckpt_name)

            with open(ckpt_path, "wb") as f:
                f.write(to_bytes(state))

            self.best_checkpoints.append((val_acc, ckpt_path))
            self.best_checkpoints.sort(key=lambda x: x[0])

            if len(self.best_checkpoints) > self.max_keep:
                worst_acc, worst_path = self.best_checkpoints.pop(0)
                if os.path.exists(worst_path):
                    os.remove(worst_path)
# --------------------------------------------------------------------

# 强化版状态机：挂载 BatchNorm 统计量与热力学 RNG
class IRLTrainState(train_state.TrainState):
    batch_stats: jax.core.FrozenDict
    rng: jax.Array

    def apply_gradients(self, *, grads, **kwargs):
        """重载更新函数，允许将 prng_key 注入到 SGLD 优化器中"""
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params, **kwargs
        )
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )

@jax.jit
def calculate_metrics_jax(policy_logits, expert_actions):
    """JAX 纯函数版本的指标计算"""
    preds = jnp.argmax(policy_logits, axis=-1)
    acc = jnp.mean(preds == expert_actions)

    # 计算 Margin (Top 1 - Top 2)
    top2_logits = jax.lax.top_k(policy_logits, 2)[0]
    margin = jnp.mean(top2_logits[:, 0] - top2_logits[:, 1])
    
    return acc, margin

@jax.jit
def get_weight_norm_jax(params):
    """计算模型所有权重的 L2 范数 (用于观测引力坍缩)"""
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves))

def get_infinite_numpy_batches(dataloader):
    """桥接函数：剥离 PyTorch Tensor，转换为极其干净的 Numpy 数组送给 JAX"""
    while True:
        for batch in dataloader:
            yield [np.array(x) for x in batch]


def train_run_jax(config, train_loader, val_loader, lr, weight_decay, run_id, save_dir="./logs"):
    # 1. 初始化伪随机数发生器 (PRNG) - JAX 的灵魂
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)

    # 2. 实例化模型架构
    model = OfflineIRLActorCriticJax(config)

    # 3. 极客初始化：喂入假数据以确定计算图形状
    dummy_board = jnp.ones((1, config['model']['num_bin_channels'], 19, 19), jnp.float32)
    dummy_global = jnp.ones((1, config['model']['num_global_channels']), jnp.float32)
    
    variables = model.init(init_rng, dummy_board, dummy_global, train=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    # 4. 优化器路由
    opt_type = config['training'].get('optimizer_type', 'AdamW')
    if opt_type == 'AdamW':
        # JAX 自带的 AdamW
        tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
        print("🚀 Using Optimizer: Deterministic AdamW (Phase 1: MAP Estimation)")
    elif opt_type == 'SGLD':
        noise_std = config['training'].get('sgld_noise_std', 0.001)
        tx = sgld_transform(learning_rate=lr, weight_decay=weight_decay, noise_std=noise_std)
        print(f"🌌 Using Optimizer: Thermodynamic SGLD (Phase 2: Bayesian Walk, Noise={noise_std})")
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    # 梯度裁剪防爆
    tx = optax.chain(optax.clip_by_global_norm(1.0), tx)

    # 5. 组装状态机
    state = IRLTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        rng=rng
    )

    # ==========================================
    # 🔴 XLA 编译核心区块 (极致性能)
    # ==========================================
    @jax.jit
    def train_step(state, board_x, global_x, expert_a, game_res):
        """XLA 编译的前向+反向原子核"""
        rng, step_rng = jax.random.split(state.rng)

        def loss_fn(params):
            # 前向推演
            (logits, q_vals), mutated_vars = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                board_x, global_x, train=True, mutable=['batch_stats']
            )
            loss, actor_l, critic_l = compute_offline_irl_loss(logits, q_vals, expert_a, config)
            return loss, (mutated_vars, logits, actor_l, critic_l)

        # 自动微分
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (mutated_vars, logits, actor_l, critic_l)), grads = grad_fn(state.params)
        
        # 状态跃迁 (注入 step_rng 供 SGLD 使用)
        state = state.apply_gradients(grads=grads, prng_key=step_rng)
        state = state.replace(batch_stats=mutated_vars['batch_stats'], rng=rng)
        
        acc, margin = calculate_metrics_jax(logits, expert_a)
        return state, loss, acc, margin

    @jax.jit
    def eval_step(state, board_x, global_x, expert_a, game_res):
        """XLA 编译的观测核"""
        (logits, q_vals) = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats},
            board_x, global_x, train=False
        )
        loss, _, _ = compute_offline_irl_loss(logits, q_vals, expert_a, config)
        acc, margin = calculate_metrics_jax(logits, expert_a)
        return loss, acc, margin
    # ==========================================

    max_steps = config['training'].get('max_steps', 200000)
    eval_interval = config['training'].get('eval_interval', 500)
    save_interval = config['training'].get('save_interval', 10000)

    train_iter = get_infinite_numpy_batches(train_loader)
    ckpt_manager = JaxCheckpointManager(save_dir=save_dir, max_keep=3)

    log_train_loss, log_train_acc, log_margin = 0.0, 0.0, 0.0
    print(f"--- JAX Ignition! Run {run_id} | Max Steps: {max_steps} ---")

    # 预热 XLA 编译器 (首次执行会慢，之后起飞)
    print("⏳ JIT Compiling... (This takes a few seconds)")
    _ = train_step(state, *[jnp.array(x) for x in next(train_iter)])
    print("⚡ Compilation complete! Engine at full thrust.")

    for step in range(1, max_steps + 1):
        # 取数据并扔进 XLA 引擎
        batch = next(train_iter)
        state, loss, acc, margin = train_step(state, *batch)

        log_train_loss += loss.item()
        log_train_acc += acc.item()
        log_margin += margin.item()

        if step % eval_interval == 0:
            total_val_loss, total_val_acc = 0.0, 0.0
            val_batches = 0

            # 评估循环
            for v_batch in val_loader:
                v_batch_np = [np.array(x) for x in v_batch]
                v_loss, v_acc, _ = eval_step(state, *v_batch_np)
                total_val_loss += v_loss.item()
                total_val_acc += v_acc.item()
                val_batches += 1

            w_norm = get_weight_norm_jax(state.params).item()
            avg_t_loss = log_train_loss / eval_interval
            avg_t_acc = log_train_acc / eval_interval
            avg_margin = log_margin / eval_interval
            avg_v_loss = total_val_loss / val_batches
            avg_v_acc = total_val_acc / val_batches

            print(f"Step {step:06d} | W_Norm: {w_norm:6.2f} | T_Acc: {avg_t_acc:.3f} | V_Acc: {avg_v_acc:.3f} | T_Loss: {avg_t_loss:.3f} | Margin: {avg_margin:.2f}")

            # 写入 CSV
            csv_path = os.path.join(save_dir, f"history_{run_id}_jax.csv")
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['step', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'weight_norm', 'margin'])
                writer.writerow([step, avg_t_loss, avg_v_loss, avg_t_acc, avg_v_acc, w_norm, avg_margin])

            log_train_loss, log_train_acc, log_margin = 0.0, 0.0, 0.0

            # 存盘判断
            ckpt_manager.save_if_best(state, step, avg_v_acc, run_id)

        if step % save_interval == 0 or step == max_steps:
            ckpt_manager.save_latest(state, step, run_id)
