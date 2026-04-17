# optim/sgld_jax.py
import optax
import jax
import jax.numpy as jnp

def sgld_transform(learning_rate, weight_decay=0.0, noise_std=0.01):
    """
    纯函数式的 Optax SGLD 优化器。
    要求每次 update 时通过 extra_args 传入 'prng_key'。
    """
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None, **extra_args):
        if params is None:
            raise ValueError("SGLD needs params to calculate weight decay.")
        
        prng_key = extra_args.get('prng_key', None)
        if prng_key is None:
            raise ValueError("SGLD needs a prng_key injected via extra_args.")

        # 分裂随机数种子，确保每个参数张量的噪声都是独立的
        treedef = jax.tree_util.tree_structure(params)
        keys = jax.random.split(prng_key, treedef.num_leaves)
        keys_tree = jax.tree_util.tree_unflatten(treedef, keys)

        def apply_sgld(grad, p, key):
            # 1. 施加 L2 重力
            grad = grad + weight_decay * p
            # 2. 生成热力学涨落 (量子扰动)
            noise = jax.random.normal(key, shape=p.shape) * noise_std
            # 3. 动力学方程
            return -learning_rate * grad + jnp.sqrt(learning_rate) * noise

        # 将 SGLD 物理法则映射到整个参数宇宙
        new_updates = jax.tree_util.tree_map(apply_sgld, updates, params, keys_tree)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)
