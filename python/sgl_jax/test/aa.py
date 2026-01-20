import jax
import jax.numpy as jnp
import time

# 模拟输入参数
num_tokens = 10000  # 调大这个值以看到明显的性能差异
half_rotary_dim = 64
mrope_section = [16, 24, 24]
split_indices = [16, 40]

# 模拟数据
cos_all = jax.random.normal(jax.random.PRNGKey(0), (3, num_tokens, half_rotary_dim))
sin_all = jax.random.normal(jax.random.PRNGKey(1), (3, num_tokens, half_rotary_dim))

# --- 方法 1: 你的原始 split 逻辑 ---
def original_method(cos_all, sin_all):
    cos_splits = jnp.split(cos_all, split_indices, axis=-1)
    sin_splits = jnp.split(sin_all, split_indices, axis=-1)
    
    final_cos_list = [split_tensor[i] for i, split_tensor in enumerate(cos_splits)]
    final_sin_list = [split_tensor[i] for i, split_tensor in enumerate(sin_splits)]
    
    cos = jnp.concatenate(final_cos_list, axis=-1)
    sin = jnp.concatenate(final_sin_list, axis=-1)
    return cos, sin

# --- 方法 2: 优化的 slicing 逻辑 ---
def optimized_method(cos_all, sin_all):
    indices = [0, 16, 40, 64]
    cos = jnp.concatenate([cos_all[i, :, indices[i]:indices[i+1]] for i in range(3)], axis=-1)
    sin = jnp.concatenate([sin_all[i, :, indices[i]:indices[i+1]] for i in range(3)], axis=-1)
    return cos, sin

def benchmark(name, func, *args):
    compiled_func = jax.jit(func).lower(*args).compile()
    
    # 1. 预热 (Warm-up)
    # 使用 jax.block_until_ready 处理返回的整个元组
    jax.block_until_ready(compiled_func(*args))
    
    # 2. 正式测量
    iters = 1000
    start_time = time.perf_counter()
    for _ in range(iters):
        # 同样在这里使用全局函数进行阻塞
        jax.block_until_ready(compiled_func(*args))
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / iters * 1000  # 毫秒
    print(f"{name} 平均耗时: {avg_time:.4f} ms")


orig_hlo = jax.jit(original_method).lower(cos_all, sin_all).compile().as_text()
opt_hlo = jax.jit(optimized_method).lower(cos_all, sin_all).compile().as_text()

print("Original Method HLO:")
print(orig_hlo)
print("Optimized Method HLO:")
print(opt_hlo)
print("-"*30)

# 执行测试
print(f"测试规模: num_tokens={num_tokens}\n" + "-"*30)
benchmark("Original (Split)", original_method, cos_all, sin_all)
benchmark("Optimized (Slicing)", optimized_method, cos_all, sin_all)