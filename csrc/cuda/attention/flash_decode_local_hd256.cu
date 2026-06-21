#include "blackwell/kernels/attention.h"
#include <cuda_bf16.h>

// Flash decode for Gemma 4 LOCAL attention layers.
//
// Specs (25 out of 30 layers):
//   num_q_heads  = 16
//   num_kv_heads = 8   → GQA ratio = 2:1
//   head_dim     = 256 (non-standard; most kernels only template HEAD_DIM ≤ 128)
//   sliding_window = 1024 tokens
//
// Key design decisions vs flash_decode.cu:
//   1. HEAD_DIM=256: register pressure doubles vs HEAD_DIM=128.
//      Use 32 threads/warp × 8 elements each to cover 256 dims without spilling.
//   2. GQA 2:1: 2 Q-heads share each KV-head — 2 warps per CTA, 1 KV load per pair.
//   3. Sliding window: only attend to the last WINDOW_SIZE=1024 KV tokens.
//      KV blocks outside the window are skipped entirely — no masking needed,
//      just limit block iteration range.
//   4. KV cache is physically circular: positions wrap modulo WINDOW_SIZE.
//      block_table maps window positions → physical blocks.

namespace blackwell {
namespace kernels {

static constexpr int GEMMA4_LOCAL_GQA   = 2;
static constexpr int GEMMA4_LOCAL_HEADS = 16;
static constexpr int GEMMA4_KV_HEADS    = 8;

// Each thread holds HEAD_DIM/WARP_SIZE = 256/32 = 8 elements per vector register.
static constexpr int ELEMS_PER_THREAD = 8;

template <int HEAD_DIM, int BLOCK_SIZE, int WINDOW_SIZE>
__global__ void flash_decode_local_kernel(
    const __nv_bfloat16* __restrict__ q,        // [S, 16, HEAD_DIM]
    const __nv_bfloat16* __restrict__ k_pool,   // [blocks, BLOCK_SIZE, 8, HEAD_DIM]
    const __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table,        // [S, max_blocks_per_window]
    const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ out,            // [S, 16, HEAD_DIM]
    const float scale,
    const int max_blocks_per_window,
    const int num_seqs
) {
    // blockIdx.x = seq_id
    // blockIdx.y = kv_head_id (0..7)
    // 2 warps = 64 threads; warp 0 → q_head = kv_head*2, warp 1 → q_head = kv_head*2+1
    const int seq_id  = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int q_head  = kv_head * GEMMA4_LOCAL_GQA + warp_id;

    const int seq_len = seq_lens[seq_id];
    // Only attend within sliding window
    const int window_start = max(0, seq_len - WINDOW_SIZE);
    const int window_len   = seq_len - window_start;
    const int num_blocks   = (window_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared KV tile: 2 warps share one load, halving smem bandwidth
    // Layout: [BLOCK_SIZE][HEAD_DIM] for K, same for V
    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_v = s_k + BLOCK_SIZE * HEAD_DIM;

    // Load Q into registers: 8 elements per thread
    float q_reg[ELEMS_PER_THREAD];
    const __nv_bfloat16* qptr = q + (seq_id * GEMMA4_LOCAL_HEADS + q_head) * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++)
        q_reg[i] = __bfloat162float(qptr[lane * ELEMS_PER_THREAD + i]);

    float m = -1e9f, l = 0.f;
    float acc[ELEMS_PER_THREAD] = {};

    for (int blk = 0; blk < num_blocks; blk++) {
        const int phys = block_table[seq_id * max_blocks_per_window + blk];
        const __nv_bfloat16* kblk = k_pool
            + (phys * BLOCK_SIZE * GEMMA4_KV_HEADS + kv_head) * HEAD_DIM;
        const __nv_bfloat16* vblk = v_pool
            + (phys * BLOCK_SIZE * GEMMA4_KV_HEADS + kv_head) * HEAD_DIM;

        // Both warps cooperate to load KV tile
        // Each thread loads 8 elements; 64 threads × 8 = 512 elements/load
        // BLOCK_SIZE=16, HEAD_DIM=256: total = 4096 elements → 8 rounds
        const int total = BLOCK_SIZE * HEAD_DIM;
        for (int i = threadIdx.x * ELEMS_PER_THREAD; i < total; i += blockDim.x * ELEMS_PER_THREAD) {
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                s_k[i + e] = __bfloat162float(kblk[i + e]);
                s_v[i + e] = __bfloat162float(vblk[i + e]);
            }
        }
        __syncthreads();

        const int valid = min(BLOCK_SIZE, window_len - blk * BLOCK_SIZE);
        for (int t = 0; t < valid; t++) {
            // Dot product Q · K[t] — each thread owns 8 dims, reduce across warp
            float dot = 0.f;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++)
                dot += q_reg[e] * s_k[t * HEAD_DIM + lane * ELEMS_PER_THREAD + e];
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, mask);
            dot *= scale;

            const float m_new   = fmaxf(m, dot);
            const float exp_m   = __expf(m - m_new);
            const float exp_dot = __expf(dot - m_new);
            l = l * exp_m + exp_dot;
            m = m_new;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++)
                acc[e] = acc[e] * exp_m + exp_dot * s_v[t * HEAD_DIM + lane * ELEMS_PER_THREAD + e];
        }
        __syncthreads();
    }

    __nv_bfloat16* optr = out + (seq_id * GEMMA4_LOCAL_HEADS + q_head) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++)
        optr[lane * ELEMS_PER_THREAD + e] = __float2bfloat16(acc[e] / l);
}

void launch_flash_decode_local_hd256(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens,
    void* out,
    int num_seqs, int num_kv_heads,
    int block_size, int max_blocks_per_window,
    float scale, cudaStream_t stream
) {
    // 2 warps per CTA (one per Q-head in GQA pair), blockIdx.y = kv_head
    dim3 grid(num_seqs, num_kv_heads);
    dim3 block(64);   // 2 warps
    // smem: 2 × BLOCK_SIZE × HEAD_DIM × sizeof(float) = 2 × 16 × 256 × 4 = 32 KB
    size_t smem = 2 * block_size * 256 * sizeof(float);

    flash_decode_local_kernel<256, 16, 1024>
        <<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k_pool),
        reinterpret_cast<const __nv_bfloat16*>(v_pool),
        block_table, seq_lens,
        reinterpret_cast<__nv_bfloat16*>(out),
        scale, max_blocks_per_window, num_seqs
    );
}

} // namespace kernels
} // namespace blackwell
