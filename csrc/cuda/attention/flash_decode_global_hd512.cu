#include "blackwell/kernels/attention.h"
#include <cuda_bf16.h>

// Flash decode for Gemma 4 GLOBAL attention layers.
//
// Specs (5 out of 30 layers, full context):
//   num_q_heads  = 16
//   num_kv_heads = 2   → GQA ratio = 8:1
//   head_dim     = 512
//   sliding_window = none (full KV context up to 256K tokens)
//
// ECOSYSTEM GAP: As of 2026-06, no public kernel in FlashInfer, vLLM,
// FlashAttention-2/3, or llama.cpp handles head_dim=512 efficiently.
// Existing kernels template HEAD_DIM ∈ {64, 128, 256} and would need to
// run two 256-dim passes or pad to 512 with wasted compute.
//
// Design strategy — two-phase dot product:
//   HEAD_DIM=512 exceeds register budget for a single warp pass.
//   Split into two 256-dim halves (phase_a, phase_b), each fitting
//   the same 8-element-per-thread register layout as flash_decode_local_hd256.cu.
//   Both halves share the same attention weight (softmax), so we compute:
//     dot  = dot_a + dot_b     (accumulated across both halves)
//     acc  = acc_a | acc_b     (output halves concatenated)
//   This doubles the smem requirement but keeps register pressure identical to HD=256.
//
// Alternative (future): use CUTLASS warp-specialized persistent kernel (Blackwell
//   ping-pong mainloop) to overlap K/V loads with QK dot-product across the two halves.

namespace blackwell {
namespace kernels {

static constexpr int GEMMA4_GLOBAL_Q_HEADS  = 16;
static constexpr int GEMMA4_GLOBAL_KV_HEADS = 2;
static constexpr int GEMMA4_GLOBAL_GQA      = 8;   // 16 / 2
static constexpr int HD512_HALF             = 256;
static constexpr int ELEMS_PER_THREAD       = 8;    // 256 / 32 lanes

template <int BLOCK_SIZE>
__global__ void flash_decode_global_hd512_kernel(
    const __nv_bfloat16* __restrict__ q,        // [S, 16, 512]
    const __nv_bfloat16* __restrict__ k_pool,   // [blocks, BLOCK_SIZE, 2, 512]
    const __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table,        // [S, max_blocks]
    const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ out,            // [S, 16, 512]
    const float scale,
    const int max_blocks_per_seq
) {
    // blockIdx.x = seq_id
    // blockIdx.y = kv_head (0 or 1)
    // 8 warps: warp i handles q_head = kv_head*8 + i
    const int seq_id  = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int q_head  = kv_head * GEMMA4_GLOBAL_GQA + warp_id;

    const int seq_len  = seq_lens[seq_id];
    const int n_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // smem: K_a[BLOCK_SIZE][256] + K_b[BLOCK_SIZE][256]
    //     + V_a[BLOCK_SIZE][256] + V_b[BLOCK_SIZE][256]
    // = 4 × BLOCK_SIZE × 256 × sizeof(float) = 4 × 16 × 256 × 4 = 64 KB
    extern __shared__ float smem[];
    float* s_ka = smem;
    float* s_kb = s_ka + BLOCK_SIZE * HD512_HALF;
    float* s_va = s_kb + BLOCK_SIZE * HD512_HALF;
    float* s_vb = s_va + BLOCK_SIZE * HD512_HALF;

    // Q registers: 8 elements per thread per half = 16 total
    // q_a covers dims [0..255], q_b covers dims [256..511]
    float q_a[ELEMS_PER_THREAD], q_b[ELEMS_PER_THREAD];
    const __nv_bfloat16* qptr = q + (seq_id * GEMMA4_GLOBAL_Q_HEADS + q_head) * 512;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        q_a[e] = __bfloat162float(qptr[lane * ELEMS_PER_THREAD + e]);
        q_b[e] = __bfloat162float(qptr[HD512_HALF + lane * ELEMS_PER_THREAD + e]);
    }

    float m = -1e9f, l = 0.f;
    float acc_a[ELEMS_PER_THREAD] = {};
    float acc_b[ELEMS_PER_THREAD] = {};

    for (int blk = 0; blk < n_blocks; blk++) {
        const int phys = block_table[seq_id * max_blocks_per_seq + blk];

        // KV base: physical_block × BLOCK_SIZE × 2 KV_heads × 512
        const __nv_bfloat16* kbase = k_pool
            + (phys * BLOCK_SIZE * GEMMA4_GLOBAL_KV_HEADS + kv_head) * 512;
        const __nv_bfloat16* vbase = v_pool
            + (phys * BLOCK_SIZE * GEMMA4_GLOBAL_KV_HEADS + kv_head) * 512;

        // Cooperative load of both K halves and both V halves.
        // 8 warps × 32 lanes = 256 threads; each loads ELEMS_PER_THREAD elements.
        // Total per half: BLOCK_SIZE × 256 = 4096 elems → 4096/256 = 16 rounds.
        const int total_half = BLOCK_SIZE * HD512_HALF;
        for (int i = threadIdx.x; i < total_half; i += blockDim.x) {
            s_ka[i] = __bfloat162float(kbase[i]);
            s_kb[i] = __bfloat162float(kbase[HD512_HALF + i]);
            s_va[i] = __bfloat162float(vbase[i]);
            s_vb[i] = __bfloat162float(vbase[HD512_HALF + i]);
        }
        __syncthreads();

        const int valid = min(BLOCK_SIZE, seq_len - blk * BLOCK_SIZE);
        for (int t = 0; t < valid; t++) {
            // Two-phase dot product: dot = <q_a, k_a[t]> + <q_b, k_b[t]>
            float dot = 0.f;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                dot += q_a[e] * s_ka[t * HD512_HALF + lane * ELEMS_PER_THREAD + e];
                dot += q_b[e] * s_kb[t * HD512_HALF + lane * ELEMS_PER_THREAD + e];
            }
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1)
                dot += __shfl_xor_sync(0xffffffff, dot, mask);
            dot *= scale;   // scale = 1/sqrt(512) for head_dim=512

            const float m_new   = fmaxf(m, dot);
            const float exp_m   = __expf(m - m_new);
            const float exp_dot = __expf(dot - m_new);
            l = l * exp_m + exp_dot;
            m = m_new;

            // Accumulate both output halves with the same attention weight
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                acc_a[e] = acc_a[e] * exp_m
                    + exp_dot * s_va[t * HD512_HALF + lane * ELEMS_PER_THREAD + e];
                acc_b[e] = acc_b[e] * exp_m
                    + exp_dot * s_vb[t * HD512_HALF + lane * ELEMS_PER_THREAD + e];
            }
        }
        __syncthreads();
    }

    __nv_bfloat16* optr = out + (seq_id * GEMMA4_GLOBAL_Q_HEADS + q_head) * 512;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        optr[lane * ELEMS_PER_THREAD + e]             = __float2bfloat16(acc_a[e] / l);
        optr[HD512_HALF + lane * ELEMS_PER_THREAD + e] = __float2bfloat16(acc_b[e] / l);
    }
}

void launch_flash_decode_global_hd512(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens,
    void* out,
    int num_seqs, int num_kv_heads,
    int block_size, int max_blocks_per_seq,
    float scale, cudaStream_t stream
) {
    // 8 warps (one per Q-head in each GQA group of 8), blockIdx.y = kv_head (0 or 1)
    dim3 grid(num_seqs, num_kv_heads);
    dim3 block(256);  // 8 warps
    // smem: 4 halves × BLOCK_SIZE × 256 × sizeof(float) = 4 × 16 × 256 × 4 = 64 KB
    // Blackwell (sm_100) supports 228 KB shared memory per CTA
    size_t smem = 4 * block_size * HD512_HALF * sizeof(float);

    flash_decode_global_hd512_kernel<16>
        <<<grid, block, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k_pool),
        reinterpret_cast<const __nv_bfloat16*>(v_pool),
        block_table, seq_lens,
        reinterpret_cast<__nv_bfloat16*>(out),
        scale, max_blocks_per_seq
    );
}

} // namespace kernels
} // namespace blackwell
