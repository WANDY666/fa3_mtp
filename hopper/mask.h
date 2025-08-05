/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

/**
 * Flash Attention Mask类 - 实现各种attention mask机制
 * 
 * 模板参数：
 * - kBlockM, kBlockN: tile的行列大小
 * - PackGQA: 是否使用Packed Group Query Attention优化
 * - TiledMma: 矩阵乘法tile配置
 * - SwapAB: 是否交换A和B矩阵的位置（用于反向传播）
 */
// 64, 64, true, TiledMmaQK, false
template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct Mask {

    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");

    // === 成员变量：mask计算所需的各种参数 ===
    int const thread_idx;                              // 当前线程索引
    int const seqlen_q, seqlen_k;                     // Q和K序列的长度
    int const window_size_left, window_size_right;    // 局部attention的左右窗口大小
    int const sink_token_length;                      // sink token长度（可以全局访问的特殊token数量）
    cutlass::FastDivmod const attention_chunk_divmod; // attention chunk的除法器（用于chunked attention）
    cutlass::FastDivmod const qhead_per_khead_divmod; // Q head到K head的映射除法器（用于GQA）

    /**
     * 构造函数：初始化mask计算所需的所有参数
     */
    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &attention_chunk_divmod,
         cutlass::FastDivmod const &qhead_per_khead_divmod)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , attention_chunk_divmod(attention_chunk_divmod)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
    {
    };

    /**
     * 核心mask应用函数：对注意力分数矩阵应用各种mask
     * 
     * 模板参数：
     * - Seqlenk_mask: 是否应用序列长度mask（处理padding）
     * - Causal_mask: 是否应用因果mask（decoder模式，只能看到之前的token）
     * - Local_mask: 是否应用局部attention mask（限制attention window）
     * 
     * 参数：
     * - tSrS: 注意力分数矩阵（Q*K^T的结果）
     * - m_block, n_block: 当前处理的block索引
     */
    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        // 编译时检查：不能同时是因果mask和局部mask
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        // 如果没有任何mask需要应用，直接返回
        if (!Seqlenk_mask && !Causal_mask && !Local_mask) { return; }

        // === 设置线程和坐标系统 ===
        // thread_idx 为 threadIdx.x - MmaThreadOffset = threadIdx.x - 128
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);  // 当前线程的MMA slice
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});       // 线程0的MMA slice（用于获取编译时已知的坐标）

        // 根据SwapAB确定行列维度（SwapAB主要用于反向传播）
        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        // === 创建坐标张量和布局转换 ===
        // 创建单位张量，用于获取每个元素在全局矩阵中的坐标
        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);  // 当前线程负责的坐标
        
        // 将注意力分数矩阵和坐标张量转换为行列布局，便于mask操作
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);  // 线程0的坐标（编译时已知）
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        
        // === 计算mask边界 ===
        // 使用线程0的列坐标进行比较，因为它在编译时已知，可以优化性能
        // 减去当前线程的列偏移，得到相对于线程0的列限制
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        
        // === 应用不同类型的mask ===
        if constexpr (!Causal_mask && !Local_mask) {
            // === 情况1：只需要序列长度mask ===
            if constexpr (Seqlenk_mask) {
                // 简单的列mask：如果列索引超出序列长度，mask整列
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { 
                            tSrS_rowcol(m, n) = -INFINITY;  // 设置为负无穷，softmax后变为0
                        }
                    }
                }
            }
        } else {
            // === 情况2：需要基于行列位置的复杂mask（因果mask或局部mask） ===
            if constexpr (!SwapAB) {
                // === 前向传播模式的mask ===
                
                // 如果使用PackGQA，需要在同一行的线程间分担divmod计算
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
                int mma_m_idx;
                
                // PackGQA模式下计算Q head到K head的映射
                if constexpr (PackGQA) {
                    mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
                }
                
                // 计算因果mask的行偏移：确保Q的第i个token只能看到K的前i个token
                int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
                
                if constexpr (Causal_mask) {
                    // === 因果mask：实现decoder的自回归特性 ===
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        // 获取当前行索引
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            : __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        
                        // 计算右边界：当前token只能看到它之前的token
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + causal_row_offset
                            : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                        
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            // 如果列索引超出右边界，mask掉这个位置
                            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { 
                                tSrS_rowcol(m, n) = -INFINITY; 
                            }
                        }
                    }
                } else {
                    // === 局部attention mask：限制attention window大小 ===
                    int const local_row_offset_right = causal_row_offset + window_size_right;  // 右窗口边界
                    int const local_row_offset_left = causal_row_offset - 1 - window_size_left; // 左窗口边界
                    int const col_limit_sink = sink_token_length - n_block * kBlockN;  // sink token边界
                    
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        // 获取当前行索引
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            : __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        
                        // 计算局部attention的左右边界
                        int col_limit_right = !Seqlenk_mask
                            ? row_idx + local_row_offset_right
                            : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                        int col_limit_left = row_idx + local_row_offset_left;
                        
                        // 如果使用chunked attention，进一步限制边界
                        if (attention_chunk_divmod.divisor > 0) {
                            int col_limit_left_chunk = flash::round_down(attention_chunk_divmod, row_idx + seqlen_k - seqlen_q) - n_block * kBlockN - thread_col_offset;
                            col_limit_left = std::max(col_limit_left, col_limit_left_chunk);
                            col_limit_right = std::min(col_limit_right, col_limit_left_chunk + attention_chunk_divmod.divisor);
                        }
                        
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
                            // mask掉窗口外的位置，但保留sink token
                            if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { 
                                tSrS_rowcol(m, n) = -INFINITY; 
                            }
                        }
                    }
                }
            } else {
                // === 反向传播模式的mask（SwapAB=true） ===
                // TODO: 反向传播暂不支持attention_chunk
                int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
                int const causal_row_offset = seqlenk_col_limit - seqlen_q + m_block * kBlockM + thread_row_offset;
                
                if constexpr (Causal_mask) {
                    // === 反向传播的因果mask ===
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // 如果列超出边界，mask整列；否则根据因果关系确定行边界
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { 
                                tSrS_rowcol(m, n) = -INFINITY; 
                            }
                        }
                    }
                } else {
                    // === 反向传播的局部attention mask ===
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // 计算局部窗口的上下边界
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset - window_size_right;
                        int const row_limit_bot = col0 < col_limit_sink ? kBlockM : col0 - causal_row_offset + window_size_left;
                        
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
                            // mask掉窗口外的位置
                            if (row_idx < row_limit_top || row_idx > row_limit_bot) { 
                                tSrS_rowcol(m, n) = -INFINITY; 
                            }
                        }
                    }
                }
            }
        }
    };

};

} // namespace flash
