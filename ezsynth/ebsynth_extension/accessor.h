// ReEzSynth/ezsynth/ebsynth_extension/accessor.h
//
// Minimal, torch-free stand-in for torch::PackedTensorAccessor{32,64}.
//
// The CUDA sources (*.cu) deliberately include no PyTorch headers: PyTorch >= 2.12
// headers require C++20, which older nvcc/MSVC pairs cannot compile, while plain
// CUDA C++17 builds everywhere. Tensors are converted to these accessors on the
// host side (cuda_dispatch.cpp) and passed to the kernels by value, exactly like
// torch's packed accessors (same data pointer, sizes and strides in elements).
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define EBS_HD __host__ __device__
#else
#define EBS_HD
#endif

namespace ebs
{

// Non-owning view into a sub-tensor; returned by PackedAccessor::operator[].
template <typename T, int N, typename index_t>
class AccessorView
{
public:
    EBS_HD AccessorView(T *data, const index_t *sizes, const index_t *strides)
        : data_(data), sizes_(sizes), strides_(strides) {}

    EBS_HD AccessorView<T, N - 1, index_t> operator[](index_t i) const
    {
        return AccessorView<T, N - 1, index_t>(data_ + strides_[0] * i, sizes_ + 1, strides_ + 1);
    }

    EBS_HD index_t size(int d) const { return sizes_[d]; }

private:
    T *data_;
    const index_t *sizes_;
    const index_t *strides_;
};

template <typename T, typename index_t>
class AccessorView<T, 1, index_t>
{
public:
    EBS_HD AccessorView(T *data, const index_t *sizes, const index_t *strides)
        : data_(data), sizes_(sizes), strides_(strides) {}

    EBS_HD T &operator[](index_t i) const { return data_[strides_[0] * i]; }

    EBS_HD index_t size(int d) const { return sizes_[d]; }

private:
    T *data_;
    const index_t *sizes_;
    const index_t *strides_;
};

// Data pointer plus sizes/strides held by value, so it can be passed to a kernel.
template <typename T, int N, typename index_t>
class PackedAccessor
{
public:
    PackedAccessor() = default;

    PackedAccessor(T *data, const int64_t *sizes, const int64_t *strides) : data_(data)
    {
        for (int d = 0; d < N; ++d)
        {
            sizes_[d] = static_cast<index_t>(sizes[d]);
            strides_[d] = static_cast<index_t>(strides[d]);
        }
    }

    EBS_HD AccessorView<T, N - 1, index_t> operator[](index_t i) const
    {
        return AccessorView<T, N - 1, index_t>(data_ + strides_[0] * i, sizes_ + 1, strides_ + 1);
    }

    EBS_HD index_t size(int d) const { return sizes_[d]; }
    EBS_HD index_t stride(int d) const { return strides_[d]; }
    EBS_HD T *data() const { return data_; }

private:
    T *data_ = nullptr;
    index_t sizes_[N] = {};
    index_t strides_[N] = {};
};

template <typename T, typename index_t>
class PackedAccessor<T, 1, index_t>
{
public:
    PackedAccessor() = default;

    PackedAccessor(T *data, const int64_t *sizes, const int64_t *strides)
        : data_(data), size_(static_cast<index_t>(sizes[0])), stride_(static_cast<index_t>(strides[0])) {}

    EBS_HD T &operator[](index_t i) const { return data_[stride_ * i]; }

    EBS_HD index_t size(int) const { return size_; }
    EBS_HD index_t stride(int) const { return stride_; }
    EBS_HD T *data() const { return data_; }

private:
    T *data_ = nullptr;
    index_t size_ = 0;
    index_t stride_ = 0;
};

template <typename T, int N>
using PackedAccessor32 = PackedAccessor<T, N, int32_t>;

template <typename T, int N>
using PackedAccessor64 = PackedAccessor<T, N, int64_t>;

} // namespace ebs
