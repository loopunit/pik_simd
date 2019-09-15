#ifndef PIK_SIMD_H_
#define PIK_SIMD_H_

#include "simd/simd.h"


namespace pik {

#define PIK_INLINE inline

template <size_t N>
using BlockDesc = SIMD_PART(float, SIMD_MIN(N, SIMD_FULL(float)::N));

// Adapters for source/destination.
//
// Block: (x, y) <-> (N * y + x)
// Lines: (x, y) <-> (stride * y + x)
//
// I.e. Block is a specialization of Lines with fixed stride.
//
// FromXXX should implement Read and Load (Read vector).
// ToXXX should implement Write and Store (Write vector).

template <size_t N>
using BlockDesc = SIMD_PART(float, SIMD_MIN(N, SIMD_FULL(float)::N));

// Here and in the following, the SZ template parameter specifies the number of
// values to load/store. Needed because we want to handle 4x4 sub-blocks of
// 16x16 blocks.
template <size_t N>
class FromBlock {
 public:
  explicit FromBlock(const float* block) : block_(block) {}

  FromBlock View(size_t dx, size_t dy) const {
    return FromBlock<N>(Address(dx, dy));
  }

  template <size_t SZ>
  SIMD_ATTR PIK_INLINE typename BlockDesc<SZ>::V LoadPart(const size_t row,
                                                          size_t i) const {
    return load(BlockDesc<SZ>(), block_ + row * N + i);
  }

  SIMD_ATTR PIK_INLINE typename BlockDesc<N>::V Load(const size_t row,
                                                     size_t i) const {
    return LoadPart<N>(row, i);
  }

  SIMD_ATTR PIK_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  constexpr PIK_INLINE const float* Address(const size_t row,
                                            const size_t i) const {
    return block_ + row * N + i;
  }

 private:
  const float* block_;
};

template <size_t N>
class ToBlock {
 public:
  explicit ToBlock(float* block) : block_(block) {}

  ToBlock View(size_t dx, size_t dy) const {
    return ToBlock<N>(Address(dx, dy));
  }

  template <size_t SZ>
  SIMD_ATTR PIK_INLINE void StorePart(const typename BlockDesc<SZ>::V& v,
                                      const size_t row, const size_t i) const {
    store(v, BlockDesc<SZ>(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE void Store(const typename BlockDesc<N>::V& v,
                                  const size_t row, size_t i) const {
    return StorePart<N>(v, row, i);
  }

  SIMD_ATTR PIK_INLINE void Write(float v, const size_t row,
                                  const size_t i) const {
    *Address(row, i) = v;
  }

  constexpr PIK_INLINE float* Address(const size_t row, const size_t i) const {
    return block_ + row * N + i;
  }

 private:
  float* block_;
};

template <size_t N>
class FromLines {
 public:
  FromLines(const float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  FromLines View(size_t dx, size_t dy) const {
    return FromLines(Address(dx, dy), stride_);
  }

  template <size_t SZ>
  SIMD_ATTR PIK_INLINE typename BlockDesc<SZ>::V LoadPart(
      const size_t row, const size_t i) const {
    return load(BlockDesc<SZ>(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE typename BlockDesc<N>::V Load(const size_t row,
                                                     size_t i) const {
    return LoadPart<N>(row, i);
  }

  SIMD_ATTR PIK_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  PIK_INLINE const float* SIMD_RESTRICT Address(const size_t row,
                                                const size_t i) const {
    return top_left_ + row * stride_ + i;
  }

 private:
  const float* SIMD_RESTRICT top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

template <size_t N>
class ToLines {
 public:
  ToLines(float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  ToLines View(const ToLines& other, size_t dx, size_t dy) const {
    return ToLines(Address(dx, dy), stride_);
  }

  template <size_t SZ>
  SIMD_ATTR PIK_INLINE void StorePart(const typename BlockDesc<SZ>::V& v,
                                      const size_t row, const size_t i) const {
    store(v, BlockDesc<SZ>(), Address(row, i));
  }

  SIMD_ATTR PIK_INLINE void Store(const typename BlockDesc<N>::V& v,
                                  const size_t row, size_t i) const {
    return StorePart<N>(v, row, i);
  }

  SIMD_ATTR PIK_INLINE void Write(float v, const size_t row,
                                  const size_t i) const {
    *Address(row, i) = v;
  }

  PIK_INLINE float* SIMD_RESTRICT Address(const size_t row,
                                          const size_t i) const {
    return top_left_ + row * stride_ + i;
  }

 private:
  float* SIMD_RESTRICT top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

template <class From, class To>
SIMD_ATTR PIK_INLINE void CopyBlock8(const From& from, const To& to) {
  const BlockDesc<8> d;
  for (size_t i = 0; i < 8; i += d.N) {
    const auto i0 = from.Load(0, i);
    const auto i1 = from.Load(1, i);
    const auto i2 = from.Load(2, i);
    const auto i3 = from.Load(3, i);
    const auto i4 = from.Load(4, i);
    const auto i5 = from.Load(5, i);
    const auto i6 = from.Load(6, i);
    const auto i7 = from.Load(7, i);
    to.Store(i0, 0, i);
    to.Store(i1, 1, i);
    to.Store(i2, 2, i);
    to.Store(i3, 3, i);
    to.Store(i4, 4, i);
    to.Store(i5, 5, i);
    to.Store(i6, 6, i);
    to.Store(i7, 7, i);
  }
}

#if (SIMD_TARGET_VALUE != SIMD_AVX2) && (SIMD_TARGET_VALUE != SIMD_NONE)

// DCT building blocks that require SIMD vector length to be 4, e.g. SSE4.
static_assert(BlockDesc<8>().N == 4, "Wrong vector size, must be 4");

template <class From, class To>
static SIMD_ATTR PIK_INLINE void TransposeBlock8_V4(const From& from,
                                                    const To& to) {
  const auto p0L = from.Load(0, 0);
  const auto p0H = from.Load(0, 4);
  const auto p1L = from.Load(1, 0);
  const auto p1H = from.Load(1, 4);
  const auto p2L = from.Load(2, 0);
  const auto p2H = from.Load(2, 4);
  const auto p3L = from.Load(3, 0);
  const auto p3H = from.Load(3, 4);
  const auto p4L = from.Load(4, 0);
  const auto p4H = from.Load(4, 4);
  const auto p5L = from.Load(5, 0);
  const auto p5H = from.Load(5, 4);
  const auto p6L = from.Load(6, 0);
  const auto p6H = from.Load(6, 4);
  const auto p7L = from.Load(7, 0);
  const auto p7H = from.Load(7, 4);

  const auto q0L = interleave_lo(p0L, p2L);
  const auto q0H = interleave_lo(p0H, p2H);
  const auto q1L = interleave_lo(p1L, p3L);
  const auto q1H = interleave_lo(p1H, p3H);
  const auto q2L = interleave_hi(p0L, p2L);
  const auto q2H = interleave_hi(p0H, p2H);
  const auto q3L = interleave_hi(p1L, p3L);
  const auto q3H = interleave_hi(p1H, p3H);
  const auto q4L = interleave_lo(p4L, p6L);
  const auto q4H = interleave_lo(p4H, p6H);
  const auto q5L = interleave_lo(p5L, p7L);
  const auto q5H = interleave_lo(p5H, p7H);
  const auto q6L = interleave_hi(p4L, p6L);
  const auto q6H = interleave_hi(p4H, p6H);
  const auto q7L = interleave_hi(p5L, p7L);
  const auto q7H = interleave_hi(p5H, p7H);

  const auto r0L = interleave_lo(q0L, q1L);
  const auto r0H = interleave_lo(q0H, q1H);
  const auto r1L = interleave_hi(q0L, q1L);
  const auto r1H = interleave_hi(q0H, q1H);
  const auto r2L = interleave_lo(q2L, q3L);
  const auto r2H = interleave_lo(q2H, q3H);
  const auto r3L = interleave_hi(q2L, q3L);
  const auto r3H = interleave_hi(q2H, q3H);
  const auto r4L = interleave_lo(q4L, q5L);
  const auto r4H = interleave_lo(q4H, q5H);
  const auto r5L = interleave_hi(q4L, q5L);
  const auto r5H = interleave_hi(q4H, q5H);
  const auto r6L = interleave_lo(q6L, q7L);
  const auto r6H = interleave_lo(q6H, q7H);
  const auto r7L = interleave_hi(q6L, q7L);
  const auto r7H = interleave_hi(q6H, q7H);

  to.Store(r0L, 0, 0);
  to.Store(r4L, 0, 4);
  to.Store(r1L, 1, 0);
  to.Store(r5L, 1, 4);
  to.Store(r2L, 2, 0);
  to.Store(r6L, 2, 4);
  to.Store(r3L, 3, 0);
  to.Store(r7L, 3, 4);
  to.Store(r0H, 4, 0);
  to.Store(r4H, 4, 4);
  to.Store(r1H, 5, 0);
  to.Store(r5H, 5, 4);
  to.Store(r2H, 6, 0);
  to.Store(r6H, 6, 4);
  to.Store(r3H, 7, 0);
  to.Store(r7H, 7, 4);
}


#endif // #if (SIMD_TARGET_VALUE != SIMD_AVX2) && (SIMD_TARGET_VALUE != SIMD_NONE)

#if SIMD_TARGET_VALUE == SIMD_AVX2

// DCT building blocks that require SIMD vector length to be 8, e.g. AVX2.
static_assert(BlockDesc<8>().N == 8, "Wrong vector size, must be 8");

// Each vector holds one row of the input/output block.
template <class V>
SIMD_ATTR PIK_INLINE void TransposeBlock8_V8(V& i0, V& i1, V& i2, V& i3, V& i4,
                                             V& i5, V& i6, V& i7) {
  // Surprisingly, this straightforward implementation (24 cycles on port5) is
  // faster than load128+insert and load_dup128+concat_hi_lo+blend.
  const auto q0 = interleave_lo(i0, i2);
  const auto q1 = interleave_lo(i1, i3);
  const auto q2 = interleave_hi(i0, i2);
  const auto q3 = interleave_hi(i1, i3);
  const auto q4 = interleave_lo(i4, i6);
  const auto q5 = interleave_lo(i5, i7);
  const auto q6 = interleave_hi(i4, i6);
  const auto q7 = interleave_hi(i5, i7);

  const auto r0 = interleave_lo(q0, q1);
  const auto r1 = interleave_hi(q0, q1);
  const auto r2 = interleave_lo(q2, q3);
  const auto r3 = interleave_hi(q2, q3);
  const auto r4 = interleave_lo(q4, q5);
  const auto r5 = interleave_hi(q4, q5);
  const auto r6 = interleave_lo(q6, q7);
  const auto r7 = interleave_hi(q6, q7);

  i0 = concat_lo_lo(r4, r0);
  i1 = concat_lo_lo(r5, r1);
  i2 = concat_lo_lo(r6, r2);
  i3 = concat_lo_lo(r7, r3);
  i4 = concat_hi_hi(r4, r0);
  i5 = concat_hi_hi(r5, r1);
  i6 = concat_hi_hi(r6, r2);
  i7 = concat_hi_hi(r7, r3);
}

template <class From, class To>
SIMD_ATTR PIK_INLINE void TransposeBlock8_V8(const From& from, const To& to) {
  auto i0 = from.Load(0, 0);
  auto i1 = from.Load(1, 0);
  auto i2 = from.Load(2, 0);
  auto i3 = from.Load(3, 0);
  auto i4 = from.Load(4, 0);
  auto i5 = from.Load(5, 0);
  auto i6 = from.Load(6, 0);
  auto i7 = from.Load(7, 0);
  TransposeBlock8_V8(i0, i1, i2, i3, i4, i5, i6, i7);
  to.Store(i0, 0, 0);
  to.Store(i1, 1, 0);
  to.Store(i2, 2, 0);
  to.Store(i3, 3, 0);
  to.Store(i4, 4, 0);
  to.Store(i5, 5, 0);
  to.Store(i6, 6, 0);
  to.Store(i7, 7, 0);
}

#endif // #if SIMD_TARGET_VALUE == SIMD_AVX2

}


#endif // #ifndef PIK_SIMD_H