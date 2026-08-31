#include "se_internal.h"

#include <stdint.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#if defined(__aarch64__)
#define SE_HAVE_NEON_FP16 1
#endif
#endif

#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#include <immintrin.h>
#define SE_HAVE_X86_F16C_TARGET 1
#endif
#endif

static se_f16_backend_t se_f16_backend = SE_F16_BACKEND_LUT;

#if !defined(SE_HAVE_NEON_FP16)
#define SE_NEED_F16_LUT 1
static float se_f16_lut[65536];
#endif

size_t se_vector_format_element_bytes(se_vector_format_t format) {
    return format == SE_VECTOR_FORMAT_F16 ? 2u : sizeof(float);
}

static void write_u16le(uint8_t *dst, uint16_t v) {
    dst[0] = (uint8_t)(v & 0xffu);
    dst[1] = (uint8_t)(v >> 8);
}

static uint16_t read_u16le(const uint8_t *src) {
    return (uint16_t)src[0] | ((uint16_t)src[1] << 8);
}

uint16_t se_float_to_f16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;

    if (exp == 0xffu) {
        if (mant == 0)
            return (uint16_t)(sign | 0x7c00u);
        mant >>= 13;
        return (uint16_t)(sign | 0x7c00u | mant | (mant == 0));
    }

    int32_t half_exp = (int32_t)exp - 127 + 15;
    if (half_exp >= 31)
        return (uint16_t)(sign | 0x7c00u);

    if (half_exp <= 0) {
        if (half_exp < -10)
            return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - half_exp);
        uint32_t rounded = (mant + (1u << (shift - 1))) >> shift;
        return (uint16_t)(sign | rounded);
    }

    mant += 0x1000u;
    if (mant & 0x800000u) {
        mant = 0;
        half_exp++;
        if (half_exp >= 31)
            return (uint16_t)(sign | 0x7c00u);
    }

    return (uint16_t)(sign | ((uint32_t)half_exp << 10) | (mant >> 13));
}

float se_f16_bits_to_float(uint16_t half) {
    uint32_t sign = ((uint32_t)half & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)half >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)half & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

void se_write_f16le(uint8_t *dst, float value) {
    write_u16le(dst, se_float_to_f16_bits(value));
}

float se_read_f16le(const uint8_t *src) {
    return se_f16_bits_to_float(read_u16le(src));
}

void se_encode_f16_from_floats(uint8_t *dst, const float *src, size_t count) {
    for (size_t i = 0; i < count; i++)
        write_u16le(dst + i * 2, se_float_to_f16_bits(src[i]));
}

void se_decode_f16_to_floats(float *dst, const uint8_t *src, size_t count) {
    for (size_t i = 0; i < count; i++)
        dst[i] = se_f16_bits_to_float(read_u16le(src + i * 2));
}

#if defined(SE_NEED_F16_LUT)
static void init_f16_lut(void) {
    for (uint32_t i = 0; i <= 0xffffu; i++)
        se_f16_lut[i] = se_f16_bits_to_float((uint16_t)i);
}
#endif

#if defined(SE_HAVE_X86_F16C_TARGET)
#ifndef bit_OSXSAVE
#define bit_OSXSAVE (1u << 27)
#endif
#ifndef bit_AVX
#define bit_AVX (1u << 28)
#endif
#ifndef bit_F16C
#define bit_F16C (1u << 29)
#endif

static int detect_x86_f16c(void) {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
        return 0;
    if ((ecx & bit_OSXSAVE) == 0 || (ecx & bit_AVX) == 0 || (ecx & bit_F16C) == 0)
        return 0;

    uint32_t xcr0_lo = 0, xcr0_hi = 0;
#if defined(_MSC_VER)
    return 0;
#else
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    (void)xcr0_hi;
    return (xcr0_lo & 0x6u) == 0x6u;
#endif
}
#endif

#if defined(SE_NEED_F16_LUT)
static float dot_product_f16_lut(const float *q, const uint8_t *row, size_t dim) {
    size_t j = 0;
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    for (; j + 3 < dim; j += 4) {
        s0 += q[j] * se_f16_lut[read_u16le(row + j * 2)];
        s1 += q[j + 1] * se_f16_lut[read_u16le(row + (j + 1) * 2)];
        s2 += q[j + 2] * se_f16_lut[read_u16le(row + (j + 2) * 2)];
        s3 += q[j + 3] * se_f16_lut[read_u16le(row + (j + 3) * 2)];
    }
    float dot = (s0 + s1) + (s2 + s3);
    for (; j < dim; j++)
        dot += q[j] * se_f16_lut[read_u16le(row + j * 2)];
    return dot;
}

static float dot_and_row_sq_f16_lut(const float *q, const uint8_t *row, size_t dim,
                                    float *row_sq_out) {
    size_t j = 0;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    for (; j + 3 < dim; j += 4) {
        float r0 = se_f16_lut[read_u16le(row + j * 2)];
        float r1 = se_f16_lut[read_u16le(row + (j + 1) * 2)];
        float r2 = se_f16_lut[read_u16le(row + (j + 2) * 2)];
        float r3 = se_f16_lut[read_u16le(row + (j + 3) * 2)];
        d0 += q[j] * r0;
        d1 += q[j + 1] * r1;
        d2 += q[j + 2] * r2;
        d3 += q[j + 3] * r3;
        s0 += r0 * r0;
        s1 += r1 * r1;
        s2 += r2 * r2;
        s3 += r3 * r3;
    }
    float dot = (d0 + d1) + (d2 + d3);
    float row_sq = (s0 + s1) + (s2 + s3);
    for (; j < dim; j++) {
        float r = se_f16_lut[read_u16le(row + j * 2)];
        dot += q[j] * r;
        row_sq += r * r;
    }
    *row_sq_out = row_sq;
    return dot;
}
#endif /* SE_NEED_F16_LUT */

#if defined(SE_HAVE_NEON_FP16)
static float dot_product_f16_neon(const float *q, const uint8_t *row, size_t dim) {
    size_t j = 0;
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    for (; j + 15 < dim; j += 16) {
        float16x8_t h0 =
            vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)(const void *)(row + j * 2)));
        float16x8_t h1 =
            vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)(const void *)(row + (j + 8) * 2)));
        float32x4_t r0 = vcvt_f32_f16(vget_low_f16(h0));
        float32x4_t r1 = vcvt_f32_f16(vget_high_f16(h0));
        float32x4_t r2 = vcvt_f32_f16(vget_low_f16(h1));
        float32x4_t r3 = vcvt_f32_f16(vget_high_f16(h1));
        a0 = vmlaq_f32(a0, vld1q_f32(q + j), r0);
        a1 = vmlaq_f32(a1, vld1q_f32(q + j + 4), r1);
        a2 = vmlaq_f32(a2, vld1q_f32(q + j + 8), r2);
        a3 = vmlaq_f32(a3, vld1q_f32(q + j + 12), r3);
    }
    for (; j + 7 < dim; j += 8) {
        float16x4_t h0 =
            vreinterpret_f16_u16(vld1_u16((const uint16_t *)(const void *)(row + j * 2)));
        float16x4_t h1 =
            vreinterpret_f16_u16(vld1_u16((const uint16_t *)(const void *)(row + (j + 4) * 2)));
        a0 = vmlaq_f32(a0, vld1q_f32(q + j), vcvt_f32_f16(h0));
        a1 = vmlaq_f32(a1, vld1q_f32(q + j + 4), vcvt_f32_f16(h1));
    }
    float32x4_t sumv = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    float dot = vaddvq_f32(sumv);
    for (; j < dim; j++)
        dot += q[j] * se_f16_bits_to_float(read_u16le(row + j * 2));
    return dot;
}

static float dot_and_row_sq_f16_neon(const float *q, const uint8_t *row, size_t dim,
                                     float *row_sq_out) {
    size_t j = 0;
    float32x4_t d0 = vdupq_n_f32(0.0f);
    float32x4_t d1 = vdupq_n_f32(0.0f);
    float32x4_t d2 = vdupq_n_f32(0.0f);
    float32x4_t d3 = vdupq_n_f32(0.0f);
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    for (; j + 15 < dim; j += 16) {
        float16x8_t h0 =
            vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)(const void *)(row + j * 2)));
        float16x8_t h1 =
            vreinterpretq_f16_u16(vld1q_u16((const uint16_t *)(const void *)(row + (j + 8) * 2)));
        float32x4_t r0 = vcvt_f32_f16(vget_low_f16(h0));
        float32x4_t r1 = vcvt_f32_f16(vget_high_f16(h0));
        float32x4_t r2 = vcvt_f32_f16(vget_low_f16(h1));
        float32x4_t r3 = vcvt_f32_f16(vget_high_f16(h1));
        d0 = vmlaq_f32(d0, vld1q_f32(q + j), r0);
        d1 = vmlaq_f32(d1, vld1q_f32(q + j + 4), r1);
        d2 = vmlaq_f32(d2, vld1q_f32(q + j + 8), r2);
        d3 = vmlaq_f32(d3, vld1q_f32(q + j + 12), r3);
        s0 = vmlaq_f32(s0, r0, r0);
        s1 = vmlaq_f32(s1, r1, r1);
        s2 = vmlaq_f32(s2, r2, r2);
        s3 = vmlaq_f32(s3, r3, r3);
    }
    for (; j + 7 < dim; j += 8) {
        float16x4_t h0 =
            vreinterpret_f16_u16(vld1_u16((const uint16_t *)(const void *)(row + j * 2)));
        float16x4_t h1 =
            vreinterpret_f16_u16(vld1_u16((const uint16_t *)(const void *)(row + (j + 4) * 2)));
        float32x4_t r0 = vcvt_f32_f16(h0);
        float32x4_t r1 = vcvt_f32_f16(h1);
        d0 = vmlaq_f32(d0, vld1q_f32(q + j), r0);
        d1 = vmlaq_f32(d1, vld1q_f32(q + j + 4), r1);
        s0 = vmlaq_f32(s0, r0, r0);
        s1 = vmlaq_f32(s1, r1, r1);
    }
    float dot = vaddvq_f32(vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3)));
    float row_sq = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; j < dim; j++) {
        float r = se_f16_bits_to_float(read_u16le(row + j * 2));
        dot += q[j] * r;
        row_sq += r * r;
    }
    *row_sq_out = row_sq;
    return dot;
}
#endif

#if defined(SE_HAVE_X86_F16C_TARGET)
__attribute__((target("f16c,avx"))) static float hsum256_f16c(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    float tmp[4];
    _mm_storeu_ps(tmp, sum);
    return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
}

__attribute__((target("f16c,avx"))) static float
dot_product_f16_f16c(const float *q, const uint8_t *row, size_t dim) {
    size_t j = 0;
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    for (; j + 15 < dim; j += 16) {
        __m256 r0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(row + j * 2)));
        __m256 r1 =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(row + (j + 8) * 2)));
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(q + j), r0));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(q + j + 8), r1));
    }
    float dot = hsum256_f16c(_mm256_add_ps(a0, a1));
    for (; j + 7 < dim; j += 8) {
        __m256 r = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(row + j * 2)));
        dot += hsum256_f16c(_mm256_mul_ps(_mm256_loadu_ps(q + j), r));
    }
    for (; j < dim; j++)
        dot += q[j] * se_f16_bits_to_float(read_u16le(row + j * 2));
    return dot;
}

__attribute__((target("f16c,avx"))) static float
dot_and_row_sq_f16_f16c(const float *q, const uint8_t *row, size_t dim, float *row_sq_out) {
    size_t j = 0;
    __m256 d0 = _mm256_setzero_ps();
    __m256 d1 = _mm256_setzero_ps();
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    for (; j + 15 < dim; j += 16) {
        __m256 r0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(row + j * 2)));
        __m256 r1 =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(row + (j + 8) * 2)));
        d0 = _mm256_add_ps(d0, _mm256_mul_ps(_mm256_loadu_ps(q + j), r0));
        d1 = _mm256_add_ps(d1, _mm256_mul_ps(_mm256_loadu_ps(q + j + 8), r1));
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(r0, r0));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(r1, r1));
    }
    float dot = hsum256_f16c(_mm256_add_ps(d0, d1));
    float row_sq = hsum256_f16c(_mm256_add_ps(s0, s1));
    for (; j + 7 < dim; j += 8) {
        __m256 r = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(row + j * 2)));
        dot += hsum256_f16c(_mm256_mul_ps(_mm256_loadu_ps(q + j), r));
        row_sq += hsum256_f16c(_mm256_mul_ps(r, r));
    }
    for (; j < dim; j++) {
        float r = se_f16_bits_to_float(read_u16le(row + j * 2));
        dot += q[j] * r;
        row_sq += r * r;
    }
    *row_sq_out = row_sq;
    return dot;
}
#endif

float se_dot_product_f16(const float *q, const uint8_t *row, size_t dim) {
#if defined(SE_HAVE_NEON_FP16)
    return dot_product_f16_neon(q, row, dim);
#else
#if defined(SE_HAVE_X86_F16C_TARGET)
    if (se_f16_backend == SE_F16_BACKEND_F16C)
        return dot_product_f16_f16c(q, row, dim);
#endif
    return dot_product_f16_lut(q, row, dim);
#endif
}

float se_dot_and_row_sq_f16(const float *q, const uint8_t *row, size_t dim, float *row_sq_out) {
#if defined(SE_HAVE_NEON_FP16)
    return dot_and_row_sq_f16_neon(q, row, dim, row_sq_out);
#else
#if defined(SE_HAVE_X86_F16C_TARGET)
    if (se_f16_backend == SE_F16_BACKEND_F16C)
        return dot_and_row_sq_f16_f16c(q, row, dim, row_sq_out);
#endif
    return dot_and_row_sq_f16_lut(q, row, dim, row_sq_out);
#endif
}

void se_select_f16_backend(void) {
#if defined(SE_HAVE_NEON_FP16)
    se_f16_backend = SE_F16_BACKEND_NEON_FP16;
#else
#if defined(SE_HAVE_X86_F16C_TARGET)
    if (detect_x86_f16c()) {
        se_f16_backend = SE_F16_BACKEND_F16C;
        return;
    }
#endif
    se_f16_backend = SE_F16_BACKEND_LUT;
    init_f16_lut();
#endif
}

se_f16_backend_t se_current_f16_backend(void) {
    return se_f16_backend;
}
