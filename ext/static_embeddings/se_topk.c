#include "se_internal.h"

#include <math.h>
#include <stdint.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SE_HAVE_NEON 1
#elif defined(__SSE__)
#include <xmmintrin.h>
#define SE_HAVE_SSE 1
#endif

float se_dot_product_f32(const float *q, const float *row, size_t dim) {
    size_t j = 0;
#if defined(SE_HAVE_NEON)
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    for (; j + 15 < dim; j += 16) {
        a0 = vmlaq_f32(a0, vld1q_f32(q + j), vld1q_f32(row + j));
        a1 = vmlaq_f32(a1, vld1q_f32(q + j + 4), vld1q_f32(row + j + 4));
        a2 = vmlaq_f32(a2, vld1q_f32(q + j + 8), vld1q_f32(row + j + 8));
        a3 = vmlaq_f32(a3, vld1q_f32(q + j + 12), vld1q_f32(row + j + 12));
    }
    float32x4_t sumv = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
#if defined(__aarch64__)
    float dot = vaddvq_f32(sumv);
#else
    float32x2_t pair = vadd_f32(vget_low_f32(sumv), vget_high_f32(sumv));
    pair = vpadd_f32(pair, pair);
    float dot = vget_lane_f32(pair, 0);
#endif
#elif defined(SE_HAVE_SSE)
    __m128 a0 = _mm_setzero_ps();
    __m128 a1 = _mm_setzero_ps();
    __m128 a2 = _mm_setzero_ps();
    __m128 a3 = _mm_setzero_ps();
    for (; j + 15 < dim; j += 16) {
        a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(q + j), _mm_loadu_ps(row + j)));
        a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(q + j + 4), _mm_loadu_ps(row + j + 4)));
        a2 = _mm_add_ps(a2, _mm_mul_ps(_mm_loadu_ps(q + j + 8), _mm_loadu_ps(row + j + 8)));
        a3 = _mm_add_ps(a3, _mm_mul_ps(_mm_loadu_ps(q + j + 12), _mm_loadu_ps(row + j + 12)));
    }
    __m128 sumv = _mm_add_ps(_mm_add_ps(a0, a1), _mm_add_ps(a2, a3));
    float tmp[4];
    _mm_storeu_ps(tmp, sumv);
    float dot = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
#else
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    float s4 = 0.0f;
    float s5 = 0.0f;
    float s6 = 0.0f;
    float s7 = 0.0f;
    for (; j + 7 < dim; j += 8) {
        s0 += q[j] * row[j];
        s1 += q[j + 1] * row[j + 1];
        s2 += q[j + 2] * row[j + 2];
        s3 += q[j + 3] * row[j + 3];
        s4 += q[j + 4] * row[j + 4];
        s5 += q[j + 5] * row[j + 5];
        s6 += q[j + 6] * row[j + 6];
        s7 += q[j + 7] * row[j + 7];
    }
    float dot = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
#endif
    for (; j < dim; j++)
        dot += q[j] * row[j];
    return dot;
}

float se_dot_and_row_sq_f32(const float *q, const float *row, size_t dim,
                            float *row_sq_out) {
    size_t j = 0;
#if defined(SE_HAVE_NEON)
    float32x4_t d0 = vdupq_n_f32(0.0f);
    float32x4_t d1 = vdupq_n_f32(0.0f);
    float32x4_t d2 = vdupq_n_f32(0.0f);
    float32x4_t d3 = vdupq_n_f32(0.0f);
    float32x4_t s0 = vdupq_n_f32(0.0f);
    float32x4_t s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f);
    float32x4_t s3 = vdupq_n_f32(0.0f);
    for (; j + 15 < dim; j += 16) {
        float32x4_t q0 = vld1q_f32(q + j);
        float32x4_t r0 = vld1q_f32(row + j);
        float32x4_t q1 = vld1q_f32(q + j + 4);
        float32x4_t r1 = vld1q_f32(row + j + 4);
        float32x4_t q2 = vld1q_f32(q + j + 8);
        float32x4_t r2 = vld1q_f32(row + j + 8);
        float32x4_t q3 = vld1q_f32(q + j + 12);
        float32x4_t r3 = vld1q_f32(row + j + 12);
        d0 = vmlaq_f32(d0, q0, r0);
        d1 = vmlaq_f32(d1, q1, r1);
        d2 = vmlaq_f32(d2, q2, r2);
        d3 = vmlaq_f32(d3, q3, r3);
        s0 = vmlaq_f32(s0, r0, r0);
        s1 = vmlaq_f32(s1, r1, r1);
        s2 = vmlaq_f32(s2, r2, r2);
        s3 = vmlaq_f32(s3, r3, r3);
    }
    float32x4_t dotv = vaddq_f32(vaddq_f32(d0, d1), vaddq_f32(d2, d3));
    float32x4_t sqv = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
#if defined(__aarch64__)
    float dot = vaddvq_f32(dotv);
    float row_sq = vaddvq_f32(sqv);
#else
    float32x2_t pair = vadd_f32(vget_low_f32(dotv), vget_high_f32(dotv));
    pair = vpadd_f32(pair, pair);
    float dot = vget_lane_f32(pair, 0);
    pair = vadd_f32(vget_low_f32(sqv), vget_high_f32(sqv));
    pair = vpadd_f32(pair, pair);
    float row_sq = vget_lane_f32(pair, 0);
#endif
#elif defined(SE_HAVE_SSE)
    __m128 d0 = _mm_setzero_ps();
    __m128 d1 = _mm_setzero_ps();
    __m128 d2 = _mm_setzero_ps();
    __m128 d3 = _mm_setzero_ps();
    __m128 s0 = _mm_setzero_ps();
    __m128 s1 = _mm_setzero_ps();
    __m128 s2 = _mm_setzero_ps();
    __m128 s3 = _mm_setzero_ps();
    for (; j + 15 < dim; j += 16) {
        __m128 q0 = _mm_loadu_ps(q + j);
        __m128 r0 = _mm_loadu_ps(row + j);
        __m128 q1 = _mm_loadu_ps(q + j + 4);
        __m128 r1 = _mm_loadu_ps(row + j + 4);
        __m128 q2 = _mm_loadu_ps(q + j + 8);
        __m128 r2 = _mm_loadu_ps(row + j + 8);
        __m128 q3 = _mm_loadu_ps(q + j + 12);
        __m128 r3 = _mm_loadu_ps(row + j + 12);
        d0 = _mm_add_ps(d0, _mm_mul_ps(q0, r0));
        d1 = _mm_add_ps(d1, _mm_mul_ps(q1, r1));
        d2 = _mm_add_ps(d2, _mm_mul_ps(q2, r2));
        d3 = _mm_add_ps(d3, _mm_mul_ps(q3, r3));
        s0 = _mm_add_ps(s0, _mm_mul_ps(r0, r0));
        s1 = _mm_add_ps(s1, _mm_mul_ps(r1, r1));
        s2 = _mm_add_ps(s2, _mm_mul_ps(r2, r2));
        s3 = _mm_add_ps(s3, _mm_mul_ps(r3, r3));
    }
    __m128 dotv = _mm_add_ps(_mm_add_ps(d0, d1), _mm_add_ps(d2, d3));
    __m128 sqv = _mm_add_ps(_mm_add_ps(s0, s1), _mm_add_ps(s2, s3));
    float tmp[4];
    _mm_storeu_ps(tmp, dotv);
    float dot = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
    _mm_storeu_ps(tmp, sqv);
    float row_sq = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
#else
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    float s0 = 0.0f;
    float s1 = 0.0f;
    float s2 = 0.0f;
    float s3 = 0.0f;
    for (; j + 3 < dim; j += 4) {
        float r0 = row[j];
        float r1 = row[j + 1];
        float r2 = row[j + 2];
        float r3 = row[j + 3];
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
#endif
    for (; j < dim; j++) {
        float r = row[j];
        dot += q[j] * r;
        row_sq += r * r;
    }
    *row_sq_out = row_sq;
    return dot;
}

static float cosine_score(float dot, float row_sq, float inv_query_norm) {
    if (row_sq > 0.0f)
        return dot * inv_query_norm / sqrtf(row_sq);
    if (row_sq == 0.0f)
        return 0.0f;
    return NAN;
}

void *se_topk_execute(void *arg) {
    se_topk_job_t *job = (se_topk_job_t *)arg;
    for (size_t r = 0; r < job->rows; r++) {
        if ((r & 1023u) == 0 && job->cancelled)
            return NULL;
        float score;

        if (job->format == SE_VECTOR_FORMAT_F16) {
            const uint8_t *row = (const uint8_t *)job->m + r * job->dim * 2u;
            if (job->cosine) {
                float row_sq = 0.0f;
                score = se_dot_and_row_sq_f16(job->q, row, job->dim, &row_sq);
                score = cosine_score(score, row_sq, job->inv_query_norm);
            } else {
                score = se_dot_product_f16(job->q, row, job->dim);
            }
        } else {
            const float *row = (const float *)job->m + r * job->dim;
            if (job->cosine) {
                float row_sq = 0.0f;
                score = se_dot_and_row_sq_f32(job->q, row, job->dim, &row_sq);
                score = cosine_score(score, row_sq, job->inv_query_norm);
            } else {
                score = se_dot_product_f32(job->q, row, job->dim);
            }
        }

        if (!(score > job->best_score[job->k - 1]))
            continue;

        long pos = job->k - 1;
        while (pos > 0 && job->best_score[pos - 1] < score) {
            job->best_score[pos] = job->best_score[pos - 1];
            job->best_idx[pos] = job->best_idx[pos - 1];
            pos--;
        }
        job->best_score[pos] = score;
        job->best_idx[pos] = r;
    }
    return NULL;
}
