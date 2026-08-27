#include "se_internal.h"

#include <math.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SE_HAVE_NEON 1
#elif defined(__SSE__)
#include <xmmintrin.h>
#define SE_HAVE_SSE 1
#endif

#if defined(HAVE_MADVISE) && !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SE_PREFETCH(addr) __builtin_prefetch((addr), 0, 1)
#else
#define SE_PREFETCH(addr)
#endif

#define SE_PREFETCH_DISTANCE 4

static void add_row(float *acc, const float *row, uint32_t dim) {
    uint32_t j = 0;
#if defined(SE_HAVE_NEON)
    for (; j + 15 < dim; j += 16) {
        vst1q_f32(acc + j, vaddq_f32(vld1q_f32(acc + j), vld1q_f32(row + j)));
        vst1q_f32(acc + j + 4, vaddq_f32(vld1q_f32(acc + j + 4), vld1q_f32(row + j + 4)));
        vst1q_f32(acc + j + 8, vaddq_f32(vld1q_f32(acc + j + 8), vld1q_f32(row + j + 8)));
        vst1q_f32(acc + j + 12, vaddq_f32(vld1q_f32(acc + j + 12), vld1q_f32(row + j + 12)));
    }
#elif defined(SE_HAVE_SSE)
    for (; j + 15 < dim; j += 16) {
        _mm_storeu_ps(acc + j, _mm_add_ps(_mm_loadu_ps(acc + j), _mm_loadu_ps(row + j)));
        _mm_storeu_ps(acc + j + 4,
                      _mm_add_ps(_mm_loadu_ps(acc + j + 4), _mm_loadu_ps(row + j + 4)));
        _mm_storeu_ps(acc + j + 8,
                      _mm_add_ps(_mm_loadu_ps(acc + j + 8), _mm_loadu_ps(row + j + 8)));
        _mm_storeu_ps(acc + j + 12,
                      _mm_add_ps(_mm_loadu_ps(acc + j + 12), _mm_loadu_ps(row + j + 12)));
    }
#endif
    for (; j < dim; j++)
        acc[j] += row[j];
}

static void scale_copy(float *out, const float *acc, uint32_t dim, float inv) {
    uint32_t j = 0;
#if defined(SE_HAVE_NEON)
    float32x4_t vinv = vdupq_n_f32(inv);
    for (; j + 15 < dim; j += 16) {
        vst1q_f32(out + j, vmulq_f32(vld1q_f32(acc + j), vinv));
        vst1q_f32(out + j + 4, vmulq_f32(vld1q_f32(acc + j + 4), vinv));
        vst1q_f32(out + j + 8, vmulq_f32(vld1q_f32(acc + j + 8), vinv));
        vst1q_f32(out + j + 12, vmulq_f32(vld1q_f32(acc + j + 12), vinv));
    }
#elif defined(SE_HAVE_SSE)
    __m128 vinv = _mm_set1_ps(inv);
    for (; j + 15 < dim; j += 16) {
        _mm_storeu_ps(out + j, _mm_mul_ps(_mm_loadu_ps(acc + j), vinv));
        _mm_storeu_ps(out + j + 4, _mm_mul_ps(_mm_loadu_ps(acc + j + 4), vinv));
        _mm_storeu_ps(out + j + 8, _mm_mul_ps(_mm_loadu_ps(acc + j + 8), vinv));
        _mm_storeu_ps(out + j + 12, _mm_mul_ps(_mm_loadu_ps(acc + j + 12), vinv));
    }
#endif
    for (; j < dim; j++)
        out[j] = acc[j] * inv;
}

void se_l2_normalize(float *vec, uint32_t dim) {
    double sum = 0.0;
    for (uint32_t j = 0; j < dim; j++)
        sum += (double)vec[j] * (double)vec[j];
    if (sum <= 0.0)
        return;
    float inv = (float)(1.0 / sqrt(sum));
    for (uint32_t j = 0; j < dim; j++)
        vec[j] *= inv;
}

static se_status_t embed_ids_core(const se_model_t *model, se_scratch_t *sc, const uint32_t *ids,
                                  size_t n, float *out, se_error_t *err,
                                  volatile sig_atomic_t *cancelled) {
    const uint32_t dim = model->meta.dim;
    float *acc = sc->acc;
    memset(acc, 0, (size_t)dim * sizeof(float));

    const uint32_t drop_unk = (model->meta.unk_policy == SE_UNK_DROP);
    const uint32_t unk_id = model->meta.unk_id;

    size_t used = 0;
    for (size_t i = 0; i < n; i++) {
        if (((i & 255u) == 0) && cancelled && *cancelled) {
            se_error_set(err, SE_ERR_INTERNAL, "operation cancelled");
            return SE_ERR_INTERNAL;
        }
        uint32_t id = ids[i];
        if (id >= model->meta.vocab_size) {
            se_error_set(err, SE_ERR_INVALID_FORMAT, "token id %u is out of range", id);
            return SE_ERR_INVALID_FORMAT;
        }
        if (drop_unk && id == unk_id)
            continue;

        if (i + SE_PREFETCH_DISTANCE < n) {
            uint32_t next = ids[i + SE_PREFETCH_DISTANCE];
            if (next < model->meta.vocab_size)
                SE_PREFETCH(model->embeddings + (size_t)next * dim);
        }

        const float *row = model->embeddings + (size_t)id * dim;

        add_row(acc, row, dim);
        used++;
    }

    if (used == 0) {
        if (model->meta.empty_policy == SE_EMPTY_RAISE) {
            se_error_set(err, SE_ERR_EMPTY_INPUT, "input produced no usable tokens");
            return SE_ERR_EMPTY_INPUT;
        }
        memset(out, 0, (size_t)dim * sizeof(float));
        return SE_OK;
    }

    if (model->meta.normalization_type == SE_NORMALIZATION_L2) {
        scale_copy(out, acc, dim, 1.0f);
        se_l2_normalize(out, dim);
    } else {
        const float inv = 1.0f / (float)used;
        scale_copy(out, acc, dim, inv);
    }

    return SE_OK;
}

se_status_t se_embed_ids(const se_model_t *model, se_scratch_t *sc, const uint32_t *ids,
                         size_t n_ids, float *out, se_token_stats_t *stats, se_error_t *err,
                         volatile sig_atomic_t *cancelled) {
    if (n_ids > UINT32_MAX) {
        se_error_set(err, SE_ERR_OOM, "too many token ids");
        return SE_ERR_OOM;
    }
    stats->token_count = (uint32_t)n_ids;
    stats->unk_count = 0;
    stats->truncated = 0;
    for (size_t i = 0; i < n_ids; i++) {
        if (ids[i] == model->meta.unk_id)
            stats->unk_count++;
    }
    return embed_ids_core(model, sc, ids, n_ids, out, err, cancelled);
}

se_status_t se_embed_one(const se_model_t *model, se_scratch_t *sc, const uint8_t *input,
                         size_t input_len, uint32_t max_tokens, float *out, se_token_stats_t *stats,
                         se_error_t *err, volatile sig_atomic_t *cancelled) {
    se_status_t rc = se_tokenize(model, sc, input, input_len, max_tokens, stats, err, cancelled);
    if (rc != SE_OK)
        return rc;
    return embed_ids_core(model, sc, sc->ids, stats->token_count, out, err, cancelled);
}

size_t se_model_warmup(const se_model_t *model) {
    const volatile uint8_t *p = (const uint8_t *)model->map_base;
    size_t n = model->map_size;
    size_t page = 4096;
#if !defined(_WIN32) && defined(_SC_PAGESIZE)
    long sys_page = sysconf(_SC_PAGESIZE);
    if (sys_page > 0)
        page = (size_t)sys_page;
#endif
    volatile uint64_t sink = 0;
    size_t touched = 0;

#if defined(HAVE_MADVISE) && !defined(_WIN32)
    (void)madvise(model->map_base, model->map_size, MADV_WILLNEED);
#endif

    for (size_t i = 0; i < n; i += page) {
        sink += p[i];
        touched++;
    }
    if (n)
        sink += p[n - 1];

    (void)sink;
    return touched;
}
