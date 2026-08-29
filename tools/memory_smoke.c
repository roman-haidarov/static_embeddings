#include "se_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int embed_text(const se_model_t *model, se_scratch_t *scratch, const char *text,
                      float *out) {
    se_error_t err;
    se_token_stats_t stats;
    se_error_clear(&err);
    se_status_t rc = se_embed_one(model, scratch, (const uint8_t *)text, strlen(text),
                                  model->meta.max_tokens_default, out, &stats, &err, NULL);
    if (rc != SE_OK) {
        fprintf(stderr, "%s\n", err.message);
        return 0;
    }
    return 1;
}

static double norm_of(const float *vec, uint32_t dim) {
    double sum = 0.0;
    for (uint32_t i = 0; i < dim; i++)
        sum += (double)vec[i] * (double)vec[i];
    return sqrt(sum);
}

static int topk_smoke(void) {
    float query[] = {1.0f, 0.0f};
    float matrix[] = {0.0f, 1.0f, 2.0f, 0.0f};
    size_t best_idx[2] = {0, 0};
    float best_score[2] = {-INFINITY, -INFINITY};
    se_topk_job_t job;

    memset(&job, 0, sizeof(job));
    job.q = query;
    job.m = matrix;
    job.format = SE_VECTOR_FORMAT_F32;
    job.dim = 2;
    job.rows = 2;
    job.k = 2;
    job.best_idx = best_idx;
    job.best_score = best_score;
    se_topk_execute(&job);

    if (best_idx[0] != 1 || fabsf(best_score[0] - 2.0f) > 1e-6f) {
        fprintf(stderr, "unexpected f32 top-k result\n");
        return 0;
    }

    uint8_t encoded[4];
    float values[] = {1.5f, -0.25f};
    se_encode_f16_from_floats(encoded, values, 2);
    if (fabsf(se_read_f16le(encoded) - 1.5f) > 1e-6f) {
        fprintf(stderr, "unexpected f16 decode result\n");
        return 0;
    }

    best_idx[0] = 0;
    best_idx[1] = 0;
    best_score[0] = -INFINITY;
    best_score[1] = -INFINITY;
    memset(&job, 0, sizeof(job));
    job.q = query;
    job.m = encoded;
    job.format = SE_VECTOR_FORMAT_F16;
    job.dim = 1;
    job.rows = 2;
    job.k = 2;
    job.best_idx = best_idx;
    job.best_score = best_score;
    se_topk_execute(&job);

    if (best_idx[0] != 0 || fabsf(best_score[0] - 1.5f) > 1e-6f) {
        fprintf(stderr, "unexpected f16 top-k result\n");
        return 0;
    }

    return 1;
}

#if SE_ENABLE_ALLOC_STATS
static int alloc_stats_expect_scratch_live(void) {
    se_alloc_stats_t stats[SE_ALLOC_CATEGORY_COUNT];
    se_alloc_stats_snapshot(stats);
    if (stats[SE_ALLOC_SCRATCH].current_bytes == 0 || stats[SE_ALLOC_SCRATCH].peak_bytes == 0) {
        fprintf(stderr, "scratch allocation stats were not recorded\n");
        return 0;
    }

    se_alloc_stats_reset();
    se_alloc_stats_snapshot(stats);
    if (stats[SE_ALLOC_SCRATCH].current_bytes == 0 ||
        stats[SE_ALLOC_SCRATCH].peak_bytes != stats[SE_ALLOC_SCRATCH].current_bytes ||
        stats[SE_ALLOC_SCRATCH].alloc_count != 0 || stats[SE_ALLOC_SCRATCH].free_count != 0) {
        fprintf(stderr, "allocation stats reset lost live scratch bytes\n");
        return 0;
    }

    return 1;
}

static int alloc_stats_expect_scratch_freed(void) {
    se_alloc_stats_t stats[SE_ALLOC_CATEGORY_COUNT];
    se_alloc_stats_snapshot(stats);
    if (stats[SE_ALLOC_SCRATCH].current_bytes != 0 || stats[SE_ALLOC_SCRATCH].free_count == 0) {
        fprintf(stderr, "scratch allocation stats were not released\n");
        return 0;
    }

    return 1;
}
#endif

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.semb\n", argv[0]);
        return 2;
    }

    se_model_t model;
    memset(&model, 0, sizeof(model));

    se_error_t err;
    se_error_clear(&err);
    se_select_f16_backend();
    if (!topk_smoke())
        return 1;

    if (se_model_open(&model, argv[1], &err) != SE_OK) {
        fprintf(stderr, "%s\n", err.message);
        return 1;
    }

    se_scratch_t scratch;
    se_scratch_init(&scratch);

    float *out = (float *)calloc(model.meta.dim ? model.meta.dim : 1, sizeof(float));
    if (!out) {
        se_scratch_free(&scratch);
        se_model_close(&model);
        return 1;
    }

    const char *texts[] = {
        "hello world",
        "postgres pipeline mode in ruby",
        "static embedding vector",
        "",
        "zzzzz unknownword qqqqq",
        "mixed Привет hello 123",
        "слово слово слово",
        "a long input that should exercise tokenizer allocation and embedding accumulation"};

    size_t count = sizeof(texts) / sizeof(texts[0]);
    for (size_t i = 0; i < count; i++) {
        if (!se_scratch_reserve(&scratch, model.meta.dim)) {
            free(out);
            se_scratch_free(&scratch);
            se_model_close(&model);
            return 1;
        }
        if (!embed_text(&model, &scratch, texts[i], out)) {
            free(out);
            se_scratch_free(&scratch);
            se_model_close(&model);
            return 1;
        }
        double norm = norm_of(out, model.meta.dim);
        if (!(norm >= 0.0 && norm <= 1.0001)) {
            fprintf(stderr, "unexpected norm %f\n", norm);
            free(out);
            se_scratch_free(&scratch);
            se_model_close(&model);
            return 1;
        }
    }

#if SE_ENABLE_ALLOC_STATS
    if (!alloc_stats_expect_scratch_live()) {
        free(out);
        se_scratch_free(&scratch);
        se_model_close(&model);
        return 1;
    }
#endif

    free(out);
    se_scratch_free(&scratch);

#if SE_ENABLE_ALLOC_STATS
    if (!alloc_stats_expect_scratch_freed()) {
        se_model_close(&model);
        return 1;
    }
#endif

    se_model_close(&model);
    return 0;
}
