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

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.semb\n", argv[0]);
        return 2;
    }

    se_model_t model;
    memset(&model, 0, sizeof(model));

    se_error_t err;
    se_error_clear(&err);
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
        if (!se_scratch_reserve(&scratch, strlen(texts[i]), model.meta.dim)) {
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

    free(out);
    se_scratch_free(&scratch);
    se_model_close(&model);
    return 0;
}
