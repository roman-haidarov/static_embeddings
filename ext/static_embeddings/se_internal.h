#ifndef SE_INTERNAL_H
#define SE_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <signal.h>
#include <limits.h>

#define SE_MAGIC          "SEMBv1\0\0"
#define SE_MAGIC_LEN      8
#define SE_FORMAT_VERSION 3u
#define SE_HEADER_SIZE    320u
#define SE_ALIGNMENT      64u

#define SE_TOKENIZER_BERT_WORDPIECE_V1 1u
#define SE_DTYPE_F32                   1u
#define SE_POOLING_MEAN                1u
#define SE_NORMALIZATION_NONE          0u
#define SE_NORMALIZATION_L2            1u

#define SE_TRUNCATE_USABLE_IDS_BEFORE_POOLING 2u

#define SE_UNK_INCLUDE 0u
#define SE_UNK_DROP    1u

#define SE_EMPTY_ZERO_VECTOR 0u
#define SE_EMPTY_RAISE       1u

#define SE_ADDED_PAD  (1u << 0)
#define SE_ADDED_UNK  (1u << 1)
#define SE_ADDED_CLS  (1u << 2)
#define SE_ADDED_SEP  (1u << 3)
#define SE_ADDED_MASK (1u << 4)
#define SE_ADDED_TOKEN_MASK_ALL \
    (SE_ADDED_PAD | SE_ADDED_UNK | SE_ADDED_CLS | SE_ADDED_SEP | SE_ADDED_MASK)

#define SE_SLOT_EMPTY 0xFFFFFFFFu
#define SE_SIZE_MAX   ((size_t)-1)

#ifndef SE_ENABLE_ALLOC_STATS
#define SE_ENABLE_ALLOC_STATS 0
#endif

#define SE_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define SE_STATIC_ASSERT(expr, name) _Static_assert((expr), #name)
#else
#define SE_STATIC_ASSERT_JOIN_INNER(a, b) a##b
#define SE_STATIC_ASSERT_JOIN(a, b)       SE_STATIC_ASSERT_JOIN_INNER(a, b)
#define SE_STATIC_ASSERT(expr, name) \
    typedef char SE_STATIC_ASSERT_JOIN(se_static_assertion_, name)[(expr) ? 1 : -1]
#endif

enum {
    SE_OFF_MAGIC = 0,
    SE_OFF_FORMAT_VERSION = 8,
    SE_OFF_HEADER_SIZE = 12,
    SE_OFF_FLAGS = 16,
    SE_OFF_DIM = 20,
    SE_OFF_VOCAB_SIZE = 24,
    SE_OFF_TOKENIZER_TYPE = 28,
    SE_OFF_EMBEDDING_DTYPE = 32,
    SE_OFF_POOLING_TYPE = 36,
    SE_OFF_NORMALIZATION_TYPE = 40,
    SE_OFF_MAX_TOKENS_DEFAULT = 44,
    SE_OFF_TRUNCATION_POLICY = 48,
    SE_OFF_ADD_SPECIAL_TOKENS = 52,
    SE_OFF_UNK_POLICY = 56,
    SE_OFF_EMPTY_POLICY = 60,
    SE_OFF_DO_LOWER_CASE = 64,
    SE_OFF_STRIP_ACCENTS = 68,
    SE_OFF_HANDLE_CHINESE_CHARS = 72,
    SE_OFF_CLEAN_TEXT = 76,
    SE_OFF_MAX_INPUT_CHARS_PER_WORD = 80,
    SE_OFF_PAD_ID = 84,
    SE_OFF_UNK_ID = 88,
    SE_OFF_CLS_ID = 92,
    SE_OFF_SEP_ID = 96,
    SE_OFF_MASK_ID = 100,
    SE_OFF_HASH_TABLE_SIZE = 104,
    SE_OFF_HASH_SEED = 108,
    SE_OFF_SUBWORD_PREFIX_LEN = 112,
    SE_OFF_SUBWORD_PREFIX = 116,
    SE_OFF_MAX_TOKEN_CHARS = 124,
    SE_OFF_SEC_VOCAB_STRINGS = 128,
    SE_OFF_SEC_VOCAB_HASH = 144,
    SE_OFF_SEC_EMBEDDINGS = 160,
    SE_OFF_SEC_NORM_TABLES = 176,
    SE_OFF_SEC_PROVENANCE = 192,
    SE_OFF_SEC_ROOT_TRIE = 208,
    SE_OFF_SEC_CONT_TRIE = 224,
    SE_OFF_CHECKSUM = 240,
    SE_OFF_MAX_PROBE = 304,
    SE_OFF_ADDED_TOKEN_MASK = 308
};

typedef struct {
    uint32_t hash;
    uint32_t str_off;
    uint32_t str_len;
    uint32_t token_id;
} se_vocab_slot_t;

typedef struct {
    uint32_t edge_start;
    uint32_t edge_count;
    uint32_t token_id;
    uint32_t reserved;
} se_trie_node_t;

typedef struct {
    uint32_t byte;
    uint32_t child;
} se_trie_edge_t;

typedef struct {
    const se_trie_node_t *nodes;
    const se_trie_edge_t *edges;
    uint32_t node_count;
    uint32_t edge_count;
} se_trie_t;

typedef struct {
    uint32_t cp;
    uint32_t len;
    uint32_t out[4];
} se_map_entry_t;

typedef struct {
    uint32_t lo;
    uint32_t hi;
} se_range_t;

typedef struct {
    const se_map_entry_t *lower;
    uint32_t lower_count;
    const se_map_entry_t *nfd;
    uint32_t nfd_count;
    const se_range_t *mn;
    uint32_t mn_count;
    const se_range_t *punct;
    uint32_t punct_count;
    const se_range_t *control;
    uint32_t control_count;
    const se_range_t *whitespace;
    uint32_t whitespace_count;
    const se_map_entry_t *lower256[256];
    const se_map_entry_t *nfd256[256];
    uint8_t mn256[256];
    uint8_t punct256[256];
    uint8_t control256[256];
    uint8_t whitespace256[256];
} se_norm_tables_t;

typedef struct {
    uint32_t dim;
    uint32_t vocab_size;
    uint32_t tokenizer_type;
    uint32_t embedding_dtype;
    uint32_t pooling_type;
    uint32_t normalization_type;
    uint32_t max_tokens_default;
    uint32_t truncation_policy;
    uint32_t add_special_tokens;
    uint32_t unk_policy;
    uint32_t empty_policy;
    uint32_t do_lower_case;
    uint32_t strip_accents;
    uint32_t handle_chinese_chars;
    uint32_t clean_text;
    uint32_t max_input_chars_per_word;
    uint32_t max_token_chars;
    uint32_t max_probe;
    uint32_t added_token_mask;
    uint32_t pad_id;
    uint32_t unk_id;
    uint32_t cls_id;
    uint32_t sep_id;
    uint32_t mask_id;
    uint32_t hash_table_size;
    uint32_t hash_seed;
    uint32_t subword_prefix_len;
    char subword_prefix[8];
} se_meta_t;

typedef struct {
    void *map_base;
    size_t map_size;
    int mapped;

    se_meta_t meta;
    se_norm_tables_t norm;

    const char *vocab_strings;
    size_t vocab_strings_size;
    const se_vocab_slot_t *vocab_hash;
    se_trie_t root_trie;
    se_trie_t cont_trie;
    const float *embeddings;
    const char *provenance;
    size_t provenance_size;
} se_model_t;

typedef struct {
    uint32_t *cps;
    size_t cps_cap;
    uint32_t *cps2;
    size_t cps2_cap;
    uint8_t *bytes;
    size_t bytes_cap;
    uint32_t *ids;
    size_t ids_cap;
    float *acc;
    size_t acc_cap;
} se_scratch_t;

typedef enum {
    SE_OK = 0,
    SE_ERR_INVALID_FORMAT,
    SE_ERR_UNSUPPORTED_VERSION,
    SE_ERR_UNSUPPORTED_TOKENIZER,
    SE_ERR_INVALID_UTF8,
    SE_ERR_OOM,
    SE_ERR_IO,
    SE_ERR_EMPTY_INPUT,
    SE_ERR_INTERNAL
} se_status_t;

typedef struct {
    se_status_t status;
    char message[256];
} se_error_t;

typedef enum { SE_VECTOR_FORMAT_F32 = 1, SE_VECTOR_FORMAT_F16 = 2 } se_vector_format_t;

typedef enum {
    SE_F16_BACKEND_LUT = 0,
    SE_F16_BACKEND_NEON_FP16 = 1,
    SE_F16_BACKEND_F16C = 2
} se_f16_backend_t;

typedef enum {
    SE_ALLOC_UNKNOWN = 0,
    SE_ALLOC_SCRATCH,
    SE_ALLOC_BATCH_INPUT,
    SE_ALLOC_BATCH_OUTPUT,
    SE_ALLOC_BATCH_STATS,
    SE_ALLOC_BATCH_INDEX,
    SE_ALLOC_TOKEN_IDS,
    SE_ALLOC_TOPK_QUERY,
    SE_ALLOC_TOPK_BEST,
    SE_ALLOC_TOPK_MATRIX_COPY,
    SE_ALLOC_FORMAT_VALIDATE,
    SE_ALLOC_MODEL_FILE,
    SE_ALLOC_CATEGORY_COUNT
} se_alloc_category_t;

typedef struct {
    size_t current_bytes;
    size_t peak_bytes;
    size_t total_allocated_bytes;
    size_t total_freed_bytes;
    size_t alloc_count;
    size_t realloc_count;
    size_t free_count;
} se_alloc_stats_t;

typedef struct {
    const float *q;
    const void *m;
    se_vector_format_t format;
    size_t dim;
    size_t rows;
    long k;
    size_t *best_idx;
    float *best_score;
    int cosine;
    float inv_query_norm;
    volatile sig_atomic_t cancelled;
} se_topk_job_t;

SE_STATIC_ASSERT(sizeof(se_vocab_slot_t) == 16, vocab_slot_size);
SE_STATIC_ASSERT(sizeof(se_trie_node_t) == 16, trie_node_size);
SE_STATIC_ASSERT(sizeof(se_trie_edge_t) == 8, trie_edge_size);
SE_STATIC_ASSERT(sizeof(se_map_entry_t) == 24, map_entry_size);
SE_STATIC_ASSERT(sizeof(se_range_t) == 8, range_size);

static inline int se_checked_add_size(size_t a, size_t b, size_t *out) {
    if (a > SE_SIZE_MAX - b)
        return 0;
    *out = a + b;
    return 1;
}

static inline int se_checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SE_SIZE_MAX / a)
        return 0;
    *out = a * b;
    return 1;
}

static inline int se_array_bytes(size_t count, size_t elem_size, size_t *out) {
    return se_checked_mul_size(count, elem_size, out);
}

static inline size_t se_alloc_count(size_t count) {
    return count ? count : 1;
}

static inline int se_alloc_bytes(size_t count, size_t elem_size, size_t *out) {
    return se_checked_mul_size(se_alloc_count(count), elem_size, out);
}

static inline int se_next_capacity(size_t current, size_t need, size_t min_capacity, size_t *out) {
    size_t next = current ? current : min_capacity;

    if (next == 0)
        next = 1;
    while (next < need) {
        if (next > SE_SIZE_MAX / 2)
            return 0;
        next *= 2;
    }

    *out = next;
    return 1;
}

static inline int se_size_fits_long(size_t n) {
    return n <= (size_t)LONG_MAX;
}

static inline int se_host_is_little_endian(void) {
    const uint16_t one = 1;
    return *((const uint8_t *)(const void *)&one) == 1;
}

#if defined(__GNUC__) || defined(__clang__)
#define SE_PRINTF_FORMAT(fmt_index, first_arg) __attribute__((format(printf, fmt_index, first_arg)))
#else
#define SE_PRINTF_FORMAT(fmt_index, first_arg)
#endif

void se_error_clear(se_error_t *err);
void se_error_set(se_error_t *err, se_status_t status, const char *fmt, ...) SE_PRINTF_FORMAT(3, 4);

const char *se_alloc_category_name(se_alloc_category_t category);
void se_alloc_stats_reset(void);
void se_alloc_stats_snapshot(se_alloc_stats_t out[SE_ALLOC_CATEGORY_COUNT]);

#if SE_ENABLE_ALLOC_STATS
void *se_alloc_stats_malloc(se_alloc_category_t category, size_t bytes);
void *se_alloc_stats_calloc(se_alloc_category_t category, size_t count, size_t elem_size);
void *se_alloc_stats_realloc(se_alloc_category_t category, void *ptr, size_t bytes);
void se_alloc_stats_free(void *ptr);

#define se_malloc(category, bytes) se_alloc_stats_malloc((category), (bytes))
#define se_calloc(category, count, elem_size) \
    se_alloc_stats_calloc((category), (count), (elem_size))
#define se_realloc(category, ptr, bytes) se_alloc_stats_realloc((category), (ptr), (bytes))
#define se_free(ptr)                     se_alloc_stats_free((ptr))
#else
#define se_malloc(category, bytes)            malloc((bytes))
#define se_calloc(category, count, elem_size) calloc((count), (elem_size))
#define se_realloc(category, ptr, bytes)      realloc((ptr), (bytes))
#define se_free(ptr)                          free((ptr))
#endif

se_status_t se_model_open(se_model_t *model, const char *path, se_error_t *err);
void se_model_close(se_model_t *model);
size_t se_model_memsize(const se_model_t *model);
uint32_t se_hash_bytes(const uint8_t *data, size_t len, uint32_t seed);
int se_vocab_lookup(const se_model_t *model, const uint8_t *bytes, size_t len, uint32_t *id_out);
int se_vocab_lookup_piece(const se_model_t *model, const uint8_t *prefix, size_t prefix_len,
                          const uint8_t *bytes, size_t len, uint32_t *id_out);
int se_trie_longest_match(const se_trie_t *trie, const uint8_t *bytes, size_t len, uint32_t *id_out,
                          size_t *matched_len_out);

size_t se_utf8_decode(const uint8_t *src, size_t len, uint32_t *out, size_t out_cap, int *ok);
size_t se_utf8_encode(uint32_t cp, uint8_t *dst);
int se_range_contains(const se_range_t *ranges, uint32_t count, uint32_t cp);
const se_map_entry_t *se_map_lookup(const se_map_entry_t *entries, uint32_t count, uint32_t cp);
int se_is_cjk(uint32_t cp);
static inline int se_is_ascii_punct(uint32_t cp) {
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
           (cp >= 123 && cp <= 126);
}

static inline int se_is_ascii_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r';
}

static inline int se_is_ascii_boundary(uint32_t cp) {
    return cp < 0x80 && (se_is_ascii_whitespace(cp) || se_is_ascii_punct(cp));
}

void se_scratch_init(se_scratch_t *s);
void se_scratch_free(se_scratch_t *s);
int se_scratch_reserve(se_scratch_t *s, uint32_t dim);
se_scratch_t *se_scratch_acquire(uint32_t dim);
void se_scratch_release(se_scratch_t *s);
void se_scratch_drop_thread(void);

size_t se_prefix_boundary_len(const se_model_t *model, const uint8_t *input, size_t input_len,
                              size_t target, size_t backscan);

typedef struct {
    uint32_t token_count;
    uint32_t unk_count;
    uint32_t truncated;
} se_token_stats_t;

typedef enum { SE_TOKEN_LIMIT_RAW = 0, SE_TOKEN_LIMIT_USABLE = 1 } se_token_limit_t;

se_status_t se_tokenize(const se_model_t *model, se_scratch_t *scratch, const uint8_t *input,
                        size_t input_len, uint32_t max_tokens, se_token_limit_t limit_mode,
                        se_token_stats_t *stats, se_error_t *err, volatile sig_atomic_t *cancelled);

se_status_t se_embed_one(const se_model_t *model, se_scratch_t *scratch, const uint8_t *input,
                         size_t input_len, uint32_t max_tokens, float *out, se_token_stats_t *stats,
                         se_error_t *err, volatile sig_atomic_t *cancelled);
se_status_t se_embed_ids(const se_model_t *model, se_scratch_t *scratch, const uint32_t *ids,
                         size_t n_ids, float *out, se_token_stats_t *stats, se_error_t *err,
                         volatile sig_atomic_t *cancelled);

void se_l2_normalize(float *vec, uint32_t dim);
size_t se_model_warmup(const se_model_t *model);

size_t se_vector_format_element_bytes(se_vector_format_t format);
uint16_t se_float_to_f16_bits(float value);
float se_f16_bits_to_float(uint16_t half);
void se_write_f16le(uint8_t *dst, float value);
float se_read_f16le(const uint8_t *src);
void se_encode_f16_from_floats(uint8_t *dst, const float *src, size_t count);
void se_decode_f16_to_floats(float *dst, const uint8_t *src, size_t count);
void se_select_f16_backend(void);
se_f16_backend_t se_current_f16_backend(void);

float se_dot_product_f32(const float *q, const float *row, size_t dim);
float se_dot_and_row_sq_f32(const float *q, const float *row, size_t dim, float *row_sq_out);
float se_dot_product_f16(const float *q, const uint8_t *row, size_t dim);
float se_dot_and_row_sq_f16(const float *q, const uint8_t *row, size_t dim, float *row_sq_out);
void *se_topk_execute(void *arg);

#endif
