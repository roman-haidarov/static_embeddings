#include "se_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const char *const se_alloc_category_names[SE_ALLOC_CATEGORY_COUNT] = {
    "unknown",    "scratch",     "batch_input",      "batch_output",
    "batch_stats", "batch_index", "token_ids",       "topk_query",
    "topk_best",  "topk_matrix_copy", "format_validate", "model_file"};

const char *se_alloc_category_name(se_alloc_category_t category) {
    if ((unsigned)category >= SE_ALLOC_CATEGORY_COUNT)
        return se_alloc_category_names[SE_ALLOC_UNKNOWN];
    return se_alloc_category_names[category];
}

#if SE_ENABLE_ALLOC_STATS

#define SE_ALLOC_STATS_MAGIC 0x5EA110C5u

typedef union {
    struct {
        uint32_t magic;
        uint32_t category;
        size_t size;
    } h;
    long double align_long_double;
    void *align_ptr;
    uint64_t align_u64;
} se_alloc_header_t;

static se_alloc_stats_t se_alloc_stats[SE_ALLOC_CATEGORY_COUNT];

static se_alloc_category_t normalize_category(se_alloc_category_t category) {
    if ((unsigned)category >= SE_ALLOC_CATEGORY_COUNT)
        return SE_ALLOC_UNKNOWN;
    return category;
}

#if defined(__GNUC__) || defined(__clang__)
static size_t atomic_add_size(size_t *ptr, size_t value) {
    return __sync_add_and_fetch(ptr, value);
}

static size_t atomic_sub_size(size_t *ptr, size_t value) {
    return __sync_sub_and_fetch(ptr, value);
}

static void atomic_inc_size(size_t *ptr) {
    __sync_fetch_and_add(ptr, (size_t)1);
}

static size_t atomic_load_size(const size_t *ptr) {
    return __sync_fetch_and_add((size_t *)ptr, (size_t)0);
}

static void atomic_store_size(size_t *ptr, size_t value) {
    __sync_lock_test_and_set(ptr, value);
}

static int atomic_cas_size(size_t *ptr, size_t old_value, size_t new_value) {
    return __sync_bool_compare_and_swap(ptr, old_value, new_value);
}
#else
static size_t atomic_add_size(size_t *ptr, size_t value) {
    *ptr += value;
    return *ptr;
}

static size_t atomic_sub_size(size_t *ptr, size_t value) {
    *ptr -= value;
    return *ptr;
}

static void atomic_inc_size(size_t *ptr) {
    (*ptr)++;
}

static size_t atomic_load_size(const size_t *ptr) {
    return *ptr;
}

static void atomic_store_size(size_t *ptr, size_t value) {
    *ptr = value;
}

static int atomic_cas_size(size_t *ptr, size_t old_value, size_t new_value) {
    if (*ptr != old_value)
        return 0;
    *ptr = new_value;
    return 1;
}
#endif

static void record_peak(se_alloc_category_t category, size_t current) {
    size_t peak = atomic_load_size(&se_alloc_stats[category].peak_bytes);
    while (current > peak) {
        if (atomic_cas_size(&se_alloc_stats[category].peak_bytes, peak, current))
            return;
        peak = atomic_load_size(&se_alloc_stats[category].peak_bytes);
    }
}

static void record_alloc(se_alloc_category_t category, size_t bytes) {
    se_alloc_stats_t *stats = &se_alloc_stats[category];
    size_t current = atomic_add_size(&stats->current_bytes, bytes);
    atomic_add_size(&stats->total_allocated_bytes, bytes);
    atomic_inc_size(&stats->alloc_count);
    record_peak(category, current);
}

static void record_realloc(se_alloc_category_t old_category, size_t old_bytes,
                           se_alloc_category_t new_category, size_t new_bytes) {
    se_alloc_stats_t *old_stats = &se_alloc_stats[old_category];
    se_alloc_stats_t *new_stats = &se_alloc_stats[new_category];
    atomic_sub_size(&old_stats->current_bytes, old_bytes);
    atomic_add_size(&old_stats->total_freed_bytes, old_bytes);
    atomic_inc_size(&old_stats->free_count);

    size_t current = atomic_add_size(&new_stats->current_bytes, new_bytes);
    atomic_add_size(&new_stats->total_allocated_bytes, new_bytes);
    atomic_inc_size(&new_stats->alloc_count);
    atomic_inc_size(&new_stats->realloc_count);
    record_peak(new_category, current);
}

static void record_free(se_alloc_category_t category, size_t bytes) {
    se_alloc_stats_t *stats = &se_alloc_stats[category];
    atomic_sub_size(&stats->current_bytes, bytes);
    atomic_add_size(&stats->total_freed_bytes, bytes);
    atomic_inc_size(&stats->free_count);
}

static se_alloc_header_t *header_from_ptr(void *ptr) {
    return ((se_alloc_header_t *)ptr) - 1;
}

void *se_alloc_stats_malloc(se_alloc_category_t category, size_t bytes) {
    size_t total = 0;
    if (!se_checked_add_size(sizeof(se_alloc_header_t), bytes, &total))
        return NULL;

    se_alloc_header_t *header = (se_alloc_header_t *)malloc(total);
    if (!header)
        return NULL;

    category = normalize_category(category);
    header->h.magic = SE_ALLOC_STATS_MAGIC;
    header->h.category = (uint32_t)category;
    header->h.size = bytes;
    record_alloc(category, bytes);
    return (void *)(header + 1);
}

void *se_alloc_stats_calloc(se_alloc_category_t category, size_t count, size_t elem_size) {
    size_t bytes = 0;
    if (!se_checked_mul_size(count, elem_size, &bytes))
        return NULL;

    void *ptr = se_alloc_stats_malloc(category, bytes);
    if (ptr)
        memset(ptr, 0, bytes);
    return ptr;
}

void *se_alloc_stats_realloc(se_alloc_category_t category, void *ptr, size_t bytes) {
    if (!ptr)
        return se_alloc_stats_malloc(category, bytes);

    size_t total = 0;
    if (!se_checked_add_size(sizeof(se_alloc_header_t), bytes, &total))
        return NULL;

    se_alloc_header_t *old_header = header_from_ptr(ptr);
    if (old_header->h.magic != SE_ALLOC_STATS_MAGIC)
        abort();

    se_alloc_category_t old_category = normalize_category((se_alloc_category_t)old_header->h.category);
    size_t old_bytes = old_header->h.size;
    se_alloc_header_t *new_header = (se_alloc_header_t *)realloc(old_header, total);
    if (!new_header)
        return NULL;

    category = normalize_category(category);
    new_header->h.magic = SE_ALLOC_STATS_MAGIC;
    new_header->h.category = (uint32_t)category;
    new_header->h.size = bytes;
    record_realloc(old_category, old_bytes, category, bytes);
    return (void *)(new_header + 1);
}

void se_alloc_stats_free(void *ptr) {
    if (!ptr)
        return;

    se_alloc_header_t *header = header_from_ptr(ptr);
    if (header->h.magic != SE_ALLOC_STATS_MAGIC)
        abort();

    se_alloc_category_t stored_category =
        normalize_category((se_alloc_category_t)header->h.category);
    size_t bytes = header->h.size;
    header->h.magic = 0;
    record_free(stored_category, bytes);
    free(header);
}

void se_alloc_stats_reset(void) {
    for (size_t i = 0; i < SE_ALLOC_CATEGORY_COUNT; i++) {
        se_alloc_stats_t *stats = &se_alloc_stats[i];
        size_t current = atomic_load_size(&stats->current_bytes);
        atomic_store_size(&stats->peak_bytes, current);
        atomic_store_size(&stats->total_allocated_bytes, 0);
        atomic_store_size(&stats->total_freed_bytes, 0);
        atomic_store_size(&stats->alloc_count, 0);
        atomic_store_size(&stats->realloc_count, 0);
        atomic_store_size(&stats->free_count, 0);
    }
}

void se_alloc_stats_snapshot(se_alloc_stats_t out[SE_ALLOC_CATEGORY_COUNT]) {
    for (size_t i = 0; i < SE_ALLOC_CATEGORY_COUNT; i++) {
        out[i].current_bytes = atomic_load_size(&se_alloc_stats[i].current_bytes);
        out[i].peak_bytes = atomic_load_size(&se_alloc_stats[i].peak_bytes);
        out[i].total_allocated_bytes =
            atomic_load_size(&se_alloc_stats[i].total_allocated_bytes);
        out[i].total_freed_bytes = atomic_load_size(&se_alloc_stats[i].total_freed_bytes);
        out[i].alloc_count = atomic_load_size(&se_alloc_stats[i].alloc_count);
        out[i].realloc_count = atomic_load_size(&se_alloc_stats[i].realloc_count);
        out[i].free_count = atomic_load_size(&se_alloc_stats[i].free_count);
    }
}

#else

void se_alloc_stats_reset(void) {
}

void se_alloc_stats_snapshot(se_alloc_stats_t out[SE_ALLOC_CATEGORY_COUNT]) {
    memset(out, 0, sizeof(se_alloc_stats_t) * SE_ALLOC_CATEGORY_COUNT);
}

#endif
