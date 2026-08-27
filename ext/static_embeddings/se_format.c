#include "se_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#endif

void se_error_clear(se_error_t *err) {
    if (!err)
        return;
    err->status = SE_OK;
    err->message[0] = '\0';
}

void se_error_set(se_error_t *err, se_status_t status, const char *fmt, ...) {
    if (!err)
        return;
    err->status = status;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err->message, sizeof(err->message), fmt, ap);
    va_end(ap);
}

static uint32_t read_u32(const uint8_t *base, size_t off) {
    return (uint32_t)base[off] | ((uint32_t)base[off + 1] << 8) | ((uint32_t)base[off + 2] << 16) |
           ((uint32_t)base[off + 3] << 24);
}

static uint64_t read_u64(const uint8_t *base, size_t off) {
    return (uint64_t)read_u32(base, off) | ((uint64_t)read_u32(base, off + 4) << 32);
}

uint32_t se_hash_bytes(const uint8_t *data, size_t len, uint32_t seed) {
    uint32_t h = seed;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint32_t)data[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t hash_bytes_continue(uint32_t h, const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        h ^= (uint32_t)data[i];
        h *= 16777619u;
    }
    return h;
}

int se_vocab_lookup_piece(const se_model_t *model, const uint8_t *prefix, size_t prefix_len,
                          const uint8_t *bytes, size_t len, uint32_t *id_out) {
    if (prefix_len > UINT32_MAX || len > UINT32_MAX || prefix_len > UINT32_MAX - len)
        return 0;

    uint32_t h = model->meta.hash_seed;
    h = hash_bytes_continue(h, prefix, prefix_len);
    h = hash_bytes_continue(h, bytes, len);

    const uint32_t mask = model->meta.hash_table_size - 1u;
    uint32_t pos = h & mask;
    uint32_t total_len = (uint32_t)(prefix_len + len);

    for (uint32_t probe = 0; probe <= mask; probe++) {
        const se_vocab_slot_t *slot = &model->vocab_hash[pos];
        if (slot->token_id == SE_SLOT_EMPTY)
            return 0;
        if (slot->hash == h && slot->str_len == total_len) {
            const char *token = model->vocab_strings + slot->str_off;
            if ((prefix_len == 0 || memcmp(token, prefix, prefix_len) == 0) &&
                (len == 0 || memcmp(token + prefix_len, bytes, len) == 0)) {
                *id_out = slot->token_id;
                return 1;
            }
        }
        pos = (pos + 1u) & mask;
    }
    return 0;
}

int se_vocab_lookup(const se_model_t *model, const uint8_t *bytes, size_t len, uint32_t *id_out) {
    return se_vocab_lookup_piece(model, NULL, 0, bytes, len, id_out);
}

static inline uint32_t trie_find_child(const se_trie_t *trie, const se_trie_node_t *node,
                                       uint32_t byte) {
    uint32_t start = node->edge_start;
    uint32_t count = node->edge_count;

    if (count == 0)
        return SE_SLOT_EMPTY;

    if (count <= 8) {
        for (uint32_t i = 0; i < count; i++) {
            const se_trie_edge_t *edge = &trie->edges[start + i];
            if (edge->byte == byte)
                return edge->child;
            if (edge->byte > byte)
                break;
        }
        return SE_SLOT_EMPTY;
    }

    uint32_t lo = start;
    uint32_t hi = start + count;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        const se_trie_edge_t *edge = &trie->edges[mid];
        if (edge->byte < byte)
            lo = mid + 1u;
        else
            hi = mid;
    }

    if (lo < start + count && trie->edges[lo].byte == byte)
        return trie->edges[lo].child;
    return SE_SLOT_EMPTY;
}

int se_trie_longest_match(const se_trie_t *trie, const uint8_t *bytes, size_t len, uint32_t *id_out,
                          size_t *matched_len_out) {
    if (!trie || !trie->nodes || trie->node_count == 0)
        return 0;

    uint32_t node_index = 0;
    uint32_t best_id = SE_SLOT_EMPTY;
    size_t best_len = 0;

    for (size_t i = 0; i < len; i++) {
        const se_trie_node_t *node = &trie->nodes[node_index];
        uint32_t next = trie_find_child(trie, node, (uint32_t)bytes[i]);
        if (next == SE_SLOT_EMPTY)
            break;

        node_index = next;
        uint32_t token_id = trie->nodes[node_index].token_id;
        if (token_id != SE_SLOT_EMPTY) {
            best_id = token_id;
            best_len = i + 1u;
        }
    }

    if (best_id == SE_SLOT_EMPTY)
        return 0;

    *id_out = best_id;
    *matched_len_out = best_len;
    return 1;
}

struct section {
    uint64_t off;
    uint64_t size;
    const char *name;
};

static int section_present(struct section s) {
    return s.off != 0 || s.size != 0;
}

static uint64_t section_end(struct section s) {
    return s.off + s.size;
}

static int section_ok(struct section s, uint64_t file_size, size_t elem_size) {
    if (!section_present(s))
        return 1;
    if (s.off < SE_HEADER_SIZE)
        return 0;
    if (s.off % SE_ALIGNMENT != 0)
        return 0;
    if (s.size > file_size)
        return 0;
    if (s.off > file_size - s.size)
        return 0;
    if (elem_size && (s.size % elem_size) != 0)
        return 0;
    return 1;
}

static struct section read_section(const uint8_t *base, size_t off, const char *name) {
    struct section s;
    s.off = read_u64(base, off);
    s.size = read_u64(base, off + 8);
    s.name = name;
    return s;
}

static int sections_overlap(struct section a, struct section b) {
    if (!section_present(a) || !section_present(b))
        return 0;
    return a.off < section_end(b) && b.off < section_end(a);
}

static se_status_t validate_no_overlaps(const struct section *sections, size_t count,
                                        se_error_t *err) {
    for (size_t i = 0; i < count; i++) {
        for (size_t j = i + 1; j < count; j++) {
            if (sections_overlap(sections[i], sections[j])) {
                se_error_set(err, SE_ERR_INVALID_FORMAT, "sections %s and %s overlap",
                             sections[i].name, sections[j].name);
                return SE_ERR_INVALID_FORMAT;
            }
        }
    }
    return SE_OK;
}

static se_status_t parse_trie(se_trie_t *trie, const uint8_t *base, struct section sec,
                              uint32_t vocab_size, const char *name, se_error_t *err) {
    memset(trie, 0, sizeof(*trie));
    if (sec.size < 16) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "%s trie section too small", name);
        return SE_ERR_INVALID_FORMAT;
    }

    const uint8_t *p = base + sec.off;
    uint32_t node_count = read_u32(p, 0);
    uint32_t edge_count = read_u32(p, 4);
    if (node_count == 0) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "%s trie has no root node", name);
        return SE_ERR_INVALID_FORMAT;
    }

    uint64_t need = 16u + (uint64_t)node_count * sizeof(se_trie_node_t) +
                    (uint64_t)edge_count * sizeof(se_trie_edge_t);
    if (need != sec.size) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "%s trie size mismatch", name);
        return SE_ERR_INVALID_FORMAT;
    }

    const se_trie_node_t *nodes = (const se_trie_node_t *)(p + 16);
    const se_trie_edge_t *edges =
        (const se_trie_edge_t *)(p + 16 + (size_t)node_count * sizeof(se_trie_node_t));

    for (uint32_t i = 0; i < node_count; i++) {
        const se_trie_node_t *node = &nodes[i];
        if (node->token_id != SE_SLOT_EMPTY && node->token_id >= vocab_size) {
            se_error_set(err, SE_ERR_INVALID_FORMAT, "%s trie node %u has token id out of range",
                         name, i);
            return SE_ERR_INVALID_FORMAT;
        }
        if (node->edge_start > edge_count || node->edge_count > edge_count - node->edge_start) {
            se_error_set(err, SE_ERR_INVALID_FORMAT, "%s trie node %u edges out of range", name, i);
            return SE_ERR_INVALID_FORMAT;
        }
        uint32_t prev = 0;
        for (uint32_t k = 0; k < node->edge_count; k++) {
            const se_trie_edge_t *edge = &edges[node->edge_start + k];
            if (edge->byte > 255u || edge->child >= node_count) {
                se_error_set(err, SE_ERR_INVALID_FORMAT, "%s trie edge out of range", name);
                return SE_ERR_INVALID_FORMAT;
            }
            if (k && edge->byte <= prev) {
                se_error_set(err, SE_ERR_INVALID_FORMAT, "%s trie edges are not strictly sorted",
                             name);
                return SE_ERR_INVALID_FORMAT;
            }
            prev = edge->byte;
        }
    }

    trie->nodes = nodes;
    trie->edges = edges;
    trie->node_count = node_count;
    trie->edge_count = edge_count;
    return SE_OK;
}

static se_status_t parse_norm_tables(se_model_t *model, const uint8_t *base, struct section sec,
                                     se_error_t *err) {
    memset(&model->norm, 0, sizeof(model->norm));
    if (sec.size == 0)
        return SE_OK;

    if (sec.size < 32) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "norm tables section too small");
        return SE_ERR_INVALID_FORMAT;
    }

    const uint8_t *p = base + sec.off;
    uint32_t n_lower = read_u32(p, 0);
    uint32_t n_nfd = read_u32(p, 4);
    uint32_t n_mn = read_u32(p, 8);
    uint32_t n_punct = read_u32(p, 12);
    uint32_t n_control = read_u32(p, 16);
    uint32_t n_ws = read_u32(p, 20);

    uint64_t map_count = (uint64_t)n_lower + (uint64_t)n_nfd;
    uint64_t range_count =
        (uint64_t)n_mn + (uint64_t)n_punct + (uint64_t)n_control + (uint64_t)n_ws;
    if (map_count > (UINT64_MAX - 32u) / sizeof(se_map_entry_t) ||
        range_count > (UINT64_MAX - 32u) / sizeof(se_range_t)) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "norm tables size overflow");
        return SE_ERR_INVALID_FORMAT;
    }

    uint64_t need = 32u + map_count * sizeof(se_map_entry_t) + range_count * sizeof(se_range_t);

    if (need != sec.size) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "norm tables size mismatch (need %llu, have %llu)",
                     (unsigned long long)need, (unsigned long long)sec.size);
        return SE_ERR_INVALID_FORMAT;
    }

    const uint8_t *cursor = p + 32;
    model->norm.lower = (const se_map_entry_t *)cursor;
    model->norm.lower_count = n_lower;
    cursor += (size_t)n_lower * sizeof(se_map_entry_t);

    model->norm.nfd = (const se_map_entry_t *)cursor;
    model->norm.nfd_count = n_nfd;
    cursor += (size_t)n_nfd * sizeof(se_map_entry_t);

    model->norm.mn = (const se_range_t *)cursor;
    model->norm.mn_count = n_mn;
    cursor += (size_t)n_mn * sizeof(se_range_t);

    model->norm.punct = (const se_range_t *)cursor;
    model->norm.punct_count = n_punct;
    cursor += (size_t)n_punct * sizeof(se_range_t);

    model->norm.control = (const se_range_t *)cursor;
    model->norm.control_count = n_control;
    cursor += (size_t)n_control * sizeof(se_range_t);

    model->norm.whitespace = (const se_range_t *)cursor;
    model->norm.whitespace_count = n_ws;

    const se_map_entry_t *maps[] = {model->norm.lower, model->norm.nfd};
    const uint32_t map_counts[] = {model->norm.lower_count, model->norm.nfd_count};
    const char *map_names[] = {"lower", "nfd"};
    for (size_t table = 0; table < 2; table++) {
        uint32_t prev = 0;
        for (uint32_t i = 0; i < map_counts[table]; i++) {
            const se_map_entry_t *entry = &maps[table][i];
            if (entry->len > 4) {
                se_error_set(err, SE_ERR_INVALID_FORMAT, "%s map entry %u has len %u",
                             map_names[table], i, entry->len);
                return SE_ERR_INVALID_FORMAT;
            }
            if (entry->cp > 0x10FFFFu || (entry->cp >= 0xD800u && entry->cp <= 0xDFFFu)) {
                se_error_set(err, SE_ERR_INVALID_FORMAT, "%s map entry %u has invalid codepoint",
                             map_names[table], i);
                return SE_ERR_INVALID_FORMAT;
            }
            for (uint32_t k = 0; k < entry->len; k++) {
                uint32_t out_cp = entry->out[k];
                if (out_cp > 0x10FFFFu || (out_cp >= 0xD800u && out_cp <= 0xDFFFu)) {
                    se_error_set(err, SE_ERR_INVALID_FORMAT,
                                 "%s map entry %u output %u is not a valid codepoint",
                                 map_names[table], i, k);
                    return SE_ERR_INVALID_FORMAT;
                }
            }
            if (i && entry->cp <= prev) {
                se_error_set(err, SE_ERR_INVALID_FORMAT, "%s map is not strictly sorted",
                             map_names[table]);
                return SE_ERR_INVALID_FORMAT;
            }
            prev = entry->cp;
        }
    }

    const se_range_t *ranges[] = {model->norm.mn, model->norm.punct, model->norm.control,
                                  model->norm.whitespace};
    const uint32_t range_counts[] = {model->norm.mn_count, model->norm.punct_count,
                                     model->norm.control_count, model->norm.whitespace_count};
    const char *range_names[] = {"mn", "punct", "control", "whitespace"};
    for (size_t table = 0; table < 4; table++) {
        uint32_t prev_hi = 0;
        for (uint32_t i = 0; i < range_counts[table]; i++) {
            const se_range_t *range = &ranges[table][i];
            if (range->hi > 0x10FFFFu) {
                se_error_set(err, SE_ERR_INVALID_FORMAT, "%s range %u exceeds the Unicode range",
                             range_names[table], i);
                return SE_ERR_INVALID_FORMAT;
            }
            if (range->lo > range->hi) {
                se_error_set(err, SE_ERR_INVALID_FORMAT, "%s range %u is inverted",
                             range_names[table], i);
                return SE_ERR_INVALID_FORMAT;
            }
            if (i && range->lo <= prev_hi) {
                se_error_set(err, SE_ERR_INVALID_FORMAT, "%s ranges overlap or are unsorted",
                             range_names[table]);
                return SE_ERR_INVALID_FORMAT;
            }
            prev_hi = range->hi;
        }
    }

    return SE_OK;
}

static se_status_t validate(se_model_t *model, se_error_t *err) {
    const uint8_t *base = (const uint8_t *)model->map_base;
    const uint64_t file_size = (uint64_t)model->map_size;

    if (file_size < SE_HEADER_SIZE) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "file shorter than header");
        return SE_ERR_INVALID_FORMAT;
    }
    if (memcmp(base, SE_MAGIC, SE_MAGIC_LEN) != 0) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "bad magic");
        return SE_ERR_INVALID_FORMAT;
    }

    uint32_t version = read_u32(base, SE_OFF_FORMAT_VERSION);
    if (version != SE_FORMAT_VERSION) {
        se_error_set(err, SE_ERR_UNSUPPORTED_VERSION, "format version %u is not supported",
                     version);
        return SE_ERR_UNSUPPORTED_VERSION;
    }
    if (read_u32(base, SE_OFF_HEADER_SIZE) != SE_HEADER_SIZE) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "unexpected header size");
        return SE_ERR_INVALID_FORMAT;
    }

    se_meta_t *m = &model->meta;
    m->dim = read_u32(base, SE_OFF_DIM);
    m->vocab_size = read_u32(base, SE_OFF_VOCAB_SIZE);
    m->tokenizer_type = read_u32(base, SE_OFF_TOKENIZER_TYPE);
    m->embedding_dtype = read_u32(base, SE_OFF_EMBEDDING_DTYPE);
    m->pooling_type = read_u32(base, SE_OFF_POOLING_TYPE);
    m->normalization_type = read_u32(base, SE_OFF_NORMALIZATION_TYPE);
    m->max_tokens_default = read_u32(base, SE_OFF_MAX_TOKENS_DEFAULT);
    m->truncation_policy = read_u32(base, SE_OFF_TRUNCATION_POLICY);
    m->add_special_tokens = read_u32(base, SE_OFF_ADD_SPECIAL_TOKENS);
    m->unk_policy = read_u32(base, SE_OFF_UNK_POLICY);
    m->empty_policy = read_u32(base, SE_OFF_EMPTY_POLICY);
    m->do_lower_case = read_u32(base, SE_OFF_DO_LOWER_CASE);
    m->strip_accents = read_u32(base, SE_OFF_STRIP_ACCENTS);
    m->handle_chinese_chars = read_u32(base, SE_OFF_HANDLE_CHINESE_CHARS);
    m->clean_text = read_u32(base, SE_OFF_CLEAN_TEXT);
    m->max_input_chars_per_word = read_u32(base, SE_OFF_MAX_INPUT_CHARS_PER_WORD);
    m->pad_id = read_u32(base, SE_OFF_PAD_ID);
    m->unk_id = read_u32(base, SE_OFF_UNK_ID);
    m->cls_id = read_u32(base, SE_OFF_CLS_ID);
    m->sep_id = read_u32(base, SE_OFF_SEP_ID);
    m->mask_id = read_u32(base, SE_OFF_MASK_ID);
    m->hash_table_size = read_u32(base, SE_OFF_HASH_TABLE_SIZE);
    m->hash_seed = read_u32(base, SE_OFF_HASH_SEED);
    m->subword_prefix_len = read_u32(base, SE_OFF_SUBWORD_PREFIX_LEN);
    m->max_token_chars = read_u32(base, SE_OFF_MAX_TOKEN_CHARS);
    m->max_probe = read_u32(base, SE_OFF_MAX_PROBE);
    memcpy(m->subword_prefix, base + SE_OFF_SUBWORD_PREFIX, 8);

    if (m->tokenizer_type != SE_TOKENIZER_BERT_WORDPIECE_V1) {
        se_error_set(err, SE_ERR_UNSUPPORTED_TOKENIZER, "tokenizer profile %u is not supported",
                     m->tokenizer_type);
        return SE_ERR_UNSUPPORTED_TOKENIZER;
    }
    if (m->embedding_dtype != SE_DTYPE_F32) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "embedding dtype %u is not supported",
                     m->embedding_dtype);
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->pooling_type != SE_POOLING_MEAN) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "pooling type %u is not supported",
                     m->pooling_type);
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->normalization_type != SE_NORMALIZATION_NONE &&
        m->normalization_type != SE_NORMALIZATION_L2) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "normalization type %u is not supported",
                     m->normalization_type);
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->truncation_policy != SE_TRUNCATE_IDS_BEFORE_POOLING) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "truncation policy %u is not supported",
                     m->truncation_policy);
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->add_special_tokens != 0) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "runtime does not add special tokens");
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->unk_policy != SE_UNK_INCLUDE && m->unk_policy != SE_UNK_DROP) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "unknown UNK policy %u", m->unk_policy);
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->empty_policy != SE_EMPTY_ZERO_VECTOR && m->empty_policy != SE_EMPTY_RAISE) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "unknown empty-input policy %u", m->empty_policy);
        return SE_ERR_INVALID_FORMAT;
    }
    if ((m->do_lower_case > 1) || (m->strip_accents > 1) || (m->handle_chinese_chars > 1) ||
        (m->clean_text > 1)) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "normalizer flags must be 0 or 1");
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->dim == 0 || m->dim > (1u << 20)) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "implausible dim %u", m->dim);
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->vocab_size == 0) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "empty vocabulary");
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->hash_table_size == 0 || (m->hash_table_size & (m->hash_table_size - 1u)) != 0) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "hash table size %u is not a power of two",
                     m->hash_table_size);
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->hash_table_size < m->vocab_size) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "hash table smaller than vocabulary");
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->subword_prefix_len > 8) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "subword prefix too long");
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->unk_id >= m->vocab_size) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "unk id out of range");
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->clean_text != 1) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "BERT_WORDPIECE_V1 requires clean_text=true");
        return SE_ERR_INVALID_FORMAT;
    }
    if (m->max_input_chars_per_word == 0)
        m->max_input_chars_per_word = 100;
    if (m->max_token_chars == 0 || m->max_token_chars > m->max_input_chars_per_word)
        m->max_token_chars = m->max_input_chars_per_word;

    struct section vocab_strings = read_section(base, SE_OFF_SEC_VOCAB_STRINGS, "vocab_strings");
    struct section vocab_hash = read_section(base, SE_OFF_SEC_VOCAB_HASH, "vocab_hash");
    struct section embeddings = read_section(base, SE_OFF_SEC_EMBEDDINGS, "embeddings");
    struct section norm = read_section(base, SE_OFF_SEC_NORM_TABLES, "norm_tables");
    struct section provenance = read_section(base, SE_OFF_SEC_PROVENANCE, "provenance");
    struct section root_trie = read_section(base, SE_OFF_SEC_ROOT_TRIE, "root_trie");
    struct section cont_trie = read_section(base, SE_OFF_SEC_CONT_TRIE, "continuation_trie");

    if (!section_ok(vocab_strings, file_size, 0) ||
        !section_ok(vocab_hash, file_size, sizeof(se_vocab_slot_t)) ||
        !section_ok(embeddings, file_size, sizeof(float)) || !section_ok(norm, file_size, 0) ||
        !section_ok(provenance, file_size, 0) || !section_ok(root_trie, file_size, 0) ||
        !section_ok(cont_trie, file_size, 0)) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "section offsets are out of bounds or unaligned");
        return SE_ERR_INVALID_FORMAT;
    }

    const struct section sections[] = {vocab_strings, vocab_hash, embeddings, norm,
                                       provenance,    root_trie,  cont_trie};
    se_status_t overlap_status =
        validate_no_overlaps(sections, sizeof(sections) / sizeof(sections[0]), err);
    if (overlap_status != SE_OK)
        return overlap_status;

    uint64_t expected_hash_bytes = (uint64_t)m->hash_table_size * sizeof(se_vocab_slot_t);
    if (vocab_hash.size != expected_hash_bytes) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "hash section size mismatch");
        return SE_ERR_INVALID_FORMAT;
    }

    if ((uint64_t)m->vocab_size > UINT64_MAX / (uint64_t)m->dim ||
        (uint64_t)m->vocab_size * (uint64_t)m->dim > UINT64_MAX / sizeof(float)) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "embedding matrix size overflow");
        return SE_ERR_INVALID_FORMAT;
    }

    uint64_t expected_matrix = (uint64_t)m->vocab_size * (uint64_t)m->dim * sizeof(float);
    if (embeddings.size != expected_matrix) {
        se_error_set(err, SE_ERR_INVALID_FORMAT,
                     "embedding section is %llu bytes, expected vocab_size * dim * 4 = %llu",
                     (unsigned long long)embeddings.size, (unsigned long long)expected_matrix);
        return SE_ERR_INVALID_FORMAT;
    }

    model->vocab_strings = (const char *)(base + vocab_strings.off);
    model->vocab_strings_size = (size_t)vocab_strings.size;
    model->vocab_hash = (const se_vocab_slot_t *)(base + vocab_hash.off);
    model->embeddings = (const float *)(base + embeddings.off);
    model->provenance = provenance.size ? (const char *)(base + provenance.off) : NULL;
    model->provenance_size = (size_t)provenance.size;

    size_t bitset_size = ((size_t)m->vocab_size + 7u) / 8u;
    uint8_t *seen = (uint8_t *)calloc(bitset_size ? bitset_size : 1u, 1u);
    if (!seen) {
        se_error_set(err, SE_ERR_OOM, "out of memory while validating vocabulary hash");
        return SE_ERR_OOM;
    }

    uint32_t filled = 0;
    for (uint32_t i = 0; i < m->hash_table_size; i++) {
        const se_vocab_slot_t *slot = &model->vocab_hash[i];
        if (slot->token_id == SE_SLOT_EMPTY)
            continue;
        if (slot->token_id >= m->vocab_size) {
            free(seen);
            se_error_set(err, SE_ERR_INVALID_FORMAT, "slot %u has token id out of range", i);
            return SE_ERR_INVALID_FORMAT;
        }
        if ((uint64_t)slot->str_off + slot->str_len > vocab_strings.size) {
            free(seen);
            se_error_set(err, SE_ERR_INVALID_FORMAT, "slot %u points outside the string blob", i);
            return SE_ERR_INVALID_FORMAT;
        }
        uint8_t mask = (uint8_t)(1u << (slot->token_id & 7u));
        uint8_t *byte = &seen[slot->token_id >> 3];
        if (*byte & mask) {
            free(seen);
            se_error_set(err, SE_ERR_INVALID_FORMAT,
                         "token id %u appears more than once in hash table", slot->token_id);
            return SE_ERR_INVALID_FORMAT;
        }
        *byte |= mask;
        filled++;
    }
    free(seen);

    if (filled != m->vocab_size) {
        se_error_set(err, SE_ERR_INVALID_FORMAT, "hash table has %u filled slots, expected %u",
                     filled, m->vocab_size);
        return SE_ERR_INVALID_FORMAT;
    }

    uint32_t unk_lookup = 0;
    if (!se_vocab_lookup(model, (const uint8_t *)"[UNK]", 5, &unk_lookup) ||
        unk_lookup != m->unk_id) {
        se_error_set(err, SE_ERR_INVALID_FORMAT,
                     "[UNK] is not reachable through the vocabulary hash");
        return SE_ERR_INVALID_FORMAT;
    }

    if (m->max_probe > 0) {
        uint32_t mask = m->hash_table_size - 1u;
        for (uint32_t i = 0; i < m->hash_table_size; i++) {
            const se_vocab_slot_t *slot = &model->vocab_hash[i];
            if (slot->token_id == SE_SLOT_EMPTY)
                continue;
            uint32_t pos = slot->hash & mask;
            uint32_t probes = 1;
            while (pos != i) {
                probes++;
                pos = (pos + 1u) & mask;
                if (probes > m->max_probe) {
                    se_error_set(err, SE_ERR_INVALID_FORMAT, "slot %u exceeds recorded max probe",
                                 i);
                    return SE_ERR_INVALID_FORMAT;
                }
            }
        }
    }

    se_status_t trie_status =
        parse_trie(&model->root_trie, base, root_trie, m->vocab_size, "root", err);
    if (trie_status != SE_OK)
        return trie_status;
    trie_status =
        parse_trie(&model->cont_trie, base, cont_trie, m->vocab_size, "continuation", err);
    if (trie_status != SE_OK)
        return trie_status;

    return parse_norm_tables(model, base, norm, err);
}

se_status_t se_model_open(se_model_t *model, const char *path, se_error_t *err) {
    memset(model, 0, sizeof(*model));
    se_error_clear(err);

#ifndef _WIN32
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        se_error_set(err, SE_ERR_IO, "cannot open %s", path);
        return SE_ERR_IO;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        se_error_set(err, SE_ERR_IO, "cannot stat %s", path);
        return SE_ERR_IO;
    }

    if ((uintmax_t)st.st_size > (uintmax_t)SIZE_MAX) {
        close(fd);
        se_error_set(err, SE_ERR_IO, "file is too large to map safely");
        return SE_ERR_IO;
    }

    void *addr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        se_error_set(err, SE_ERR_IO, "mmap failed for %s", path);
        return SE_ERR_IO;
    }

    model->map_base = addr;
    model->map_size = (size_t)st.st_size;
    model->mapped = 1;
#else
    FILE *f = fopen(path, "rb");
    if (!f) {
        se_error_set(err, SE_ERR_IO, "cannot open %s", path);
        return SE_ERR_IO;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        se_error_set(err, SE_ERR_IO, "empty file %s", path);
        return SE_ERR_IO;
    }
    void *buf = malloc((size_t)size);
    if (!buf) {
        fclose(f);
        se_error_set(err, SE_ERR_OOM, "out of memory");
        return SE_ERR_OOM;
    }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        se_error_set(err, SE_ERR_IO, "short read on %s", path);
        return SE_ERR_IO;
    }
    fclose(f);
    model->map_base = buf;
    model->map_size = (size_t)size;
    model->mapped = 0;
#endif

    se_status_t rc = validate(model, err);
    if (rc != SE_OK) {
        se_model_close(model);
        return rc;
    }
    return SE_OK;
}

void se_model_close(se_model_t *model) {
    if (!model || !model->map_base)
        return;
#ifndef _WIN32
    if (model->mapped)
        munmap(model->map_base, model->map_size);
    else
        free(model->map_base);
#else
    free(model->map_base);
#endif
    memset(model, 0, sizeof(*model));
}

size_t se_model_memsize(const se_model_t *model) {
    if (!model)
        return 0;

    return sizeof(se_model_t) + model->map_size;
}
