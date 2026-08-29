#include "se_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SE_CANCEL_CHECK_MASK 0x3ffu

void se_scratch_init(se_scratch_t *s) {
    memset(s, 0, sizeof(*s));
}

void se_scratch_free(se_scratch_t *s) {
    se_free(s->cps);
    se_free(s->cps2);
    se_free(s->bytes);
    se_free(s->ids);
    se_free(s->acc);
    memset(s, 0, sizeof(*s));
}

static void *grow_buffer(void *buf, size_t *cap, size_t need, size_t elem_size,
                         size_t min_capacity) {
    if (*cap >= need)
        return buf;

    size_t next = 0;
    size_t bytes = 0;
    if (!se_next_capacity(*cap, need, min_capacity, &next) ||
        !se_array_bytes(next, elem_size, &bytes)) {
        return NULL;
    }

    void *p = se_realloc(SE_ALLOC_SCRATCH, buf, bytes);
    if (!p)
        return NULL;

    *cap = next;
    return p;
}

static int grow_u32(uint32_t **buf, size_t *cap, size_t need) {
    uint32_t *p = (uint32_t *)grow_buffer(*buf, cap, need, sizeof(uint32_t), 256);
    if (!p)
        return 0;

    *buf = p;
    return 1;
}

static int grow_bytes(uint8_t **buf, size_t *cap, size_t need) {
    uint8_t *p = (uint8_t *)grow_buffer(*buf, cap, need, sizeof(uint8_t), 256);
    if (!p)
        return 0;

    *buf = p;
    return 1;
}

static int grow_float(float **buf, size_t *cap, size_t need) {
    float *p = (float *)grow_buffer(*buf, cap, need, sizeof(float), need);
    if (!p)
        return 0;

    *buf = p;
    return 1;
}

static int push_id(se_scratch_t *sc, size_t *n_ids, uint32_t id) {
    if (*n_ids >= UINT32_MAX)
        return 0;
    if (!grow_u32(&sc->ids, &sc->ids_cap, *n_ids + 1))
        return 0;
    sc->ids[(*n_ids)++] = id;
    return 1;
}

int se_scratch_reserve(se_scratch_t *s, uint32_t dim) {
    if (!grow_u32(&s->cps, &s->cps_cap, 256))
        return 0;
    if (!grow_u32(&s->cps2, &s->cps2_cap, 256))
        return 0;
    if (!grow_u32(&s->ids, &s->ids_cap, 256))
        return 0;
    if (!grow_bytes(&s->bytes, &s->bytes_cap, 256))
        return 0;
    if (!grow_float(&s->acc, &s->acc_cap, dim))
        return 0;

    return 1;
}

static int is_whitespace(const se_model_t *m, uint32_t cp) {
    if (cp < 0x80)
        return se_is_ascii_whitespace(cp);
    return se_range_contains(m->norm.whitespace, m->norm.whitespace_count, cp);
}

static int is_control(const se_model_t *m, uint32_t cp) {
    if (cp < 0x80)
        return (cp < 0x20 || cp == 0x7f) && cp != '\t' && cp != '\n' && cp != '\r';
    return se_range_contains(m->norm.control, m->norm.control_count, cp);
}

static int is_punct(const se_model_t *m, uint32_t cp) {
    if (cp < 0x80)
        return se_is_ascii_punct(cp);
    return se_range_contains(m->norm.punct, m->norm.punct_count, cp);
}

static int decode_one(const uint8_t *src, size_t len, size_t *i, uint32_t *cp_out) {
    uint8_t b0 = src[*i];
    uint32_t cp;
    size_t need;

    if (b0 < 0x80) {
        *cp_out = b0;
        (*i)++;
        return 1;
    }
    if ((b0 & 0xE0) == 0xC0) {
        cp = b0 & 0x1Fu;
        need = 1;
    } else if ((b0 & 0xF0) == 0xE0) {
        cp = b0 & 0x0Fu;
        need = 2;
    } else if ((b0 & 0xF8) == 0xF0) {
        cp = b0 & 0x07u;
        need = 3;
    } else {
        return 0;
    }

    if (*i + need >= len)
        return 0;

    for (size_t k = 1; k <= need; k++) {
        uint8_t bk = src[*i + k];
        if ((bk & 0xC0) != 0x80)
            return 0;
        cp = (cp << 6) | (uint32_t)(bk & 0x3Fu);
    }

    if ((need == 1 && cp < 0x80) || (need == 2 && cp < 0x800) || (need == 3 && cp < 0x10000))
        return 0;
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
        return 0;

    *cp_out = cp;
    *i += need + 1;
    return 1;
}

static int normalization_stable(const se_model_t *m, uint32_t cp) {
    if (cp < 0x80)
        return 1;
    if (m->meta.do_lower_case && se_map_lookup(m->norm.lower, m->norm.lower_count, cp))
        return 0;
    if (m->meta.strip_accents) {
        if (se_map_lookup(m->norm.nfd, m->norm.nfd_count, cp))
            return 0;
        if (se_range_contains(m->norm.mn, m->norm.mn_count, cp))
            return 0;
    }
    return 1;
}

static int cp_is_cjk_segment(const se_model_t *m, uint32_t cp) {
    return m->meta.handle_chinese_chars && cp >= 0x3400 && se_is_cjk(cp);
}

size_t se_prefix_boundary_len(const se_model_t *model, const uint8_t *input, size_t input_len,
                              size_t target, size_t backscan) {
    if (target >= input_len)
        return input_len;
    if (target == 0)
        return 0;

    size_t floor = target > backscan ? target - backscan : 0;
    size_t pos = target;

    while (pos > floor) {
        pos--;
        while (pos > floor && (input[pos] & 0xC0) == 0x80)
            pos--;
        if ((input[pos] & 0xC0) == 0x80)
            return 0;

        uint32_t cp = 0;
        size_t scan = pos;
        if (!decode_one(input, input_len, &scan, &cp))
            return 0;
        size_t after = scan;

        if (!normalization_stable(model, cp))
            continue;

        if (cp_is_cjk_segment(model, cp)) {
            if (after <= target)
                return after;
            if (pos > 0)
                return pos;
            return 0;
        }

        if ((is_whitespace(model, cp) || is_punct(model, cp)) && after <= target)
            return after;
    }

    return 0;
}

typedef enum {
    SE_ASCII_DROP = 0,
    SE_ASCII_SPACE = 1,
    SE_ASCII_PUNCT = 2,
    SE_ASCII_WORD = 3
} se_ascii_class_t;

static const uint8_t SE_ASCII_CLASS[128] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2,
    2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2,
    2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 0,
};

#define SE_ASCII_CANCEL_STRIDE (SE_CANCEL_CHECK_MASK + 1u)

typedef struct {
    const se_model_t *model;
    se_scratch_t *scratch;
    uint32_t max_tokens;
    size_t n_ids;
    size_t n_unk;
    size_t segment_len;
    se_token_stats_t *stats;
    se_error_t *err;
    volatile sig_atomic_t *cancelled;
} token_state_t;

static int token_cancelled(const token_state_t *st) {
    return st->cancelled && *st->cancelled;
}

static se_status_t oom(token_state_t *st, const char *where) {
    se_error_set(st->err, SE_ERR_OOM, "out of memory while %s", where);
    return SE_ERR_OOM;
}

static int append_segment_cp(token_state_t *st, uint32_t cp) {
    if (!grow_u32(&st->scratch->cps, &st->scratch->cps_cap, st->segment_len + 1))
        return 0;
    st->scratch->cps[st->segment_len++] = cp;
    return 1;
}

static int cap_after_append(token_state_t *st) {
    if (st->max_tokens == 0 || st->n_ids <= (size_t)st->max_tokens)
        return 0;

    size_t dropped_unk = 0;
    for (size_t k = st->max_tokens; k < st->n_ids; k++) {
        if (st->scratch->ids[k] == st->model->meta.unk_id)
            dropped_unk++;
    }
    st->n_unk -= dropped_unk;
    st->n_ids = st->max_tokens;
    st->stats->truncated = 1;
    return 1;
}

typedef enum { WORDPIECE_OK = 1, WORDPIECE_OOM = 0, WORDPIECE_INVALID = -1 } wordpiece_status_t;

static wordpiece_status_t wordpiece(const se_model_t *m, se_scratch_t *sc, const uint32_t *word,
                                    size_t word_len, size_t *n_ids, size_t *n_unk) {
    const se_meta_t *meta = &m->meta;

    if (word_len == 0)
        return 1;

    if (word_len > meta->max_input_chars_per_word) {
        if (!push_id(sc, n_ids, meta->unk_id))
            return 0;
        (*n_unk)++;
        return 1;
    }

    size_t bytes_need = 0;
    if (!se_checked_mul_size(word_len, 4, &bytes_need) ||
        !se_checked_add_size(bytes_need, 4, &bytes_need))
        return 0;
    if (!grow_bytes(&sc->bytes, &sc->bytes_cap, bytes_need))
        return 0;
    if (!grow_u32(&sc->cps2, &sc->cps2_cap, word_len + 1))
        return 0;

    size_t blen = 0;
    for (size_t k = 0; k < word_len; k++) {
        if (blen > UINT32_MAX)
            return 0;
        sc->cps2[k] = (uint32_t)blen;
        uint32_t cp = word[k];
        if (cp < 0x80) {
            sc->bytes[blen++] = (uint8_t)cp;
        } else {
            blen += se_utf8_encode(cp, sc->bytes + blen);
        }
    }
    if (blen > UINT32_MAX)
        return 0;
    sc->cps2[word_len] = (uint32_t)blen;

    if (word_len <= meta->max_token_chars) {
        uint32_t exact_id = 0;
        if (se_vocab_lookup(m, sc->bytes, blen, &exact_id))
            return push_id(sc, n_ids, exact_id);
    }

    size_t start = 0;
    size_t emitted_before = *n_ids;

    while (start < word_len) {
        size_t max_end = start + meta->max_token_chars;
        if (max_end > word_len)
            max_end = word_len;

        size_t off = sc->cps2[start];
        size_t max_off = sc->cps2[max_end];
        size_t matched_len = 0;
        uint32_t found_id = 0;
        const se_trie_t *trie = start > 0 ? &m->cont_trie : &m->root_trie;

        int found =
            se_trie_longest_match(trie, sc->bytes + off, max_off - off, &found_id, &matched_len);

        if (!found) {
            *n_ids = emitted_before;
            if (!push_id(sc, n_ids, meta->unk_id))
                return 0;
            (*n_unk)++;
            return 1;
        }

        size_t target = off + matched_len;
        size_t end = start + 1;
        while (end <= max_end && sc->cps2[end] < target)
            end++;
        if (end > max_end || sc->cps2[end] != target)
            return WORDPIECE_INVALID;

        if (!push_id(sc, n_ids, found_id))
            return 0;
        start = end;
    }
    return 1;
}

static se_status_t append_wordpiece(token_state_t *st, const uint32_t *word, size_t word_len,
                                    int *stop) {
    wordpiece_status_t wp =
        wordpiece(st->model, st->scratch, word, word_len, &st->n_ids, &st->n_unk);
    if (wp == WORDPIECE_OOM)
        return oom(st, "tokenizing");
    if (wp == WORDPIECE_INVALID) {
        se_error_set(st->err, SE_ERR_INTERNAL, "invalid WordPiece byte boundary");
        return SE_ERR_INTERNAL;
    }
    if (cap_after_append(st))
        *stop = 1;
    return SE_OK;
}

static se_status_t flush_segment(token_state_t *st, int *stop) {
    if (st->segment_len == 0)
        return SE_OK;
    se_status_t rc = append_wordpiece(st, st->scratch->cps, st->segment_len, stop);
    st->segment_len = 0;
    return rc;
}

static se_status_t feed_token_cp(token_state_t *st, uint32_t cp, int *stop) {
    if (is_whitespace(st->model, cp))
        return flush_segment(st, stop);

    if (is_punct(st->model, cp)) {
        se_status_t rc = flush_segment(st, stop);
        if (rc != SE_OK || *stop)
            return rc;
        return append_wordpiece(st, &cp, 1, stop);
    }

    if (!append_segment_cp(st, cp))
        return oom(st, "building token segment");
    return SE_OK;
}

static se_status_t emit_lowered(token_state_t *st, uint32_t cp, int *stop) {
    if (st->model->meta.do_lower_case) {
        if (cp >= 'A' && cp <= 'Z')
            cp += 32;
        else if (cp >= 0x80) {
            const se_map_entry_t *e =
                se_map_lookup(st->model->norm.lower, st->model->norm.lower_count, cp);
            if (e) {
                for (uint32_t k = 0; k < e->len; k++) {
                    se_status_t rc = feed_token_cp(st, e->out[k], stop);
                    if (rc != SE_OK || *stop)
                        return rc;
                }
                return SE_OK;
            }
        }
    }
    return feed_token_cp(st, cp, stop);
}

static se_status_t emit_stripped(token_state_t *st, uint32_t cp, int *stop) {
    if (!st->model->meta.strip_accents || cp < 0x80)
        return emit_lowered(st, cp, stop);

    const se_map_entry_t *e = se_map_lookup(st->model->norm.nfd, st->model->norm.nfd_count, cp);
    if (!e) {
        if (se_range_contains(st->model->norm.mn, st->model->norm.mn_count, cp))
            return SE_OK;
        return emit_lowered(st, cp, stop);
    }

    for (uint32_t k = 0; k < e->len; k++) {
        uint32_t d = e->out[k];
        if (se_range_contains(st->model->norm.mn, st->model->norm.mn_count, d))
            continue;
        se_status_t rc = emit_lowered(st, d, stop);
        if (rc != SE_OK || *stop)
            return rc;
    }
    return SE_OK;
}

static se_status_t emit_cleaned(token_state_t *st, uint32_t cp, int *stop) {
    if (st->model->meta.clean_text) {
        if (cp == 0 || cp == 0xFFFD || is_control(st->model, cp))
            return SE_OK;
        if (is_whitespace(st->model, cp))
            cp = ' ';
    }

    if (st->model->meta.handle_chinese_chars && cp >= 0x3400 && se_is_cjk(cp)) {
        se_status_t rc = emit_stripped(st, ' ', stop);
        if (rc != SE_OK || *stop)
            return rc;
        rc = emit_stripped(st, cp, stop);
        if (rc != SE_OK || *stop)
            return rc;
        return emit_stripped(st, ' ', stop);
    }

    return emit_stripped(st, cp, stop);
}

static inline se_ascii_class_t ascii_class(uint8_t byte) {
    return (se_ascii_class_t)SE_ASCII_CLASS[byte];
}

static size_t ascii_word_reserve_len(const uint8_t *input, size_t input_len, size_t i) {
    size_t run = 1;
    size_t limit = input_len - i;

    if (limit > SE_ASCII_CANCEL_STRIDE)
        limit = SE_ASCII_CANCEL_STRIDE;

    while (run < limit && input[i + run] < 0x80 && ascii_class(input[i + run]) == SE_ASCII_WORD)
        run++;

    return run;
}

static se_status_t tokenize_ascii_run(token_state_t *st, const uint8_t *input, size_t input_len,
                                      size_t *ip, int *stop) {
    size_t i = *ip;
    size_t next_cancel_check = i + SE_ASCII_CANCEL_STRIDE;
    const int lc = (int)st->model->meta.do_lower_case;
    se_scratch_t *sc = st->scratch;

    while (i < input_len) {
        uint8_t b = input[i];
        if (b >= 0x80)
            break;

        if (i >= next_cancel_check) {
            if (token_cancelled(st)) {
                *ip = i;
                se_error_set(st->err, SE_ERR_INTERNAL, "operation cancelled");
                return SE_ERR_INTERNAL;
            }
            next_cancel_check = i + SE_ASCII_CANCEL_STRIDE;
        }

        se_ascii_class_t cls = ascii_class(b);

        if (cls == SE_ASCII_WORD) {
            if (st->segment_len >= sc->cps_cap) {
                if (!grow_u32(&sc->cps, &sc->cps_cap,
                              st->segment_len + ascii_word_reserve_len(input, input_len, i))) {
                    *ip = i;
                    return oom(st, "building token segment");
                }
            }
            sc->cps[st->segment_len++] =
                (lc && b >= 'A' && b <= 'Z') ? (uint32_t)(b + 32u) : (uint32_t)b;
            i++;
            continue;
        }

        if (cls == SE_ASCII_DROP) {
            i++;
            continue;
        }

        se_status_t rc = flush_segment(st, stop);
        if (rc != SE_OK) {
            *ip = i;
            return rc;
        }
        if (*stop) {
            *ip = i;
            return SE_OK;
        }

        if (cls == SE_ASCII_PUNCT) {
            uint32_t cp = b;
            rc = append_wordpiece(st, &cp, 1, stop);
            if (rc != SE_OK) {
                *ip = i;
                return rc;
            }
        }

        i++;
        if (*stop)
            break;
    }

    *ip = i;
    return SE_OK;
}

se_status_t se_tokenize(const se_model_t *model, se_scratch_t *sc, const uint8_t *input,
                        size_t input_len, uint32_t max_tokens, se_token_stats_t *stats,
                        se_error_t *err, volatile sig_atomic_t *cancelled) {
    memset(stats, 0, sizeof(*stats));

    token_state_t st;
    memset(&st, 0, sizeof(st));
    st.model = model;
    st.scratch = sc;
    st.max_tokens = max_tokens;
    st.stats = stats;
    st.err = err;
    st.cancelled = cancelled;

    size_t i = 0;
    size_t iterations = 0;
    while (i < input_len) {
        if (((iterations++ & SE_CANCEL_CHECK_MASK) == 0) && token_cancelled(&st)) {
            se_error_set(err, SE_ERR_INTERNAL, "operation cancelled");
            return SE_ERR_INTERNAL;
        }

        int stop = 0;

        if (input[i] < 0x80) {
            se_status_t frc = tokenize_ascii_run(&st, input, input_len, &i, &stop);
            if (frc != SE_OK)
                return frc;
            if (stop)
                break;
            if (i >= input_len)
                break;
        }

        uint32_t cp;
        if (!decode_one(input, input_len, &i, &cp)) {
            se_error_set(err, SE_ERR_INVALID_UTF8, "input is not valid UTF-8");
            return SE_ERR_INVALID_UTF8;
        }

        se_status_t rc = emit_cleaned(&st, cp, &stop);
        if (rc != SE_OK)
            return rc;
        if (stop)
            break;
    }

    if (!stats->truncated) {
        int stop = 0;
        se_status_t rc = flush_segment(&st, &stop);
        if (rc != SE_OK)
            return rc;
    }

    if (st.n_ids > UINT32_MAX || st.n_unk > UINT32_MAX) {
        se_error_set(err, SE_ERR_OOM, "input produced too many tokens");
        return SE_ERR_OOM;
    }

    stats->token_count = (uint32_t)st.n_ids;
    stats->unk_count = (uint32_t)st.n_unk;
    return SE_OK;
}
