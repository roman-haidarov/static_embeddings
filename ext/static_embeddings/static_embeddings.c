#include <ruby.h>
#include <ruby/encoding.h>
#include <ruby/thread.h>

#ifdef HAVE_RUBY_FIBER_SCHEDULER_H
#include <ruby/fiber/scheduler.h>
#endif

#include <float.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SE_HAVE_NEON 1
#elif defined(__SSE__)
#include <xmmintrin.h>
#define SE_HAVE_SSE 1
#endif

#include "se_internal.h"

#if defined(__GNUC__) || defined(__clang__)
#define SE_NORETURN __attribute__((noreturn))
#else
#define SE_NORETURN
#endif

#define SE_GVL_UNLOCK_THRESHOLD      2048
#define SE_FIBER_THRESHOLD           2048
#define SE_IDS_GVL_UNLOCK_THRESHOLD  256
#define SE_TOPK_GVL_UNLOCK_THRESHOLD (1024 * 1024)
#define SE_PREFIX_BYTES_PER_TOKEN    16
#define SE_PREFIX_MIN_BYTES          4096
#define SE_PREFIX_MAX_BYTES          65536
#define SE_PREFIX_BACKSCAN_BYTES     8192
#define SE_SIZE_MAX                  ((size_t)-1)

typedef enum { SE_VECTOR_FORMAT_F32 = 1, SE_VECTOR_FORMAT_F16 = 2 } se_vector_format_t;

static size_t vector_format_element_bytes(se_vector_format_t format) {
    return format == SE_VECTOR_FORMAT_F16 ? 2u : sizeof(float);
}

static int string_equals_literal(VALUE str, const char *lit) {
    size_t n = strlen(lit);
    return (size_t)RSTRING_LEN(str) == n && memcmp(RSTRING_PTR(str), lit, n) == 0;
}

static se_vector_format_t resolve_vector_format(VALUE opt) {
    if (opt == Qundef || opt == Qnil)
        return SE_VECTOR_FORMAT_F32;

    VALUE name;
    if (SYMBOL_P(opt)) {
        name = rb_sym2str(opt);
    } else if (RB_TYPE_P(opt, T_STRING)) {
        name = opt;
    } else {
        rb_raise(rb_eArgError, "format must be :f32, :float32, :f16, or :float16");
    }

    if (string_equals_literal(name, "f32") || string_equals_literal(name, "float32"))
        return SE_VECTOR_FORMAT_F32;
    if (string_equals_literal(name, "f16") || string_equals_literal(name, "float16"))
        return SE_VECTOR_FORMAT_F16;

    rb_raise(rb_eArgError, "unsupported embedding format %" PRIsVALUE " (expected :f32 or :f16)",
             opt);
}

static void write_u16le(uint8_t *dst, uint16_t v) {
    dst[0] = (uint8_t)(v & 0xffu);
    dst[1] = (uint8_t)(v >> 8);
}

static uint16_t read_u16le(const uint8_t *src) {
    return (uint16_t)src[0] | ((uint16_t)src[1] << 8);
}

static uint16_t float_to_f16_bits(float value) {
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

static float f16_bits_to_float(uint16_t half) {
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

static void encode_f16_from_floats(uint8_t *dst, const float *src, size_t count) {
    for (size_t i = 0; i < count; i++)
        write_u16le(dst + i * 2, float_to_f16_bits(src[i]));
}

static void decode_f16_to_floats(float *dst, const uint8_t *src, size_t count) {
    for (size_t i = 0; i < count; i++)
        dst[i] = f16_bits_to_float(read_u16le(src + i * 2));
}

static int checked_add_size(size_t a, size_t b, size_t *out) {
    if (a > SE_SIZE_MAX - b)
        return 0;
    *out = a + b;
    return 1;
}

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SE_SIZE_MAX / a)
        return 0;
    *out = a * b;
    return 1;
}

static int size_fits_long(size_t n) {
    return n <= (size_t)LONG_MAX;
}

static VALUE mStaticEmbeddings;
static VALUE cModel;
static VALUE cFiber;
static VALUE eError;
static VALUE eInvalidModel;
static VALUE eUnsupportedModel;
static VALUE eEncodingError;
static VALUE eEmptyInput;

static rb_encoding *binary_encoding;
static rb_encoding *utf8_encoding;

static ID id_join;
static ID id_kill;
static ID id_max_tokens;
static ID id_threads;
static ID id_format;
static ID id_blocking_p;
static ID id_vector;
static ID id_token_count;
static ID id_unk_count;
static ID id_truncated;

RUBY_FUNC_EXPORTED void Init_static_embeddings(void);

typedef struct {
    se_model_t model;
    int open;
} model_wrapper_t;

static void model_free(void *ptr) {
    model_wrapper_t *w = (model_wrapper_t *)ptr;
    if (!w)
        return;
    if (w->open)
        se_model_close(&w->model);
    xfree(w);
}

static size_t model_memsize(const void *ptr) {
    const model_wrapper_t *w = (const model_wrapper_t *)ptr;
    if (!w)
        return 0;
    return sizeof(model_wrapper_t) + se_model_memsize(&w->model);
}

static const rb_data_type_t model_type = {"StaticEmbeddings/Model",
                                          {NULL, model_free, model_memsize, NULL, {0}},
                                          0,
                                          0,
                                          RUBY_TYPED_FREE_IMMEDIATELY};

static model_wrapper_t *get_model(VALUE self) {
    model_wrapper_t *w;
    TypedData_Get_Struct(self, model_wrapper_t, &model_type, w);
    if (!w->open)
        rb_raise(eError, "model is closed");
    return w;
}

static VALUE error_class_for(se_status_t status) {
    switch (status) {
    case SE_ERR_INVALID_FORMAT:
    case SE_ERR_IO:
        return eInvalidModel;
    case SE_ERR_UNSUPPORTED_VERSION:
    case SE_ERR_UNSUPPORTED_TOKENIZER:
        return eUnsupportedModel;
    case SE_ERR_INVALID_UTF8:
        return eEncodingError;
    case SE_ERR_EMPTY_INPUT:
        return eEmptyInput;
    default:
        return eError;
    }
}

static void raise_se(const se_error_t *err) SE_NORETURN;

static void raise_se(const se_error_t *err) {
    rb_raise(error_class_for(err->status), "%s", err->message);
}

static void check_text_encoding_at(VALUE str, long index) {
    rb_encoding *enc = rb_enc_get(str);
    if (enc != utf8_encoding && enc != rb_usascii_encoding()) {
        if (index >= 0) {
            rb_raise(eEncodingError,
                     "input[%ld]: expected UTF-8 or US-ASCII, got %s (transcode explicitly)", index,
                     rb_enc_name(enc));
        }
        rb_raise(eEncodingError, "expected UTF-8 or US-ASCII, got %s (transcode explicitly)",
                 rb_enc_name(enc));
    }
    int cr = rb_enc_str_coderange(str);
    if (cr != ENC_CODERANGE_VALID && cr != ENC_CODERANGE_7BIT) {
        if (index >= 0)
            rb_raise(eEncodingError, "input[%ld]: string is not valid %s", index, rb_enc_name(enc));
        rb_raise(eEncodingError, "string is not valid %s", rb_enc_name(enc));
    }
}

static void check_text_encoding(VALUE str) {
    check_text_encoding_at(str, -1);
}

typedef struct {
    const uint8_t *base;
    size_t *offsets;
    size_t *lengths;
    size_t count;
    void *allocation;
} batch_input_t;

typedef struct {
    const se_model_t *model;
    batch_input_t input;
    uint32_t max_tokens;
    float *out;
    se_token_stats_t *stats;
    const size_t *out_index;
    size_t next_index;
    int failed;
    size_t failed_index;
    volatile sig_atomic_t cancelled;
    se_error_t error;
} batch_job_t;

static void job_fail(batch_job_t *job, se_error_t err, size_t index) {
    if (job->failed)
        return;
    job->failed = 1;
    job->failed_index = index;
    job->error = err;
}

static void batch_worker_run(batch_job_t *job, se_scratch_t *scratch) {
    const uint32_t dim = job->model->meta.dim;

    for (;;) {
        if (job->cancelled) {
            se_error_set(&job->error, SE_ERR_INTERNAL, "operation cancelled");
            job->failed = 1;
            return;
        }

        size_t i = job->next_index++;
        if (i >= job->input.count || job->failed)
            return;

        const uint8_t *text = job->input.base + job->input.offsets[i];
        size_t len = job->input.lengths[i];
        size_t row = job->out_index ? job->out_index[i] : i;

        if (!se_scratch_reserve(scratch, len, dim)) {
            se_error_t err;
            se_error_set(&err, SE_ERR_OOM, "out of memory while sizing scratch buffers");
            job_fail(job, err, i);
            return;
        }

        se_error_t err;
        se_error_clear(&err);
        se_status_t rc =
            se_embed_one(job->model, scratch, text, len, job->max_tokens, job->out + row * dim,
                         &job->stats[row], &err, &job->cancelled);
        if (rc != SE_OK) {
            job_fail(job, err, i);
            return;
        }
    }
}

static void *batch_execute(void *arg) {
    batch_job_t *job = (batch_job_t *)arg;
    se_scratch_t scratch;
    se_scratch_init(&scratch);
    batch_worker_run(job, &scratch);
    se_scratch_free(&scratch);
    return NULL;
}

static void unblock_cancel(void *arg) {
    batch_job_t *job = (batch_job_t *)arg;
    if (job)
        job->cancelled = 1;
}

static VALUE current_fiber_scheduler(void) {
#ifdef HAVE_RUBY_FIBER_SCHEDULER_H
    VALUE sched = rb_fiber_scheduler_current();
    if (sched == Qnil || sched == Qfalse)
        return Qnil;
    return sched;
#else
    return Qnil;
#endif
}

static int current_fiber_is_blocking(void) {
#ifdef HAVE_RUBY_FIBER_SCHEDULER_H
    if (NIL_P(cFiber) || !rb_respond_to(cFiber, id_blocking_p))
        return 1;
    return RTEST(rb_funcall(cFiber, id_blocking_p, 0));
#else
    return 1;
#endif
}

typedef struct {
    batch_job_t *job;
    VALUE thread;
} fiber_worker_t;

static void fiber_worker_mark(void *ptr) {
    fiber_worker_t *w = (fiber_worker_t *)ptr;
    if (w)
        rb_gc_mark(w->thread);
}

static void fiber_worker_free(void *ptr) {
    xfree(ptr);
}

static const rb_data_type_t fiber_worker_type = {
    "StaticEmbeddings/FiberWorker",
    {fiber_worker_mark, fiber_worker_free, NULL, NULL, {0}},
    0,
    0,
    RUBY_TYPED_FREE_IMMEDIATELY};

static VALUE fiber_thread_body(void *arg) {
    fiber_worker_t *w = (fiber_worker_t *)arg;
    rb_thread_call_without_gvl(batch_execute, w->job, unblock_cancel, w->job);
    return Qnil;
}

static VALUE fiber_worker_wait(VALUE wrapper) {
    fiber_worker_t *w;
    TypedData_Get_Struct(wrapper, fiber_worker_t, &fiber_worker_type, w);
    w->thread = rb_thread_create(fiber_thread_body, w);
    rb_funcall(w->thread, id_join, 0);
    w->thread = Qnil;
    return Qnil;
}

static VALUE fiber_worker_ensure(VALUE wrapper) {
    fiber_worker_t *w;
    TypedData_Get_Struct(wrapper, fiber_worker_t, &fiber_worker_type, w);
    if (!NIL_P(w->thread)) {
        w->job->cancelled = 1;
        rb_funcall(w->thread, id_kill, 0);
        rb_funcall(w->thread, id_join, 0);
    }
    return Qnil;
}

static void run_job(batch_job_t *job, size_t total_bytes) {
    VALUE scheduler = current_fiber_scheduler();

    if (scheduler != Qnil && !current_fiber_is_blocking() && total_bytes >= SE_FIBER_THRESHOLD) {
        fiber_worker_t *w;
        VALUE wrapper = TypedData_Make_Struct(rb_cObject, fiber_worker_t, &fiber_worker_type, w);
        w->job = job;
        w->thread = Qnil;
        rb_ensure(fiber_worker_wait, wrapper, fiber_worker_ensure, wrapper);
        RB_GC_GUARD(wrapper);
        return;
    }

    if (total_bytes >= SE_GVL_UNLOCK_THRESHOLD) {
        rb_thread_call_without_gvl(batch_execute, job, unblock_cancel, job);
        rb_thread_check_ints();
        return;
    }

    batch_execute(job);
    rb_thread_check_ints();
}

static VALUE protected_run_job(VALUE arg) {
    batch_job_t *job = (batch_job_t *)(uintptr_t)arg;
    size_t total_bytes = 0;
    for (size_t i = 0; i < job->input.count; i++)
        total_bytes += job->input.lengths[i];
    run_job(job, total_bytes);
    return Qnil;
}

static VALUE model_alloc(VALUE klass) {
    model_wrapper_t *w;
    VALUE obj = TypedData_Make_Struct(klass, model_wrapper_t, &model_type, w);
    memset(w, 0, sizeof(*w));
    return obj;
}

static VALUE model_initialize(VALUE self, VALUE path) {
    model_wrapper_t *w;
    TypedData_Get_Struct(self, model_wrapper_t, &model_type, w);

    Check_Type(path, T_STRING);
    if (memchr(RSTRING_PTR(path), '\0', (size_t)RSTRING_LEN(path)))
        rb_raise(rb_eArgError, "path contains a null byte");

    se_error_t err;
    se_error_clear(&err);
    if (se_model_open(&w->model, StringValueCStr(path), &err) != SE_OK)
        raise_se(&err);

    w->open = 1;
    rb_ivar_set(self, rb_intern("@path"), rb_str_new_frozen(path));
    return self;
}

static VALUE model_close(VALUE self) {
    model_wrapper_t *w;
    TypedData_Get_Struct(self, model_wrapper_t, &model_type, w);
    if (w->open) {
        se_model_close(&w->model);
        w->open = 0;
    }
    return Qnil;
}

static VALUE model_closed_p(VALUE self) {
    model_wrapper_t *w;
    TypedData_Get_Struct(self, model_wrapper_t, &model_type, w);
    return w->open ? Qfalse : Qtrue;
}

static VALUE model_dim(VALUE self) {
    return UINT2NUM(get_model(self)->model.meta.dim);
}

static VALUE model_vocab_size(VALUE self) {
    return UINT2NUM(get_model(self)->model.meta.vocab_size);
}

static VALUE model_max_tokens(VALUE self) {
    return UINT2NUM(get_model(self)->model.meta.max_tokens_default);
}

static VALUE model_normalized_p(VALUE self) {
    return get_model(self)->model.meta.normalization_type == SE_NORMALIZATION_L2 ? Qtrue : Qfalse;
}

static VALUE model_lowercase_p(VALUE self) {
    return get_model(self)->model.meta.do_lower_case ? Qtrue : Qfalse;
}

static VALUE model_unk_id(VALUE self) {
    return UINT2NUM(get_model(self)->model.meta.unk_id);
}

static VALUE model_provenance_json(VALUE self) {
    model_wrapper_t *w = get_model(self);
    if (!w->model.provenance)
        return Qnil;
    return rb_enc_str_new(w->model.provenance, (long)w->model.provenance_size, utf8_encoding);
}

static VALUE model_mapped_bytes(VALUE self) {
    return SIZET2NUM(get_model(self)->model.map_size);
}

typedef struct {
    const se_model_t *model;
    size_t pages;
} warmup_job_t;

static void *warmup_execute(void *arg) {
    warmup_job_t *job = (warmup_job_t *)arg;
    job->pages = se_model_warmup(job->model);
    return NULL;
}

static VALUE model_warmup(VALUE self) {
    model_wrapper_t *w = get_model(self);
    warmup_job_t job;
    job.model = &w->model;
    job.pages = 0;
    rb_thread_call_without_gvl(warmup_execute, &job, NULL, NULL);
    RB_GC_GUARD(self);
    return self;
}

static uint32_t resolve_max_tokens(const se_model_t *model, VALUE opt) {
    if (opt == Qundef || opt == Qnil)
        return model->meta.max_tokens_default;
    if (opt == Qfalse)
        return 0;
    unsigned long long v = NUM2ULL(opt);
    if (v == 0 || v > UINT32_MAX)
        rb_raise(rb_eArgError,
                 "max_tokens must be between 1 and 4294967295, or false for unlimited");
    return (uint32_t)v;
}

static VALUE lookup_option(VALUE opts, ID id) {
    if (NIL_P(opts))
        return Qundef;
    return rb_hash_lookup2(opts, ID2SYM(id), Qundef);
}

static void reject_parallel_threads(VALUE opts) {
    VALUE v = lookup_option(opts, id_threads);
    if (v == Qundef || v == Qnil)
        return;
    long threads = NUM2LONG(v);
    if (threads != 1)
        rb_raise(
            rb_eArgError,
            "threads: is not supported by the runtime; run batches in application workers instead");
}

static size_t prefix_initial_target(uint32_t max_tokens) {
    if (max_tokens == 0)
        return 0;

    size_t target = (size_t)max_tokens;
    if (target > SE_PREFIX_MAX_BYTES / SE_PREFIX_BYTES_PER_TOKEN)
        return SE_PREFIX_MAX_BYTES;

    target *= SE_PREFIX_BYTES_PER_TOKEN;
    if (target < SE_PREFIX_MIN_BYTES)
        target = SE_PREFIX_MIN_BYTES;
    if (target > SE_PREFIX_MAX_BYTES)
        target = SE_PREFIX_MAX_BYTES;
    return target;
}

static size_t prefix_boundary_len(VALUE text, size_t target) {
    size_t full_len = (size_t)RSTRING_LEN(text);
    if (target >= full_len)
        return full_len;

    const uint8_t *ptr = (const uint8_t *)RSTRING_PTR(text);
    size_t floor = target > SE_PREFIX_BACKSCAN_BYTES ? target - SE_PREFIX_BACKSCAN_BYTES : 0;
    for (size_t i = target; i > floor; i--) {
        if (se_is_ascii_boundary(ptr[i - 1]))
            return i;
    }
    return 0;
}

static size_t grow_target(size_t target, size_t full_len) {
    if (target == 0 || target > full_len / 2)
        return full_len;
    return target * 2;
}

static size_t resolve_copy_len(VALUE text, size_t *target, uint32_t max_tokens) {
    size_t full_len = (size_t)RSTRING_LEN(text);
    if (max_tokens == 0 || *target == 0 || *target >= full_len) {
        *target = full_len;
        return full_len;
    }

    for (;;) {
        size_t copy_len = prefix_boundary_len(text, *target);
        if (copy_len)
            return copy_len;
        if (*target >= full_len) {
            *target = full_len;
            return full_len;
        }
        *target = grow_target(*target, full_len);
    }
}

typedef struct {
    VALUE texts;
    int is_array;
} text_source_t;

static VALUE text_at(const text_source_t *src, size_t i) {
    return src->is_array ? rb_ary_entry(src->texts, (long)i) : src->texts;
}

static void validate_texts(const text_source_t *src, size_t count) {
    for (size_t i = 0; i < count; i++) {
        VALUE s = text_at(src, i);
        Check_Type(s, T_STRING);
        check_text_encoding_at(s, src->is_array ? (long)i : -1);
    }
}

static int build_input_indexed(const text_source_t *src, const size_t *indices,
                               const size_t *copy_lens, size_t count, batch_input_t *input,
                               size_t *total_bytes) {
    memset(input, 0, sizeof(*input));

    size_t total = 0;
    for (size_t j = 0; j < count; j++) {
        if (!checked_add_size(total, copy_lens[j], &total))
            return 0;
    }

    size_t offsets_bytes;
    size_t lengths_bytes;
    size_t meta_bytes;
    size_t allocation_bytes;
    size_t slots = count ? count : 1;

    if (!checked_mul_size(slots, sizeof(size_t), &offsets_bytes) ||
        !checked_mul_size(slots, sizeof(size_t), &lengths_bytes) ||
        !checked_add_size(offsets_bytes, lengths_bytes, &meta_bytes) ||
        !checked_add_size(meta_bytes, total ? total : 1, &allocation_bytes)) {
        return 0;
    }

    uint8_t *allocation = (uint8_t *)malloc(allocation_bytes);
    if (!allocation)
        return 0;

    size_t *offsets = (size_t *)allocation;
    size_t *lengths = (size_t *)(allocation + offsets_bytes);
    uint8_t *buf = allocation + meta_bytes;

    size_t cursor = 0;
    for (size_t j = 0; j < count; j++) {
        VALUE s = text_at(src, indices[j]);
        size_t len = copy_lens[j];
        if (len)
            memcpy(buf + cursor, RSTRING_PTR(s), len);
        offsets[j] = cursor;
        lengths[j] = len;
        cursor += len;
    }

    input->base = buf;
    input->offsets = offsets;
    input->lengths = lengths;
    input->count = count;
    input->allocation = allocation;
    *total_bytes = total;
    return 1;
}

static void free_input(batch_input_t *input) {
    free(input->allocation);
    memset(input, 0, sizeof(*input));
}

typedef struct {
    const void *ptr;
    size_t bytes;
} binary_string_job_t;

static VALUE binary_string_create(VALUE arg) {
    binary_string_job_t *job = (binary_string_job_t *)(uintptr_t)arg;
    return rb_enc_str_new((const char *)job->ptr, (long)job->bytes, binary_encoding);
}

static VALUE binary_string_from_malloc(void *ptr, size_t bytes) {
    if (!size_fits_long(bytes)) {
        free(ptr);
        rb_raise(rb_eArgError, "embedding output is too large");
    }

    binary_string_job_t job;
    job.ptr = ptr;
    job.bytes = bytes;

    int state = 0;
    VALUE result = rb_protect(binary_string_create, (VALUE)(uintptr_t)&job, &state);
    free(ptr);
    if (state)
        rb_jump_tag(state);
    return result;
}

static VALUE binary_string_from_floats(float *ptr, size_t count, se_vector_format_t format) {
    size_t bytes;
    if (!checked_mul_size(count, vector_format_element_bytes(format), &bytes) ||
        !size_fits_long(bytes)) {
        free(ptr);
        rb_raise(rb_eArgError, "embedding output is too large");
    }

    if (format == SE_VECTOR_FORMAT_F32)
        return binary_string_from_malloc(ptr, bytes);

    uint8_t *encoded = (uint8_t *)malloc(bytes ? bytes : 1);
    if (!encoded) {
        free(ptr);
        rb_raise(rb_eNoMemError, "out of memory while encoding embedding output");
    }
    encode_f16_from_floats(encoded, ptr, count);
    free(ptr);
    return binary_string_from_malloc(encoded, bytes);
}

typedef struct {
    float *out;
    se_token_stats_t *stats;
    size_t *targets;
    size_t *pending;
    size_t *copy_lens;
} embed_run_t;

static void embed_run_free(embed_run_t *run) {
    free(run->out);
    free(run->stats);
    free(run->targets);
    free(run->pending);
    free(run->copy_lens);
    memset(run, 0, sizeof(*run));
}

static VALUE embed_texts_internal(VALUE self, const text_source_t *src, size_t count,
                                  VALUE max_tokens_opt, se_vector_format_t format,
                                  se_token_stats_t *stats_out) {
    model_wrapper_t *w = get_model(self);
    const se_model_t *model = &w->model;
    const uint32_t dim = model->meta.dim;
    const uint32_t max_tokens = resolve_max_tokens(model, max_tokens_opt);

    validate_texts(src, count);

    size_t floats;
    size_t out_bytes;
    if (!checked_mul_size(count, dim, &floats) ||
        !checked_mul_size(floats, vector_format_element_bytes(format), &out_bytes) ||
        !size_fits_long(out_bytes))
        rb_raise(rb_eArgError, "embedding output is too large");

    embed_run_t run;
    memset(&run, 0, sizeof(run));
    size_t slots = count ? count : 1;
    run.out = (float *)calloc(floats ? floats : 1, sizeof(float));
    run.stats = (se_token_stats_t *)calloc(slots, sizeof(se_token_stats_t));
    run.targets = (size_t *)calloc(slots, sizeof(size_t));
    run.pending = (size_t *)calloc(slots, sizeof(size_t));
    run.copy_lens = (size_t *)calloc(slots, sizeof(size_t));
    if (!run.out || !run.stats || !run.targets || !run.pending || !run.copy_lens) {
        embed_run_free(&run);
        rb_raise(rb_eNoMemError, "out of memory");
    }

    size_t initial = prefix_initial_target(max_tokens);
    size_t npending = count;
    for (size_t i = 0; i < count; i++) {
        run.targets[i] = initial;
        run.pending[i] = i;
    }

    while (npending) {
        for (size_t j = 0; j < npending; j++) {
            size_t i = run.pending[j];
            VALUE s = text_at(src, i);
            run.copy_lens[j] = resolve_copy_len(s, &run.targets[i], max_tokens);
        }

        batch_job_t job;
        memset(&job, 0, sizeof(job));
        job.model = model;
        job.max_tokens = max_tokens;
        job.out = run.out;
        job.stats = run.stats;
        job.out_index = run.pending;

        size_t total_bytes = 0;
        if (!build_input_indexed(src, run.pending, run.copy_lens, npending, &job.input,
                                 &total_bytes)) {
            embed_run_free(&run);
            rb_raise(rb_eNoMemError, "out of memory while staging the input batch");
        }

        int state = 0;
        rb_protect(protected_run_job, (VALUE)(uintptr_t)&job, &state);
        if (state) {
            job.cancelled = 1;
            free_input(&job.input);
            embed_run_free(&run);
            rb_jump_tag(state);
        }

        free_input(&job.input);

        if (job.failed) {
            size_t sub = job.failed_index < npending ? job.failed_index : 0;
            size_t original = run.pending[sub];

            if (job.error.status == SE_ERR_INVALID_UTF8 &&
                run.copy_lens[sub] < (size_t)RSTRING_LEN(text_at(src, original))) {
                run.targets[original] =
                    grow_target(run.targets[original], (size_t)RSTRING_LEN(text_at(src, original)));
                continue;
            }

            se_error_t err = job.error;
            embed_run_free(&run);
            if (src->is_array) {
                se_error_t wrapped;
                wrapped.status = err.status;
                snprintf(wrapped.message, sizeof(wrapped.message), "input[%zu]: %s", original,
                         err.message);
                raise_se(&wrapped);
            }
            raise_se(&err);
        }

        size_t next = 0;
        for (size_t j = 0; j < npending; j++) {
            size_t i = run.pending[j];
            size_t full_len = (size_t)RSTRING_LEN(text_at(src, i));
            if (max_tokens != 0 && run.copy_lens[j] < full_len && !run.stats[i].truncated) {
                run.targets[i] = grow_target(run.targets[i], full_len);
                run.pending[next++] = i;
            }
        }
        npending = next;
    }

    if (stats_out && count)
        *stats_out = run.stats[0];

    free(run.stats);
    run.stats = NULL;
    free(run.targets);
    run.targets = NULL;
    free(run.pending);
    run.pending = NULL;
    free(run.copy_lens);
    run.copy_lens = NULL;

    float *out = run.out;
    run.out = NULL;
    VALUE texts_guard = src->texts;
    RB_GC_GUARD(texts_guard);
    (void)out_bytes;
    return binary_string_from_floats(out, floats, format);
}

static VALUE embed_batch_internal(VALUE self, VALUE texts, VALUE max_tokens_opt,
                                  se_vector_format_t format) {
    Check_Type(texts, T_ARRAY);
    text_source_t src;
    src.texts = texts;
    src.is_array = 1;
    return embed_texts_internal(self, &src, (size_t)RARRAY_LEN(texts), max_tokens_opt, format,
                                NULL);
}

static VALUE model_embed_batch(int argc, VALUE *argv, VALUE self) {
    VALUE texts, opts;
    rb_scan_args(argc, argv, "1:", &texts, &opts);
    reject_parallel_threads(opts);

    VALUE max_tokens = lookup_option(opts, id_max_tokens);
    se_vector_format_t format = resolve_vector_format(lookup_option(opts, id_format));
    return embed_batch_internal(self, texts, max_tokens, format);
}

static VALUE embed_one_via_batch(VALUE self, VALUE text, VALUE max_tokens_opt,
                                 se_vector_format_t format, se_token_stats_t *stats) {
    text_source_t src;
    src.texts = text;
    src.is_array = 0;
    return embed_texts_internal(self, &src, 1, max_tokens_opt, format, stats);
}

static VALUE embed_one_value(VALUE self, VALUE text, VALUE max_tokens_opt,
                             se_vector_format_t format, se_token_stats_t *stats) {
    model_wrapper_t *w = get_model(self);
    Check_Type(text, T_STRING);
    check_text_encoding(text);

    if (format != SE_VECTOR_FORMAT_F32 || (size_t)RSTRING_LEN(text) >= SE_GVL_UNLOCK_THRESHOLD)
        return embed_one_via_batch(self, text, max_tokens_opt, format, stats);

    const uint32_t dim = w->model.meta.dim;
    size_t out_bytes;
    if (!checked_mul_size(dim, sizeof(float), &out_bytes) || !size_fits_long(out_bytes))
        rb_raise(rb_eArgError, "embedding output is too large");

    VALUE result = rb_str_new(NULL, (long)out_bytes);
    rb_enc_associate(result, binary_encoding);
    float *out = (float *)RSTRING_PTR(result);

    se_scratch_t scratch;
    se_scratch_init(&scratch);
    if (!se_scratch_reserve(&scratch, (size_t)RSTRING_LEN(text), dim)) {
        se_scratch_free(&scratch);
        rb_raise(rb_eNoMemError, "out of memory");
    }

    se_error_t err;
    se_error_clear(&err);
    se_token_stats_t local_stats;
    se_status_t rc =
        se_embed_one(&w->model, &scratch, (const uint8_t *)RSTRING_PTR(text),
                     (size_t)RSTRING_LEN(text), resolve_max_tokens(&w->model, max_tokens_opt), out,
                     stats ? stats : &local_stats, &err, NULL);
    se_scratch_free(&scratch);
    RB_GC_GUARD(text);

    if (rc != SE_OK)
        raise_se(&err);

    return result;
}

static VALUE model_embed(int argc, VALUE *argv, VALUE self) {
    VALUE text, opts;
    rb_scan_args(argc, argv, "1:", &text, &opts);
    reject_parallel_threads(opts);
    return embed_one_value(self, text, lookup_option(opts, id_max_tokens),
                           resolve_vector_format(lookup_option(opts, id_format)), NULL);
}

static VALUE model_embed_with_stats(int argc, VALUE *argv, VALUE self) {
    VALUE text, opts;
    rb_scan_args(argc, argv, "1:", &text, &opts);
    reject_parallel_threads(opts);

    se_token_stats_t stats;
    VALUE vector = embed_one_value(self, text, lookup_option(opts, id_max_tokens),
                                   resolve_vector_format(lookup_option(opts, id_format)), &stats);

    VALUE hash = rb_hash_new();
    rb_hash_aset(hash, ID2SYM(id_vector), vector);
    rb_hash_aset(hash, ID2SYM(id_token_count), UINT2NUM(stats.token_count));
    rb_hash_aset(hash, ID2SYM(id_unk_count), UINT2NUM(stats.unk_count));
    rb_hash_aset(hash, ID2SYM(id_truncated), stats.truncated ? Qtrue : Qfalse);
    return hash;
}

static VALUE model_tokenize(int argc, VALUE *argv, VALUE self) {
    VALUE text, opts;
    rb_scan_args(argc, argv, "1:", &text, &opts);

    model_wrapper_t *w = get_model(self);
    Check_Type(text, T_STRING);
    check_text_encoding(text);

    uint32_t max_tokens = resolve_max_tokens(&w->model, lookup_option(opts, id_max_tokens));

    se_scratch_t scratch;
    se_scratch_init(&scratch);
    if (!se_scratch_reserve(&scratch, (size_t)RSTRING_LEN(text), w->model.meta.dim)) {
        se_scratch_free(&scratch);
        rb_raise(rb_eNoMemError, "out of memory");
    }

    se_token_stats_t stats;
    se_error_t err;
    se_error_clear(&err);
    se_status_t rc = se_tokenize(&w->model, &scratch, (const uint8_t *)RSTRING_PTR(text),
                                 (size_t)RSTRING_LEN(text), max_tokens, &stats, &err, NULL);
    if (rc != SE_OK) {
        se_scratch_free(&scratch);
        raise_se(&err);
    }

    VALUE ids = rb_ary_new_capa((long)stats.token_count);
    for (uint32_t i = 0; i < stats.token_count; i++)
        rb_ary_push(ids, UINT2NUM(scratch.ids[i]));

    se_scratch_free(&scratch);
    RB_GC_GUARD(text);
    return ids;
}

typedef struct {
    const se_model_t *model;
    const uint32_t *ids;
    size_t n_ids;
    float *out;
    se_token_stats_t stats;
    volatile sig_atomic_t cancelled;
    se_error_t error;
} ids_job_t;

typedef struct {
    VALUE ids_value;
    const se_model_t *model;
    uint32_t max_tokens;
    uint32_t *ids;
    size_t n;
    float *out;
    size_t out_floats;
    se_vector_format_t format;
    se_token_stats_t *stats_out;
    se_token_stats_t stats;
    int truncated;
} ids_run_t;

typedef struct {
    VALUE array;
    long index;
    unsigned long long value;
} num2ull_job_t;

static VALUE num2ull_at_value(VALUE arg) {
    num2ull_job_t *job = (num2ull_job_t *)(uintptr_t)arg;
    job->value = NUM2ULL(rb_ary_entry(job->array, job->index));
    return Qnil;
}

static void *ids_execute(void *arg) {
    ids_job_t *job = (ids_job_t *)arg;
    se_scratch_t scratch;
    se_scratch_init(&scratch);
    if (!se_scratch_reserve(&scratch, job->n_ids, job->model->meta.dim)) {
        se_error_set(&job->error, SE_ERR_OOM, "out of memory while sizing scratch buffers");
        se_scratch_free(&scratch);
        return NULL;
    }
    se_error_clear(&job->error);
    se_embed_ids(job->model, &scratch, job->ids, job->n_ids, job->out, &job->stats, &job->error,
                 &job->cancelled);
    se_scratch_free(&scratch);
    return NULL;
}

static void ids_unblock_cancel(void *arg) {
    ids_job_t *job = (ids_job_t *)arg;
    if (job)
        job->cancelled = 1;
}

static VALUE embed_token_ids_body(VALUE arg) {
    ids_run_t *run = (ids_run_t *)(uintptr_t)arg;
    const uint32_t vocab_size = run->model->meta.vocab_size;

    for (size_t i = 0; i < run->n; i++) {
        num2ull_job_t conv;
        conv.array = run->ids_value;
        conv.index = (long)i;
        conv.value = 0;

        int state = 0;
        rb_protect(num2ull_at_value, (VALUE)(uintptr_t)&conv, &state);
        if (state)
            rb_jump_tag(state);

        if (conv.value >= vocab_size)
            rb_raise(rb_eArgError, "token id at index %zu is out of range", i);
        run->ids[i] = (uint32_t)conv.value;
    }

    run->out = (float *)calloc(run->model->meta.dim ? run->model->meta.dim : 1, sizeof(float));
    if (!run->out)
        rb_raise(rb_eNoMemError, "out of memory");

    ids_job_t job;
    memset(&job, 0, sizeof(job));
    job.model = run->model;
    job.ids = run->ids;
    job.n_ids = run->n;
    job.out = run->out;
    se_error_clear(&job.error);

    if (run->n >= SE_IDS_GVL_UNLOCK_THRESHOLD) {
        rb_thread_call_without_gvl(ids_execute, &job, ids_unblock_cancel, &job);
        rb_thread_check_ints();
        if (job.cancelled)
            rb_raise(rb_eInterrupt, "operation cancelled");
    } else {
        ids_execute(&job);
        rb_thread_check_ints();
    }

    if (job.error.status != SE_OK)
        raise_se(&job.error);

    run->stats = job.stats;
    if (run->truncated)
        run->stats.truncated = 1u;
    if (run->stats_out)
        *run->stats_out = run->stats;

    free(run->ids);
    run->ids = NULL;

    float *out = run->out;
    run->out = NULL;
    return binary_string_from_floats(out, run->out_floats, run->format);
}

static VALUE embed_token_ids_ensure(VALUE arg) {
    ids_run_t *run = (ids_run_t *)(uintptr_t)arg;
    free(run->ids);
    free(run->out);
    run->ids = NULL;
    run->out = NULL;
    return Qnil;
}

static VALUE embed_token_ids_value(VALUE self, VALUE ids_value, VALUE max_tokens_opt,
                                   se_vector_format_t format, se_token_stats_t *stats_out) {
    model_wrapper_t *w = get_model(self);
    Check_Type(ids_value, T_ARRAY);

    const uint32_t max_tokens = resolve_max_tokens(&w->model, max_tokens_opt);
    long n_long = RARRAY_LEN(ids_value);
    size_t n = (size_t)n_long;
    int truncated = 0;

    if (max_tokens != 0 && n > (size_t)max_tokens) {
        n = (size_t)max_tokens;
        truncated = 1;
    }

    size_t ids_bytes;
    if (!checked_mul_size(n ? n : 1, sizeof(uint32_t), &ids_bytes))
        rb_raise(rb_eArgError, "token id array is too large");

    size_t out_bytes;
    if (!checked_mul_size(w->model.meta.dim, vector_format_element_bytes(format), &out_bytes) ||
        !size_fits_long(out_bytes))
        rb_raise(rb_eArgError, "embedding output is too large");

    ids_run_t run;
    memset(&run, 0, sizeof(run));
    run.ids_value = ids_value;
    run.model = &w->model;
    run.max_tokens = max_tokens;
    run.n = n;
    run.out_floats = w->model.meta.dim;
    run.format = format;
    run.stats_out = stats_out;
    run.truncated = truncated;
    run.ids = (uint32_t *)malloc(ids_bytes);
    if (!run.ids)
        rb_raise(rb_eNoMemError, "out of memory");

    VALUE result = rb_ensure(embed_token_ids_body, (VALUE)(uintptr_t)&run, embed_token_ids_ensure,
                             (VALUE)(uintptr_t)&run);
    RB_GC_GUARD(ids_value);
    return result;
}

static VALUE model_embed_token_ids(int argc, VALUE *argv, VALUE self) {
    VALUE ids_value, opts;
    rb_scan_args(argc, argv, "1:", &ids_value, &opts);
    reject_parallel_threads(opts);
    return embed_token_ids_value(self, ids_value, lookup_option(opts, id_max_tokens),
                                 resolve_vector_format(lookup_option(opts, id_format)), NULL);
}

static VALUE model_embed_token_ids_with_stats(int argc, VALUE *argv, VALUE self) {
    VALUE ids_value, opts;
    rb_scan_args(argc, argv, "1:", &ids_value, &opts);
    reject_parallel_threads(opts);

    se_token_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    VALUE vector =
        embed_token_ids_value(self, ids_value, lookup_option(opts, id_max_tokens),
                              resolve_vector_format(lookup_option(opts, id_format)), &stats);

    VALUE hash = rb_hash_new();
    rb_hash_aset(hash, ID2SYM(id_vector), vector);
    rb_hash_aset(hash, ID2SYM(id_token_count), UINT2NUM(stats.token_count));
    rb_hash_aset(hash, ID2SYM(id_unk_count), UINT2NUM(stats.unk_count));
    rb_hash_aset(hash, ID2SYM(id_truncated), stats.truncated ? Qtrue : Qfalse);
    return hash;
}

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
} topk_job_t;

typedef struct {
    VALUE query;
    VALUE matrix;
    topk_job_t job;
    float *q_copy;
    float *matrix_copy;
    size_t matrix_bytes;
    int matrix_locked;
    int release_gvl;
    VALUE result;
} topk_run_t;

static int ptr_is_float_aligned(const void *ptr) {
    return ((uintptr_t)ptr % sizeof(float)) == 0;
}

static void topk_unblock_cancel(void *arg) {
    topk_job_t *job = (topk_job_t *)arg;
    if (job)
        job->cancelled = 1;
}

static float dot_product_unrolled(const float *q, const float *row, size_t dim) {
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

static float dot_and_row_sq_unrolled(const float *q, const float *row, size_t dim,
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

static float dot_product_f16(const float *q, const uint8_t *row, size_t dim) {
    float dot = 0.0f;
    for (size_t j = 0; j < dim; j++)
        dot += q[j] * f16_bits_to_float(read_u16le(row + j * 2));
    return dot;
}

static float dot_and_row_sq_f16(const float *q, const uint8_t *row, size_t dim, float *row_sq_out) {
    float dot = 0.0f;
    float row_sq = 0.0f;
    for (size_t j = 0; j < dim; j++) {
        float r = f16_bits_to_float(read_u16le(row + j * 2));
        dot += q[j] * r;
        row_sq += r * r;
    }
    *row_sq_out = row_sq;
    return dot;
}

static void *topk_execute(void *arg) {
    topk_job_t *job = (topk_job_t *)arg;
    for (size_t r = 0; r < job->rows; r++) {
        if ((r & 1023u) == 0 && job->cancelled)
            return NULL;
        float score;

        if (job->format == SE_VECTOR_FORMAT_F16) {
            const uint8_t *row = (const uint8_t *)job->m + r * job->dim * 2u;
            if (job->cosine) {
                float row_sq = 0.0f;
                score = dot_and_row_sq_f16(job->q, row, job->dim, &row_sq);
                score = row_sq > 0.0f ? score * job->inv_query_norm / sqrtf(row_sq) : 0.0f;
            } else {
                score = dot_product_f16(job->q, row, job->dim);
            }
        } else {
            const float *row = (const float *)job->m + r * job->dim;
            if (job->cosine) {
                float row_sq = 0.0f;
                score = dot_and_row_sq_unrolled(job->q, row, job->dim, &row_sq);
                score = row_sq > 0.0f ? score * job->inv_query_norm / sqrtf(row_sq) : 0.0f;
            } else {
                score = dot_product_unrolled(job->q, row, job->dim);
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

static VALUE topk_body(VALUE arg) {
    topk_run_t *run = (topk_run_t *)(uintptr_t)arg;
    const char *matrix_ptr = RSTRING_PTR(run->matrix);

    if (run->job.format == SE_VECTOR_FORMAT_F32 && !ptr_is_float_aligned(matrix_ptr)) {
        run->matrix_copy = (float *)malloc(run->matrix_bytes ? run->matrix_bytes : 1);
        if (!run->matrix_copy)
            rb_raise(rb_eNoMemError, "out of memory");
        memcpy(run->matrix_copy, matrix_ptr, run->matrix_bytes);
        run->job.m = run->matrix_copy;
    } else {
        run->job.m = matrix_ptr;
    }

    if (run->job.cosine) {
        float query_sq = dot_product_unrolled(run->job.q, run->job.q, run->job.dim);
        if (!(query_sq > 0.0f))
            rb_raise(rb_eArgError, "cosine_top_k needs a query with a non-zero norm");
        run->job.inv_query_norm = 1.0f / sqrtf(query_sq);
    }

    if (run->release_gvl && !run->matrix_copy) {
        rb_str_locktmp(run->matrix);
        run->matrix_locked = 1;
        run->job.m = RSTRING_PTR(run->matrix);
        rb_thread_call_without_gvl(topk_execute, &run->job, topk_unblock_cancel, &run->job);
        rb_thread_check_ints();
        rb_str_unlocktmp(run->matrix);
        run->matrix_locked = 0;
    } else if (run->release_gvl) {
        rb_thread_call_without_gvl(topk_execute, &run->job, topk_unblock_cancel, &run->job);
        rb_thread_check_ints();
    } else {
        topk_execute(&run->job);
        rb_thread_check_ints();
    }

    if (run->job.cancelled)
        rb_raise(rb_eInterrupt, "operation cancelled");

    VALUE result = rb_ary_new_capa(run->job.k);
    for (long i = 0; i < run->job.k; i++) {
        if (run->job.best_score[i] == -INFINITY)
            break;
        VALUE pair = rb_ary_new_capa(2);
        rb_ary_push(pair, SIZET2NUM(run->job.best_idx[i]));
        rb_ary_push(pair, DBL2NUM((double)run->job.best_score[i]));
        rb_ary_push(result, pair);
    }
    run->result = result;
    return result;
}

static VALUE topk_ensure(VALUE arg) {
    topk_run_t *run = (topk_run_t *)(uintptr_t)arg;
    if (run->matrix_locked) {
        rb_str_unlocktmp(run->matrix);
        run->matrix_locked = 0;
    }
    free(run->q_copy);
    free(run->matrix_copy);
    free(run->job.best_idx);
    free(run->job.best_score);
    run->q_copy = NULL;
    run->matrix_copy = NULL;
    run->job.best_idx = NULL;
    run->job.best_score = NULL;
    return Qnil;
}

static VALUE top_k_impl(int argc, VALUE *argv, VALUE self, int cosine) {
    VALUE query, matrix, k_val, opts;
    rb_scan_args(argc, argv, "3:", &query, &matrix, &k_val, &opts);
    (void)self;

    Check_Type(query, T_STRING);
    Check_Type(matrix, T_STRING);

    se_vector_format_t format = resolve_vector_format(lookup_option(opts, id_format));
    size_t element_bytes = vector_format_element_bytes(format);

    long qbytes = RSTRING_LEN(query);
    if (qbytes == 0 || qbytes % (long)element_bytes != 0)
        rb_raise(rb_eArgError, "query must be a non-empty embedding blob matching format");

    size_t dim = (size_t)qbytes / element_bytes;
    long mbytes = RSTRING_LEN(matrix);
    size_t row_bytes;
    if (!checked_mul_size(dim, element_bytes, &row_bytes) || !size_fits_long(row_bytes))
        rb_raise(rb_eArgError, "query dimension is too large");
    if (mbytes % (long)row_bytes != 0)
        rb_raise(rb_eArgError, "matrix length is not a multiple of the query dimension");

    size_t rows = (size_t)mbytes / row_bytes;
    long k = NUM2LONG(k_val);
    if (k < 1)
        rb_raise(rb_eArgError, "k must be >= 1");
    if (rows == 0)
        return rb_ary_new();
    if ((size_t)k > rows)
        k = (long)rows;

    size_t matrix_bytes = (size_t)mbytes;

    topk_run_t run;
    memset(&run, 0, sizeof(run));
    run.query = query;
    run.matrix = matrix;
    run.matrix_bytes = matrix_bytes;
    run.release_gvl = matrix_bytes >= SE_TOPK_GVL_UNLOCK_THRESHOLD;
    run.job.format = format;
    run.job.dim = dim;
    run.job.rows = rows;
    run.job.k = k;
    run.job.cosine = cosine;
    run.job.inv_query_norm = 1.0f;

    size_t q_float_bytes;
    if (!checked_mul_size(dim, sizeof(float), &q_float_bytes))
        rb_raise(rb_eArgError, "query dimension is too large");

    run.q_copy = (float *)malloc(q_float_bytes);
    run.job.best_idx = (size_t *)calloc((size_t)k, sizeof(size_t));
    run.job.best_score = (float *)malloc((size_t)k * sizeof(float));
    if (!run.q_copy || !run.job.best_idx || !run.job.best_score) {
        topk_ensure((VALUE)(uintptr_t)&run);
        rb_raise(rb_eNoMemError, "out of memory");
    }
    if (format == SE_VECTOR_FORMAT_F16)
        decode_f16_to_floats(run.q_copy, (const uint8_t *)RSTRING_PTR(query), dim);
    else
        memcpy(run.q_copy, RSTRING_PTR(query), row_bytes);
    run.job.q = run.q_copy;

    for (long i = 0; i < k; i++)
        run.job.best_score[i] = -INFINITY;

    VALUE result =
        rb_ensure(topk_body, (VALUE)(uintptr_t)&run, topk_ensure, (VALUE)(uintptr_t)&run);
    RB_GC_GUARD(query);
    RB_GC_GUARD(matrix);
    return result;
}

static VALUE se_cosine_top_k(int argc, VALUE *argv, VALUE self) {
    return top_k_impl(argc, argv, self, 1);
}

static VALUE se_dot_top_k(int argc, VALUE *argv, VALUE self) {
    return top_k_impl(argc, argv, self, 0);
}

RUBY_FUNC_EXPORTED void Init_static_embeddings(void) {
    binary_encoding = rb_ascii8bit_encoding();
    utf8_encoding = rb_utf8_encoding();
    id_join = rb_intern("join");
    id_kill = rb_intern("kill");
    id_max_tokens = rb_intern("max_tokens");
    id_threads = rb_intern("threads");
    id_format = rb_intern("format");
    id_blocking_p = rb_intern("blocking?");
    id_vector = rb_intern("vector");
    id_token_count = rb_intern("token_count");
    id_unk_count = rb_intern("unk_count");
    id_truncated = rb_intern("truncated");

    mStaticEmbeddings = rb_define_module("StaticEmbeddings");
    cFiber = rb_const_get(rb_cObject, rb_intern("Fiber"));

    eError = rb_define_class_under(mStaticEmbeddings, "Error", rb_eStandardError);
    eInvalidModel = rb_define_class_under(mStaticEmbeddings, "InvalidModelError", eError);
    eUnsupportedModel = rb_define_class_under(mStaticEmbeddings, "UnsupportedModelError", eError);
    eEncodingError = rb_define_class_under(mStaticEmbeddings, "EncodingError", eError);
    eEmptyInput = rb_define_class_under(mStaticEmbeddings, "EmptyInputError", eError);

    cModel = rb_define_class_under(mStaticEmbeddings, "Model", rb_cObject);
    rb_define_alloc_func(cModel, model_alloc);
    rb_define_method(cModel, "initialize", model_initialize, 1);
    rb_define_method(cModel, "close", model_close, 0);
    rb_define_method(cModel, "closed?", model_closed_p, 0);
    rb_define_method(cModel, "dim", model_dim, 0);
    rb_define_method(cModel, "vocab_size", model_vocab_size, 0);
    rb_define_method(cModel, "max_tokens", model_max_tokens, 0);
    rb_define_method(cModel, "normalized?", model_normalized_p, 0);
    rb_define_method(cModel, "lowercase?", model_lowercase_p, 0);
    rb_define_method(cModel, "unk_id", model_unk_id, 0);
    rb_define_method(cModel, "provenance_json", model_provenance_json, 0);
    rb_define_method(cModel, "mapped_bytes", model_mapped_bytes, 0);
    rb_define_method(cModel, "warmup!", model_warmup, 0);
    rb_define_method(cModel, "embed", model_embed, -1);
    rb_define_method(cModel, "embed_batch", model_embed_batch, -1);
    rb_define_method(cModel, "embed_with_stats", model_embed_with_stats, -1);
    rb_define_method(cModel, "tokenize", model_tokenize, -1);
    rb_define_method(cModel, "embed_token_ids", model_embed_token_ids, -1);
    rb_define_method(cModel, "embed_token_ids_with_stats", model_embed_token_ids_with_stats, -1);

    rb_define_singleton_method(mStaticEmbeddings, "cosine_top_k", se_cosine_top_k, -1);
    rb_define_singleton_method(mStaticEmbeddings, "dot_top_k", se_dot_top_k, -1);

    rb_define_const(mStaticEmbeddings, "FORMAT_VERSION", UINT2NUM(SE_FORMAT_VERSION));
    rb_define_const(mStaticEmbeddings, "TOKENIZER_BERT_WORDPIECE_V1",
                    UINT2NUM(SE_TOKENIZER_BERT_WORDPIECE_V1));
}
