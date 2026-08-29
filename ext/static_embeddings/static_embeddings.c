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
static ID id_dim;
static ID id_allow_unfrozen;
static ID id_validate_encoding;
static ID id_full;
static ID id_prefix;

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

static VALUE se_simd_backend(VALUE self) {
    (void)self;
    switch (se_current_f16_backend()) {
    case SE_F16_BACKEND_NEON_FP16:
        return rb_str_new_cstr("neon-fp16");
    case SE_F16_BACKEND_F16C:
        return rb_str_new_cstr("f16c");
    default:
        return rb_str_new_cstr("lut");
    }
}

typedef enum {
    SE_VALIDATE_ENCODING_FULL = 0,
    SE_VALIDATE_ENCODING_PREFIX = 1
} se_encoding_validation_t;

static void check_text_encoding_mode(VALUE str, long index, se_encoding_validation_t mode) {
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

    int cr = ENC_CODERANGE(str);
    if (cr == ENC_CODERANGE_UNKNOWN) {
        if (mode == SE_VALIDATE_ENCODING_PREFIX)
            return;
        cr = rb_enc_str_coderange(str);
    }

    if (cr != ENC_CODERANGE_VALID && cr != ENC_CODERANGE_7BIT) {
        if (index >= 0)
            rb_raise(eEncodingError, "input[%ld]: string is not valid %s", index, rb_enc_name(enc));
        rb_raise(eEncodingError, "string is not valid %s", rb_enc_name(enc));
    }
}

static se_encoding_validation_t resolve_encoding_validation(VALUE opt) {
    if (opt == Qundef || NIL_P(opt))
        return SE_VALIDATE_ENCODING_FULL;

    if (SYMBOL_P(opt)) {
        ID sym = SYM2ID(opt);
        if (sym == id_full)
            return SE_VALIDATE_ENCODING_FULL;
        if (sym == id_prefix)
            return SE_VALIDATE_ENCODING_PREFIX;
    }

    rb_raise(rb_eArgError, "validate_encoding: must be :full or :prefix");
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

        if (!se_scratch_reserve(scratch, dim)) {
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

static size_t prefix_boundary_len(const se_model_t *model, VALUE text, size_t target) {
    size_t full_len = (size_t)RSTRING_LEN(text);
    if (target >= full_len)
        return full_len;

    const uint8_t *ptr = (const uint8_t *)RSTRING_PTR(text);
    return se_prefix_boundary_len(model, ptr, full_len, target, SE_PREFIX_BACKSCAN_BYTES);
}

static size_t grow_target(size_t target, size_t full_len) {
    if (target == 0 || target > full_len / 2)
        return full_len;
    return target * 2;
}

static size_t resolve_copy_len(const se_model_t *model, VALUE text, size_t *target,
                               uint32_t max_tokens) {
    size_t full_len = (size_t)RSTRING_LEN(text);
    if (max_tokens == 0 || *target == 0 || *target >= full_len) {
        *target = full_len;
        return full_len;
    }

    for (;;) {
        size_t copy_len = prefix_boundary_len(model, text, *target);
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
    VALUE snapshot;
    int is_array;
} text_source_t;

static VALUE text_at(const text_source_t *src, size_t i) {
    return RARRAY_AREF(src->snapshot, (long)i);
}

static VALUE snapshot_texts(VALUE texts, int is_array, size_t count,
                            se_encoding_validation_t validation) {
    VALUE snapshot = rb_ary_new_capa((long)count);
    for (size_t i = 0; i < count; i++) {
        VALUE s = is_array ? rb_ary_entry(texts, (long)i) : texts;
        Check_Type(s, T_STRING);
        check_text_encoding_mode(s, is_array ? (long)i : -1, validation);
        rb_ary_push(snapshot, s);
    }
    return snapshot;
}

static int build_input_indexed(const text_source_t *src, const size_t *indices,
                               const size_t *copy_lens, size_t count, batch_input_t *input,
                               size_t *total_bytes) {
    memset(input, 0, sizeof(*input));

    size_t total = 0;
    for (size_t j = 0; j < count; j++) {
        if (!se_checked_add_size(total, copy_lens[j], &total))
            return 0;
    }

    size_t offsets_bytes = 0;
    size_t lengths_bytes = 0;
    size_t meta_bytes = 0;
    size_t allocation_bytes = 0;
    size_t payload_bytes = total;
    size_t malloc_bytes = 0;

    if (!se_array_bytes(count, sizeof(size_t), &offsets_bytes) ||
        !se_array_bytes(count, sizeof(size_t), &lengths_bytes) ||
        !se_checked_add_size(offsets_bytes, lengths_bytes, &meta_bytes) ||
        !se_checked_add_size(meta_bytes, payload_bytes, &allocation_bytes) ||
        !se_alloc_bytes(allocation_bytes, 1u, &malloc_bytes)) {
        return 0;
    }

    uint8_t *allocation = (uint8_t *)se_malloc(SE_ALLOC_BATCH_INPUT, malloc_bytes);
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
    se_free(input->allocation);
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
    if (!se_size_fits_long(bytes)) {
        se_free(ptr);
        rb_raise(rb_eArgError, "embedding output is too large");
    }

    binary_string_job_t job;
    job.ptr = ptr;
    job.bytes = bytes;

    int state = 0;
    VALUE result = rb_protect(binary_string_create, (VALUE)(uintptr_t)&job, &state);
    se_free(ptr);
    if (state)
        rb_jump_tag(state);
    return result;
}

static VALUE binary_string_from_floats(float *ptr, size_t count, se_vector_format_t format) {
    size_t bytes;
    if (!se_checked_mul_size(count, se_vector_format_element_bytes(format), &bytes) ||
        !se_size_fits_long(bytes)) {
        se_free(ptr);
        rb_raise(rb_eArgError, "embedding output is too large");
    }

    if (format == SE_VECTOR_FORMAT_F32)
        return binary_string_from_malloc(ptr, bytes);

    se_encode_f16_from_floats((uint8_t *)ptr, ptr, count);
    return binary_string_from_malloc(ptr, bytes);
}

typedef struct {
    float *out;
    se_token_stats_t *stats;
    size_t *targets;
    size_t *pending;
    size_t *copy_lens;
} embed_run_t;

typedef enum {
    EMBED_RUN_ALLOC_OK = 0,
    EMBED_RUN_ALLOC_OVERFLOW,
    EMBED_RUN_ALLOC_OOM
} embed_run_alloc_status_t;

static void embed_run_free(embed_run_t *run) {
    se_free(run->out);
    se_free(run->stats);
    se_free(run->targets);
    se_free(run->pending);
    se_free(run->copy_lens);
    memset(run, 0, sizeof(*run));
}

static embed_run_alloc_status_t embed_run_alloc(embed_run_t *run, size_t count, size_t floats) {
    size_t out_bytes = 0;
    size_t stats_bytes = 0;
    size_t targets_bytes = 0;

    memset(run, 0, sizeof(*run));
    if (!se_alloc_bytes(floats, sizeof(float), &out_bytes) ||
        !se_alloc_bytes(count, sizeof(se_token_stats_t), &stats_bytes) ||
        !se_alloc_bytes(count, sizeof(size_t), &targets_bytes))
        return EMBED_RUN_ALLOC_OVERFLOW;

    run->out = (float *)se_calloc(SE_ALLOC_BATCH_OUTPUT, 1, out_bytes);
    run->stats = (se_token_stats_t *)se_calloc(SE_ALLOC_BATCH_STATS, 1, stats_bytes);
    run->targets = (size_t *)se_calloc(SE_ALLOC_BATCH_INDEX, 1, targets_bytes);
    run->pending = (size_t *)se_calloc(SE_ALLOC_BATCH_INDEX, 1, targets_bytes);
    run->copy_lens = (size_t *)se_calloc(SE_ALLOC_BATCH_INDEX, 1, targets_bytes);
    if (run->out && run->stats && run->targets && run->pending && run->copy_lens)
        return EMBED_RUN_ALLOC_OK;

    embed_run_free(run);
    return EMBED_RUN_ALLOC_OOM;
}

static VALUE embed_texts_internal(VALUE self, VALUE texts, int is_array, size_t count,
                                  VALUE max_tokens_opt, se_vector_format_t format,
                                  se_encoding_validation_t validation,
                                  se_token_stats_t *stats_out) {
    model_wrapper_t *w = get_model(self);
    const se_model_t *model = &w->model;
    const uint32_t dim = model->meta.dim;
    const uint32_t max_tokens = resolve_max_tokens(model, max_tokens_opt);

    text_source_t source;
    source.snapshot = snapshot_texts(texts, is_array, count, validation);
    source.is_array = is_array;
    const text_source_t *src = &source;

    size_t floats;
    size_t out_bytes;
    if (!se_checked_mul_size(count, dim, &floats) ||
        !se_checked_mul_size(floats, se_vector_format_element_bytes(format), &out_bytes) ||
        !se_size_fits_long(out_bytes))
        rb_raise(rb_eArgError, "embedding output is too large");

    embed_run_t run;
    embed_run_alloc_status_t alloc_status = embed_run_alloc(&run, count, floats);
    if (alloc_status == EMBED_RUN_ALLOC_OVERFLOW)
        rb_raise(rb_eArgError, "embedding output is too large");
    if (alloc_status == EMBED_RUN_ALLOC_OOM)
        rb_raise(rb_eNoMemError, "out of memory");

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
            run.copy_lens[j] = resolve_copy_len(model, s, &run.targets[i], max_tokens);
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

    se_free(run.stats);
    run.stats = NULL;
    se_free(run.targets);
    run.targets = NULL;
    se_free(run.pending);
    run.pending = NULL;
    se_free(run.copy_lens);
    run.copy_lens = NULL;

    float *out = run.out;
    run.out = NULL;
    VALUE snapshot_guard = source.snapshot;
    RB_GC_GUARD(snapshot_guard);
    (void)out_bytes;
    return binary_string_from_floats(out, floats, format);
}

static VALUE embed_batch_internal(VALUE self, VALUE texts, VALUE max_tokens_opt,
                                  se_vector_format_t format, se_encoding_validation_t validation) {
    Check_Type(texts, T_ARRAY);
    return embed_texts_internal(self, texts, 1, (size_t)RARRAY_LEN(texts), max_tokens_opt, format,
                                validation, NULL);
}

static VALUE model_embed_batch(int argc, VALUE *argv, VALUE self) {
    VALUE texts, opts;
    rb_scan_args(argc, argv, "1:", &texts, &opts);
    reject_parallel_threads(opts);

    VALUE max_tokens = lookup_option(opts, id_max_tokens);
    se_vector_format_t format = resolve_vector_format(lookup_option(opts, id_format));
    se_encoding_validation_t validation =
        resolve_encoding_validation(lookup_option(opts, id_validate_encoding));
    return embed_batch_internal(self, texts, max_tokens, format, validation);
}

static VALUE embed_one_via_batch(VALUE self, VALUE text, VALUE max_tokens_opt,
                                 se_vector_format_t format, se_encoding_validation_t validation,
                                 se_token_stats_t *stats) {
    return embed_texts_internal(self, text, 0, 1, max_tokens_opt, format, validation, stats);
}

static VALUE embed_one_value(VALUE self, VALUE text, VALUE max_tokens_opt,
                             se_vector_format_t format, se_encoding_validation_t validation,
                             se_token_stats_t *stats) {
    model_wrapper_t *w = get_model(self);
    Check_Type(text, T_STRING);
    check_text_encoding_mode(text, -1, validation);

    if (format != SE_VECTOR_FORMAT_F32 || (size_t)RSTRING_LEN(text) >= SE_GVL_UNLOCK_THRESHOLD)
        return embed_one_via_batch(self, text, max_tokens_opt, format, validation, stats);

    const uint32_t dim = w->model.meta.dim;
    size_t out_bytes;
    if (!se_checked_mul_size(dim, sizeof(float), &out_bytes) || !se_size_fits_long(out_bytes))
        rb_raise(rb_eArgError, "embedding output is too large");

    VALUE result = rb_str_new(NULL, (long)out_bytes);
    rb_enc_associate(result, binary_encoding);
    float *out = (float *)RSTRING_PTR(result);

    se_scratch_t scratch;
    se_scratch_init(&scratch);
    if (!se_scratch_reserve(&scratch, dim)) {
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
                           resolve_vector_format(lookup_option(opts, id_format)),
                           resolve_encoding_validation(lookup_option(opts, id_validate_encoding)),
                           NULL);
}

static VALUE model_embed_with_stats(int argc, VALUE *argv, VALUE self) {
    VALUE text, opts;
    rb_scan_args(argc, argv, "1:", &text, &opts);
    reject_parallel_threads(opts);

    se_token_stats_t stats;
    VALUE vector = embed_one_value(
        self, text, lookup_option(opts, id_max_tokens),
        resolve_vector_format(lookup_option(opts, id_format)),
        resolve_encoding_validation(lookup_option(opts, id_validate_encoding)), &stats);

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
    check_text_encoding_mode(
        text, -1, resolve_encoding_validation(lookup_option(opts, id_validate_encoding)));

    uint32_t max_tokens = resolve_max_tokens(&w->model, lookup_option(opts, id_max_tokens));

    se_scratch_t scratch;
    se_scratch_init(&scratch);
    if (!se_scratch_reserve(&scratch, w->model.meta.dim)) {
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
    if (!se_scratch_reserve(&scratch, job->model->meta.dim)) {
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

    size_t out_bytes;
    if (!se_alloc_bytes(run->model->meta.dim, sizeof(float), &out_bytes))
        rb_raise(rb_eArgError, "embedding output is too large");

    run->out = (float *)se_calloc(SE_ALLOC_BATCH_OUTPUT, 1, out_bytes);
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

    se_free(run->ids);
    run->ids = NULL;

    float *out = run->out;
    run->out = NULL;
    return binary_string_from_floats(out, run->out_floats, run->format);
}

static VALUE embed_token_ids_ensure(VALUE arg) {
    ids_run_t *run = (ids_run_t *)(uintptr_t)arg;
    se_free(run->ids);
    se_free(run->out);
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
    if (!se_alloc_bytes(n, sizeof(uint32_t), &ids_bytes))
        rb_raise(rb_eArgError, "token id array is too large");

    size_t out_bytes;
    if (!se_checked_mul_size(w->model.meta.dim, se_vector_format_element_bytes(format),
                             &out_bytes) ||
        !se_size_fits_long(out_bytes))
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
    run.ids = (uint32_t *)se_malloc(SE_ALLOC_TOKEN_IDS, ids_bytes);
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
    VALUE matrix;
    se_topk_job_t job;
    float *q_copy;
    float *matrix_copy;
    size_t matrix_bytes;
    int release_gvl;
} topk_run_t;

static int ptr_is_float_aligned(const void *ptr) {
    return ((uintptr_t)ptr % sizeof(float)) == 0;
}

static void topk_unblock_cancel(void *arg) {
    se_topk_job_t *job = (se_topk_job_t *)arg;
    if (job)
        job->cancelled = 1;
}

static VALUE topk_body(VALUE arg) {
    topk_run_t *run = (topk_run_t *)(uintptr_t)arg;
    const char *matrix_ptr = RSTRING_PTR(run->matrix);

    if (run->matrix_copy) {
        memcpy(run->matrix_copy, matrix_ptr, run->matrix_bytes);
        run->job.m = run->matrix_copy;
    } else {
        run->job.m = matrix_ptr;
    }

    if (run->job.cosine) {
        float query_sq = se_dot_product_f32(run->job.q, run->job.q, run->job.dim);
        if (!(query_sq > 0.0f))
            rb_raise(rb_eArgError, "cosine_top_k needs a query with a non-zero norm");
        run->job.inv_query_norm = 1.0f / sqrtf(query_sq);
    }

    if (run->release_gvl) {
        rb_thread_call_without_gvl(se_topk_execute, &run->job, topk_unblock_cancel, &run->job);
        rb_thread_check_ints();
    } else {
        se_topk_execute(&run->job);
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
    return result;
}

static VALUE topk_ensure(VALUE arg) {
    topk_run_t *run = (topk_run_t *)(uintptr_t)arg;
    se_free(run->q_copy);
    se_free(run->matrix_copy);
    se_free(run->job.best_idx);
    se_free(run->job.best_score);
    run->q_copy = NULL;
    run->matrix_copy = NULL;
    run->job.best_idx = NULL;
    run->job.best_score = NULL;
    return Qnil;
}

static size_t topk_required_dim(VALUE opts) {
    VALUE v = lookup_option(opts, id_dim);
    if (v == Qundef || v == Qnil)
        rb_raise(rb_eArgError, "dim: is required (raw blobs carry no dimension or format tag); "
                               "pass dim: model.dim, or call model.cosine_top_k / model.dot_top_k");

    long dim = NUM2LONG(v);
    if (dim < 1)
        rb_raise(rb_eArgError, "dim: must be >= 1");
    return (size_t)dim;
}

static void topk_check_matrix(VALUE matrix, size_t matrix_bytes, se_vector_format_t format,
                              VALUE opts, int *release_gvl, int *needs_copy) {
    int large = matrix_bytes >= SE_TOPK_GVL_UNLOCK_THRESHOLD;
    int aligned = format != SE_VECTOR_FORMAT_F32 || ptr_is_float_aligned(RSTRING_PTR(matrix));
    VALUE allow_unfrozen = lookup_option(opts, id_allow_unfrozen);

    *release_gvl = 0;
    *needs_copy = 0;

    if (!large) {
        *needs_copy = !aligned;
        return;
    }

    if (!aligned)
        rb_raise(rb_eArgError,
                 "matrix blob is not 4-byte aligned; copying %zu bytes would be silently "
                 "expensive - pass a freshly packed String instead of a byteslice",
                 matrix_bytes);

    if (RB_OBJ_FROZEN(matrix)) {
        *release_gvl = 1;
        return;
    }

    if (allow_unfrozen != Qundef && RTEST(allow_unfrozen))
        return;

    rb_raise(rb_eArgError,
             "a matrix of %zu bytes is scanned with the GVL released and must be frozen so that "
             "no other thread can mutate it; call matrix.freeze, or pass allow_unfrozen: true to "
             "scan it while holding the GVL",
             matrix_bytes);
}

static VALUE top_k_impl(int argc, VALUE *argv, VALUE self, int cosine) {
    VALUE query, matrix, k_val, opts;
    rb_scan_args(argc, argv, "3:", &query, &matrix, &k_val, &opts);
    (void)self;

    Check_Type(query, T_STRING);
    Check_Type(matrix, T_STRING);

    se_vector_format_t format = resolve_vector_format(lookup_option(opts, id_format));
    size_t element_bytes = se_vector_format_element_bytes(format);
    size_t dim = topk_required_dim(opts);

    size_t row_bytes;
    if (!se_checked_mul_size(dim, element_bytes, &row_bytes) || !se_size_fits_long(row_bytes))
        rb_raise(rb_eArgError, "dim: is too large");

    if ((size_t)RSTRING_LEN(query) != row_bytes)
        rb_raise(rb_eArgError, "query is %ld bytes but dim: %zu with format %s needs %zu bytes",
                 RSTRING_LEN(query), dim, format == SE_VECTOR_FORMAT_F16 ? ":f16" : ":f32",
                 row_bytes);

    long mbytes = RSTRING_LEN(matrix);
    if ((size_t)mbytes % row_bytes != 0)
        rb_raise(rb_eArgError, "matrix is %ld bytes, which is not a whole number of %zu-byte rows",
                 mbytes, row_bytes);

    size_t rows = (size_t)mbytes / row_bytes;
    long k = NUM2LONG(k_val);
    if (k < 1)
        rb_raise(rb_eArgError, "k must be >= 1");
    if (rows == 0)
        return rb_ary_new();
    if ((size_t)k > rows)
        k = (long)rows;

    size_t matrix_bytes = (size_t)mbytes;
    int release_gvl = 0;
    int needs_copy = 0;
    topk_check_matrix(matrix, matrix_bytes, format, opts, &release_gvl, &needs_copy);

    topk_run_t run;
    memset(&run, 0, sizeof(run));
    run.matrix = matrix;
    run.matrix_bytes = matrix_bytes;
    run.release_gvl = release_gvl;
    run.job.format = format;
    run.job.dim = dim;
    run.job.rows = rows;
    run.job.k = k;
    run.job.cosine = cosine;
    run.job.inv_query_norm = 1.0f;

    size_t q_float_bytes;
    if (!se_checked_mul_size(dim, sizeof(float), &q_float_bytes))
        rb_raise(rb_eArgError, "dim: is too large");

    run.q_copy = (float *)se_malloc(SE_ALLOC_TOPK_QUERY, q_float_bytes);
    size_t best_idx_bytes = 0;
    size_t best_score_bytes = 0;
    if (!se_array_bytes((size_t)k, sizeof(size_t), &best_idx_bytes) ||
        !se_array_bytes((size_t)k, sizeof(float), &best_score_bytes)) {
        topk_ensure((VALUE)(uintptr_t)&run);
        rb_raise(rb_eArgError, "k is too large");
    }

    run.job.best_idx = (size_t *)se_calloc(SE_ALLOC_TOPK_BEST, 1, best_idx_bytes);
    run.job.best_score = (float *)se_malloc(SE_ALLOC_TOPK_BEST, best_score_bytes);
    if (needs_copy)
        run.matrix_copy = (float *)se_malloc(SE_ALLOC_TOPK_MATRIX_COPY, matrix_bytes);
    if (!run.q_copy || !run.job.best_idx || !run.job.best_score ||
        (needs_copy && !run.matrix_copy)) {
        topk_ensure((VALUE)(uintptr_t)&run);
        rb_raise(rb_eNoMemError, "out of memory");
    }

    if (format == SE_VECTOR_FORMAT_F16)
        se_decode_f16_to_floats(run.q_copy, (const uint8_t *)RSTRING_PTR(query), dim);
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

static VALUE se_encode_f16(VALUE self, VALUE ary) {
    (void)self;
    Check_Type(ary, T_ARRAY);

    long n = RARRAY_LEN(ary);
    size_t bytes;
    if (!se_checked_mul_size((size_t)n, 2, &bytes) || !se_size_fits_long(bytes))
        rb_raise(rb_eArgError, "vector is too large");

    VALUE out = rb_str_new(NULL, (long)bytes);
    rb_enc_associate(out, binary_encoding);
    for (long i = 0; i < n; i++) {
        double v = NUM2DBL(rb_ary_entry(ary, i));
        se_write_f16le((uint8_t *)RSTRING_PTR(out) + (size_t)i * 2, (float)v);
    }
    return out;
}

static VALUE se_decode_f16(VALUE self, VALUE blob) {
    (void)self;
    Check_Type(blob, T_STRING);

    long n = RSTRING_LEN(blob);
    if (n % 2 != 0)
        rb_raise(rb_eArgError, "f16 blob byte size must be a multiple of 2");

    VALUE out = rb_ary_new_capa(n / 2);
    for (long i = 0; i < n / 2; i++) {
        const uint8_t *src = (const uint8_t *)RSTRING_PTR(blob) + (size_t)i * 2;
        rb_ary_push(out, DBL2NUM((double)se_read_f16le(src)));
    }
    RB_GC_GUARD(blob);
    return out;
}

#if SE_ENABLE_ALLOC_STATS
static VALUE se_alloc_stats_hash(VALUE self) {
    (void)self;
    se_alloc_stats_t stats[SE_ALLOC_CATEGORY_COUNT];
    se_alloc_stats_snapshot(stats);

    VALUE out = rb_hash_new();
    ID id_current_bytes = rb_intern("current_bytes");
    ID id_peak_bytes = rb_intern("peak_bytes");
    ID id_total_allocated_bytes = rb_intern("total_allocated_bytes");
    ID id_total_freed_bytes = rb_intern("total_freed_bytes");
    ID id_alloc_count = rb_intern("alloc_count");
    ID id_realloc_count = rb_intern("realloc_count");
    ID id_free_count = rb_intern("free_count");

    for (int i = 0; i < SE_ALLOC_CATEGORY_COUNT; i++) {
        VALUE item = rb_hash_new();
        rb_hash_aset(item, ID2SYM(id_current_bytes), SIZET2NUM(stats[i].current_bytes));
        rb_hash_aset(item, ID2SYM(id_peak_bytes), SIZET2NUM(stats[i].peak_bytes));
        rb_hash_aset(item, ID2SYM(id_total_allocated_bytes),
                     SIZET2NUM(stats[i].total_allocated_bytes));
        rb_hash_aset(item, ID2SYM(id_total_freed_bytes), SIZET2NUM(stats[i].total_freed_bytes));
        rb_hash_aset(item, ID2SYM(id_alloc_count), SIZET2NUM(stats[i].alloc_count));
        rb_hash_aset(item, ID2SYM(id_realloc_count), SIZET2NUM(stats[i].realloc_count));
        rb_hash_aset(item, ID2SYM(id_free_count), SIZET2NUM(stats[i].free_count));
        rb_hash_aset(out, ID2SYM(rb_intern(se_alloc_category_name((se_alloc_category_t)i))), item);
    }

    return out;
}

static VALUE se_alloc_stats_reset_bang(VALUE self) {
    (void)self;
    se_alloc_stats_reset();
    return Qnil;
}
#endif

RUBY_FUNC_EXPORTED void Init_static_embeddings(void) {
    se_select_f16_backend();
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
    id_dim = rb_intern("dim");
    id_allow_unfrozen = rb_intern("allow_unfrozen");

    id_validate_encoding = rb_intern("validate_encoding");
    id_full = rb_intern("full");
    id_prefix = rb_intern("prefix");
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
    rb_define_singleton_method(mStaticEmbeddings, "encode_f16", se_encode_f16, 1);
    rb_define_singleton_method(mStaticEmbeddings, "decode_f16", se_decode_f16, 1);
    rb_define_singleton_method(mStaticEmbeddings, "simd_backend", se_simd_backend, 0);
#if SE_ENABLE_ALLOC_STATS
    rb_define_singleton_method(mStaticEmbeddings, "__alloc_stats__", se_alloc_stats_hash, 0);
    rb_define_singleton_method(mStaticEmbeddings, "__alloc_stats_reset__",
                               se_alloc_stats_reset_bang, 0);
#endif

    rb_define_const(mStaticEmbeddings, "FORMAT_VERSION", UINT2NUM(SE_FORMAT_VERSION));
    rb_define_const(mStaticEmbeddings, "TOKENIZER_BERT_WORDPIECE_V1",
                    UINT2NUM(SE_TOKENIZER_BERT_WORDPIECE_V1));
}
