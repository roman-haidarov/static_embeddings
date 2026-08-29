require "mkmf"

def drop_unsupported_ruby_warnflags
  return if RbConfig::CONFIG["CC"].to_s.include?("clang")

  flags = %w[
    -Wno-self-assign
    -Wno-parentheses-equality
    -Wno-constant-logical-operand
  ]

  [RbConfig::CONFIG, RbConfig::MAKEFILE_CONFIG].each do |config|
    warnflags = config["warnflags"].to_s.split
    config["warnflags"] = (warnflags - flags).join(" ")
  end
end

drop_unsupported_ruby_warnflags

MINIMUM_RUBY_VERSION = Gem::Version.new("3.1.0")
if Gem::Version.new(RUBY_VERSION) < MINIMUM_RUBY_VERSION
  abort "static_embeddings requires Ruby >= #{MINIMUM_RUBY_VERSION}; current Ruby is #{RUBY_VERSION}"
end

$srcs = %w[
  static_embeddings.c
  se_alloc_stats.c
  se_f16.c
  se_topk.c
  se_format.c
  se_unicode.c
  se_tokenizer.c
  se_embed.c
]

have_header("ruby/fiber/scheduler.h")
have_header("sys/mman.h")
have_func("madvise", "sys/mman.h")

unless [RbConfig::CONFIG["LIBS"], RbConfig::CONFIG["LIBRUBYARG_SHARED"], RbConfig::CONFIG["DLDFLAGS"]].compact.any? { |v| v.include?("-lm") }
  have_library("m")
end

$CFLAGS += " -O3 -std=gnu99 -fvisibility=hidden"
$CFLAGS += " -Wall -Wextra -Wno-unused-parameter"
$defs << "-DSE_ENABLE_ALLOC_STATS=1" if ENV["STATIC_EMBEDDINGS_ALLOC_STATS"] == "1"

create_makefile("static_embeddings/static_embeddings")
