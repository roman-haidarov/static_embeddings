require "minitest/autorun"
require "fileutils"
require "tmpdir"

$LOAD_PATH.unshift(File.expand_path("../lib", __dir__))
require "static_embeddings"
require "static_embeddings/reference"

require_relative "../tools/make_fixture_model"

module TestSupport
  ROOT = File.expand_path("..", __dir__)
  SOURCE_DIR = File.join(ROOT, "test", "fixtures", "tiny-wordpiece")
  MODEL_PATH = File.join(ROOT, "tmp", "test-tiny.semb")

  module_function

  def source_dir
    FixtureModel.write(SOURCE_DIR) unless File.file?(File.join(SOURCE_DIR, "tokenizer.json"))
    SOURCE_DIR
  end

  def model_path
    return MODEL_PATH if current_model_file?(MODEL_PATH)

    FileUtils.mkdir_p(File.dirname(MODEL_PATH))
    StaticEmbeddings.convert(source_dir, output_path: MODEL_PATH, model_id: "fixture/tiny-wordpiece")
    MODEL_PATH
  end

  def current_model_file?(path)
    return false unless File.file?(path)
    return false if File.size(path) < StaticEmbeddings::Format::HEADER_SIZE

    data = File.binread(path, 16)
    data.byteslice(0, 8) == StaticEmbeddings::Format::MAGIC.b &&
      data.byteslice(8, 4).unpack1("V") == StaticEmbeddings::Format::VERSION &&
      data.byteslice(12, 4).unpack1("V") == StaticEmbeddings::Format::HEADER_SIZE
  end

  def model
    @model ||= StaticEmbeddings.load(model_path)
  end

  def reference
    @reference ||= StaticEmbeddings::Reference.from_source_dir(source_dir)
  end

  def timing_sensitive_tests?
    ENV["STATIC_EMBEDDINGS_TIMING_TESTS"] == "1"
  end

  def thread_ticks_during
    ticks = 0
    running = true
    ticker = Thread.new do
      while running
        ticks += 1
        Thread.pass
      end
    end

    Thread.pass
    before = ticks
    yield
    ticks - before
  ensure
    running = false
    ticker&.join
  end

  TRICKY_TEXTS = [
    "",
    " ",
    "\t\n\r",
    "hello",
    "Hello WORLD",
    "hello, world!",
    "café",
    "CAFÉ",
    "cafe\u0301",
    "naïve",
    "привет мир",
    "ПРИВЕТ, МИР!",
    "приве\u0301т",
    "локальный поиск по тексту",
    "中文测试",
    "🧬 emoji 🧬",
    "zero\u200bwidth",
    "non\u00a0breaking\u2009space",
    "!!!???...,,,",
    "mixed Привет hello 123",
    "testing ##ing edge",
    "zzzzz unknownword qqqqq",
    "a" * 300,
    "слово " * 400
  ].freeze
end
