#!/usr/bin/env ruby

require "json"
require "fileutils"

$LOAD_PATH.unshift(File.expand_path("../lib", __dir__))
require "static_embeddings/version"
require "static_embeddings/safetensors"

module FixtureModel
  DIM = 8
  SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"].freeze

  WORDS = %w[
    hello world ruby gem static embedding vector search text token
    the a of and for local fast model cafe naive
    привет мир поиск текст вектор модель локальный быстрый
  ].freeze

  PUNCTUATION = %w[. , ! ? - : ; ' " ( ) /].freeze
  SUBWORDS = %w[##ing ##ed ##s ##er ##ик ##ов ##ый].freeze

  module_function

  def vocab_tokens
    [SPECIALS, ascii_letters, digits, cyrillic_letters, ascii_subwords,
     cyrillic_subwords, PUNCTUATION, WORDS, SUBWORDS].each_with_object([]) do |group, tokens|
      group.each { |token| tokens << token }
    end.uniq
  end

  def matrix_floats(vocab_size)
    state = 0x12345678
    Array.new(vocab_size * DIM) do
      state = (state * 1_103_515_245 + 12_345) & 0x7FFFFFFF
      ((state % 20_000) - 10_000) / 10_000.0
    end
  end

  def write(dir)
    FileUtils.mkdir_p(dir)
    tokens = vocab_tokens
    write_tokenizer(dir, tokens)
    write_config(dir)
    write_tokenizer_config(dir)
    write_tensor(dir, tokens.length)
    { dir: dir, vocab_size: tokens.length, dim: DIM }
  end

  def ascii_letters
    ("a".."z").to_a
  end

  def digits
    ("0".."9").to_a
  end

  def cyrillic_letters
    ("а".."я").to_a
  end

  def ascii_subwords
    ascii_letters.map { |char| "###{char}" }
  end

  def cyrillic_subwords
    cyrillic_letters.map { |char| "###{char}" }
  end

  def write_tokenizer(dir, tokens)
    File.write(File.join(dir, "tokenizer.json"), JSON.pretty_generate(tokenizer_payload(tokens)))
  end

  def tokenizer_payload(tokens)
    {
      "version" => "1.0",
      "truncation" => nil,
      "padding" => nil,
      "added_tokens" => added_tokens,
      "normalizer" => normalizer,
      "pre_tokenizer" => { "type" => "BertPreTokenizer" },
      "post_processor" => nil,
      "decoder" => nil,
      "model" => wordpiece_model(tokens)
    }
  end

  def added_tokens
    SPECIALS.each_with_index.map do |content, index|
      {
        "id" => index,
        "content" => content,
        "single_word" => false,
        "lstrip" => false,
        "rstrip" => false,
        "normalized" => false,
        "special" => true
      }
    end
  end

  def normalizer
    {
      "type" => "BertNormalizer",
      "clean_text" => true,
      "handle_chinese_chars" => true,
      "strip_accents" => nil,
      "lowercase" => true
    }
  end

  def wordpiece_model(tokens)
    {
      "type" => "WordPiece",
      "unk_token" => "[UNK]",
      "continuing_subword_prefix" => "##",
      "max_input_chars_per_word" => 100,
      "vocab" => tokens.each_with_index.to_h
    }
  end

  def write_config(dir)
    File.write(File.join(dir, "config.json"), JSON.pretty_generate(config_payload))
  end

  def config_payload
    {
      "model_type" => "model2vec",
      "hidden_dim" => DIM,
      "seq_length" => 1_000_000,
      "normalize" => true,
      "tokenizer_name" => "fixture/tiny-wordpiece"
    }
  end

  def write_tokenizer_config(dir)
    File.write(File.join(dir, "tokenizer_config.json"), JSON.pretty_generate(tokenizer_config_payload))
  end

  def tokenizer_config_payload
    {
      "tokenizer_class" => "BertTokenizer",
      "do_lower_case" => true,
      "strip_accents" => nil,
      "tokenize_chinese_chars" => true,
      "model_max_length" => 1_000_000,
      "unk_token" => "[UNK]"
    }
  end

  def write_tensor(dir, vocab_size)
    StaticEmbeddings::Safetensors.write(
      File.join(dir, "model.safetensors"),
      "embeddings",
      [vocab_size, DIM],
      matrix_floats(vocab_size)
    )
  end
end

if $PROGRAM_NAME == __FILE__
  target = ARGV[0] || File.expand_path("../test/fixtures/tiny-wordpiece", __dir__)
  info = FixtureModel.write(target)
  puts "wrote #{info[:vocab_size]} tokens, dim #{info[:dim]} -> #{info[:dir]}"
end
