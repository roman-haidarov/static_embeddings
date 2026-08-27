#!/usr/bin/env ruby

$LOAD_PATH.unshift(File.expand_path("../lib", __dir__))
require "static_embeddings"
require_relative "make_fixture_model"
require "fileutils"
require "tmpdir"

target = File.expand_path("../lib/models/demo.semb", __dir__)
FileUtils.mkdir_p(File.dirname(target))

Dir.mktmpdir do |dir|
  FixtureModel.write(dir)
  report = StaticEmbeddings.convert(dir, output_path: target, model_id: "demo/tiny-wordpiece")
  puts "wrote #{target} (#{report[:bytes]} bytes, vocab #{report[:vocab_size]}, dim #{report[:dim]})"
end
