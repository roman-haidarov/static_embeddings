# frozen_string_literal: true
require_relative "_support"

SEBench.header("gvl_threshold")

THRESHOLD_BYTES = [512, 1_024, 2_040, 2_048, 2_056, 4_096, 8_192, 16_384, 65_536].freeze
PROGRESS_BYTES = [1_024, 2_048, 8_192, 65_536].freeze
PROGRESS_ITERATIONS = SEBench.env_integer("BENCH_GVL_PROGRESS_ITERATIONS", 200)
PROGRESS_THREAD_YIELD_EVERY = 4_096

def text_with_bytes(target_bytes)
  rng = Random.new(target_bytes)
  text = +""
  text << SEBench.words[rng.rand(SEBench.words.length)] << " " while text.bytesize < target_bytes
  text = text.byteslice(0, target_bytes).freeze
  text.valid_encoding?
  text
end

THRESHOLD_BYTES.each do |target_bytes|
  text = text_with_bytes(target_bytes)
  stats = SEBench.measure(initial_iterations: 200) { |n| n.times { SEBench.model.embed(text) } }
  SEBench.report("embed", stats, bytes: text.bytesize, coderange: "cached")
end

puts
puts "second-thread progress during repeated embed calls (higher is better for the caller's neighbours)"
PROGRESS_BYTES.each do |target_bytes|
  text = text_with_bytes(target_bytes)
  counter = 0
  running = true
  neighbour = Thread.new do
    while running
      counter += 1
      Thread.pass if (counter % PROGRESS_THREAD_YIELD_EVERY).zero?
    end
  end

  elapsed =
    begin
      SEBench.realtime { PROGRESS_ITERATIONS.times { SEBench.model.embed(text) } }
    ensure
      running = false
      neighbour.join
    end

  printf("%-34s %12d ticks %9.3f s\n", "bytes=#{target_bytes}", counter, elapsed)
end
