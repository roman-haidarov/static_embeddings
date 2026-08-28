require "test_helper"

class F16CodecTest < Minitest::Test
  def bits(value)
    StaticEmbeddings.encode_f16([value]).unpack1("v")
  end

  def value(bits)
    StaticEmbeddings.decode_f16([bits].pack("v")).first
  end

  def test_exact_values
    [
      [0.0, 0x0000],
      [-0.0, 0x8000],
      [1.0, 0x3C00],
      [-1.0, 0xBC00],
      [2.0, 0x4000],
      [0.5, 0x3800],
      [65_504.0, 0x7BFF]
    ].each { |float, expected| assert_equal expected, bits(float), "encoding #{float}" }
  end

  def test_overflow_saturates_to_infinity
    assert_equal 0x7C00, bits(65_520.0)
    assert_equal 0x7C00, bits(1.0e30)
    assert_equal 0xFC00, bits(-1.0e30)
    assert_equal 0x7C00, bits(Float::INFINITY)
    assert_equal 0xFC00, bits(-Float::INFINITY)
  end

  def test_mantissa_carry_rounds_the_exponent_up
    assert_equal 2048.0, value(bits(2047.5))
    assert_equal 1.0, value(bits(0.99975586 + 0.00012))
  end

  def test_subnormals
    min_subnormal = 2.0**-24
    min_normal = 2.0**-14

    assert_equal 0x0001, bits(min_subnormal)
    assert_equal 0x0400, bits(min_normal)
    assert_equal 0x03FF, bits(min_normal - min_subnormal)
    assert_in_delta min_subnormal, value(0x0001), 1e-30
    assert_in_delta min_normal, value(0x0400), 1e-12
  end

  def test_underflow_to_zero_keeps_the_sign
    assert_equal 0x0000, bits(1.0e-30)
    assert_equal 0x8000, bits(-1.0e-30)
  end

  def test_nan_stays_nan
    assert_predicate value(bits(Float::NAN)), :nan?
    assert_predicate value(0x7E00), :nan?
    assert_predicate value(0xFE00), :nan?

    refute_equal 0x7C00, bits(Float::NAN) & 0x7FFF
  end

  def test_every_bit_pattern_round_trips_through_the_encoder
    mismatches = []
    (0...0x10000).each do |pattern|
      decoded = value(pattern)
      next if decoded.nan?

      re_encoded = bits(decoded)
      mismatches << format("0x%04x -> %s -> 0x%04x", pattern, decoded, re_encoded) if re_encoded != pattern
    end

    assert_empty mismatches.first(10)
  end

  def test_pack_and_unpack_are_inverses
    rows = [[1.5, -2.25, 0.0, 65_504.0], [0.125, -0.125, 2.0**-24, 1.0]]

    %i[f32 f16].each do |format|
      blob = StaticEmbeddings.pack(rows, format: format)
      assert_equal rows, StaticEmbeddings.unpack(blob, 4, format: format)
    end
  end

  def test_unpack_uses_the_same_decoder_as_top_k
    query = StaticEmbeddings.encode_f16([1.0])
    matrix = StaticEmbeddings.encode_f16([1.5, -0.25, 2.0**-24])
    scores = StaticEmbeddings.dot_top_k(query, matrix, 3, dim: 1, format: :f16)
                             .to_h.sort.map(&:last)

    assert_equal StaticEmbeddings.unpack(matrix, 1, format: :f16).flatten, scores
  end

  def test_embed_f16_matches_the_public_encoder
    model = TestSupport.model
    f32 = model.embed_array("hello world")

    assert_equal StaticEmbeddings.encode_f16(f32), model.embed("hello world", format: :f16)
  end

  def test_f16_top_k_matches_the_f32_kernel_for_every_tail_shape
    srand 20_260_828
    mismatches = []

    [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 63, 65, 127, 512].each do |dim|
      rows = 17
      values = Array.new(rows * dim) { rand * 4 - 2 }
      query = Array.new(dim) { rand * 4 - 2 }

      m16 = StaticEmbeddings.pack(values, format: :f16)
      q16 = StaticEmbeddings.pack(query, format: :f16)

      m32 = StaticEmbeddings.pack(StaticEmbeddings.unpack(m16, dim, format: :f16).flatten)
      q32 = StaticEmbeddings.pack(StaticEmbeddings.unpack(q16, dim, format: :f16).flatten)

      %i[dot_top_k cosine_top_k].each do |method|
        got = StaticEmbeddings.send(method, q16, m16, 5, dim: dim, format: :f16)
        expected = StaticEmbeddings.send(method, q32, m32, 5, dim: dim)

        mismatches << "dim=#{dim} #{method} order" if got.map(&:first) != expected.map(&:first)
        got.zip(expected).each do |(_, a), (_, b)|
          tolerance = method == :cosine_top_k ? 1e-5 : 1e-4 * dim
          mismatches << "dim=#{dim} #{method} #{a} vs #{b}" if (a - b).abs > tolerance
        end
      end
    end

    assert_empty mismatches.first(5)
  end
end
