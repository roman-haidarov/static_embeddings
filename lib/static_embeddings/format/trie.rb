require "static_embeddings/format/constants"

module StaticEmbeddings
  module Format
    class Trie
      Node = Struct.new(:terminal, :children, keyword_init: true)

      def initialize
        @nodes = [Node.new(terminal: SLOT_EMPTY, children: {})]
      end

      def insert(bytes, id)
        return if bytes.empty?

        node_index = 0
        bytes.each_byte do |byte|
          node = @nodes[node_index]
          node_index = node.children[byte] ||= append_node
        end

        node = @nodes[node_index]
        raise ArgumentError, "duplicate trie key for token id #{id}" unless node.terminal == SLOT_EMPTY

        node.terminal = id
      end

      def pack
        edges = []
        nodes = @nodes.map do |node|
          start = edges.length
          node.children.sort.each { |byte, child| edges << [byte, child] }
          [start, node.children.length, node.terminal, 0]
        end

        nodes.each_with_object(header(nodes.length, edges.length)) { |record, out| out << record.pack("V4") }
             .tap { |out| edges.each { |edge| out << edge.pack("V2") } }
      end

      private

      def append_node
        @nodes << Node.new(terminal: SLOT_EMPTY, children: {})
        @nodes.length - 1
      end

      def header(node_count, edge_count)
        capacity = 16 + node_count * 16 + edge_count * 8
        Format.binary_string(capacity).tap { |out| out << [node_count, edge_count, 0, 0].pack("V4") }
      end
    end

    module WordPieceTrie
      module_function

      def build(tokens, prefix)
        root = Trie.new
        continuation = Trie.new
        prefix_bytes = prefix.b

        tokens.each_with_index do |token, id|
          bytes = token.b
          if continuation?(bytes, prefix_bytes)
            body = bytes.byteslice(prefix_bytes.bytesize, bytes.bytesize - prefix_bytes.bytesize)
            continuation.insert(body, id)
          else
            root.insert(bytes, id)
          end
        end

        [root.pack, continuation.pack]
      end

      def continuation?(bytes, prefix)
        !prefix.empty? && bytes.start_with?(prefix) && bytes.bytesize > prefix.bytesize
      end
    end
  end
end
