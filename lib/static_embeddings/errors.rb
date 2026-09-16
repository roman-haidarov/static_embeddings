module StaticEmbeddings
  class ConversionError < Error; end
  class InvalidSourceError < ConversionError; end
  class InvalidOptionError < ArgumentError; end
  class ModelNotFound < Error; end
end
