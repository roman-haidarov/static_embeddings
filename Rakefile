require "bundler/gem_tasks" rescue nil
require "rake/testtask"

begin
  require "rake/extensiontask"
  Rake::ExtensionTask.new("static_embeddings") do |ext|
    ext.lib_dir = "lib/static_embeddings"
    ext.ext_dir = "ext/static_embeddings"
  end
rescue LoadError
  desc "Compile the extension without rake-compiler"
  task :compile do
    ext = File.expand_path("ext/static_embeddings", __dir__)
    lib = File.expand_path("lib/static_embeddings", __dir__)
    build = File.expand_path("tmp/build", __dir__)
    FileUtils.mkdir_p(build)
    Dir.chdir(build) do
      sh "ruby #{ext}/extconf.rb --with-cflags=-I#{ext}"
      sh "make -j4"
    end
    FileUtils.cp(File.join(build, "static_embeddings.so"), lib)
  end
end

desc "Regenerate the synthetic source model used by the test suite"
task :fixtures do
  ruby "tools/make_fixture_model.rb"
end

desc "Build the demo .semb shipped inside the gem"
task demo_model: %i[fixtures compile] do
  ruby "tools/build_demo_model.rb"
end

desc "Check the C runtime against the Python model2vec.StaticModel oracle"
task parity: :compile do
  model = ENV["MODEL"] || File.expand_path("~/.cache/static_embeddings/models/potion-retrieval-32m.semb")
  oracle = ENV["ORACLE"] || "tmp/model2vec_oracle.json"

  unless File.file?(oracle)
    abort <<~MSG
      oracle not found: #{oracle}

      Generate it first (needs Python and the model2vec package):

        python3 -m venv .venv-model2vec && . .venv-model2vec/bin/activate
        pip install -U model2vec numpy
        python tools/model2vec_oracle.py /path/to/source-model --out #{oracle}
    MSG
  end
  abort "model not found: #{model}" unless File.file?(model)

  ruby "-Ilib tools/check_model2vec_parity.rb --model #{model.inspect} --oracle #{oracle.inspect}"
end

Rake::TestTask.new(:test) do |t|
  t.libs << "test" << "lib"
  t.test_files = FileList["test/**/*_test.rb"].exclude("test/cancellation_timing_test.rb")
  t.warning = false
end

Rake::TestTask.new(:cancellation_timing) do |t|
  t.libs << "test" << "lib"
  t.test_files = ["test/cancellation_timing_test.rb"]
  t.warning = false
end

task test: %i[compile fixtures]
task cancellation_timing: %i[compile fixtures]
task default: %i[compile fixtures test]
