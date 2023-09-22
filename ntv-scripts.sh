# source ntv-scripts.sh && whisper-run base 64
# source ntv-scripts.sh && ggmlir-compile ../llvm-project/mlir/test/Dialect/Linalg/ggml-q5-1-reference.mlir && whisper-run mlir 64
function ggmlir-compile() {( set -e
  DOT_VERSION=${2:-0}

  if [[ -z "$1" || ! -f "$1" || ! "${DOT_VERSION}" =~ ^(0|1|10|11)$ ]]; then
    echo "Usage: mlir-compile <ggmlir-impl-filename> <dot-impl-number> , where:"
    echo "    <ggmlir-impl-filename> must be a valid file name"
    echo "    <dot-impl-number> must be a valid integer specifying a dot implementation:"
    echo "        0: ggml_vec_dot_q5_1_q8_1_interim_1x (default)"
    echo "        1: ggml_vec_dot_q5_1_q8_1_interim_32x"
    echo "       10: ggml_vec_dot_q5_1_q8_1_interim_1x_avx2"
    echo "       11: ggml_vec_dot_q5_1_q8_1_interim_8x_avx2"
    echo "Got: mlir-compile $1 $2"
    return 1
  fi

   # if the $(which mlir-opt) command does not return the path to the mlir-opt executable, then fail with an error.
  if [[ -z "$(which mlir-opt)" || -z "$(which llc)" ]]; then
    echo "Either mlir-opt of llv commands not found. Please add the LLVM build directory to your path."
    return 1
  fi

  # sed command to choose the version of the dotvec implementation to use.
  local sed_cmd="cat $1 | sed -e 's/%switch_cst = arith.constant 0: index/%switch_cst = arith.constant "$DOT_VERSION": index/g' > /tmp/ggmlir.mlir"
  echo ${sed_cmd}
  eval ${sed_cmd}

  local mlir_cmd="mlir-opt /tmp/ggmlir.mlir --inline -test-transform-dialect-interpreter -test-transform-dialect-erase-schedule | mlir-opt -test-lower-to-llvm -canonicalize -cse | mlir-translate -mlir-to-llvmir > /tmp/ggmlir.ll"
  echo ${mlir_cmd}
  eval ${mlir_cmd}

  # Atm we use a simpler memref<vector<32xf32>> impl in MLIR which has alignment
  # of 128. This does not match the GGML allocations, so we need to change it to 4.
  local sed_cmd='sed -i "s/align 128/align 4/g" /tmp/ggmlir.ll'
  echo ${sed_cmd}
  eval ${sed_cmd}

  # local clang_cmd='clang-head -O3 -ffast-math -x ir -c /tmp/ggmlir.ll -S -emit-llvm -o /tmp/ggmlir.2.ll'
  # echo ${clang_cmd}
  # eval ${clang_cmd}
   
  local opt_cmd='opt -O3 --enable-no-infs-fp-math --enable-no-nans-fp-math --enable-no-signed-zeros-fp-math --enable-unsafe-fp-math  --enable-no-trapping-fp-math  /tmp/ggmlir.ll -o /tmp/ggmlir.2.ll'
  echo ${opt_cmd}
  eval ${opt_cmd}
   
  local llc_cmd='llc -O3 -mcpu=native --function-sections -filetype=obj /tmp/ggmlir.2.ll -o /tmp/ggmlir.o'
  echo ${llc_cmd}
  eval ${llc_cmd}

  # Optionally, disassemble the object file and inspect with llvm-mca.
  # objdump -d --disassemble=ggml_vec_dot_q5_1_q8_1 --no-addresses --no-show-raw-insn -M att ggmlir.o | llvm-mca -mcpu=native | less
)}

function whisper-configure() {( set -e
  if [ -z "$1" ]; then
    echo "Usage: whisper-configure <branch>"
    return 1
  fi
  local branch=$1
  # if the build-$branch directory exists and is not empty, then we assume that the build is already configured and we skip the configuration step.
  if [ -d "build-$branch" ] && [ -n "$(ls -A build-$branch)" ]; then
    echo "build-$branch directory exists and is not empty. Skipping configuration step."
    return 0
  fi

  local config_cmd="git checkout $branch && (mkdir -p build-$branch && cd build-$branch && CC=clang-15 CXX=clang++-15 cmake -DCMAKE_BUILD_TYPE=Release ..)"
  
  # Note: build with relwithdebinfo allows to run perf and FlameGraph:
  # source ntv-scripts.sh && \
  # rm -Rf build-base/ && \
  # whisper-run base 1 && \
  # (sudo perf record -F 99 -a -g -- ./build-base/bin/whisper -m /usr/local/google/home/ntv/model-mlir.bin /usr/local/google/home/ntv/test.wav -t 1 2>&1 > /dev/null) && \
  # sudo perf script > out.perf && \
  # sudo ../FlameGraph/stackcollapse-perf.pl out.perf > perf.folded && \
  # ../FlameGraph/flamegraph.pl perf.folded > perf.svg

  # Note: Build with default system compiler (GCC in my case is bad):
  # local config_cmd="git checkout $branch && (mkdir -p build-$branch && cd build-$branch && cmake -DCMAKE_BUILD_TYPE=Release ..)"

  echo ${config_cmd}
  eval ${config_cmd}
)}


function whisper-build() {( set -e
  if [ -z "$1" ]; then
    echo "Usage: whisper-build <branch>"
    return 1
  fi
  if [ -z "$2" ]; then
    echo "Usage: whisper-build <branch> <bin>"
    return 1
  fi
  local branch=$1
  local bin=$2
  whisper-configure $branch
  local build_cmd="git checkout $branch && (cd build-$branch && make -j 16 $bin)"
  echo ${build_cmd}
  eval ${build_cmd}
)}


function whisper-quantize() {( set -e
  local branch=$1
  whisper-build $branch whisper-quantize
  local quantize_cmd="time (cd build-$branch/bin && ./whisper-quantize ${HOME}/ggml-tiny.en.bin ${HOME}/model-$branch.bin 9 > /dev/null)"
  echo ${quantize_cmd}
  eval ${quantize_cmd}
)}


function whisper-run() {( set -e
  local branch=$1
  whisper-build $branch whisper
  whisper-quantize $branch
  local num_threads=${2:-16}
  local run_cmd="(cd build-$branch/bin && ./whisper -m ${HOME}/model-$branch.bin ${HOME}/test.wav -t ${num_threads})"
  echo ${run_cmd}
  eval ${run_cmd}
)}

function test-quantize-perf() {( set -e
  local branch=$1
  whisper-build $branch test-quantize-perf
  local size=${2:-256}
  local run_cmd="(cd build-$branch/bin && ./test-quantize-perf --type q5_1 --size ${size} --op quantize_row_q,dequantize_row_q)"
  echo ${run_cmd}
  eval ${run_cmd}
)}


# opt -O3 /tmp/aaa.ll -o /tmp/bbb.ll -S --print-after=instcombine
# llc -O3 ${TARGET_CPU} --function-sections -filetype=obj -o /tmp/aaa.o --debug-only=isel > /tmp/foo 2>&1
