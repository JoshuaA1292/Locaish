# Building OpenMVS on macOS (Apple Silicon)

OpenMVS supplies the CPU dense-stereo stage (`DensifyPointCloud`) that the
video pipeline prefers when CUDA is absent. There is no Homebrew formula; these
are the steps that produced a working build on an M4 running Sequoia, against
OpenMVS v2.4.0. The Linux equivalent is automated in the `Dockerfile`.

```bash
# Dependencies. opencv@4 specifically: OpenMVS does not compile against
# OpenCV 5 (cv::DataType specialisation clashes).
brew install cmake boost eigen cgal glew libomp nanoflann tinyxml2 opencv@4

mkdir -p ~/tools && cd ~/tools

# Two tiny libraries by the OpenMVS author, expected as CMake packages:
git clone https://github.com/cdcseacave/TinyEXIF.git
cmake -S TinyEXIF -B TinyEXIF/build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMO=OFF
cmake --build TinyEXIF/build -j4 && cmake --install TinyEXIF/build --prefix ~/tools/local

git clone https://github.com/cdcseacave/TinyNPY.git
cmake -S TinyNPY -B TinyNPY/build -DCMAKE_BUILD_TYPE=Release
cmake --build TinyNPY/build -j4 && cmake --install TinyNPY/build --prefix ~/tools/local

git clone https://github.com/cdcseacave/VCG.git
git clone --recursive https://github.com/cdcseacave/openMVS.git
cd openMVS

# NOTE: the repo's own cmake helpers live in the tracked `build/` directory --
# do not use it as the binary dir (and never rm -rf it).
cmake -S . -B out -DCMAKE_BUILD_TYPE=Release \
  -DVCG_ROOT=$HOME/tools/VCG \
  -DCMAKE_PREFIX_PATH=$HOME/tools/local \
  -DOpenCV_DIR="$(brew --prefix opencv@4)/lib/cmake/opencv4" \
  -DOpenMVS_USE_CUDA=OFF -DOpenMVS_ENABLE_TESTS=OFF

# Homebrew's libheif ships both a static archive and a dylib, and the
# generated link lines pick the archive, whose codec dependencies (x265, aom,
# sharpyuv, de265) are then missing. Point the link lines at the dylib:
grep -rl "libheif.a" out --include=link.txt | xargs sed -i '' \
  's|/opt/homebrew/opt/libheif/lib/libheif.a|/opt/homebrew/opt/libheif/lib/libheif.dylib|g'

cmake --build out -j8 --target DensifyPointCloud InterfaceCOLMAP
```

`locaish` finds the binaries automatically at
`~/tools/openMVS/out/bin/DensifyPointCloud` (see
`locaish/video/dense.py::openmvs_binary`); putting them on PATH works too.
`LOCAISH_MVS_LEVEL=2` halves the matching resolution again for small cloud
CPUs.
