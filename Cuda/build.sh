mkdir third_party
git clone https://github.com/pybind/pybind11.git third_party/pybind11

mkdir build && cd build
cmake -DCUDA_ARCHS=75 ..
make -j
