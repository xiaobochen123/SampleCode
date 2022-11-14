#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "transpose.h"
#include "utils.h"

namespace py = pybind11;

template <typename T>
void TestNumpyTranspose(std::vector<T> &data_in, std::vector<T> &data_out,
                        std::vector<uint32_t> dims_in,
                        std::vector<uint32_t> perms) {
    py::array_t<T> data_in_np(dims_in, static_cast<T *>(data_in.data()));
    auto np = py::module::import("numpy");

    py::object transpose_func = np.attr("transpose");
    py::object result;
    result = np.attr("ascontiguousarray")(
        transpose_func(data_in_np, py::cast(perms)));

    py::array_t<T> data_out_np = result.cast<py::array_t<T>>();
    py::buffer_info data_out_buf = data_out_np.request();
    T *ptr = (T *)data_out_buf.ptr;
    memcpy(static_cast<T *>(data_out.data()), ptr, data_out.size() * sizeof(T));
}

template <>
void TestNumpyTranspose<float16>(std::vector<float16> &data_in,
                                 std::vector<float16> &data_out,
                                 std::vector<uint32_t> dims_in,
                                 std::vector<uint32_t> perms) {
    std::vector<float> data_in_fp32(data_in.size());
    std::vector<float> data_out_fp32(data_out.size());
    for (int i = 0; i < data_in_fp32.size(); i++) {
        data_in_fp32[i] = float(data_in[i]);
    }
    py::array_t<float> data_in_np(dims_in,
                                  static_cast<float *>(data_in_fp32.data()));
    auto np = py::module::import("numpy");

    py::object transpose_func = np.attr("transpose");
    py::object result;
    result = np.attr("ascontiguousarray")(
        transpose_func(data_in_np, py::cast(perms)));

    py::array_t<float> data_out_np = result.cast<py::array_t<float>>();
    py::buffer_info data_out_buf = data_out_np.request();
    float *ptr = (float *)data_out_buf.ptr;
    memcpy(static_cast<float *>(data_out_fp32.data()), ptr,
           data_out_fp32.size() * sizeof(float));
    for (int i = 0; i < data_out_fp32.size(); i++) {
        data_out[i] = float16(data_out_fp32[i]);
    }
}

template <typename T, int NUM_AXES>
void TestCase(std::vector<uint32_t> dims_in, std::vector<uint32_t> perms) {
    std::cout << "Dims[ ";
    for (int i = 0; i < dims_in.size(); ++i) {
        std::cout << dims_in[i] << " ";
    }
    std::cout << "] Perms[ ";
    for (int i = 0; i < perms.size(); ++i) {
        std::cout << perms[i] << " ";
    }
    std::cout << "] ElementSize = " << sizeof(T) << std::endl;

    std::vector<T> data_in(AccMul(dims_in));
    GenerateRandomData<T>(data_in, data_in.size());
    std::vector<T> data_out(AccMul(dims_in));
    TestNumpyTranspose<T>(data_in, data_out, dims_in, perms);
    TestTranspose<T, NUM_AXES>(data_in, data_out, dims_in, perms);
}

int main() {
    py::scoped_interpreter guard{};

    std::cout << "Test Transpose" << std::endl;

    using DataType = float;
    const int NUM_AXES = 3;
    std::vector<uint32_t> shapes = {256, 256, 128};
    std::vector<uint32_t> perms = {0, 1, 2};

    TestCase<DataType, NUM_AXES>(shapes, perms);
    return 0;
}