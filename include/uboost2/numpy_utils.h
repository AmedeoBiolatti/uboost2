#pragma once

#include <pybind11/numpy.h>

#include <uboost2/data.h>

namespace py = pybind11;

DMatrix<double> numpyToDMatrix(py::array_t<double> arr) {
	auto r = arr.request();
	if (r.ndim != 2) {
		throw std::runtime_error("NDIM Must be == 2");
	}
	auto p = reinterpret_cast<double*>(r.ptr);

	DMatrix<double> data(r.shape[0], r.shape[1]);
	for (size_t i = 0; i < r.shape[0]; i++) {
		for (size_t j = 0; j < r.shape[1]; j++) {
			data(i, j) = p[i * r.shape[1] + j];
		}
	}

	return data;
}

DColumn<double> numpyToDColumn(py::array_t<double> arr) {
	auto r = arr.request();
	if (r.ndim != 1) {
		throw std::runtime_error("NDIM Must be == 1");
	}
	auto p = reinterpret_cast<double*>(r.ptr);

	DColumn<double> data(r.shape[0]);
	for (size_t i = 0; i < r.shape[0]; i++) {
		data(i) = p[i];
	}

	return data;
}