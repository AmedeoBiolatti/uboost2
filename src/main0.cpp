#pragma once

#include <iostream>
#include <chrono>
#include <thread>

#include <uboost2/data.h>
#include <uboost2/metrics.h>
#include <uboost2/tree/tree.h>
#include <uboost2/tree/builder/builder_base.h>
#include <uboost2/tree/builder/builder_layerwise.h>
#include <uboost2/tree/builder/builder_layerwise_gh.h>
#include <uboost2/tree/builder/builder_nodewise.h>

void main() {

	size_t nrows = 500000, ncols = 5;
	auto x = random::uniform(nrows, ncols);
	auto y = random::normal(nrows);
	auto f = matrix::zeros(nrows);
	auto h = matrix::ones(nrows);
	auto t1_ = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < nrows; i++) {
		f(i) = x(i, 0) + x(i, 1) + x(i, 2) + x(i, 3);
		y(i) = f(i) + 0.5 * y(i);
	}
	auto t2_ = std::chrono::high_resolution_clock::now();
	auto duration_ = std::chrono::duration_cast<std::chrono::microseconds>(t2_ - t1_).count();
	std::cout << duration_ << std::endl;

	{
		Tree t(12);
		TreeBuilder* b = new GHLayerWiseTreeBuilder(x, y, h, 1, 2, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0);
		auto t1 = std::chrono::high_resolution_clock::now();
		b->update(t);
		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << duration << std::endl;
		std::cout << "N leaves " << t.get_n_leaves() << std::endl;
		std::printf("Train MSE = %.6f\n", mean_squared_error(y, t.predict_value(x)));
		std::printf("True  MSE = %.6f\n", mean_squared_error(f, t.predict_value(x)));
	}
	{
		Tree t(12);
		TreeBuilder* b = new LayerWiseTreeBuilder(x, y);
		auto t1 = std::chrono::high_resolution_clock::now();
		b->update(t);
		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << duration << std::endl;
		std::cout << "N leaves " << t.get_n_leaves() << std::endl;
		std::printf("Train MSE = %.6f\n", mean_squared_error(y, t.predict_value(x)));
		std::printf("True  MSE = %.6f\n", mean_squared_error(f, t.predict_value(x)));
	}
}
