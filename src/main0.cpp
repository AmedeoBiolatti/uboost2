#pragma once

#include <iostream>

#include <uboost2/data.h>
#include <uboost2/tree/tree.h>
#include <uboost2/tree/builder/builder_base.h>
#include <uboost2/tree/builder/builder_layerwise.h>

void main() {
	size_t nrows = 100, ncols = 50;
	auto x = random::uniform(nrows, ncols);
	auto y = random::uniform(nrows);
	for (size_t i = 0; i < nrows; i++) {
		y(i) = y(i) + x(i, 0) + x(i, 1) + x(i, 2);
	}

	Tree t(12);
	TreeBuilder* b = new LayerWiseTreeBuilder(x, y, 1, 2, 1.0, 0.8);
	b->update(t);
	
	for (size_t i = 0; i < 10; i++)
		std::printf("%d\t%.2f\t%.2f\n", i, y(i), t.predict_value_row(x, i));
}