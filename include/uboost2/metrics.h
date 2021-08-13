#pragma once

#include <cmath>
#include <cassert>

#include <uboost2/data.h>

double mean_squared_error(const DColumn<>& y, const DColumn<>& p) {
	assert(y.nrows() == p.nrows());
	double tot = 0.0;
	for (size_t i = 0; i < y.nrows(); i++) tot += (y(i) - p(i)) * (y(i) - p(i));
	return tot / y.nrows();
}