#pragma once

#include <vector>
#include <random>

class ColumnProposer {
	std::vector<size_t> columns;
	double colsample_bylevel = 1.0;
	std::default_random_engine generator;
public:
	ColumnProposer(size_t ncols, double colsample_bytree, double colsample_bylevel) {
		this->colsample_bylevel = colsample_bylevel;

		size_t n = (size_t)(ncols * colsample_bytree);
		if (n < 1) n = 1;
		for (size_t col = 0; col < ncols; col++) {
			columns.push_back(col);
		}

		std::shuffle(columns.begin(), columns.end(), generator);

		columns.resize(n);
	}
	std::vector<size_t> get_columns() {
		std::vector<size_t> out(this->columns);
		std::shuffle(out.begin(), out.end(), generator);
		if (colsample_bylevel < 1.0) {
			size_t n = (size_t)(out.size() * colsample_bylevel);
			if (n < 1) n = 1;
			out.resize(n);
		}
		return out;
	}
};