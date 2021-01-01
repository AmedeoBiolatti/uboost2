#pragma once

#include <vector>
#include <algorithm>

#include <uboost2/data.h>

struct Entry {
	size_t i;
	double f;
	double y;
};

class EntryMatrix : public DMatrix<Entry> {
	std::vector<std::vector<Entry>> entries;
public:
	EntryMatrix(const DMatrix<>& x, const DColumn<>& y) : DMatrix<Entry>{ x.nrows(), x.ncols() } {
		for (size_t i = 0; i < x.nrows(); i++) {
			for (size_t j = 0; j < x.ncols(); j++) {
				auto xij = x(i, j);
				auto e = Entry();
				e.i = i;
				e.f = xij;
				e.y = y(i);
				DMatrix<Entry>::operator()(i, j) = e;
			}
		}
	}
	//
	void sort_columns() {
		for (size_t col = 0; col < ncols(); col++) {
			size_t i_start = col * m_nrows;
			size_t i_end = (col + 1) * m_nrows;
			std::sort(
				m_data->begin() + i_start,
				m_data->begin() + i_end,
				[](Entry& a, Entry& b) { return a.f < b.f; }
			);
		}
	}
};
