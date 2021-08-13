#pragma once

#include <vector>
#include <algorithm>
#include <cassert>

#include <uboost2/data.h>

struct Entry {
	size_t i = 0;
	double x = NAN;
	double y = NAN;
	double w = 1.0;
	Entry() {}
	Entry(size_t i, double x, double y) {
		this->i = i;
		this->x = x;
		this->y = y;
		this->w = 1.0;
	}
	Entry(size_t i, double x, double y, double w) {
		this->i = i;
		this->x = x;
		this->y = y;
		this->w = w;
	}
};

class EntryMatrix : public DMatrix<Entry> {
	double y_mean;
public:
	EntryMatrix(const DMatrix<>& x, const DColumn<>& y) : DMatrix<Entry>{ x.nrows(), x.ncols() } {
		assert(x.nrows() == y.nrows());
		for (size_t i = 0; i < x.nrows(); i++) {
			for (size_t j = 0; j < x.ncols(); j++) {
				auto xij = x(i, j);
				Entry e(i, xij, y(i));
				DMatrix<Entry>::operator()(i, j) = e;
			}
		}
		y_mean = y.sum() / y.nrows();
	}
	EntryMatrix(const DMatrix<>& x, const DColumn<>& y, const DColumn<>& w) : DMatrix<Entry>{ x.nrows(), x.ncols() } {
		assert(x.nrows() == y.nrows());
		assert(x.nrows() == w.nrows());
		for (size_t i = 0; i < x.nrows(); i++) {
			for (size_t j = 0; j < x.ncols(); j++) {
				auto xij = x(i, j);
				Entry e(i, xij, y(i), w(i));
				DMatrix<Entry>::operator()(i, j) = e;
			}
		}
		y_mean = y.sum() / y.nrows();
	}
	//
	void set_x(const DMatrix<>& x) {
		assert(nrows() == y.nrows());
		for (size_t k = 0; k < nrows(); k++) {
			for (size_t j = 0; j < ncols(); j++) {
				auto& e = DMatrix<Entry>::operator()(k, j);
				size_t i = e.i;
				e.x = x(i, j);
			}
		}
	}
	void set_y(const DColumn<>& y) {
		assert(nrows() == y.nrows());
		for (size_t k = 0; k < nrows(); k++) {
			for (size_t j = 0; j < ncols(); j++) {
				auto& e = DMatrix<Entry>::operator()(k, j);
				size_t i = e.i;
				e.y = y(i);
			}
		}
		y_mean = y.sum() / y.nrows();
	}
	void set_w(const DColumn<>& w) {
		assert(nrows() == y.nrows());
		for (size_t k = 0; k < nrows(); k++) {
			for (size_t j = 0; j < ncols(); j++) {
				auto& e = DMatrix<Entry>::operator()(k, j);
				size_t i = e.i;
				e.w = w(i);
			}
		}
	}
	//
	double get_y_mean() const {
		return y_mean;
	}
	//
	void sort_columns() {
		for (size_t col = 0; col < ncols(); col++) {
			size_t i_start = col * m_nrows;
			size_t i_end = (col + 1) * m_nrows;
			std::sort(
				m_data->begin() + i_start,
				m_data->begin() + i_end,
				[](Entry& a, Entry& b) { return a.x < b.x; }
			);
		}
	}
};

// GH

struct GHEntry {
	size_t i = 0;
	double x = NAN;
	double g = NAN;
	double h = NAN;
	double w = 1.0;
	GHEntry() {}
	GHEntry(size_t i, double x, double g, double h) {
		this->i = i;
		this->x = x;
		this->g = g;
		this->h = h;
		this->w = 1.0;
	}
	GHEntry(size_t i, double x, double g, double h, double w) {
		this->i = i;
		this->x = x;
		this->g = g;
		this->h = h;
		this->w = w;
	}
};

class GHEntryMatrix : public DMatrix<GHEntry> {
public:
	GHEntryMatrix(
		const DMatrix<>& x, const DColumn<>& g, const DColumn<>& h
	) : DMatrix<GHEntry>{ x.nrows(), x.ncols() } {
		assert(x.nrows() == g.nrows());
		assert(x.nrows() == h.nrows());
		for (size_t i = 0; i < x.nrows(); i++) {
			for (size_t j = 0; j < x.ncols(); j++) {
				auto xij = x(i, j);
				auto e = GHEntry(i, x(i, j), g(i), h(i));
				DMatrix<GHEntry>::operator()(i, j) = e;
			}
		}
	}
	GHEntryMatrix(
		const DMatrix<>& x, const DColumn<>& g, const DColumn<>& h, const DColumn<>& w) : DMatrix<GHEntry>{ x.nrows(), x.ncols() } {
		assert(x.nrows() == g.nrows());
		assert(x.nrows() == h.nrows());
		for (size_t i = 0; i < x.nrows(); i++) {
			for (size_t j = 0; j < x.ncols(); j++) {
				auto xij = x(i, j);
				auto e = GHEntry(i, x(i, j), g(i), h(i), w(i));
				DMatrix<GHEntry>::operator()(i, j) = e;
			}
		}
	}
	//
	void set_x(const DMatrix<>& x) {
		assert(nrows() == y.nrows());
		for (size_t k = 0; k < nrows(); k++) {
			for (size_t j = 0; j < ncols(); j++) {
				auto& e = DMatrix<GHEntry>::operator()(k, j);
				size_t i = e.i;
				e.x = x(i, j);
			}
		}
	}
	void set_g(const DColumn<>& g) {
		assert(nrows() == y.nrows());
		for (size_t k = 0; k < nrows(); k++) {
			for (size_t j = 0; j < ncols(); j++) {
				auto& e = DMatrix<GHEntry>::operator()(k, j);
				size_t i = e.i;
				e.g = g(i);
			}
		}
	}
	void set_h(const DColumn<>& h) {
		assert(nrows() == y.nrows());
		for (size_t k = 0; k < nrows(); k++) {
			for (size_t j = 0; j < ncols(); j++) {
				auto& e = DMatrix<GHEntry>::operator()(k, j);
				size_t i = e.i;
				e.h = h(i);
			}
		}
	}
	void set_w(const DColumn<>& w) {
		assert(nrows() == y.nrows());
		for (size_t k = 0; k < nrows(); k++) {
			for (size_t j = 0; j < ncols(); j++) {
				auto& e = DMatrix<GHEntry>::operator()(k, j);
				size_t i = e.i;
				e.w = w(i);
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
				[](GHEntry& a, GHEntry& b) { return a.x < b.x; }
			);
		}
	}
};
