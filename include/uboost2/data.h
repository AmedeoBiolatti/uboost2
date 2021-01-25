#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <iostream>
#include <iomanip>
#include <random>
#include <cassert>

// interfaces

template <typename T>
class AbstractDMatrix {
public:
	virtual size_t nrows() const = 0;
	virtual size_t ncols() const = 0;
	//
	virtual inline T& operator()(size_t i, size_t j) = 0;
	virtual inline const T& operator()(size_t i, size_t j) const = 0;
	//
	friend std::ostream& operator<<(std::ostream& os, const AbstractDMatrix<T>& d) {
		for (size_t i = 0; i < d.nrows(); i++) {
			for (size_t j = 0; j < d.ncols(); j++) {
				os << d(i, j) << "\t";
			}
			os << std::endl;
		}
		return os;
	}
};

template <typename T>
class AbstractDColumn {
public:
	virtual size_t nrows() const = 0;
	//
	virtual inline T& operator()(size_t i) = 0;
	virtual inline const T& operator()(size_t i) const = 0;
	// iterator TODO
	class iterator {
		AbstractDColumn* col;
		size_t i = 0;
	public:
		iterator(AbstractDColumn* col) : col{ col }, i{ 0 } {}
		iterator(AbstractDColumn* col, size_t i) : col{ col }, i{ i } {}
		T& operator*() const {
			return (*col)(i);
		}
		iterator& operator++() {
			i++;
			return (*this);
		}
		bool operator==(const iterator& b) { return (col == b.col) && (i == b.i); }
		//bool operator!=(const iterator& b) { return (col != b.col) || (i != b.i); }
	};
	class const_iterator {
		const AbstractDColumn* col;
		size_t i = 0;
	public:
		const_iterator(const AbstractDColumn* col) : col{ col }, i{ 0 } {}
		const_iterator(const AbstractDColumn* col, size_t i) : col{ col }, i{ i } {}
		const T& operator*() const {
			return (*col)(i);
		}
		const_iterator& operator++() {
			i++;
			return (*this);
		}
		bool operator==(const const_iterator& b) { return (col == b.col) && (i == b.i); }
		//bool operator!=(const iterator& b) { return (col != b.col) || (i != b.i); }
	};
	iterator begin() {
		return iterator(this);
	}
	iterator end() {
		return iterator(this, nrows());
	}
	const_iterator begin() const {
		return const_iterator(this);
	}
	const_iterator end() const {
		return const_iterator(this, nrows());
	}
	//
	inline T sum() const {
		T t = T();
		for (auto xi : (*this)) t = t + xi;
		return t;
	}
};

template <typename T>
class AbstractDRow{
public:
	virtual size_t ncols() const = 0;
	//
	virtual inline T& operator()(size_t j) = 0;
	virtual inline const T& operator()(size_t j) const = 0;
	// iterator TODO
};

// implementation

template <typename T=double>
class DMatrix : public AbstractDMatrix<T> {
protected:
	std::shared_ptr<std::vector<T>> m_data;
	size_t m_nrows, m_ncols;
	bool m_owning = false;
public:
	size_t nrows() const override {
		return m_nrows;
	}
	size_t ncols() const override {
		return m_ncols;
	}
	//
	DMatrix(size_t nrows, size_t ncols) {
		assert(nrows > 0);
		assert(ncols > 0);
		m_nrows = nrows;
		m_ncols = ncols;
		m_data = std::make_shared<std::vector<T>>();
		m_data->resize(nrows * ncols);
		m_owning = true;
	}
	DMatrix(size_t nrows, size_t ncols, const T& t) {
		assert(nrows > 0);
		assert(ncols > 0);
		m_nrows = nrows;
		m_ncols = ncols;
		m_data = std::make_shared<std::vector<T>>();
		m_data->resize(nrows * ncols, t);
		m_owning = true;
	}
	DMatrix(const DMatrix<T>& mat) {
		m_nrows = mat.m_nrows;
		m_ncols = mat.m_ncols;
		m_data = mat.m_data;
		assert(m_data->size() > 0);
		m_owning = false;
	}
	~DMatrix() {}
	//
	inline T& operator()(size_t i, size_t j) override {
#ifdef DEBUG
		if (j * m_nrows + i >= m_data->size()) std::printf("Out of bound! (%d, %d)\n", i, j);
#endif // DEBUG
		return (*m_data)[j * m_nrows + i];
	}
	inline const T& operator()(size_t i, size_t j) const override {
#ifdef DEBUG
		if (j * m_nrows + i >= m_data->size()) std::printf("Out of bound! (%d, %d)\n", i, j);
#endif // DEBUG
		return (*m_data)[j * m_nrows + i];
	}
};

template <typename T=double>
class DColumn : public AbstractDColumn<T>, protected DMatrix<T> {
	size_t m_column = 0;
public:
	DColumn(const DMatrix<T>& mat, size_t col) : DMatrix<T>{ mat }, m_column{ col } {}
	DColumn(size_t nrows) : DMatrix<T>{ nrows, 1 }, m_column{ 0 } {}
	DColumn(size_t nrows, const T& t) : DMatrix<T>{ nrows, 1, t }, m_column{ 0 } {}
	~DColumn() {

	}
	//
	size_t nrows() const {
		return DMatrix<T>::nrows();
	}
	//
	inline T& operator()(size_t i) override {
		return DMatrix<T>::operator()(i, m_column);
	}
	inline const T& operator()(size_t i) const override {
		return DMatrix<T>::operator()(i, m_column);
	}
	//
	friend std::ostream& operator<<(std::ostream& os, const DColumn<T>& d) {
		for (size_t i = 0; i < d.nrows(); i++) {
			os << d(i) << std::endl;
		}
		return os;
	}
};


template <typename T = double>
class DRow : public AbstractDRow<T>, protected DMatrix<T> {
	size_t m_row;
public:
	DRow(const DMatrix<T>& mat, size_t row) : DMatrix<T>{ mat } {
		m_row = row;
	}
	size_t ncols() const {
		return DMatrix<T>::ncols();
	}
	//
	inline T& operator()(size_t j) override {
		return DMatrix<T>::operator()(m_row, j);
	}
	inline const T& operator()(size_t j) const override {
		return DMatrix<T>::operator()(m_row, j);
	}
};


//
namespace random {
	DMatrix<> uniform(size_t nrows, size_t ncols) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0.0, 1.0);
		DMatrix<> m(nrows, ncols);
		for (size_t i = 0; i < nrows; i++) for (size_t j = 0; j < ncols; j++) {
			m(i, j) = dis(gen);
		}
		return m;
	}
	DColumn<> uniform(size_t nrows) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(0.0, 1.0);
		DColumn<> m(nrows);
		for (size_t i = 0; i < nrows; i++) {
			m(i) = dis(gen);
		}
		return m;
	}
	//
	DColumn<> normal(size_t nrows) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<> dis(0.0, 1.0);
		DColumn<> m(nrows);
		for (size_t i = 0; i < nrows; i++) {
			m(i) = dis(gen);
		}
		return m;
	}

};

namespace matrix {
	DColumn<> ones(size_t nrows) {
		DColumn<> m(nrows);
		for (size_t i = 0; i < nrows; i++) {
			m(i) = 1.0;
		}
		return m;
	}
	DColumn<> zeros(size_t nrows) {
		DColumn<> m(nrows);
		for (size_t i = 0; i < nrows; i++) {
			m(i) = 0.0;
		}
		return m;
	}
};
