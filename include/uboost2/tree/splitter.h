#pragma once

#include <uboost2/tree/split.h>
#include <uboost2/tree/entry.h>

class Splitter {
public:
	Splitter() {}
	virtual void add(const Entry& e) = 0;
	virtual void start_splitting(size_t col=0) = 0;
	virtual const Split build_split(const Entry& e) = 0;
};


class MSESplitter : public Splitter {
	size_t min_samples_leaf = 1;
	size_t column;
	//
	double s, s2;
	size_t n;
	double sl, s2l;
	size_t nl;
	double sr, s2r;
	size_t nr;
	//
	bool first;
	double previous_f;
	double p_criterion;
public:
	MSESplitter(size_t min_samples_leaf=1) {
		this->min_samples_leaf = min_samples_leaf;
		s = 0.0;
		s2 = 0.0;
		n = 0;
	}
	void add(const Entry& e) override {
		s += e.y;
		s2 += e.y * e.y;
		n++;
	}
	void start_splitting(size_t col = 0) override {
		column = col;
		//
		sl = 0.0;
		s2l = 0.0;
		nl = 0;
		sr = s;
		s2r = s2;
		nr = n;
		//
		p_criterion = s * s / n - s2;
		first = true;
		previous_f = NAN;
	}
	inline const Split build_split(const Entry& e) override {
		Split split = Split::build_unsuccessful_split();
		split.succesful = true;
		double delta_x = e.f - previous_f;
		
		//if (first) {
		//	first = false;
		//	split.succesful = false;
		//}
		if (nl < min_samples_leaf || nr < min_samples_leaf) {
			split.succesful = false;
		}

		if (delta_x < 1e-6) {
			split.succesful = false;
		}

		if (split.succesful) {
			split.column = column;
			split.threshold = 0.5 * (e.f + previous_f);
			split.i = e.i;
			split.l_criterion = sl * sl / nl - s2l;
			split.r_criterion = sr * sr / nr - s2r;
			split.p_criterion = this->p_criterion;
			split.criterion_gain = split.l_criterion + split.r_criterion - split.p_criterion;
			split.l_n = nl;
			split.r_n = nr;
			split.p_n = n;
			split.l_value = sl / nl;
			split.r_value = sr / nr;
			split.p_value = s / n;
			//split.succesful = true;
		}

		// update statistics for next split
		sl += e.y;
		sr -= e.y;
		s2l += e.y * e.y;
		s2r -= e.y * e.y;
		nl++;
		nr--;
		previous_f = e.f;

		return split;
	}
};