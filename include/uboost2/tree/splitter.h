#pragma once

#include <uboost2/tree/split.h>
#include <uboost2/tree/entry.h>

class UnarySplitter {
public:
	UnarySplitter() {}
	virtual void add(const Entry& e) = 0;
	virtual void start_splitting(size_t col=0) = 0;
	virtual const Split build_split(const Entry& e) = 0;
};

class BinarySplitter {
public:
	BinarySplitter() {}
	virtual void add(const GHEntry& e) = 0;
	virtual void start_splitting(size_t col = 0) = 0;
	virtual const Split build_split(const GHEntry& e) = 0;
};

//

class MSESplitter : public UnarySplitter {
	size_t min_samples_leaf = 1;
	double min_weight_leaf = 0.0;
	size_t column;
	//
	double s, s2, w;
	size_t n;
	double sl, s2l, wl;
	size_t nl;
	double sr, s2r, wr;
	size_t nr;
	//
	bool first;
	double previous_x;
	double p_criterion;
public:
	MSESplitter(size_t min_samples_leaf = 1, double min_weight_leaf = 0.0) {
		this->min_samples_leaf = min_samples_leaf;
		this->min_weight_leaf = min_weight_leaf;
		s = 0.0;
		s2 = 0.0;
		n = 0;
		w = 0.0;
	}
	void add(const Entry& e) override {
		s += e.y * e.w;
		s2 += e.y * e.y * e.w;
		w += e.w;
		n++;
	}
	void start_splitting(size_t col = 0) override {
		column = col;
		//
		sl = 0.0;
		s2l = 0.0;
		nl = 0;
		wl = 0.0;
		sr = s;
		s2r = s2;
		nr = n;
		wr = w;
		//
		p_criterion = s * s / w - s2;
		first = true;
		previous_x = NAN;
	}
	inline const Split build_split(const Entry& e) override {
		Split split = Split::build_unsuccessful_split();
		split.succesful = true;
		double delta_x = e.x - previous_x;
		
		if (nl < min_samples_leaf || nr < min_samples_leaf) split.succesful = false;
		if (wl < min_weight_leaf || wr < min_weight_leaf) split.succesful = false;

		if (delta_x < 1e-6) split.succesful = false;

		if (split.succesful) {
			split.column = column;
			split.threshold = 0.5 * (e.x + previous_x);
			split.i = e.i;
			split.l_criterion = sl * sl / wl - s2l;
			split.r_criterion = sr * sr / wr - s2r;
			split.p_criterion = this->p_criterion;
			split.criterion_gain = split.l_criterion + split.r_criterion - split.p_criterion;
			split.l_n = nl;
			split.r_n = nr;
			split.p_n = n;
			split.l_value = sl / wl;
			split.r_value = sr / wr;
			split.p_value = s / w;
			split.l_w = wl;
			split.r_w = wr;
			split.p_w = w;
		}

		// update statistics for next split
		sl += e.y * e.w;
		sr -= e.y * e.w;
		s2l += e.y * e.y * e.w;
		s2r -= e.y * e.y * e.w;
		nl++;
		nr--;
		wl += e.w;
		wr -= e.w;
		previous_x = e.x;

		return split;
	}
};



class GHSplitter : public BinarySplitter {
	double G, H;
	double GL, HL, GR, HR;
	double w, wl, wr;
	double reg_lambda = 1.0;
	size_t n, nl, nr;
	size_t column, min_samples_leaf;
	double min_weight_leaf = 0.0;
	double previous_x;
	double p_criterion, p_value;
public:
	GHSplitter(size_t min_samples_leaf = 1, double min_weight_leaf = 0.0) {
		this->min_samples_leaf = min_samples_leaf;
		this->min_weight_leaf = min_weight_leaf;
		G = 0.0;
		H = 0.0;
		n = 0;
		w = 0.0;
	}
	void add(const GHEntry& e) {
		G += e.g * e.w;
		H += e.h * e.w;
		n++;
		w += e.w;
	}
	void start_splitting(size_t col = 0) {
		column = col;
		//
		GL = 0.0;
		GR = G;
		HL = 0.0;
		HR = H;
		nl = 0;
		nr = n;
		wl = 0.0;
		wr = w;
		//
		p_criterion = G * G / (reg_lambda + H);
		p_value = G / (reg_lambda + H);
		previous_x = NAN;
	}
	inline const Split build_split(const GHEntry& e) {
		Split split = Split::build_unsuccessful_split();
		split.succesful = true;
		double delta_x = e.x - previous_x;

		if (nl < this->min_samples_leaf || nr < this->min_samples_leaf) {
			split.succesful = false;
		}
		if (wl < min_weight_leaf || wr < min_weight_leaf) {
			split.succesful = false;
		}
		if (delta_x < 1e-6) {
			split.succesful = false;
		}

		if (split.succesful) {
			split.column = column;
			split.threshold = 0.5 * (e.x + previous_x);
			split.i = e.i;
			split.l_criterion = GL * GL / (reg_lambda + HL);
			split.r_criterion = GR * GR / (reg_lambda + HR);
			split.p_criterion = this->p_criterion;
			split.criterion_gain = split.l_criterion + split.r_criterion - split.p_criterion;
			split.l_n = nl;
			split.r_n = nr;
			split.p_n = n;
			split.l_value = GL / (reg_lambda + HL);
			split.r_value = GR / (reg_lambda + HR);
			split.p_value = p_value;
			split.l_w = wl;
			split.r_w = wr;
			split.p_w = w;
		}

		// update statistics for next split 
		GL += e.g * e.w;
		GR -= e.g * e.w;
		HL += e.h * e.w;
		HR -= e.h * e.w;
		nl++;
		nr--;
		wl += e.w;
		wr -= e.w;
		previous_x = e.x;

		return split;
	}
};
