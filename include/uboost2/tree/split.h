#pragma once

class Split {
public:
	bool succesful;
	size_t column;
	double threshold;
	size_t i;
	//
	double criterion_gain;
	double l_criterion = NAN, r_criterion = NAN, p_criterion = NAN;
	size_t l_n = 0, r_n = 0, p_n = 0;
	double l_value = NAN, r_value = NAN, p_value = NAN;

	static Split build_unsuccessful_split() {
		Split s;
		s.succesful = false;
		s.criterion_gain = -INFINITY;
		return s;
	}
	Split() {
		this->succesful = false;
		this->criterion_gain = -INFINITY;
	}
	operator bool() const {
		return this->succesful;
	}
	bool operator<(const Split& s) const {
		return criterion_gain < s.criterion_gain;
	}
	bool operator>(const Split& s) const {
		return criterion_gain > s.criterion_gain;
	}
};
