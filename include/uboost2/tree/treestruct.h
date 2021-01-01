#pragma once

namespace trees {

	constexpr size_t ROOTID = 0;

	inline size_t min_idx_at_depth(size_t d) {
		return std::pow(2.0, d) - 1;
	}

	inline size_t max_idx_at_depth(size_t d) {
		return std::pow(2.0, d + 1) - 2;
	}

	inline size_t depth_at_idx(size_t idx) {
		return std::floor(std::log2(idx + 1));
	}

	inline size_t left_child(size_t idx) {
		return 2 * idx + 1;
	}

	inline size_t right_child(size_t idx) {
		return left_child(idx) + 1;
	}
}
