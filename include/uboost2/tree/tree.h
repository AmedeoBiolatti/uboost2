#pragma once

#include <vector>
#include <limits>
#include <cassert>

#include <uboost2/tree/treestruct.h>
#include <uboost2/data.h>

constexpr size_t NOCOLUMN = std::numeric_limits<size_t>::max();

struct TreeNode {
	bool is_leaf = true;
	double value = 0.0;
	size_t column = NOCOLUMN;
	double threshold = NAN;
	// stats
	double criterion = NAN;
	double gain = NAN;
	size_t n = 0;
};

class Tree {
	size_t max_depth = 15;

	std::vector<TreeNode> nodes;
protected:
	void reserve(size_t depth) {
		nodes.resize(trees::max_idx_at_depth(depth) + 1);
	}
public:
	void init_node_as_leaf(size_t nid) {
		(*this)[nid] = TreeNode();
	}
	Tree(size_t max_depth=5) {
		assert(max_depth <= 15);
		this->max_depth = max_depth;
		reserve(max_depth);
		init_node_as_leaf(trees::ROOTID);
	}
	//
	inline TreeNode& operator[](size_t nid) {
		return nodes[nid];
	}
	inline const TreeNode& operator[](size_t nid) const {
		return nodes[nid];
	}
	inline TreeNode& get_node(size_t nid) {
		return nodes[nid];
	}
	//
	size_t predict_leaf(const DMatrix<>& x, size_t i) const {
		const auto& xi = DRow<>(x, i);
		return this->predict_leaf(xi);
	}
	size_t predict_leaf(const DRow<>& xi) const {
		size_t nid{ trees::ROOTID };
		while (!nodes[nid].is_leaf) {
			auto xicol = xi(nodes[nid].column);
			if (xicol == NAN) break;
			if (xi(nodes[nid].column) >= nodes[nid].threshold) nid = trees::right_child(nid);
			else nid = trees::left_child(nid);
		}
		return nid;
	}
	//
	double predict_value_row(const DMatrix<>& x, size_t i) const {
		return nodes[predict_leaf(x, i)].value;
	}
	double predict_value_row(const DRow<>& xi) const {
		return nodes[predict_leaf(xi)].value;
	}
	//
	DColumn<> predict_value(const DMatrix<>& x) const {
		DColumn<> out(x.nrows());
		for (size_t i = 0; i < x.nrows(); i++) {
			out(i) = this->predict_value_row(x, i);
		}
		return out;
	}
	//
	size_t get_n_leaves(size_t nid = trees::ROOTID) const {
		if (nodes[nid].is_leaf) return 1;
		return get_n_leaves(trees::left_child(nid)) + get_n_leaves(trees::right_child(nid));
	}
	size_t get_max_depth() const {
		return max_depth;
	}
	//
	void print_i(size_t nid) const {
		size_t depth = trees::depth_at_idx(nid);
		auto node = nodes[nid];
		for (size_t i = 0; i < depth; i++) std::printf("  ");

		std::printf("[%.3d|%.2d]", nid, depth);

		if (node.is_leaf) {
			std::printf(" %d - %.2f - %.4f\n", node.n, node.criterion, node.value);
		}
		else {
			printf("\n");
			print_i(trees::right_child(nid));
			print_i(trees::left_child(nid));
		}
	}
	void print() const {
		print_i(trees::ROOTID);
	}
};
