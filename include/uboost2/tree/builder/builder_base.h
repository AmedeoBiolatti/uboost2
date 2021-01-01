#pragma once

#include <uboost2/tree/builder/builder.h>

class BaseTreeBuilder : public NodeWiseTreeBuilder {
	EntryMatrix entries;
	std::vector<size_t> position;
	double y_mean;
	size_t nrows, ncols;
	//
	size_t min_samples_split = 2;
	size_t min_samples_leaf = 1;
	//
public:
	BaseTreeBuilder(const DMatrix<>& x, const DColumn<>& y) : entries{ x, y } {
		nrows = x.nrows();
		ncols = x.ncols();
		y_mean = y.sum() / nrows;
		node_proposer = new LowerFirstNodeProposer();
		entries.sort_columns();
	}
protected:
	void init(Tree& tree) override {
		assert(tree[trees::ROOTID].is_leaf);
		node_proposer->push(trees::ROOTID);
		tree[trees::ROOTID].value = y_mean;
		tree[trees::ROOTID].n = nrows;
		this->position.clear();
		this->position.resize(nrows, trees::ROOTID);
	}
	void expand_node(Tree& tree, size_t nid) override {

		if (tree[nid].n < min_samples_split || tree[nid].n < 2 * min_samples_leaf) {
			return;
		}

		MSESplitter splitter;
		const auto& ecol = DColumn<Entry>(entries, 0);
		for (auto e : ecol) if (position[e.i] == nid) {
			splitter.add(e);
		}

		Split best_split = Split::build_unsuccessful_split();
		for (size_t col = 0; col < ncols; col++) {
			auto entry_column = DColumn<Entry>(entries, col);
			splitter.start_splitting(col);
			for (const Entry& e : entry_column) if (position[e.i] == nid) {
				const Split candidate_split = splitter.build_split(e);
				if (!candidate_split.succesful) continue;
				if (candidate_split > best_split) best_split = candidate_split;
			}
		}

		if (!best_split.succesful) {
			return;
		}

		tree[nid].is_leaf = false;
		tree[nid].column = best_split.column;
		tree[nid].threshold = best_split.threshold;
		tree[nid].value = best_split.p_value;
		tree[nid].criterion = best_split.p_criterion;
		tree[nid].gain = best_split.criterion_gain;
		tree[nid].n = best_split.p_n;

		size_t lchild = trees::left_child(nid);
		size_t rchild = trees::right_child(nid);

		tree.init_node_as_leaf(lchild);
		tree[lchild].value = best_split.l_value;
		tree[lchild].criterion = best_split.l_criterion;
		tree[lchild].n = best_split.l_n;

		tree.init_node_as_leaf(rchild);
		tree[rchild].value = best_split.r_value;
		tree[rchild].criterion = best_split.r_criterion;
		tree[rchild].n = best_split.r_n;

		node_proposer->push(lchild, best_split.l_criterion);
		node_proposer->push(rchild, best_split.r_criterion);

		// update position
		auto entry_column = DColumn<Entry>(entries, best_split.column);
		for (const Entry& e : entry_column) {
			if (position[e.i] == nid) {
				if (e.f >= best_split.threshold) position[e.i] = rchild;
				else position[e.i] = lchild;
			}
		}
	}
};
