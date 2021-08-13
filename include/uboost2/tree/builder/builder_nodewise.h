#pragma once

#include <uboost2/tree/builder/builder.h>
#include <uboost2/tree/node_proposer.h>
#include <uboost2/tree/column_proposer.h>
#include <uboost2/tree/entry.h>

class SplitTreeBuilder : public NodeWiseTreeBuilder {
	const DMatrix<>& x;
	ColumnProposer *column_proposer;
	size_t nrows, ncols;
	EntryMatrix entries;
	double y_mean;
	//
	size_t min_samples_leaf, min_samples_split;
	double colsample_bytree, colsample_bylevel;
	//
	struct range {
		size_t start;
		size_t end;
	};
	std::vector<std::vector<range>> positions;
public:
	SplitTreeBuilder(const DMatrix<>& x, const DColumn<>& y, size_t min_samples_leaf = 1, size_t min_samples_split = 2,
		double colsample_bytree = 1.0, double colsample_bylevel = 1.0) : x{ x }, entries{ x, y } {
		this->nrows = x.nrows();
		this->ncols = x.ncols();
		y_mean = y.sum() / nrows;
		entries.sort_columns();
		this->min_samples_leaf = min_samples_leaf;
		this->min_samples_split = min_samples_split;
		this->colsample_bytree = colsample_bytree;
		this->colsample_bylevel = colsample_bylevel;
		//
		this->column_proposer = new ColumnProposer(ncols, this->colsample_bytree, this->colsample_bylevel);
		this->node_proposer = new LowerFirstNodeProposer();
	}
	~SplitTreeBuilder() {
		delete this->node_proposer;
	}
protected:
	void init(Tree& tree) {
		size_t max_depth = tree.get_max_depth();
		size_t size = trees::min_idx_at_depth(max_depth) + 1;
		positions.resize(size);
		for (size_t nid = 0; nid < size; nid++) {
			for (size_t col = 0; col < ncols; col++) {
				positions[nid].push_back(range{ 0, nrows });
			}
		}
		node_proposer->push(trees::ROOTID);
	}
	void expand_node(Tree& tree, size_t nid) {
		const auto columns = column_proposer->get_columns();
		std::vector<range>& position = this->positions[nid];

		Split best_split = Split::build_unsuccessful_split();
		MSESplitter splitter(this->min_samples_leaf);
		const range& r0 = position[0];
		for (size_t k = r0.start; k < r0.end; k++) {
			const Entry& e = entries(k, 0);
			splitter.add(e);
		}
		for (const auto& col : columns) {
			const range& r = position[col];
			splitter.start_splitting(col);
			for (size_t k = r.start; k < r.end; k++) {
				const Entry& e = entries(k, col);
				const Split& split = splitter.build_split(e);
				if (!split.succesful) continue;
				if (split > best_split) best_split = split;
			}
		}
		if (!best_split.succesful) return;
		size_t lchild = trees::left_child(nid);
		size_t rchild = trees::right_child(nid);
		

		// update tree
		tree[nid].is_leaf = false;
		tree[nid].column = best_split.column;
		tree[nid].threshold = best_split.threshold;
		tree[nid].value = best_split.p_value;
		tree[nid].criterion = best_split.p_criterion;
		tree[nid].gain = best_split.criterion_gain;
		tree[nid].n = best_split.p_n;

		tree.init_node_as_leaf(lchild);
		tree[lchild].value = best_split.l_value;
		tree[lchild].criterion = best_split.l_criterion;
		tree[lchild].n = best_split.l_n;

		tree.init_node_as_leaf(rchild);
		tree[rchild].value = best_split.r_value;
		tree[rchild].criterion = best_split.r_criterion;
		tree[rchild].n = best_split.r_n;

		if (trees::depth_at_idx(nid) >= tree.get_max_depth() - 1) return;
		// update splits
		for (size_t col = 0; col < ncols; col++) {
			const range& p_range = position[col];
			range& l_range = positions[lchild][col];
			range& r_range = positions[rchild][col];

			l_range.start = p_range.start;
			l_range.end = p_range.start + best_split.l_n;
			r_range.start = p_range.start + best_split.l_n;
			r_range.end = p_range.end;
			const auto& x = this->x;
			if (col != best_split.column) {
				std::stable_partition(
					&entries(p_range.start, col),
					&entries(p_range.end, col),
					[x, best_split](const Entry& e) {
						return x(e.i, best_split.column) < best_split.threshold;
					}
				);
			}
		}

		// add nodes
		node_proposer->push(lchild, best_split.l_criterion);
		node_proposer->push(rchild, best_split.r_criterion);
	}
};
