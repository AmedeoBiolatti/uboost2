#pragma once

#include <random>

#include <uboost2/tree/builder/builder.h>
#include <uboost2/tree/column_proposer.h>


class LayerWiseTreeBuilder : public TreeBuilder {
	const DMatrix<>& x;
	EntryMatrix* entries;
	size_t nrows, ncols;
	double y_mean;
	std::vector<int> position;
	std::vector<size_t> nodes;
	//
	size_t min_samples_leaf = 1, min_samples_split = 2;
	double min_weight_leaf = 0.0, min_weight_split = 0.0;
	double colsample_bytree = 1.0, colsample_bylevel = 1.0;
	double reg_alpha = 0.0;
protected:
	void init(Tree& tree) {
		tree[trees::ROOTID].value = y_mean;
		position.clear();
		position.resize(nrows, trees::ROOTID);
		nodes.reserve(100);
		nodes.push_back(trees::ROOTID);
	}
public:
	LayerWiseTreeBuilder(const DMatrix<>& x, const DColumn<>& y, 
		size_t min_samples_leaf = 1, size_t min_samples_split = 2,
		double min_weight_leaf = 0.0, double min_weight_split = 0.0,
		double colsample_bytree = 1.0, double colsample_bylevel = 1.0, double reg_alpha=0.0) : entries{ new EntryMatrix(x, y) }, x{ x } {
		nrows = x.nrows();
		ncols = x.ncols();
		y_mean = entries->get_y_mean();
		entries->sort_columns();
		this->min_samples_leaf = min_samples_leaf;
		this->min_samples_split = min_samples_split;
		this->min_weight_leaf = min_weight_leaf;
		this->min_weight_split = min_weight_split;
		this->colsample_bytree = colsample_bytree;
		this->colsample_bylevel = colsample_bylevel;
		this->reg_alpha = reg_alpha;
	}
	LayerWiseTreeBuilder(EntryMatrix& e, size_t min_samples_leaf = 1, size_t min_samples_split = 2,
		double min_weight_leaf = 0.0, double min_weight_split = 0.0,
		double colsample_bytree = 1.0, double colsample_bylevel = 1.0) : entries{ &e }, x{ x } {
		nrows = e.nrows();
		ncols = e.ncols();
		y_mean = entries->get_y_mean();
		entries->sort_columns();
		this->min_samples_leaf = min_samples_leaf;
		this->min_samples_split = min_samples_split;
		this->min_weight_leaf = min_weight_leaf;
		this->min_weight_split = min_weight_split;
		this->colsample_bytree = colsample_bytree;
		this->colsample_bylevel = colsample_bylevel;
	}
	void update(Tree& tree) override {
		assert(tree[trees::ROOTID].is_leaf);

		ColumnProposer column_proposer(ncols, colsample_bytree, colsample_bylevel);

		//std::unordered_map<size_t, Split> best_splits;
		std::vector<Split> best_splits;
		best_splits.resize(trees::max_idx_at_depth(tree.get_max_depth()), Split::build_unsuccessful_split(reg_alpha));
		//std::unordered_map<size_t, MSESplitter> splitters;
		std::vector<MSESplitter> splitters;
		splitters.resize(trees::max_idx_at_depth(tree.get_max_depth()), MSESplitter(min_samples_leaf, min_weight_leaf));
		
		init(tree);
		for (size_t curr_depth = 0; curr_depth < tree.get_max_depth(); curr_depth++) {
			if (nodes.size() == 0) break;
			for (const auto& e : DColumn<Entry>(*entries, 0)) {
				if (position[e.i] >= 0) {
					splitters[position[e.i]].add(e);
				}
			}

			// search splits
			const auto columns = column_proposer.get_columns();
			for (size_t col : columns) {
				for (auto nid : nodes) splitters[nid].start_splitting(col);
				for (size_t i = 0; i < nrows; i++) {
					const auto& e = entries->operator()(i, col);
					const int& nid = position[e.i];
					if (nid < 0) continue;
					const Split& candidate_split = splitters[nid].build_split(e);
					if (!candidate_split.succesful) continue;
					if (candidate_split > best_splits[nid]) best_splits[nid] = candidate_split;
				}
			}
			
			// update position			
			for (size_t i = 0; i < nrows; i++) {
				int nid = position[i];
				if (nid < 0) continue;
				const Split& split = best_splits[nid];
				if (!split.succesful) {
					position[i] = -1;
					continue;
				}	
				if (x(i, split.column) >= split.threshold) {
					if (split.r_n >= this->min_samples_split) position[i] = trees::right_child(nid);
					else position[i] = -1;
				}
				else {
					if (split.l_n >= this->min_samples_split) position[i] = trees::left_child(nid);
					else position[i] = -1;					
				}
			}

			// update tree
			for (auto nid : nodes){
				const Split& split = best_splits[nid];
				if (split.succesful) {
					tree[nid].is_leaf = false;
					tree[nid].column = split.column;
					tree[nid].threshold = split.threshold;
					tree[nid].value = split.p_value;
					tree[nid].criterion = split.p_criterion;
					tree[nid].gain = split.criterion_gain;
					tree[nid].n = split.p_n;

					size_t lchild = trees::left_child(nid);
					size_t rchild = trees::right_child(nid);

					tree.init_node_as_leaf(lchild);
					tree[lchild].value = split.l_value;
					tree[lchild].criterion = split.l_criterion;
					tree[lchild].n = split.l_n;

					tree.init_node_as_leaf(rchild);
					tree[rchild].value = split.r_value;
					tree[rchild].criterion = split.r_criterion;
					tree[rchild].n = split.r_n;
				}
			}

			// update nodes
			std::vector<size_t> nodes_old(nodes);
			nodes.clear();
			for (auto parent : nodes_old) {
				const Split& split = best_splits[parent];
				if (split.succesful) {
					if (split.r_n >= this->min_samples_split && split.r_w >= min_weight_split)
						nodes.push_back(trees::right_child(parent));
					if (split.l_n >= this->min_samples_split && split.l_w >= min_weight_split) 
						nodes.push_back(trees::left_child(parent));
				}
			}
		}

	}
};