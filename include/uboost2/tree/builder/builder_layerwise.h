#pragma once

#include <unordered_map>

#include <uboost2/tree/builder/builder.h>


class LayerWiseTreeBuilder : public TreeBuilder {
	const DMatrix<>& x;
	EntryMatrix entries;
	size_t nrows, ncols;
	double y_mean;
	std::vector<int> position;
	std::vector<size_t> nodes;
protected:
	void init(Tree& tree) {
		tree[trees::ROOTID].value = y_mean;
		position.clear();
		position.resize(nrows, trees::ROOTID);
		nodes.reserve(100);
		nodes.push_back(trees::ROOTID);
	}
public:
	LayerWiseTreeBuilder(const DMatrix<>& x, const DColumn<>& y) : entries{ x, y }, x{ x } {
		nrows = x.nrows();
		ncols = x.ncols();
		y_mean = y.sum() / nrows;
		entries.sort_columns();
	}
	void update(Tree& tree) override {
		assert(tree[trees::ROOTID].is_leaf);

		//std::unordered_map<size_t, Split> best_splits;
		std::vector<Split> best_splits;
		best_splits.resize(trees::max_idx_at_depth(tree.get_max_depth()), Split::build_unsuccessful_split());
		//std::unordered_map<size_t, MSESplitter> splitters;
		std::vector<MSESplitter> splitters;
		splitters.resize(trees::max_idx_at_depth(tree.get_max_depth()));
		
		init(tree);
		for (size_t curr_depth = 0; curr_depth < tree.get_max_depth(); curr_depth++) {
			if (nodes.size() == 0) break;

			for (const auto& e : DColumn<Entry>(entries, 0)) {
				if (position[e.i] >= 0) {
					splitters[position[e.i]].add(e);
				}
			}

			// search splits
			for (size_t col = 0; col < ncols; col++) {
				for (auto nid : nodes) splitters[nid].start_splitting(col);
				for (size_t i = 0; i < nrows; i++){
					const auto& e = entries(i, col);
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
				if (x(i, split.column) >= split.threshold) position[i] = trees::right_child(nid);
				else position[i] = trees::left_child(nid);
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
			for (auto parent : nodes_old) if (best_splits[parent].succesful) {
				nodes.push_back(trees::right_child(parent));
				nodes.push_back(trees::left_child(parent));
			}
		}

	}
};