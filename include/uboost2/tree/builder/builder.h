#pragma once

#include <vector>
#include <iostream>

#include <uboost2/data.h>
#include <uboost2/tree/entry.h>
#include <uboost2/tree/tree.h>
#include <uboost2/tree/splitter.h>
#include <uboost2/tree/node_proposer.h>

class TreeBuilder {
public:
	virtual void update(Tree& tree) = 0;
};

class NodeWiseTreeBuilder : public TreeBuilder{
protected:
	size_t max_leaves = 9999999;
	size_t max_iter = 99999;
	size_t iter = 0;
	NodeProposer* node_proposer = nullptr;
public:
	void update(Tree& tree) {
		init(tree);
		iter = 0;
		while (node_proposer->size() > 0) {
			if (iter++ > max_iter) break;
			size_t nid = node_proposer->get_and_pop();

			if (tree.get_n_leaves() >= max_leaves) break;
			if (trees::depth_at_idx(nid) >= tree.get_max_depth()) {
				tree[nid].is_leaf = true;
				continue;
			}

			expand_node(tree, nid);
		}
	}
protected:
	virtual void init(Tree& tree) = 0;
	virtual void expand_node(Tree& tree, size_t nid) = 0;
};
