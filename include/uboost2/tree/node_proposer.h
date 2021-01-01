#pragma once

#include <stack>
#include <queue>

class NodeProposer {
public:
	virtual void push(size_t nid, double criterion = 0.0) = 0;
	virtual size_t get_and_pop() = 0;
	virtual size_t size() const = 0;
};

class StackNodeProposer : public NodeProposer{
	std::stack<size_t> stack;
public:
	void push(size_t nid, double criterion = 0.0) override {
		stack.push(nid);
	}
	size_t get_and_pop() override {
		size_t nid = stack.top();
		stack.pop();
		return nid;
	}
	size_t size() const override {
		return stack.size();
	}
};

class QueueNodeProposer : public NodeProposer {
	std::queue<size_t> queue;
public:
	void push(size_t nid, double criterion = 0.0) override {
		queue.push(nid);
	}
	size_t get_and_pop() override {
		size_t nid = queue.front();
		queue.pop();
		return nid;
	}
	size_t size() const override {
		return queue.size();
	}
};

class LowerFirstNodeProposer : public NodeProposer {
	std::vector<size_t> ids;
	std::vector<double> scores;
public:
	void push(size_t nid, double criterion = 0.0) override {
		ids.push_back(nid);
		scores.push_back(criterion);
	}
	size_t get_and_pop() override {
		size_t k_best = 0;
		double crit_best = scores[0];
		for (size_t k = 0; k < size(); k++) {
			if (scores[k] < crit_best) {
				crit_best = scores[k];
				k_best = k;
			}
		}
		size_t nid = ids[k_best];
		ids.erase(ids.begin() + k_best);
		scores.erase(scores.begin() + k_best);
		return nid;
	}
	size_t size() const override {
		return ids.size();
	}
};