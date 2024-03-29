#pragma once

#include <pybind11/pybind11.h>

#include <uboost2/numpy_utils.h>

#include <uboost2/tree/tree.h>
#include <uboost2/tree/builder/builder_layerwise.h>
#include <uboost2/tree/builder/builder_layerwise_gh.h>
#include <uboost2/tree/builder/builder_base.h>


namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
	m.doc() = "A python module";

	py::class_<DMatrix<>>(m, "DMatrix");
	m.def("numpyToDMatrix", &numpyToDMatrix, "...");

	py::class_<DColumn<>>(m, "DColumn");
	m.def("numpyToDColumn", &numpyToDColumn, "...");
	m.def("DColumntoNumpyInplace", &DColumntoNumpyInplace, "...");

	// standard decision tree & builders
	py::class_<TreeNode>(m, "TreeNode")
		.def_readwrite("is_leaf", &TreeNode::is_leaf)
		.def_readwrite("value", &TreeNode::value)
		.def_readwrite("column", &TreeNode::column)
		.def_readwrite("threshold", &TreeNode::threshold)
		.def_readwrite("criterion", &TreeNode::criterion)
		.def_readwrite("gain", &TreeNode::gain)
		.def_readwrite("n", &TreeNode::n)
		;

	py::class_<Tree>(m, "Tree")
		.def(py::init<size_t>(), py::arg("max_depth")=10)
		.def("predict_value", &Tree::predict_value)
		.def("predict_leaf", py::overload_cast<const DMatrix<double>&, size_t>(&Tree::predict_leaf, py::const_))
		.def("get_node", &Tree::get_node)
		;

	py::class_<LayerWiseTreeBuilder>(m, "LayerWiseTreeBuilder")
		.def(py::init<const DMatrix<>&, const DColumn<>&, size_t, size_t, double, double, double, double, double>(),
			py::arg("x"), py::arg("y"),
			py::arg("min_samples_leaf") = 1,
			py::arg("min_samples_split") = 2,
			py::arg("min_weight_leaf") = 0.0,
			py::arg("min_weight_split") = 0.0,
			py::arg("colsample_bytree") = 1.0, 
			py::arg("colsample_bylevel") = 1.0,
			py::arg("reg_alpha") = 0.0
			)
		.def("update", &LayerWiseTreeBuilder::update)
		;

	py::class_<BaseTreeBuilder>(m, "BaseTreeBuilder")
		.def(py::init<const DMatrix<>&, const DColumn<>&>())
		.def("update", &BaseTreeBuilder::update)
		;

	py::class_<GHLayerWiseTreeBuilder>(m, "GHLayerWiseTreeBuilder")
		.def(py::init<const DMatrix<>&, const DColumn<>&, const DColumn<>&, size_t, size_t, double, double, double, double, double, double>(),
			py::arg("x"), py::arg("g"), py::arg("h"), 
			py::arg("min_samples_leaf") = 1, py::arg("min_samples_split") = 2, 
			py::arg("min_weight_leaf") = 0.0, py::arg("min_weight_split") = 0.0,
			py::arg("colsample_bytree") = 1.0, py::arg("colsample_bylevel") = 1.0,
			py::arg("reg_lambda")=1.0, py::arg("reg_alpha")=0.0)
		.def("update", &GHLayerWiseTreeBuilder::update)
		;

}
