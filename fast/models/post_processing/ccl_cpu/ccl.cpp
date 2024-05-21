//#pragma once
#include <torch/script.h>
#include <torch/extension.h>

// Include your ported CPU functions
torch::Tensor connected_componnets_labeling_2d(const torch::Tensor &input);
torch::Tensor connected_componnets_labeling_2d_batch(const torch::Tensor &input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("connected_componnets_labeling_2d", &connected_componnets_labeling_2d, "Connected Components Labeling 2D (CPU)");
    m.def("connected_componnets_labeling_2d_batch", &connected_componnets_labeling_2d_batch, "Connected Components Labeling 2D Batch (CPU)");
}
