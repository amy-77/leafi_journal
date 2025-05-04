//
// Created by Qitong Wang on 2023/5/30.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_MODEL_DATASET_H_
#define DSTREE_SRC_MODEL_DATASET_H_

#include <torch/data/example.h>
#include <torch/data/datasets/base.h>

namespace upcite {

class TORCH_API SeriesDataset
    : public torch::data::datasets::Dataset<SeriesDataset> {
 public:
  SeriesDataset(torch::Tensor &series,
                std::vector<VALUE_TYPE> &targets,
                int num_instances,
                torch::Device device) :
      series_(std::move(series.clone())),
      targets_(torch::from_blob(targets.data(),
                                num_instances,
                                torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(device)) {
    assert(series_.size(0) == targets_.size(0));
  }

  SeriesDataset(torch::Tensor &series,
                torch::Tensor &targets) :
      series_(std::move(series.clone())),
      targets_(std::move(targets.clone())) {
    assert(series_.size(0) == targets_.size(0));
  }

  torch::data::Example<> get(size_t index) override {
    return {series_[index], targets_[index]};
  }

  torch::optional<size_t> size() const override {
    return targets_.size(0);
  }

 private:
  torch::Tensor series_, targets_;
};

}

#endif //DSTREE_SRC_MODEL_DATASET_H_
