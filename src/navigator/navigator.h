//
// Created by Qitong Wang on 2023/5/30.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_NAVIGATOR_NAVIGATOR_H_
#define DSTREE_SRC_NAVIGATOR_NAVIGATOR_H_

#include <vector>
#include <memory>
#include <functional>

#include <torch/torch.h>
#include <spdlog/spdlog.h>

#include "global.h"
#include "config.h"
#include "navigator_core.h"

namespace upcite {
namespace dstree {

class Navigator {
 public:
  Navigator(dstree::Config &config,
            std::vector<ID_TYPE> node_id_map,
            std::reference_wrapper<torch::Tensor> train_queries,
            std::vector<VALUE_TYPE> nn_residence_distributions,
            std::reference_wrapper<torch::Device> device);
  ~Navigator() = default;

  RESPONSE train();
  std::vector<VALUE_TYPE> &infer(torch::Tensor &query_series);

  ID_TYPE get_id_from_pos(ID_TYPE pos) const {
    return node_id_map_[pos];
  }

//  RESPONSE dump(std::ofstream &node_fos) const;
//  RESPONSE load(std::ifstream &node_ifs, void *ifs_buf);

 private:
  std::reference_wrapper<dstree::Config> config_;

  std::shared_ptr<NavigatorCore> model_;

  std::reference_wrapper<torch::Device> device_;

  bool is_trained_;

  std::reference_wrapper<torch::Tensor> train_queries_;

  std::vector<ID_TYPE> node_id_map_;
  std::vector<VALUE_TYPE> nn_residence_distributions_;

  std::vector<VALUE_TYPE> pred_residence_distributions_;
};

}
}

#endif //DSTREE_SRC_NAVIGATOR_NAVIGATOR_H_
