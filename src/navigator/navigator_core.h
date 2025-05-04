//
// Created by Qitong Wang on 2023/5/29.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_NAVIGATOR_NAVIGATOR_CORE_H_
#define DSTREE_SRC_NAVIGATOR_NAVIGATOR_CORE_H_

#include <cmath>
#include <vector>
#include <string>

#include <torch/torch.h>
#include "spdlog/spdlog.h"

#include "global.h"
#include "vec.h"
#include "str.h"

namespace upcite {
namespace dstree {

class NavigatorCore : public torch::nn::Module {
 public:
  ~NavigatorCore() override = default;

  virtual torch::Tensor forward(torch::Tensor &x) = 0;
};

class MLPNavigator : public NavigatorCore {
 public:
  MLPNavigator(ID_TYPE dim_in, ID_TYPE dim_out, ID_TYPE dim_latent = 500) {
    fc1_ = register_module("fc1", torch::nn::Linear(dim_in, dim_latent));
    fc2_ = register_module("fc2", torch::nn::Linear(dim_latent, dim_latent));
    fc3_ = register_module("fc3", torch::nn::Linear(dim_latent, dim_out));

    activate_ = register_module("sigmoid", torch::nn::Sigmoid());
  }

  torch::Tensor forward(torch::Tensor &x) {
    auto a1 = fc1_->forward(x);
    auto z1 = activate_->forward(a1);

    auto a2 = fc2_->forward(z1);
    auto z2 = activate_->forward(a2);

    auto a3 = fc3_->forward(z2);
//    auto z3 = activate_->forward(a3);

    return at::squeeze(a3);
  }

 private:
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, fc3_{nullptr};

  torch::nn::Sigmoid activate_{nullptr};
};

}
}

#endif //DSTREE_SRC_NAVIGATOR_NAVIGATOR_CORE_H_
