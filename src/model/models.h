//
// Created by Qitong Wang on 2022/10/24.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_MODEL_H_
#define DSTREE_SRC_EXEC_MODEL_H_

#include <cmath>
#include <vector>
#include <string>

#include <torch/torch.h>
#include "spdlog/spdlog.h"

#include "global.h"
#include "config.h"
#include "vec.h"
#include "str.h"

namespace upcite {

enum MODEL_TYPE {
  MLP = 0,
  CNN = 1,
  RNN = 2,
  SAN = 3, // self-attention networks, i.e., the Transformer variants
  size = 4
};

static std::vector<MODEL_TYPE> MODEL_TYPE_LIST{
    MLP, CNN, RNN, SAN
};

struct MODEL_SETTING {
 public:
  MODEL_SETTING() {
    model_setting_str = "";

    model_type = MLP;
    num_layer = 2;
    layer_size = 256;
    has_skip_connections = false;

    gpu_mem_mb = -1;

    gpu_ms_per_query = -1;
    cpu_ms_per_query = -1;

    pruning_prob = 0;
  };

  explicit MODEL_SETTING(const std::string &setting_str, std::string delim = "_") {
    model_setting_str = setting_str;

    std::vector<std::string> setting_segments = upcite::split_str(setting_str, delim);

#ifdef DEBUG
#ifndef DEBUGGED
    spdlog::debug("model_setting {:d} segments in {:s}",
                  setting_segments.size(), setting_str);
#endif
#endif

    // coding-version_model-type_num-layer_dim-layer_skip-connected, e.g., v0_mlp_3_256_f
    if (setting_segments[0][1] == '0') { // version
      if (setting_segments[1] == "mlp") {
        model_type = MLP;

        num_layer = std::stol(setting_segments[2]);
        layer_size = std::stol(setting_segments[3]);
        has_skip_connections = setting_segments[4] == "t";
      } else {
        goto default_branch;  // default
      }
    } else {
      default_branch: // default

      model_type = MLP;
      num_layer = 2;
      layer_size = 256;
      has_skip_connections = false;
    }

    gpu_mem_mb = -1;
    gpu_ms_per_query = -1;
    cpu_ms_per_query = -1;

    pruning_prob = 0;
  };

  ~MODEL_SETTING() = default;

  std::string model_setting_str;

  MODEL_TYPE model_type;
  ID_TYPE num_layer;
  ID_TYPE layer_size;
  bool has_skip_connections;

  VALUE_TYPE gpu_mem_mb;
  double_t gpu_ms_per_query;
  double_t cpu_ms_per_query;

  VALUE_TYPE pruning_prob;
};

extern MODEL_SETTING MODEL_SETTING_PLACEHOLDER;
extern std::reference_wrapper<MODEL_SETTING> MODEL_SETTING_PLACEHOLDER_REF;

namespace dstree {

class FilterModel : public torch::nn::Module {
 public:
  ~FilterModel() override {};

  virtual torch::Tensor forward(torch::Tensor &x) = 0;

  MODEL_TYPE model_type_;
};

//class BasicRNN(nn.Module):
//def __init__(self, input_size=1, hidden_size=256, output_size=1):
//super(BasicRNN, self).__init__()
//self.hidden_size = hidden_size
//self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
//self.activation = nn.LeakyReLU(0.1)
//self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
//self.fc = nn.Linear(hidden_size, output_size)
//
//def forward(self, x):
//out, _ = self.rnn1(x)
//out = self.activation(out)
//out, _ = self.rnn2(out)
//out = self.fc(out[:, -1, :])  # Use the last output for regression
//return out.squeeze()

class BasicRNN : public FilterModel {
 public:
  explicit BasicRNN(dstree::Config &config) :
      lstm1_(torch::nn::LSTM(torch::nn::LSTMOptions(1, config.filter_rnn_hidden_dim_).batch_first(true))),
      lstm2_(torch::nn::LSTM(torch::nn::LSTMOptions(config.filter_rnn_hidden_dim_,
                                                    config.filter_rnn_hidden_dim_).batch_first(true))),
      activate_(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(config.filter_leaky_relu_negative_slope_))),
      fc1_(torch::nn::Linear(config.filter_rnn_hidden_dim_, 1)) {

    register_module("lstm1", lstm1_);
    register_module("lkrelu", activate_);
    register_module("lstm2", lstm2_);
    register_module("fc", fc1_);

    model_type_ = RNN;
  }

  torch::Tensor forward(torch::Tensor &x) override {
    auto x1 = x.unsqueeze(2);

    auto o1 = std::get<0>(lstm1_->forward(x1));
    auto z1 = activate_->forward(o1);

    auto o2 = std::get<0>(lstm2_->forward(z1));
    auto z2 = activate_->forward(o2.index({torch::indexing::Slice(), -1, torch::indexing::Slice()}));

    auto a3 = fc1_->forward(z2);
    return at::squeeze(a3);
  }

 private:
  torch::nn::LSTM lstm1_{nullptr}, lstm2_{nullptr};
  torch::nn::LeakyReLU activate_{nullptr};
  torch::nn::Linear fc1_{nullptr};
};

class BasicCNN : public FilterModel {
 public:
  explicit BasicCNN(dstree::Config &config) :
      negative_slope_(config.filter_leaky_relu_negative_slope_) {
    conv1_ = register_module("conv1", torch::nn::Conv1d(torch::nn::Conv1dOptions(
        1, config.filter_cnn_num_channel_, config.filter_cnn_kernel_size_).padding(
        config.filter_cnn_kernel_size_ / 2)));
    conv2_ = register_module("conv2", torch::nn::Conv1d(torch::nn::Conv1dOptions(
        config.filter_cnn_num_channel_, config.filter_cnn_num_channel_, config.filter_cnn_kernel_size_).padding(
        config.filter_cnn_kernel_size_ / 2)));

    pool_ = register_module("pool", torch::nn::AvgPool1d(config.series_length_));

    activate_ = register_module("lkrelu", torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(negative_slope_)));

    fc1_ = register_module("fc1", torch::nn::Linear(config.filter_cnn_num_channel_, 1));

    model_type_ = CNN;
  }

  torch::Tensor forward(torch::Tensor &x) override {
    auto x1 = x.unsqueeze(1);

    auto a1 = conv1_->forward(x1);
    auto z1 = activate_->forward(a1);

    auto a2 = conv2_->forward(z1);
    auto p2 = at::squeeze(pool_->forward(a2));
    auto z2 = activate_->forward(p2);

    auto a3 = fc1_->forward(z2);
    return at::squeeze(a3);
  }

 private:
  VALUE_TYPE negative_slope_;

  torch::nn::Conv1d conv1_{nullptr}, conv2_{nullptr};
  torch::nn::Linear fc1_{nullptr};

  torch::nn::LeakyReLU activate_{nullptr};
  torch::nn::AvgPool1d pool_{nullptr};
};

class BasicMLP : public FilterModel {
 public:
  explicit BasicMLP(dstree::Config &config) :
      dropout_p_(config.filter_train_dropout_p_),
      negative_slope_(config.filter_leaky_relu_negative_slope_) {
    fc1_ = register_module("fc1", torch::nn::Linear(config.series_length_, config.filter_dim_latent_));
//    fc2 = register_module("fc2", torch::nn::Linear(dim_latent, dim_latent));
    fc3_ = register_module("fc3", torch::nn::Linear(config.filter_dim_latent_, 1));

    activate_ = register_module("lkrelu", torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(negative_slope_)));

    model_type_ = MLP;
  }

  torch::Tensor forward(torch::Tensor &x) override {
    auto a1 = fc1_->forward(x);
    auto z1 = activate_->forward(a1);
//    x = torch::dropout(x, dropout_p_, is_training());

//    x = activate_->forward(fc2->forward(x));
//    x = torch::dropout(x, dropout_p_, is_training());

    auto a3 = fc3_->forward(z1);
    return at::squeeze(a3);
  }

 private:
  VALUE_TYPE dropout_p_, negative_slope_;

//  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  torch::nn::Linear fc1_{nullptr}, fc3_{nullptr};

  torch::nn::LeakyReLU activate_{nullptr};
//  torch::nn::Softplus activate_{nullptr};
//  torch::nn::Sigmoid activate_;
//  torch::nn::Tanh activate_;
};

std::shared_ptr<FilterModel> get_model(Config &config);

VALUE_TYPE get_memory_footprint(FilterModel const &model);

}
}

#endif //DSTREE_SRC_EXEC_MODEL_H_
