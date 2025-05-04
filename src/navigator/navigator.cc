//
// Created by Qitong Wang on 2023/5/30.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "navigator.h"

#include <utility>

#include <tuple>

#include "dataset.h"
#include "scheduler.h"

namespace dstree = upcite::dstree;

dstree::Navigator::Navigator(dstree::Config &config,
                             std::vector<ID_TYPE> node_id_map,
                             std::reference_wrapper<torch::Tensor> train_queries,
                             std::vector<VALUE_TYPE> nn_residence_distributions,
                             std::reference_wrapper<torch::Device> device) :
    config_(config),
    is_trained_(false),
    node_id_map_(std::move(node_id_map)),
    train_queries_(train_queries),
    nn_residence_distributions_(std::move(nn_residence_distributions)),
    device_(device) {
  pred_residence_distributions_ = make_reserved<VALUE_TYPE>(node_id_map_.size());
}

RESPONSE dstree::Navigator::train() {
  assert(nn_residence_distributions_.size() % node_id_map_.size() == 0);
  ID_TYPE nnode = node_id_map_.size();
  ID_TYPE train_size = nn_residence_distributions_.size() / node_id_map_.size();

#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("navigator nnode = {:d}", nnode);
  spdlog::debug("navigator train_size = {:d}", train_size);
  spdlog::debug("navigator config_.get().filter_train_nexample_ = {:d}", config_.get().filter_train_nexample_);
#endif
#endif

  torch::Tensor train_target_tsr = torch::from_blob(nn_residence_distributions_.data(),
                                                    {train_size, nnode},
                                                    torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(device_);

  ID_TYPE num_train_examples = train_size * config_.get().navigator_train_val_split_;

  auto train_data = train_queries_.get().index({torch::indexing::Slice(0, num_train_examples)}).clone();
  auto train_target = train_target_tsr.index({torch::indexing::Slice(0, num_train_examples)}).clone();
  auto train_dataset = upcite::SeriesDataset(train_data, train_target);
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      train_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);

  ID_TYPE num_valid_examples = train_size - num_train_examples;
  auto valid_data = train_queries_.get().index({torch::indexing::Slice(num_train_examples, train_size)}).clone();
  auto valid_target = train_target_tsr.index({torch::indexing::Slice(num_train_examples, train_size)}).clone();
  auto valid_dataset = upcite::SeriesDataset(valid_data, valid_target);
  auto valid_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      valid_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);

#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("navigator train_data = [{:d}, {:d}]", train_data.size(0), train_data.size(1));
  spdlog::debug("navigator train_target = [{:d}, {:d}]", train_target.size(0), train_target.size(1));
  spdlog::debug("navigator train_dataset = [{:d}]", train_dataset.size().value());
  spdlog::debug("navigator valid_data = [{:d}, {:d}]", valid_data.size(0), valid_data.size(1));
  spdlog::debug("navigator valid_target = [{:d}, {:d}]", valid_target.size(0), valid_target.size(1));
  spdlog::debug("navigator valid_dataset = [{:d}]", valid_dataset.size().value());
#endif
#endif

  model_ = std::make_shared<dstree::MLPNavigator>(config_.get().series_length_, nnode);
  model_->to(device_);

  torch::optim::SGD optimizer(model_->parameters(), config_.get().filter_train_learning_rate_);
  upcite::optim::ReduceLROnPlateau lr_scheduler = upcite::optim::ReduceLROnPlateau(optimizer);

  torch::nn::MSELoss mse_loss(torch::nn::MSELossOptions().reduction(torch::kMean));

#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("navigator config_.get().filter_train_nepoch_ = {:d}", config_.get().filter_train_nepoch_);
  spdlog::debug("navigator num_train_examples = {:d}", num_train_examples);
  spdlog::debug("navigator config_.get().filter_train_batchsize_ = {:d}", config_.get().filter_train_batchsize_);
  spdlog::debug("navigator num_train_examples / config_.get().filter_train_batchsize_ + 1 = {:d}",
                num_train_examples / config_.get().filter_train_batchsize_ + 1);
#endif
#endif

#ifdef DEBUG
  std::vector<float> train_losses, valid_losses, batch_train_losses;
  train_losses.reserve(config_.get().filter_train_nepoch_);
  batch_train_losses.reserve(num_train_examples / config_.get().filter_train_batchsize_ + 1);

  valid_losses.reserve(config_.get().filter_train_nepoch_);
#endif

  torch::Tensor batch_data, batch_target;
  for (ID_TYPE epoch = 0; epoch < config_.get().filter_train_nepoch_; ++epoch) {
    model_->train();

    for (auto &batch : *train_data_loader) {
      batch_data = batch.data;
      batch_target = batch.target;

      optimizer.zero_grad();

      torch::Tensor prediction = model_->forward(batch_data);

      torch::Tensor loss = mse_loss->forward(prediction, batch_target);
      loss.backward();

      if (config_.get().filter_train_clip_grad_) {
        auto norm = torch::nn::utils::clip_grad_norm_(model_->parameters(),
                                                      config_.get().filter_train_clip_grad_max_norm_,
                                                      config_.get().filter_train_clip_grad_norm_type_);
      }
      optimizer.step();

#ifdef DEBUG
      batch_train_losses.push_back(loss.detach().item<float>());
#endif
    }

#ifdef DEBUG
    train_losses.push_back(std::accumulate(batch_train_losses.begin(), batch_train_losses.end(), 0.0)
                               / static_cast<VALUE_TYPE>(batch_train_losses.size()));
    batch_train_losses.clear();
#endif

    { // evaluate
      VALUE_TYPE valid_loss = 0;

      c10::InferenceMode guard;
      model_->eval();

      torch::Tensor prediction = model_->forward(valid_data);

      valid_loss = mse_loss->forward(prediction, valid_target).detach().item<VALUE_TYPE>();

#ifdef DEBUG
      valid_losses.push_back(valid_loss);
#endif
    }
  }

#ifdef DEBUG
  spdlog::debug("navigator t_losses = {:s}",
                upcite::array2str(train_losses.data(), config_.get().filter_train_nepoch_));
  spdlog::debug("navigator v_losses = {:s}",
                upcite::array2str(valid_losses.data(), config_.get().filter_train_nepoch_));
#endif

  return SUCCESS;
}

std::vector<VALUE_TYPE> &dstree::Navigator::infer(torch::Tensor &query_series) {
  c10::InferenceMode guard;

  auto prediction = model_->forward(query_series).detach().cpu();
  auto predictions_array = prediction.detach().cpu().contiguous().data_ptr<VALUE_TYPE>();

  pred_residence_distributions_.insert(pred_residence_distributions_.begin(),
                                       predictions_array,
                                       predictions_array + node_id_map_.size());

  return pred_residence_distributions_;
}
