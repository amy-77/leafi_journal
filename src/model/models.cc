//
// Created by Qitong Wang on 2023/5/16.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "models.h"

#include <string>
#include "spdlog/spdlog.h"

namespace dstree = upcite::dstree;

upcite::MODEL_SETTING upcite::MODEL_SETTING_PLACEHOLDER = upcite::MODEL_SETTING();
std::reference_wrapper<upcite::MODEL_SETTING>
    upcite::MODEL_SETTING_PLACEHOLDER_REF = std::ref(upcite::MODEL_SETTING_PLACEHOLDER);

std::shared_ptr<dstree::FilterModel> dstree::get_model(dstree::Config &config) {
  std::shared_ptr<dstree::FilterModel> model = nullptr;

  std::string model_setting = config.filter_model_setting_str_;
  if (model_setting.compare(0, 3, "cnn") == 0) {
    model = std::make_shared<dstree::BasicCNN>(config);
  } else if (model_setting.compare(0, 3, "rnn") == 0) {
    model = std::make_shared<dstree::BasicRNN>(config);
  } else if (model_setting.compare(0, 3, "mlp") == 0) {
    model = std::make_shared<dstree::BasicMLP>(config);
  } else {
    spdlog::error("{:s} is unsupported", model_setting);
    std::exit(-1);
  }

  return model;
}

VALUE_TYPE dstree::get_memory_footprint(dstree::FilterModel const &model) {
  size_t memory_size = 0;

  for (const auto &parameter : model.parameters()) {
    memory_size += parameter.nbytes() * 2; // x2 inflates to cater the peak
  }

  for (const auto &buffer : model.buffers()) {
    memory_size += buffer.nbytes();
  }

  return static_cast<VALUE_TYPE>(memory_size) / (1024 * 1024);
}
