//
// Created by Qitong Wang on 2023/2/10.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "global.h"

namespace constant = upcite::constant;

torch::Tensor constant::TENSOR_PLACEHOLDER = at::empty(0);
std::reference_wrapper<torch::Tensor> constant::TENSOR_PLACEHOLDER_REF = std::ref(constant::TENSOR_PLACEHOLDER);
