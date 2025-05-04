//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_GLOBAL_H
#define DSTREE_GLOBAL_H

#include <string>

#include <ATen/ATen.h>
#include <torch/types.h>

#define DEBUG
#define DEBUGGED

typedef int64_t ID_TYPE;
typedef float VALUE_TYPE;
// match GSL and avoid extra conversions
typedef double ERROR_TYPE;

constexpr auto TORCH_VALUE_TYPE = torch::kFloat32;

enum RESPONSE {
  SUCCESS = 0,
  FAILURE = 1
};

namespace upcite {
namespace constant {

const VALUE_TYPE EPSILON = 1e-5;
const VALUE_TYPE EPSILON_GAP = 1e-4;

const VALUE_TYPE MIN_VALUE = -1e6;
const VALUE_TYPE MAX_VALUE = 1e6;

const ID_TYPE MAX_ID = 9999999999;

const VALUE_TYPE PI_APPROX_7 = 3.1415926;

const ID_TYPE STR_DEFAULT_SIZE = 65536;

const std::string LOGGER_NAME = "file_logger_mt";

extern torch::Tensor TENSOR_PLACEHOLDER;
extern std::reference_wrapper<torch::Tensor> TENSOR_PLACEHOLDER_REF;

}
}

#endif //DSTREE_GLOBAL_H
