//
// Created by Qitong Wang on 2023/2/20.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_COMMON_INTERVEL_H_
#define DSTREE_SRC_COMMON_INTERVEL_H_

#include "global.h"

namespace upcite {

struct INTERVAL {
 public:
  INTERVAL(VALUE_TYPE lower_bound, VALUE_TYPE upper_bound) : left_bound_(lower_bound), right_bound_(upper_bound) {};
  ~INTERVAL() = default;

  VALUE_TYPE right_bound_;
  VALUE_TYPE left_bound_;
};

} // namespace upcite

namespace std {

inline std::string to_string(upcite::INTERVAL interval) {
  return "[" + std::to_string(interval.left_bound_) + ", " + std::to_string(interval.right_bound_) + "]";
}

} // namespace std

#endif //DSTREE_SRC_COMMON_INTERVEL_H_
