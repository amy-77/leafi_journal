//
// Created by Qitong Wang on 2022/10/19.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_COMMON_H_
#define DSTREE_SRC_EXEC_COMMON_H_

#include <vector>

#include "global.h"

namespace upcite {

template<class T>
std::vector<T> make_reserved(const ID_TYPE n) {
  std::vector<T> v;
  v.reserve(n);
  return v;
}

}

#endif //DSTREE_SRC_EXEC_COMMON_H_
