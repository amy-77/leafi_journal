//
// Created by Qitong Wang on 2022/10/24.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_UTIL_STR_H_
#define DSTREE_SRC_UTIL_STR_H_

#include <string>
#include <vector>

#include "global.h"
#include "interval.h"

namespace upcite {

template<class T>
static std::string array2str(T *values, ID_TYPE length) {
  return std::accumulate(values + 1,
                         values + length,
                         std::to_string(values[0]),
                         [](const std::string &a, T b) {
                           return a + " " + std::to_string(b);
                         });
}

// credit: https://stackoverflow.com/a/37454181
static std::vector<std::string> split_str(const std::string &str, const std::string &delim) {
  std::vector<std::string> tokens;
  size_t prev = 0, pos = 0;
  do {
    pos = str.find(delim, prev);
    if (pos == std::string::npos) pos = str.length();
    std::string token = str.substr(prev, pos - prev);
    if (!token.empty()) tokens.push_back(token);
    prev = pos + delim.length();
  } while (pos < str.length() && prev < str.length());
  return tokens;
}

}

#endif //DSTREE_SRC_UTIL_STR_H_
