//
// Created by Qitong Wang on 2024/3/26.
// Copyright (c) 2024 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_QUERY_SYNTHESIZER_H_
#define DSTREE_SRC_EXEC_QUERY_SYNTHESIZER_H_

#include <vector>
#include <map>
#include <tuple>

#include "global.h"
#include "config.h"
#include "models.h"
#include "node.h"

namespace upcite {
namespace dstree {

class Synthesizer {
 public:
  explicit Synthesizer(Config &config,
                       ID_TYPE num_leaves);
  ~Synthesizer() = default;

  RESPONSE push_node(Node &leaf_node);

  RESPONSE generate_global_data(VALUE_TYPE *generated_queries);
  RESPONSE generate_local_data();

 private:
  std::reference_wrapper<Config> config_;

  ID_TYPE num_leaves_;
  std::vector<std::reference_wrapper<Node>> leaves_;
  std::vector<ID_TYPE> accumulated_leaf_sizes_;
};

} // namespace dstree
} // namespace upcite

#endif //DSTREE_SRC_EXEC_QUERY_SYNTHESIZER_H_
