//
// Created by Qitong Wang on 2022/10/10.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_SPLIT_H_
#define DSTREE_SRC_EXEC_SPLIT_H_

#include <memory>
#include <vector>
#include <tuple>

#include "global.h"
#include "config.h"

namespace upcite {
namespace dstree {

enum HORIZONTAL_SPLIT_MODE {
  MEAN = 0,
  STD = 1
};

class Split {
 public:
  Split() = default;
  ~Split() = default;

  ID_TYPE route(VALUE_TYPE value) const;

  RESPONSE dump(std::ofstream &node_ofs, void *ofs_buf) const;
  RESPONSE load(std::ifstream &node_ifs, void *ifs_buf);

  Split &operator=(const Split &split);

  bool is_vertical_split_;
  ID_TYPE split_segment_id_, split_subsegment_id_;
  ID_TYPE split_segment_offset_, split_segment_length_;

  HORIZONTAL_SPLIT_MODE horizontal_split_mode_;
  std::vector<VALUE_TYPE> horizontal_breakpoints_;
};

}
}

#endif //DSTREE_SRC_EXEC_SPLIT_H_
