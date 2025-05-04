//
// Created by Qitong Wang on 2022/10/10.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "split.h"

#include <fstream>

namespace dstree = upcite::dstree;

ID_TYPE dstree::Split::route(VALUE_TYPE value) const {
  auto horizontal_segment_id = static_cast<ID_TYPE>(horizontal_breakpoints_.size());

  for (ID_TYPE i = 0; i < horizontal_breakpoints_.size(); ++i) {
    if (value < horizontal_breakpoints_[i]) {
      horizontal_segment_id = i;
    }
  }

  return horizontal_segment_id;
}

dstree::Split &dstree::Split::operator=(const dstree::Split &split) {
  is_vertical_split_ = split.is_vertical_split_;

  split_segment_id_ = split.split_segment_id_;
  split_subsegment_id_ = split.split_subsegment_id_;

  horizontal_split_mode_ = split.horizontal_split_mode_;
  horizontal_breakpoints_ = split.horizontal_breakpoints_;

  return *this;
}

RESPONSE dstree::Split::dump(std::ofstream &node_ofs, void *ofs_buf) const {
  node_ofs.write(reinterpret_cast<const char *>(&is_vertical_split_), sizeof(bool));

  node_ofs.write(reinterpret_cast<const char *>(&split_segment_id_), sizeof(ID_TYPE));
  node_ofs.write(reinterpret_cast<const char *>(&split_subsegment_id_), sizeof(ID_TYPE));
  node_ofs.write(reinterpret_cast<const char *>(&split_segment_offset_), sizeof(ID_TYPE));
  node_ofs.write(reinterpret_cast<const char *>(&split_segment_length_), sizeof(ID_TYPE));

  node_ofs.write(reinterpret_cast<const char *>(&horizontal_split_mode_), sizeof(HORIZONTAL_SPLIT_MODE));

  // TODO horizontal_breakpoints_.size() is fixed and redundant
  auto ofs_id_buf = reinterpret_cast<ID_TYPE *>(ofs_buf);
  ofs_id_buf[0] = static_cast<ID_TYPE>(horizontal_breakpoints_.size());
  node_ofs.write(reinterpret_cast<char *>(ofs_id_buf), sizeof(ID_TYPE));

  node_ofs.write(reinterpret_cast<const char *>(horizontal_breakpoints_.data()),
                 sizeof(VALUE_TYPE) * horizontal_breakpoints_.size());

  return SUCCESS;
}

RESPONSE dstree::Split::load(std::ifstream &node_ifs, void *ifs_buf) {
  auto ofs_bool_buf = reinterpret_cast<bool *>(ifs_buf);
  auto ofs_mode_buf = reinterpret_cast<HORIZONTAL_SPLIT_MODE *>(ifs_buf);
  auto ofs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);
  auto ofs_value_buf = reinterpret_cast<VALUE_TYPE *>(ifs_buf);

  ID_TYPE read_nbytes = sizeof(bool);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  is_vertical_split_ = ofs_bool_buf[0];

  // split_segment_id_, split_subsegment_id_, split_segment_offset_, split_segment_length_
  read_nbytes = sizeof(ID_TYPE) * 4;
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  split_segment_id_ = ofs_id_buf[0];
  split_subsegment_id_ = ofs_id_buf[1];
  split_segment_offset_ = ofs_id_buf[2];
  split_segment_length_ = ofs_id_buf[3];

  read_nbytes = sizeof(HORIZONTAL_SPLIT_MODE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  horizontal_split_mode_ = ofs_mode_buf[0];

//  auto ofs_id_buf = reinterpret_cast<ID_TYPE *>(ofs_buf);
//  ofs_id_buf[0] = static_cast<ID_TYPE>(horizontal_breakpoints_.size());
//  node_ofs.write(reinterpret_cast<char *>(ofs_id_buf), sizeof(ID_TYPE));
//  node_ofs.write(reinterpret_cast<const char *>(horizontal_breakpoints_.data()),
//                 sizeof(VALUE_TYPE) * horizontal_breakpoints_.size());

  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  ID_TYPE nbreakpoints = ofs_mode_buf[0];

  read_nbytes = sizeof(VALUE_TYPE) * nbreakpoints;
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  horizontal_breakpoints_.insert(horizontal_breakpoints_.begin(), ofs_value_buf, ofs_value_buf + nbreakpoints);

  return SUCCESS;
}
