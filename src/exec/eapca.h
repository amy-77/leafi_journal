//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_EAPCA_H
#define DSTREE_EAPCA_H

#include <vector>
#include <list>
#include <memory>

#include "global.h"
#include "config.h"
#include "split.h"

namespace upcite {
namespace dstree {

class EAPCA {
 public:
  EAPCA(const VALUE_TYPE *series_ptr,
        ID_TYPE series_length,
        ID_TYPE vertical_split_nsegment);
  ~EAPCA() = default;

  VALUE_TYPE get_segment_value(ID_TYPE segment_id,
                               bool is_mean = true) const;
  VALUE_TYPE get_subsegment_value(ID_TYPE subsegment_id,
                                  bool is_mean = true) const;

  RESPONSE split(const Config &config,
                 const Split &split,
                 const std::vector<ID_TYPE> &segment_lengths,
                 const std::vector<ID_TYPE> &subsegment_lengths);

  ID_TYPE nsegment_, nsubsegment_;
  std::list<VALUE_TYPE> segment_means_, segment_stds_;
  std::list<VALUE_TYPE> subsegment_means_, subsegment_stds_;

 private:
  ID_TYPE nvalues_;
  const VALUE_TYPE *series_ptr_;
};

class EAPCAEnvelope {
 public:
  explicit EAPCAEnvelope(const EAPCAEnvelope &eapca_envelope);
  EAPCAEnvelope(const Config &config,
                ID_TYPE nsegment);
  EAPCAEnvelope(const Config &config,
                const EAPCAEnvelope &parent_eapca_envelope,
                const Split &parent_split);

//  EAPCAEnvelope() = default; // only used for loading
  ~EAPCAEnvelope() = default;

  RESPONSE update(const dstree::EAPCA &series_eapca);

  VALUE_TYPE cal_lower_bound_EDsquare(const VALUE_TYPE *series_ptr) const;
  VALUE_TYPE cal_upper_bound_EDsquare(const VALUE_TYPE *series_ptr) const;

  RESPONSE dump(std::ofstream &node_ofs) const;
  RESPONSE load(std::ifstream &node_ifs, void *ifs_buf);

  EAPCAEnvelope &operator=(const EAPCAEnvelope &eapca_envelope) = default;

  ID_TYPE nsegment_, nsubsegment_;
  std::vector<ID_TYPE> segment_lengths_, subsegment_lengths_;

  std::vector<VALUE_TYPE> segment_min_means_, segment_max_means_, segment_min_stds_, segment_max_stds_;
  std::vector<VALUE_TYPE> subsegment_min_means_, subsegment_max_means_, subsegment_min_stds_, subsegment_max_stds_;

 private:
  RESPONSE initialize_stats();
};

}
}

#endif //DSTREE_EAPCA_H
