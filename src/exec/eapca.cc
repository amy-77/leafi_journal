//
// Created by Qitong Wang on 2022/10/10.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "eapca.h"

#include <tuple>
#include <cmath>
#include <fstream>

#include <spdlog/spdlog.h>

#include "stat.h"

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::EAPCA::EAPCA(const VALUE_TYPE *series_ptr,
                     ID_TYPE series_length,
                     ID_TYPE vertical_split_nsegment) :
    nvalues_(series_length),
    series_ptr_(series_ptr) {
#ifdef DEBUG
  // TODO support vertical_split_nsegment > 2
  assert(vertical_split_nsegment == 2);
#endif

  VALUE_TYPE mean = 0, mean_first_half = 0, mean_last_half = 0;
  VALUE_TYPE std = 0, std_first_half = 0, std_last_half = 0;

  ID_TYPE breakpoint = nvalues_ / 2;

  ID_TYPE value_id = 0;
  while (value_id < breakpoint) {
    mean_first_half += series_ptr[value_id];
    value_id += 1;
  }
  while (value_id < nvalues_) {
    mean_last_half += series_ptr[value_id];
    value_id += 1;
  }
  mean = (mean_first_half + mean_last_half) / static_cast<VALUE_TYPE>(nvalues_);
  mean_first_half /= static_cast<VALUE_TYPE>(breakpoint);
  mean_last_half /= static_cast<VALUE_TYPE>(nvalues_ - breakpoint);

  value_id = 0;
  while (value_id < breakpoint) {
    std_first_half += (series_ptr[value_id] - mean_first_half) * (series_ptr[value_id] - mean_first_half);
    std += (series_ptr[value_id] - mean) * (series_ptr[value_id] - mean);
    value_id += 1;
  }
  while (value_id < nvalues_) {
    std_last_half += (series_ptr[value_id] - mean_last_half) * (series_ptr[value_id] - mean_last_half);
    std += (series_ptr[value_id] - mean) * (series_ptr[value_id] - mean);
    value_id += 1;
  }

  std = sqrt(std / static_cast<VALUE_TYPE>(nvalues_));
  std_first_half = sqrt(std_first_half / static_cast<VALUE_TYPE>(nvalues_));
  std_last_half = sqrt(std_last_half / static_cast<VALUE_TYPE>(nvalues_));

  segment_means_.push_back(mean);
  segment_stds_.push_back(std);
  nsegment_ = 1;

  subsegment_means_.push_back(mean_first_half);
  subsegment_means_.push_back(mean_last_half);
  subsegment_stds_.push_back(std_first_half);
  subsegment_stds_.push_back(std_last_half);
  nsubsegment_ = 2;
}

RESPONSE dstree::EAPCA::split(const dstree::Config &config,
                              const Split &split,
                              const std::vector<ID_TYPE> &segment_lengths,
                              const std::vector<ID_TYPE> &subsegment_lengths) {
  // TODO enable non-splittable segment
  // assume all segments could be vertical split, i.e., segment_length >= config->vertical_split_nsubsegment_

  if (split.is_vertical_split_) {
    auto segment_mean_iter = segment_means_.begin();
    std::advance(segment_mean_iter, split.split_segment_id_);

    auto subsegment_mean_iter = subsegment_means_.begin();
    std::advance(subsegment_mean_iter, config.vertical_split_nsubsegment_ * split.split_segment_id_);

    auto segment_std_iter = segment_stds_.begin();
    std::advance(segment_std_iter, split.split_segment_id_);

    auto subsegment_std_iter = subsegment_stds_.begin();
    std::advance(subsegment_std_iter, config.vertical_split_nsubsegment_ * split.split_segment_id_);

    for (ID_TYPE i = 1; i < config.vertical_split_nsubsegment_; ++i) {
      segment_means_.insert(segment_mean_iter, *subsegment_mean_iter);
      subsegment_mean_iter = subsegment_means_.erase(subsegment_mean_iter);

      segment_stds_.insert(segment_std_iter, *subsegment_std_iter);
      subsegment_std_iter = subsegment_stds_.erase(subsegment_std_iter);
    }

    *segment_mean_iter = *subsegment_mean_iter;
    subsegment_mean_iter = subsegment_means_.erase(subsegment_mean_iter);

    *segment_std_iter = *subsegment_std_iter;
    subsegment_std_iter = subsegment_stds_.erase(subsegment_std_iter);

    ID_TYPE value_offset = 0;
    for (ID_TYPE i = 0; i < split.split_segment_id_; ++i) {
      value_offset += segment_lengths[i];
    }

    nsegment_ += config.vertical_split_nsubsegment_ - 1;

#ifdef DEBUG
    assert(nsegment_ == segment_means_.size());
    assert(nsegment_ == segment_stds_.size());
#endif

    VALUE_TYPE mean, std;
    ID_TYPE subsegment_offset = config.vertical_split_nsubsegment_ * split.split_segment_id_;

    for (ID_TYPE subsegment_local_id = 0;
         subsegment_local_id < config.vertical_split_nsubsegment_;
         ++subsegment_local_id) {
      ID_TYPE subsegment_length = subsegment_lengths[subsegment_offset + subsegment_local_id];
      ID_TYPE subsubsegment_length = subsegment_length / config.vertical_split_nsubsegment_;
      ID_TYPE last_subsubsegment_length =
          subsegment_length - subsubsegment_length * (config.vertical_split_nsubsegment_ - 1);

#ifdef DEBUG
      assert(subsubsegment_length > 0);
      assert(last_subsubsegment_length > 0);
#endif

      for (ID_TYPE i = 1; i < config.vertical_split_nsubsegment_; ++i) {
        std::tie(mean, std) = upcite::cal_mean_std(series_ptr_ + value_offset, subsubsegment_length);

        subsegment_means_.insert(subsegment_mean_iter, mean);
        subsegment_stds_.insert(subsegment_std_iter, std);

        value_offset += subsubsegment_length;
      }

      std::tie(mean, std) = upcite::cal_mean_std(series_ptr_ + value_offset, last_subsubsegment_length);

      subsegment_means_.insert(subsegment_mean_iter, mean);
      subsegment_stds_.insert(subsegment_std_iter, std);

      value_offset += last_subsubsegment_length;
    }

    nsubsegment_ += (config.vertical_split_nsubsegment_ - 1) * config.vertical_split_nsubsegment_;

#ifdef DEBUG
    assert(nsubsegment_ == subsegment_means_.size());
    assert(nsubsegment_ == subsegment_stds_.size());
#endif
  }

  return SUCCESS;
}

VALUE_TYPE dstree::EAPCA::get_segment_value(ID_TYPE segment_id, bool is_mean) const {
  if (is_mean) {
    auto mean_iter = segment_means_.cbegin();
    std::advance(mean_iter, segment_id);
    return *mean_iter;
  } else {
    auto std_iter = segment_stds_.cbegin();
    std::advance(std_iter, segment_id);
    return *std_iter;
  }
}

VALUE_TYPE dstree::EAPCA::get_subsegment_value(ID_TYPE subsegment_id, bool is_mean) const {
  if (is_mean) {
    auto mean_iter = subsegment_means_.cbegin();
    std::advance(mean_iter, subsegment_id);
    return *mean_iter;
  } else {
    auto std_iter = subsegment_stds_.cbegin();
    std::advance(std_iter, subsegment_id);
    return *std_iter;
  }
}

dstree::EAPCAEnvelope::EAPCAEnvelope(const dstree::EAPCAEnvelope &eapca_envelope) {
  nsegment_ = eapca_envelope.nsegment_;
  nsubsegment_ = eapca_envelope.nsubsegment_;

  segment_lengths_ = eapca_envelope.segment_lengths_;
  subsegment_lengths_ = eapca_envelope.subsegment_lengths_;

  segment_min_means_ = eapca_envelope.segment_min_means_;
  segment_max_means_ = eapca_envelope.segment_max_means_;
  segment_min_stds_ = eapca_envelope.segment_min_stds_;
  segment_max_stds_ = eapca_envelope.segment_max_stds_;

  subsegment_min_means_ = eapca_envelope.subsegment_min_means_;
  subsegment_max_means_ = eapca_envelope.subsegment_max_means_;
  subsegment_min_stds_ = eapca_envelope.subsegment_min_stds_;
  subsegment_max_stds_ = eapca_envelope.subsegment_max_stds_;
}

dstree::EAPCAEnvelope::EAPCAEnvelope(const Config &config,
                                     ID_TYPE nsegment) {
  nsegment_ = nsegment;
  nsubsegment_ = nsegment_ * config.vertical_split_nsubsegment_;

  ID_TYPE series_length = config.series_length_;
  ID_TYPE segment_length = series_length / nsegment_;

#ifdef DEBUG
  assert(segment_length > config.vertical_split_nsubsegment_);
#endif

  segment_lengths_.assign(nsegment_ - 1, segment_length);

  ID_TYPE subsegment_length = segment_length / config.vertical_split_nsubsegment_;
  ID_TYPE last_subsegment_length = segment_length - subsegment_length * (config.vertical_split_nsubsegment_ - 1);

#ifdef DEBUG
  assert(subsegment_length > 0);
  assert(last_subsegment_length > 0);
#endif

  for (ID_TYPE segment_id = 0; segment_id < nsegment_ - 1; ++segment_id) {
    subsegment_lengths_.insert(subsegment_lengths_.end(), config.vertical_split_nsubsegment_ - 1, subsegment_length);
    subsegment_lengths_.push_back(last_subsegment_length);
  }

  ID_TYPE last_segment_length = series_length - segment_length * (nsegment_ - 1);

#ifdef DEBUG
  assert(last_segment_length > config.vertical_split_nsubsegment_);
#endif

  segment_lengths_.push_back(last_segment_length);

  subsegment_length = last_segment_length / config.vertical_split_nsubsegment_;
  last_subsegment_length = last_segment_length - subsegment_length * (config.vertical_split_nsubsegment_ - 1);

#ifdef DEBUG
  assert(subsegment_length > 0);
  assert(last_subsegment_length > 0);
#endif

  subsegment_lengths_.insert(subsegment_lengths_.end(), config.vertical_split_nsubsegment_ - 1, subsegment_length);
  subsegment_lengths_.push_back(last_subsegment_length);

#ifdef DEBUG
  assert(nsubsegment_ == subsegment_lengths_.size());
#endif

  initialize_stats();
}

dstree::EAPCAEnvelope::EAPCAEnvelope(const Config &config,
                                     const dstree::EAPCAEnvelope &parent_eapca_envelope,
                                     const dstree::Split &parent_split) {
  if (parent_split.is_vertical_split_) {
    nsegment_ = parent_eapca_envelope.nsegment_ + config.vertical_split_nsubsegment_ - 1;
    nsubsegment_ = parent_eapca_envelope.nsubsegment_ +
        +config.vertical_split_nsubsegment_ * (config.vertical_split_nsubsegment_ - 1);

    for (ID_TYPE segment_id = 0; segment_id < parent_eapca_envelope.nsegment_; ++segment_id) {
      if (segment_id == parent_split.split_segment_id_) {
        for (ID_TYPE subsegment_id = segment_id * config.vertical_split_nsubsegment_;
             subsegment_id < (segment_id + 1) * config.vertical_split_nsubsegment_;
             ++subsegment_id) {
          segment_lengths_.push_back(parent_eapca_envelope.subsegment_lengths_[subsegment_id]);

          ID_TYPE subsegment_length = parent_eapca_envelope.subsegment_lengths_[subsegment_id];
          ID_TYPE subsubsegment_length = subsegment_length / config.vertical_split_nsubsegment_;
          ID_TYPE last_subsubsegment_length =
              subsegment_length - subsubsegment_length * (config.vertical_split_nsubsegment_ - 1);

#ifdef DEBUG
          if (subsubsegment_length < 1) {
            spdlog::error(
                "eapca seg_id {:d} subseg_id {:d} subseg_len {:d} subsubseg_len {:d} last_subsubseg_len {:d}",
                segment_id,
                subsegment_id,
                subsegment_length,
                subsubsegment_length,
                last_subsubsegment_length);
          }

          assert(subsubsegment_length > 0);
          assert(last_subsubsegment_length > 0);
#endif

          subsegment_lengths_.insert(subsegment_lengths_.end(),
                                     config.vertical_split_nsubsegment_ - 1,
                                     subsubsegment_length);
          subsegment_lengths_.push_back(last_subsubsegment_length);
        }
      } else {
        segment_lengths_.push_back(parent_eapca_envelope.segment_lengths_[segment_id]);

        for (ID_TYPE subsegment_id = segment_id * config.vertical_split_nsubsegment_;
             subsegment_id < (segment_id + 1) * config.vertical_split_nsubsegment_;
             ++subsegment_id) {
          subsegment_lengths_.push_back(parent_eapca_envelope.subsegment_lengths_[subsegment_id]);
        }
      }
    }
  } else {
    nsegment_ = parent_eapca_envelope.nsegment_;
    nsubsegment_ = parent_eapca_envelope.nsubsegment_;

    segment_lengths_ = parent_eapca_envelope.segment_lengths_;
    subsegment_lengths_ = parent_eapca_envelope.subsegment_lengths_;
  }

  initialize_stats();
}

RESPONSE dstree::EAPCAEnvelope::initialize_stats() {
  segment_min_means_.assign(nsegment_, constant::MAX_VALUE);
  segment_max_means_.assign(nsegment_, constant::MIN_VALUE);
  segment_min_stds_.assign(nsegment_, constant::MAX_VALUE);
  segment_max_stds_.assign(nsegment_, constant::MIN_VALUE);

  subsegment_min_means_.assign(nsubsegment_, constant::MAX_VALUE);
  subsegment_max_means_.assign(nsubsegment_, constant::MIN_VALUE);
  subsegment_min_stds_.assign(nsubsegment_, constant::MAX_VALUE);
  subsegment_max_stds_.assign(nsubsegment_, constant::MIN_VALUE);

  return SUCCESS;
}

RESPONSE dstree::EAPCAEnvelope::update(const dstree::EAPCA &series_eapca) {
  if (nsegment_ == series_eapca.nsegment_ && nsubsegment_ == series_eapca.nsubsegment_) {
    auto mean_iter = series_eapca.segment_means_.cbegin();
    auto std_iter = series_eapca.segment_stds_.cbegin();

    for (ID_TYPE i = 0; i < series_eapca.nsegment_; ++i, ++mean_iter, ++std_iter) {
      if (segment_min_means_[i] > *mean_iter) {
        segment_min_means_[i] = *mean_iter;
      } else if (segment_max_means_[i] < *mean_iter) {
        segment_max_means_[i] = *mean_iter;
      }

      if (segment_min_stds_[i] > *std_iter) {
        segment_min_stds_[i] = *std_iter;
      } else if (segment_max_stds_[i] < *std_iter) {
        segment_max_stds_[i] = *std_iter;
      }
    }

    mean_iter = series_eapca.subsegment_means_.cbegin();
    std_iter = series_eapca.subsegment_stds_.cbegin();

    for (ID_TYPE i = 0; i < series_eapca.nsubsegment_; ++i, ++mean_iter, ++std_iter) {
      if (subsegment_min_means_[i] > *mean_iter) {
        subsegment_min_means_[i] = *mean_iter;
      } else if (subsegment_max_means_[i] < *mean_iter) {
        subsegment_max_means_[i] = *mean_iter;
      }

      if (subsegment_min_stds_[i] > *std_iter) {
        subsegment_min_stds_[i] = *std_iter;
      } else if (subsegment_max_stds_[i] < *std_iter) {
        subsegment_max_stds_[i] = *std_iter;
      }
    }

    return SUCCESS;
  }

  return FAILURE;
}

VALUE_TYPE dstree::EAPCAEnvelope::cal_lower_bound_EDsquare(const VALUE_TYPE *series_ptr) const {
  VALUE_TYPE lowe_bound_distance = 0;
  VALUE_TYPE mean_diff, std_diff;
  VALUE_TYPE query_segment_mean, query_segment_std;
  ID_TYPE segment_offset = 0;

  for (ID_TYPE i = 0; i < nsegment_; ++i) {
    std::tie(query_segment_mean, query_segment_std) = upcite::cal_mean_std(
        series_ptr + segment_offset, segment_lengths_[i]);

    if (query_segment_mean < segment_min_means_[i]) {
      mean_diff = segment_min_means_[i] - query_segment_mean;
    } else if (query_segment_mean > segment_max_means_[i]) {
      mean_diff = query_segment_mean - segment_max_means_[i];
    } else {
      mean_diff = 0;
    }

    if (query_segment_std < segment_min_stds_[i]) {
      std_diff = segment_min_stds_[i] - query_segment_std;
    } else if (query_segment_std > segment_max_stds_[i]) {
      std_diff = query_segment_std - segment_max_stds_[i];
    } else {
      std_diff = 0;
    }

    lowe_bound_distance += static_cast<VALUE_TYPE>(segment_lengths_[i]) * (mean_diff * mean_diff + std_diff * std_diff);
    segment_offset += segment_lengths_[i];

#ifdef DEBUG
#ifndef DEBUGGED
    MALAT_LOG(logger->logger, trivial::debug) << boost::format(
          "([%.3f %.3f %.3f] = %.3f + [%.3f %.3f %.3f] = %.3f) * %d = %.3f, %d")
          % segment_min_means_[i]
          % query_segment_mean
          % segment_max_means_[i]
          % mean_diff
          % segment_min_stds_[i]
          % query_segment_std
          % segment_max_stds_[i]
          % std_diff
          % segment_lengths_[i]
          % lowe_bound_distance
          % segment_offset;
#endif
#endif
  }
  return lowe_bound_distance;
}





VALUE_TYPE dstree::EAPCAEnvelope::cal_upper_bound_EDsquare(const VALUE_TYPE *series_ptr) const {
  // TODO
  return 0;
}

RESPONSE dstree::EAPCAEnvelope::load(std::ifstream &node_ifs, void *ifs_buf) {
  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);
  auto ifs_value_buf = reinterpret_cast<VALUE_TYPE *>(ifs_buf);

  // nsegment_
  ID_TYPE read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  nsegment_ = ifs_id_buf[0];

  read_nbytes = sizeof(ID_TYPE) * nsegment_;
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  segment_lengths_.insert(segment_lengths_.begin(), ifs_id_buf, ifs_id_buf + nsegment_);

  read_nbytes = sizeof(VALUE_TYPE) * nsegment_;
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  segment_min_means_.insert(segment_min_means_.begin(), ifs_value_buf, ifs_value_buf + nsegment_);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  segment_max_means_.insert(segment_max_means_.begin(), ifs_value_buf, ifs_value_buf + nsegment_);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  segment_min_stds_.insert(segment_min_stds_.begin(), ifs_value_buf, ifs_value_buf + nsegment_);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  segment_max_stds_.insert(segment_max_stds_.begin(), ifs_value_buf, ifs_value_buf + nsegment_);

  // nsubsegment_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  nsubsegment_ = ifs_id_buf[0];

  read_nbytes = sizeof(ID_TYPE) * nsubsegment_;
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  subsegment_lengths_.insert(subsegment_lengths_.begin(), ifs_id_buf, ifs_id_buf + nsubsegment_);

  read_nbytes = sizeof(VALUE_TYPE) * nsubsegment_;
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  subsegment_min_means_.insert(subsegment_min_means_.begin(), ifs_value_buf, ifs_value_buf + nsubsegment_);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  subsegment_max_means_.insert(subsegment_max_means_.begin(), ifs_value_buf, ifs_value_buf + nsubsegment_);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  subsegment_min_stds_.insert(subsegment_min_stds_.begin(), ifs_value_buf, ifs_value_buf + nsubsegment_);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  subsegment_max_stds_.insert(subsegment_max_stds_.begin(), ifs_value_buf, ifs_value_buf + nsubsegment_);

  return SUCCESS;
}

RESPONSE dstree::EAPCAEnvelope::dump(std::ofstream &node_ofs) const {
  node_ofs.write(reinterpret_cast<const char *>(&nsegment_), sizeof(ID_TYPE));
  node_ofs.write(reinterpret_cast<const char *>(segment_lengths_.data()), sizeof(ID_TYPE) * nsegment_);
  node_ofs.write(reinterpret_cast<const char *>(segment_min_means_.data()), sizeof(VALUE_TYPE) * nsegment_);
  node_ofs.write(reinterpret_cast<const char *>(segment_max_means_.data()), sizeof(VALUE_TYPE) * nsegment_);
  node_ofs.write(reinterpret_cast<const char *>(segment_min_stds_.data()), sizeof(VALUE_TYPE) * nsegment_);
  node_ofs.write(reinterpret_cast<const char *>(segment_max_stds_.data()), sizeof(VALUE_TYPE) * nsegment_);

  node_ofs.write(reinterpret_cast<const char *>(&nsubsegment_), sizeof(ID_TYPE));
  node_ofs.write(reinterpret_cast<const char *>(subsegment_lengths_.data()), sizeof(ID_TYPE) * nsubsegment_);
  node_ofs.write(reinterpret_cast<const char *>(subsegment_min_means_.data()), sizeof(VALUE_TYPE) * nsubsegment_);
  node_ofs.write(reinterpret_cast<const char *>(subsegment_max_means_.data()), sizeof(VALUE_TYPE) * nsubsegment_);
  node_ofs.write(reinterpret_cast<const char *>(subsegment_min_stds_.data()), sizeof(VALUE_TYPE) * nsubsegment_);
  node_ofs.write(reinterpret_cast<const char *>(subsegment_max_stds_.data()), sizeof(VALUE_TYPE) * nsubsegment_);

  return SUCCESS;
}
