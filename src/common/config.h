//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_CONFIG_H
#define DSTREE_CONFIG_H

#include <string>
#include <memory>

#include "global.h"

namespace upcite {
namespace dstree {

class Config {
 public:
  Config(int argc, char *argv[]);
  ~Config() = default;

  void log();
  // 结果保存路径
  std::string results_path_;
  std::string log_filepath_;
  std::string log_filepath_qyl_; 
  std::string db_filepath_;
  std::string query_filepath_;

  bool is_sketch_provided_;
  ID_TYPE sketch_length_;
  std::string train_sketch_filepath_;
  std::string query_sketch_filepath_;

  ID_TYPE series_length_;
  bool is_znormalized_;

  ID_TYPE db_nseries_;
  ID_TYPE query_nseries_;
  ID_TYPE leaf_max_nseries_;

  ID_TYPE batch_load_nseries_;
  ID_TYPE default_nbuffer_;

  bool on_disk_;

  ID_TYPE node_nchild_;
  ID_TYPE vertical_split_nsubsegment_;
  VALUE_TYPE vertical_split_gain_tradeoff_factor_;

  bool is_exact_search_;
  ID_TYPE search_max_nseries_;
  ID_TYPE search_max_nnode_;
  // batch 
  ID_TYPE filter_conformal_num_batches_;
  VALUE_TYPE filter_valid_ratio_;
  
  // New parameters for combinatorial calibration batch generation
  ID_TYPE filter_conformal_n_parts_;
  ID_TYPE filter_conformal_k_parts_;
  bool filter_conformal_use_combinatorial_;

  ID_TYPE n_nearest_neighbor_;
  bool examine_ground_truth_;
  bool require_neurofilter_;
  ID_TYPE filter_dim_latent_;
  VALUE_TYPE filter_leaky_relu_negative_slope_;
  VALUE_TYPE filter_train_dropout_p_;
  bool filter_train_is_gpu_;
  bool filter_infer_is_gpu_;
  ID_TYPE filter_device_id_;
  ID_TYPE filter_train_batchsize_;
  ID_TYPE filter_train_nepoch_;
  VALUE_TYPE filter_train_learning_rate_;
  VALUE_TYPE filter_train_min_lr_;
  bool filter_train_clip_grad_;
  VALUE_TYPE filter_train_clip_grad_norm_type_;
  VALUE_TYPE filter_train_clip_grad_max_norm_;

  VALUE_TYPE filter_noise_level_;

  /* obtaining training set */
  // option 1: from a user-defined file
  // disable option 1 by leaving filter_query_filepath_ default (nullptr)
  std::string filter_query_filepath_;
  // if method 2 or 3 is deployed, filter_train_nexample_ will be set after queries are generated
  ID_TYPE filter_train_nexample_;
  // option 2: several synthetic queries per filter that is large enough
  // disable option 2 by leaving filter_num_synthetic_query_per_filter_ default (-1)
  ID_TYPE filter_num_synthetic_query_per_filter_;
  // option 3: a set of global synthetic queries, shared by all filters,
  // and a set of local synthetic queries, exclusive to each filter (that is large enough)
  ID_TYPE filter_train_num_global_example_;
  ID_TYPE filter_train_num_local_example_;

  VALUE_TYPE filter_query_min_noise_;
  VALUE_TYPE filter_query_max_noise_;

  std::string dump_query_folderpath_; // predefined child folder

  bool filter_train_is_mthread_;
  bool filter_collect_is_mthread_;
  ID_TYPE filter_collect_nthread_;
  ID_TYPE filter_train_nthread_;

  bool filter_remove_square_;
  VALUE_TYPE filter_train_val_split_;

  std::string filter_query_id_filename_;
  std::string filter_query_filename_;

  VALUE_TYPE filter_query_noise_level_;

  bool to_dump_index_;
  std::string index_dump_folderpath_;
  std::string dump_node_info_folderpath_; // predefined child folder
  std::string dump_filters_folderpath_; // predefined child folder
  std::string dump_data_folderpath_; // predefined child folder

  std::string index_dump_file_postfix_;
  std::string model_dump_file_postfix_;

  bool to_load_index_;
  bool to_load_filters_;
  std::string index_load_folderpath_;
  std::string load_node_info_folderpath_; // predefined child folder
  std::string load_filters_folderpath_; // predefined child folder
  std::string load_data_folderpath_; // predefined child folder

  bool filter_is_conformal_;
  std::string filter_conformal_core_type_;
  VALUE_TYPE filter_conformal_confidence_;
  VALUE_TYPE filter_conformal_default_confidence_;
  VALUE_TYPE filter_conformal_recall_;
  VALUE_TYPE filter_conformal_gamma_;
  VALUE_TYPE filter_conformal_coverage_;
  
  bool filter_conformal_adjust_confidence_by_recall_;

  VALUE_TYPE filter_max_gpu_memory_mb_;
  std::string filter_model_setting_str_;
  std::string filter_candidate_settings_filepath_;
  bool filter_allocate_is_gain_;
  ID_TYPE filter_node_size_threshold_;
  ID_TYPE filter_fixed_node_size_threshold_;

  bool filter_conformal_is_smoothen_;
  std::string filter_conformal_smoothen_method_;
  std::string filter_conformal_smoothen_core_;

  VALUE_TYPE filter_trial_confidence_level_;
  ID_TYPE filter_trial_iterations_;
  ID_TYPE filter_trial_nnode_;
  ID_TYPE filter_default_node_size_threshold_;

  bool filter_retrain_;
  bool filter_reallocate_single_;
  bool filter_reallocate_multi_;

  ID_TYPE allocator_cpu_trial_iterations_;

  bool navigator_is_learned_;
  ID_TYPE navigator_train_k_nearest_neighbor_;
  bool navigator_is_combined_;
  VALUE_TYPE navigator_combined_lambda_;
  bool navigator_is_gpu_;
  VALUE_TYPE navigator_train_val_split_;

  bool to_profile_search_;
  bool to_profile_search_exhausting_;
  bool to_profile_filters_;

  ID_TYPE filter_cnn_num_channel_;
  ID_TYPE filter_cnn_kernel_size_;
  VALUE_TYPE filter_lr_adjust_factor_;

  ID_TYPE filter_rnn_hidden_dim_;
};

}
}

#endif //DSTREE_CONFIG_H
