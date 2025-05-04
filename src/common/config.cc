//
// Created by Qitong Wang on 2022/10/2.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "config.h"

#include <iostream>

#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "comp.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace constant = upcite::constant;

namespace dstree = upcite::dstree;

dstree::Config::Config(int argc, char *argv[]) :
    filter_conformal_num_batches_(1),
    filter_valid_ratio_(0.6),
    filter_conformal_n_parts_(10),
    filter_conformal_k_parts_(3),
    filter_conformal_use_combinatorial_(false),
    db_nseries_(-1),
    query_nseries_(-1),
    series_length_(-1),
    is_znormalized_(true),
    leaf_max_nseries_(100),
    batch_load_nseries_(-1),
    default_nbuffer_(1024 * 64),
    on_disk_(false),
    index_dump_folderpath_("."),
    index_dump_file_postfix_(".bin"),
    node_nchild_(2),
    vertical_split_nsubsegment_(2),
    vertical_split_gain_tradeoff_factor_(2),
    is_exact_search_(true),
    search_max_nseries_(-1),
    search_max_nnode_(-1),
    n_nearest_neighbor_(10),
    examine_ground_truth_(false),
    require_neurofilter_(false),
    filter_dim_latent_(-1),
    filter_train_dropout_p_(0.5),
    filter_leaky_relu_negative_slope_(0.1),
    filter_train_is_gpu_(false),
    filter_infer_is_gpu_(false),
    filter_device_id_(0),
    filter_train_nexample_(-1),
    filter_train_batchsize_(-1),
    filter_train_nepoch_(100),
    filter_train_learning_rate_(0.01),
    filter_train_min_lr_(0.0001),
    filter_train_clip_grad_(false),
    filter_train_clip_grad_norm_type_(2),
    filter_train_clip_grad_max_norm_(1),
    filter_query_filepath_(""),
    filter_train_is_mthread_(false),
    filter_collect_is_mthread_(false),
    filter_collect_nthread_(-1),
    filter_train_nthread_(4),
    filter_remove_square_(false),
    filter_train_val_split_(0.9),
    is_sketch_provided_(false),
    sketch_length_(-1),
    train_sketch_filepath_(""),
    query_sketch_filepath_(""),
    filter_query_id_filename_("global_sampled_indices.bin"),
    filter_query_filename_("global_generated_queries.bin"),
    filter_query_noise_level_(0.1),
    to_dump_index_(false),
    model_dump_file_postfix_(".pickle"),
    dump_node_info_folderpath_(""),
    dump_filters_folderpath_(""),
    dump_data_folderpath_(""),
    to_load_index_(false),
    to_load_filters_(false),
    index_load_folderpath_(""),
    results_path_(""),
    load_node_info_folderpath_(""),
    load_filters_folderpath_(""),
    load_data_folderpath_(""),
    filter_is_conformal_(false),
    filter_conformal_core_type_("discrete"),
    filter_conformal_confidence_(-1),
    filter_conformal_default_confidence_(0.95),
//    filter_conformal_train_val_split_(0.9),
    filter_max_gpu_memory_mb_(constant::MAX_VALUE),
    filter_model_setting_str_(""),
    filter_candidate_settings_filepath_(""),
    filter_allocate_is_gain_(false),
    filter_node_size_threshold_(0),
    filter_conformal_recall_(-1),
    filter_conformal_gamma_(0.1),
    filter_conformal_adjust_confidence_by_recall_(false),
    filter_conformal_is_smoothen_(false),
    filter_conformal_smoothen_method_("spline"),
    filter_conformal_smoothen_core_("steffen"),
    filter_trial_confidence_level_(0.95),
    filter_trial_iterations_(20000),
    filter_trial_nnode_(32),
    filter_default_node_size_threshold_(64),
    filter_retrain_(false),
    filter_reallocate_single_(false),
    filter_reallocate_multi_(false),
    allocator_cpu_trial_iterations_(10000),
    navigator_is_learned_(false),
    navigator_train_k_nearest_neighbor_(5),
    navigator_is_combined_(false),
    navigator_combined_lambda_(0.2),
    navigator_is_gpu_(false),
    navigator_train_val_split_(0.9),
    filter_num_synthetic_query_per_filter_(-1),
    filter_train_num_global_example_(-1),
    filter_train_num_local_example_(-1),
    filter_query_min_noise_(0.1),
    filter_query_max_noise_(0.4),
    dump_query_folderpath_(""),
    to_profile_search_(false),
    to_profile_search_exhausting_(false),
    to_profile_filters_(false),
    filter_fixed_node_size_threshold_(-1),
    filter_cnn_num_channel_(-1),
    filter_cnn_kernel_size_(-1),
    filter_lr_adjust_factor_(0.1),
    filter_rnn_hidden_dim_(64),
    filter_conformal_coverage_(0.9) {
  po::options_description po_desc("DSTree C++ implementation. Copyright (c) 2022 UPCité.");

  po_desc.add_options()
      ("help", "produce help message")
      ("log_filepath_qyl", po::value<std::string>(&log_filepath_qyl_)->default_value("./dstree_qyl.log"),
      "Logging file path")
      ("log_filepath", po::value<std::string>(&log_filepath_)->default_value("./dstree.log"),
       "Logging file path")
      ("results_path", po::value<std::string>(&results_path_)->default_value("./results/"),
       "Results save path")
       //batch
      ("filter_conformal_num_batches", po::value<ID_TYPE>(&filter_conformal_num_batches_)->default_value(1),
       "Number of calibration batches")
      ("filter_valid_ratio", po::value<VALUE_TYPE>(&filter_valid_ratio_)->default_value(0.6),
       "Validation ratio for calibration")
      ("filter_conformal_n_parts", po::value<ID_TYPE>(&filter_conformal_n_parts_)->default_value(10),
       "Number of parts to divide calibration data for combinatorial approach")
      ("filter_conformal_k_parts", po::value<ID_TYPE>(&filter_conformal_k_parts_)->default_value(3),
       "Number of parts to use in each batch for combinatorial approach")
      ("filter_conformal_use_combinatorial", po::bool_switch(&filter_conformal_use_combinatorial_)->default_value(false),
       "Whether to use the combinatorial approach for generating calibration batches")
      
      ("db_filepath", po::value<std::string>(&db_filepath_)->required(),
       "Database file path")
      ("query_filepath", po::value<std::string>(&query_filepath_)->required(),
       "Query file path")
      ("is_sketch_provided", po::bool_switch(&is_sketch_provided_)->default_value(false),
       "Whether the sketch file has been provided")
      ("sketch_length", po::value<ID_TYPE>(&sketch_length_)->default_value(-1),
       "Sketch length")
      ("train_sketch_filepath", po::value<std::string>(&train_sketch_filepath_),
       "Database summarization file path")
      ("query_sketch_filepath", po::value<std::string>(&query_sketch_filepath_),
       "Query summarization file path")
      ("series_length", po::value<ID_TYPE>(&series_length_)->required(),
       "Series length")
      ("is_znormalized", po::bool_switch(&is_znormalized_)->default_value(false),
       "Whether data has been z-normalized")
      ("db_size", po::value<ID_TYPE>(&db_nseries_)->required(),
       "Number of series in database")
      ("query_size", po::value<ID_TYPE>(&query_nseries_)->required(),
       "Number of query series")
      ("leaf_size", po::value<ID_TYPE>(&leaf_max_nseries_)->default_value(100),
       "Maximal leaf node size")
      ("batch_load_size", po::value<ID_TYPE>(&batch_load_nseries_),
       "Maximal number of series for batch loading")
      ("default_nbuffer", po::value<ID_TYPE>(&default_nbuffer_),
       "Default number of node buffers (will increase when needed)")
      ("on_disk", po::bool_switch(&on_disk_)->default_value(false),
       "Whether to build an on-disk index (otherwise an in-memory index)")
      ("index_dump_folderpath", po::value<std::string>(&index_dump_folderpath_),
       "Index dump (or on-disk) root folderpath")
      ("node_nchildren", po::value<ID_TYPE>(&node_nchild_)->default_value(2),
       "Number of child nodes (i.e., fanout) for each parent node")
      ("vsplit_nsubsegment", po::value<ID_TYPE>(&vertical_split_nsubsegment_)->default_value(2),
       "Number of subsegments for vertical split")
      ("vsplit_gain_factor", po::value<VALUE_TYPE>(&vertical_split_gain_tradeoff_factor_)->default_value(2),
       "Increase factor of vertical splits versus horizontal splits")
      ("exact_search", po::bool_switch(&is_exact_search_)->default_value(false),
       "Whether to conduct exact search (or approximate search)")
      ("search_max_nseries", po::value<ID_TYPE>(&search_max_nseries_)->default_value(-1),
       "Maximal number of series to be checked during query answering")
      ("search_max_nnode", po::value<ID_TYPE>(&search_max_nnode_)->default_value(-1),
       "Maximal number of nodes to be checked during query answering")
      ("n_nearest_neighbor", po::value<ID_TYPE>(&n_nearest_neighbor_)->default_value(10),
       "Number of nearest neighbors to be returned")
      ("ground_truth", po::bool_switch(&examine_ground_truth_)->default_value(false),
       "Whether to fetch the ground truths, i.e., linear scan without pruning")
      ("require_neurofilter", po::bool_switch(&require_neurofilter_)->default_value(false),
       "Whether to implant neurofilters")
      ("dim_latent", po::value<ID_TYPE>(&filter_dim_latent_),
       "Dimension of neural model latent variables")
      ("dropout_p", po::value<VALUE_TYPE>(&filter_train_dropout_p_)->default_value(0.5),
       "Dropout probability for MLP latent layer")
      ("leaky_relu_negative_slope", po::value<VALUE_TYPE>(&filter_leaky_relu_negative_slope_)->default_value(0.1),
       "Leaky ReLU negative slope for MLP")
      ("filter_train_is_gpu", po::bool_switch(&filter_train_is_gpu_)->default_value(false),
       "Whether to train neurofilters on GPU (other on CPU)")
      ("filter_infer_is_gpu", po::bool_switch(&filter_infer_is_gpu_)->default_value(false),
       "Whether to run neurofilters on GPU (other on CPU)")
      ("device_id", po::value<ID_TYPE>(&filter_device_id_)->default_value(0),
       "GPU device id")
      ("filter_train_nexample", po::value<ID_TYPE>(&filter_train_nexample_),
       "Number of train examples for neurofilters")
      ("filter_train_batchsize", po::value<ID_TYPE>(&filter_train_batchsize_)->default_value(-1),
       "Neurofilter train batch size")
      ("filter_train_nepoch", po::value<ID_TYPE>(&filter_train_nepoch_)->default_value(100),
       "Neurofilter train (maximal) number of epochs")
      ("learning_rate", po::value<VALUE_TYPE>(&filter_train_learning_rate_)->default_value(0.01),
       "Neurofilter train learning rate")
      ("filter_train_min_lr", po::value<VALUE_TYPE>(&filter_train_min_lr_)->default_value(0.0001),
       "Neurofilter train minimal learning rate, for adjusting learning rates")
      ("filter_train_clip_grad", po::bool_switch(&filter_train_clip_grad_)->default_value(false),
       "Whether to train neurofilters with gradient clipping")
      ("clip_grad_norm_type", po::value<VALUE_TYPE>(&filter_train_clip_grad_norm_type_)->default_value(2),
       "Gradient clipping norm type")
      ("clip_grad_max_norm", po::value<VALUE_TYPE>(&filter_train_clip_grad_max_norm_)->default_value(1),
       "Gradient clipping max norm")
      ("filter_query_filepath", po::value<std::string>(&filter_query_filepath_),
       "Query file path to train neurofilters")
      ("filter_train_mthread", po::bool_switch(&filter_train_is_mthread_)->default_value(false),
       "Whether to train neurofilters multithreadingly")
      //qyl add filter_collect_mthread
      ("filter_collect_mthread", po::bool_switch(&filter_collect_is_mthread_)->default_value(false),
       "Whether to collect knn distances multithreadingly")
       

      ("filter_collect_nthread", po::value<ID_TYPE>(&filter_collect_nthread_)->default_value(-1),
       "Number of threads to collect neurofilter train set; default filter_train_nthread")
      ("filter_train_nthread", po::value<ID_TYPE>(&filter_train_nthread_)->default_value(4),
       "Number of threads to train neurofilters")
      ("filter_remove_square", po::bool_switch(&filter_remove_square_)->default_value(false),
       "Whether to use real distance (instead of square distances) to train filters")
      ("filter_train_val_split", po::value<VALUE_TYPE>(&filter_train_val_split_)->default_value(0.9),
       "Neurofilter train train/val split ratio")
      ("filter_query_noise_level", po::value<VALUE_TYPE>(&filter_query_noise_level_)->default_value(0.1),
       "Neurofilter train query noise level")
      ("dump_index", po::bool_switch(&to_dump_index_)->default_value(false),
       "Whether to dump the index structure (defaults: true for on-disk while false for in-memory)")
      ("load_index", po::bool_switch(&to_load_index_)->default_value(false),
       "Whether to load the index structure")
      ("load_filters", po::bool_switch(&to_load_filters_)->default_value(false),
       "Whether to load the trained filters")
      ("index_load_folderpath", po::value<std::string>(&index_load_folderpath_),
       "Index load root folderpath")
      ("index_load_node_info_folderpath", po::value<std::string>(&load_node_info_folderpath_),
       "Index load node info folderpath")
      ("index_load_filters_folderpath", po::value<std::string>(&load_filters_folderpath_),
       "Index load filter folderpath")
      ("index_load_data_folderpath", po::value<std::string>(&load_data_folderpath_),
       "Index load raw data folderpath")
      ("filter_is_conformal", po::bool_switch(&filter_is_conformal_)->default_value(false),
       "Whether to use conformal filters")
      ("filter_conformal_core_type", po::value<std::string>(&filter_conformal_core_type_)->default_value("discrete"),
       "Filter conformal core type (discrete, spline (i.e., smoothened); defaults: discrete)")
      ("filter_conformal_confidence", po::value<VALUE_TYPE>(&filter_conformal_confidence_)->default_value(-1),
       "Filter conformal confidence level ([0, 1])")
//      ("filter_conformal_train_val_split",
//       po::value<VALUE_TYPE>(&filter_conformal_train_val_split_)->default_value(0.9),
//       "Filter conformal train/val split ratio")
      ("filter_max_gpu_memory_mb",
       po::value<VALUE_TYPE>(&filter_max_gpu_memory_mb_)->default_value(constant::MAX_VALUE),
       "Filter max gpu memory to be used")
      ("filter_model_setting", po::value<std::string>(&filter_model_setting_str_),
       "Filter model setting string")
      ("filter_candidate_settings_filepath", po::value<std::string>(&filter_candidate_settings_filepath_),
       "Filter model candidate model setting filepath")
      ("filter_allocate_is_gain", po::bool_switch(&filter_allocate_is_gain_)->default_value(false),
       "Whether to allocate filters based on the expected runtime gain (default: false, i.e., by size)")
      ("filter_node_size_threshold", po::value<ID_TYPE>(&filter_node_size_threshold_)->default_value(0),
       "Filter node size threshold (default: 0)")
      ("filter_conformal_recall", po::value<VALUE_TYPE>(&filter_conformal_recall_)->default_value(-1),
       "Filter conformal recall level ([0, 1])")
       
      ("filter_conformal_gamma",
       po::value<VALUE_TYPE>(&filter_conformal_gamma_)->default_value(0.1),
       "Filter conformal gamma level ([0, 1])")
       
      ("filter_conformal_coverage",
       po::value<VALUE_TYPE>(&filter_conformal_coverage_)->default_value(0.9),
       "Filter conformal coverage: required proportion of batches meeting recall threshold ([0, 1])")
       
      ("filter_conformal_is_smoothened", po::bool_switch(&filter_conformal_is_smoothen_)->default_value(false),
       "Whether to use smoothened conformal models")
      ("filter_conformal_smoothen_method",
       po::value<std::string>(&filter_conformal_smoothen_method_)->default_value("spline"),
       "Filter conformal smoothening method (default: spline)")
      ("filter_conformal_smoothen_core",
       po::value<std::string>(&filter_conformal_smoothen_core_)->default_value("steffen"),
       "Filter conformal smoothening method (default: steffen; options: cubic)")
      ("filter_trial_confidence_level",
       po::value<VALUE_TYPE>(&filter_trial_confidence_level_)->default_value(0.95),
       "Filter conformal confidence level for allocator trial runs (default: 0.95)")
      ("filter_trial_iterations",
       po::value<ID_TYPE>(&filter_trial_iterations_)->default_value(20000),
       "Filter no. queries for model speed test (default: 20000)")
      ("filter_default_node_size_threshold",
       po::value<ID_TYPE>(&filter_default_node_size_threshold_)->default_value(8),
       "Filter size threshold to run trials (default: 8)")
      ("filter_trial_nnode",
       po::value<ID_TYPE>(&filter_trial_nnode_)->default_value(32),
       "Filter size threshold to run trials (default: 32)")
      ("allocator_cpu_trial_iterations",
       po::value<ID_TYPE>(&allocator_cpu_trial_iterations_)->default_value(10000),
       "Allocator no. full nodes for cpu time estimation (default: 10000)")
      ("navigator_is_learned",
       po::bool_switch(&navigator_is_learned_)->default_value(false),
       "Whether the leaf node visiting order is learned")
      ("navigator_train_k_nearest_neighbor",
       po::value<ID_TYPE>(&navigator_train_k_nearest_neighbor_)->default_value(5),
       "The no. nearest neighbors used to learn node visiting order (default: 5)")
      ("navigator_is_combined",
       po::bool_switch(&navigator_is_combined_)->default_value(false),
       "Whether to use the combine strategy for the learned navigator (default: false)")
      ("navigator_combined_lambda",
       po::value<VALUE_TYPE>(&navigator_combined_lambda_)->default_value(0.2),
       "The lambda used to combine the learned navigator with conventional orders (default: 0.2)")
      ("navigator_is_gpu",
       po::bool_switch(&navigator_is_gpu_)->default_value(false),
       "Whether to train nagivator on GPU (other on CPU; default: CPU)")
      ("navigator_train_val_split",
       po::value<VALUE_TYPE>(&navigator_train_val_split_)->default_value(0.9),
       "Navigator train train/val split ratio")
      ("filter_retrain_",
       po::bool_switch(&filter_retrain_)->default_value(false),
       "Whether to (re-assign and) re-train filters")
      ("filter_reallocate_single",
       po::bool_switch(&filter_reallocate_single_)->default_value(false),
       "Whether to re-allocate the learned filters")
      ("filter_reallocate_multi",
       po::bool_switch(&filter_reallocate_multi_)->default_value(false),
       "Whether to re-allocate the learned filters")
      ("filter_num_synthetic_query_per_filter",
       po::value<ID_TYPE>(&filter_num_synthetic_query_per_filter_)->default_value(-1),
       "The no. synthetic queries to generate per node")
      ("filter_train_num_global_example",
       po::value<ID_TYPE>(&filter_train_num_global_example_)->default_value(-1),
       "The no. global synthetic queries (per index; default: -1)")
      ("filter_train_num_local_example",
       po::value<ID_TYPE>(&filter_train_num_local_example_)->default_value(-1),
       "The no. local synthetic queries (per node; default: -1)")
      ("filter_query_min_noise",
       po::value<VALUE_TYPE>(&filter_query_min_noise_)->default_value(0.1),
       "The min noise level to add to a random series to generate a synthetic query")
      ("filter_query_max_noise",
       po::value<VALUE_TYPE>(&filter_query_max_noise_)->default_value(0.4),
       "The max noise level to add to a random series to generate a synthetic query")
      ("to_profile_search",
       po::bool_switch(&to_profile_search_)->default_value(false),
       "Whether to profile query answering (with early stopping)")
      ("to_profile_search_exhausting",
       po::bool_switch(&to_profile_search_exhausting_)->default_value(false),
       "Whether to profile query answering (without early stopping)")
      ("to_profile_filters",
       po::bool_switch(&to_profile_filters_)->default_value(false),
       "Whether to profile query answering")
      ("filter_fixed_node_size_threshold",
       po::value<ID_TYPE>(&filter_fixed_node_size_threshold_)->default_value(-1),
       "Whether to use a fixed filter leaf node threshold, instead of run time-based estimation")
      ("filter_cnn_num_channel",
       po::value<ID_TYPE>(&filter_cnn_num_channel_)->default_value(-1),
       "The number of channels, if the filter model is CNN")
      ("filter_cnn_kernel_size",
       po::value<ID_TYPE>(&filter_cnn_kernel_size_)->default_value(-1),
       "The kernel size, if the filter model is CNN")
      ("filter_lr_factor",
       po::value<VALUE_TYPE>(&filter_lr_adjust_factor_)->default_value(0.1),
       "The reducing factor for LR scheduler")
      ("filter_rnn_hidden_dim",
       po::value<ID_TYPE>(&filter_rnn_hidden_dim_)->default_value(64),
       "The hidden dim, if the filter model is RNN");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, po_desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << po_desc << std::endl;
    exit(0);
  }

  if (batch_load_nseries_ < 0) {
    batch_load_nseries_ = db_nseries_;
  }

  if (vertical_split_nsubsegment_ != 2) {
    if (vm.count("vsplit_gain_factor") == 0) {
      vertical_split_gain_tradeoff_factor_ = static_cast<VALUE_TYPE>(vertical_split_nsubsegment_);
    }
  }

  if (search_max_nseries_ > 0 || search_max_nnode_ > 0) {
    is_exact_search_ = true;
  }

  if (is_exact_search_) {
    if (search_max_nseries_ < 1) {
      search_max_nseries_ = db_nseries_;
    }

    if (search_max_nnode_ < 1) {
      search_max_nnode_ = db_nseries_;
    }
  }

  if (require_neurofilter_) {
    if (filter_dim_latent_ < 0) {
      filter_dim_latent_ = series_length_;
    }

    if ((filter_train_nexample_ <= 0 && filter_num_synthetic_query_per_filter_ <= 0
        && filter_train_num_global_example_ <= 0) && !to_load_index_) {
      std::cout << "Please specify the number of neurofilter train examples by setting --neurofilter_train_nexample"
                << std::endl;
      exit(-1);
    }

    if (filter_train_batchsize_ < 0) {
      if (filter_train_nexample_ > 0) {
        filter_train_batchsize_ = filter_train_nexample_;
      } else if (filter_train_num_global_example_ > 0 && filter_train_num_local_example_ >= 0) {
        filter_train_batchsize_ = filter_train_num_global_example_ + filter_train_num_local_example_;
      } else {
        filter_train_batchsize_ = 1024;
      }
    }

    if (vm.count("filter_infer_is_gpu") < 1) {
      filter_infer_is_gpu_ = filter_train_is_gpu_;
    }

    if (vm.count("dim_latent") < 1) {
      filter_dim_latent_ = series_length_;
    }

    // if (filter_train_is_mthread_) {
    //   if (filter_collect_nthread_ < 0) {
    //     filter_collect_nthread_ = filter_train_nthread_;
    //   }
    // }

    if (filter_is_conformal_) {
      if (filter_conformal_recall_ >= 0 && filter_conformal_recall_ <= 1) {
        filter_conformal_adjust_confidence_by_recall_ = true;
      } else if (filter_conformal_confidence_ < 0 || filter_conformal_confidence_ > 1) {
        if (!upcite::is_equal(filter_conformal_confidence_, static_cast<VALUE_TYPE>(-1))) {
          std::cout << "Filter conformal confidence level " << filter_conformal_confidence_
                    << " is invalid; revert to default (" << filter_conformal_default_confidence_ << ")"
                    << std::endl;
        }

        filter_conformal_confidence_ = filter_conformal_default_confidence_;
      }

      if (filter_conformal_is_smoothen_) {
        if (filter_conformal_core_type_ != "spline") {
          spdlog::info("leafi conformal core {:s} is not supported for smoothening; roll back to spline",
                       filter_conformal_core_type_);
        }
      }
    }

    if (filter_model_setting_str_.empty() && filter_candidate_settings_filepath_.empty()) {
      std::cout
          << "Please specify model settings by setting --filter_model_setting or --filter_candidate_settings_filepath"
          << std::endl;
      exit(-1);
    }

    if (filter_fixed_node_size_threshold_ >= 0) {
      if (filter_allocate_is_gain_) {
        std::cout << "gain-based allocation not supported under fixed node size threshold" << std::endl;
        exit(-1);
      }
    }

    if (filter_model_setting_str_.compare(0, 3, "cnn") == 0) {
      if (filter_cnn_num_channel_ < 1 && filter_cnn_kernel_size_ < 1) {
        std::cout
            << "Please specify number of channels and kernel size for CNN"
            << std::endl;
        exit(-1);
      }
    }
  }

  if (is_sketch_provided_) {
    if (sketch_length_ == -1 || train_sketch_filepath_.size() < 2 || query_sketch_filepath_.size() < 2) {
      std::cout << "Please specify sketch_length, train_sketch_filepath and query_sketch_filepath" << std::endl;
      exit(-1);
    }
  } else if (train_sketch_filepath_.size() > 1) {
    if (sketch_length_ == -1 || query_sketch_filepath_.size() < 2) {
      std::cout << "Please specify sketch_length and query_sketch_filepath" << std::endl;
      exit(-1);
    }

    is_sketch_provided_ = true;
  }

  if (on_disk_ || to_dump_index_) {
    if (vm.count("index_dump_folderpath")) {
      index_dump_folderpath_ = fs::system_complete(index_dump_folderpath_).string();
    } else {
      index_dump_folderpath_ = fs::absolute(fs::path(log_filepath_).parent_path()).string();
    }

    if (!fs::is_directory(index_dump_folderpath_)) {
      fs::create_directories(index_dump_folderpath_);
    }

    if (!boost::algorithm::ends_with(index_dump_folderpath_, "/")) {
      index_dump_folderpath_ += "/";
    }

    if (dump_node_info_folderpath_.empty()) {
      dump_node_info_folderpath_ = index_dump_folderpath_ + "node/";
    } else if (!boost::algorithm::ends_with(dump_node_info_folderpath_, "/")) {
      dump_node_info_folderpath_ += "/";
    }
    if (!fs::is_directory(dump_node_info_folderpath_)) {
      fs::create_directory(dump_node_info_folderpath_);
    }

    if (dump_data_folderpath_.empty()) {
      dump_data_folderpath_ = index_dump_folderpath_ + "data/";
    } else if (!boost::algorithm::ends_with(dump_data_folderpath_, "/")) {
      dump_data_folderpath_ += "/";
    }
    if (!fs::is_directory(dump_data_folderpath_)) {
      fs::create_directory(dump_data_folderpath_);
    }

    if (require_neurofilter_) {
      if (dump_filters_folderpath_.empty()) {
        dump_filters_folderpath_ = index_dump_folderpath_ + "filter/";
      } else if (!boost::algorithm::ends_with(dump_filters_folderpath_, "/")) {
        dump_filters_folderpath_ += "/";
      }
      if (!fs::is_directory(dump_filters_folderpath_)) {
        fs::create_directory(dump_filters_folderpath_);
      }

      if (filter_train_num_local_example_ > 0) {
        if (dump_query_folderpath_.empty()) {
          dump_query_folderpath_ = index_dump_folderpath_ + "query/";
        } else if (!boost::algorithm::ends_with(dump_query_folderpath_, "/")) {
          dump_query_folderpath_ += "/";
        }
        if (!fs::is_directory(dump_query_folderpath_)) {
          fs::create_directory(dump_query_folderpath_);
        }
      }
    }
  }

  if (to_load_index_) {
    if (vm.count("index_load_folderpath")) {
      index_load_folderpath_ = fs::system_complete(index_load_folderpath_).string();
    } else {
      std::cout << "Please specify index_load_folderpath" << std::endl;
      exit(-1);
    }

    if (!fs::is_directory(index_load_folderpath_)) {
      std::cout << "Empty index_load_folderpath found: " << index_load_folderpath_ << std::endl;
      exit(-1);
    } else if (index_load_folderpath_ == index_dump_folderpath_) {
      std::cout << "index_load_folderpath should be different from index_dump_folderpath" << std::endl;
      exit(-1);
    }

    if (!boost::algorithm::ends_with(index_load_folderpath_, "/")) {
      index_load_folderpath_ += "/";
    }

    if (load_node_info_folderpath_.empty()) {
      load_node_info_folderpath_ = index_load_folderpath_ + "node/";
    } else if (!boost::algorithm::ends_with(load_node_info_folderpath_, "/")) {
      load_node_info_folderpath_ += "/";
    }
    if (!fs::is_directory(load_node_info_folderpath_)) {
      std::cout << "Empty load_node_info_folderpath found: " << load_node_info_folderpath_ << std::endl;
      exit(-1);
    } else if (load_node_info_folderpath_ == dump_node_info_folderpath_) {
      std::cout << "load_node_info_folderpath should be different from dump_node_info_folderpath" << std::endl;
      exit(-1);
    }

    if (load_data_folderpath_.empty()) {
      load_data_folderpath_ = index_load_folderpath_ + "data/";
    } else if (!boost::algorithm::ends_with(load_data_folderpath_, "/")) {
      load_data_folderpath_ += "/";
    }
    if (!fs::is_directory(load_data_folderpath_)) {
      std::cout << "Empty load_data_folderpath found: " << load_data_folderpath_ << std::endl;
      exit(-1);
    } else if (load_data_folderpath_ == dump_data_folderpath_) {
      std::cout << "load_data_folderpath should be different from dump_data_folderpath" << std::endl;
      exit(-1);
    }

    if (require_neurofilter_ && to_load_filters_) {
      if (load_filters_folderpath_.empty()) {
        load_filters_folderpath_ = index_load_folderpath_ + "filter/";
      } else if (!boost::algorithm::ends_with(load_filters_folderpath_, "/")) {
        load_filters_folderpath_ += "/";
      }

      if (!fs::is_directory(load_filters_folderpath_)) {
        std::cout << "Empty load_filters_folderpath found: " << load_filters_folderpath_ << std::endl;
        exit(-1);
      } else if (load_filters_folderpath_ == dump_filters_folderpath_) {
        std::cout << "load_filters_folderpath should be different from dump_filters_folderpath" << std::endl;
        exit(-1);
      }
    }
  }

  if (filter_retrain_ || filter_reallocate_single_ || filter_reallocate_multi_) {
    assert(to_load_index_ && to_load_filters_);

    if (filter_retrain_) {
      filter_reallocate_single_ = false;
      filter_reallocate_multi_ = false;

      assert(fs::is_regular_file(filter_query_filepath_));
    } else {
      assert(!(filter_reallocate_single_ && filter_reallocate_multi_));
    }
  }

  if (require_neurofilter_ && navigator_is_learned_) {
    // TODO support
    std::cout << "Learned filters and learned navigators are not supported at the same time yet" << std::endl;
    exit(-1);
  }

  if (navigator_is_learned_) {
    if (filter_train_batchsize_ < 0) {
      filter_train_batchsize_ = filter_train_nexample_;
    }
  }

  if (to_profile_search_exhausting_) {
    to_profile_search_ = true;
  }
}

void dstree::Config::log() {
  spdlog::info("db_filepath = {:s}", db_filepath_);
  spdlog::info("query_filepath = {:s}", query_filepath_);
  spdlog::info("is_znormalized = {:b}", is_znormalized_);

  spdlog::info("is_sketch_provided = {:b}", is_sketch_provided_);
  spdlog::info("sketch_length = {:d}", sketch_length_);
  spdlog::info("train_sketch_filepath = {:s}", train_sketch_filepath_);
  spdlog::info("query_sketch_filepath = {:s}", query_sketch_filepath_);

  spdlog::info("db_nseries = {:d}", db_nseries_);
  spdlog::info("query_nseries = {:d}", query_nseries_);
  spdlog::info("leaf_max_nseries = {:d}", leaf_max_nseries_);

  spdlog::info("batch_load_nseries = {:d}", batch_load_nseries_);
  spdlog::info("default_nbuffer = {:d}", default_nbuffer_);

  spdlog::info("on_disk = {:b}", on_disk_);
  spdlog::info("index_dump_folderpath = {:s}", index_dump_folderpath_);
  spdlog::info("index_dump_file_postfix = {:s}", index_dump_file_postfix_);

  spdlog::info("node_nchild = {:d}", node_nchild_);
  spdlog::info("vertical_split_nsubsegment = {:d}", vertical_split_nsubsegment_);
  spdlog::info("vertical_split_gain_tradeoff_factor = {:.3f}", vertical_split_gain_tradeoff_factor_);

  spdlog::info("is_exact_search = {:b}", is_exact_search_);
  spdlog::info("search_max_nseries = {:d}", search_max_nseries_);
  spdlog::info("search_max_nnode = {:d}", search_max_nnode_);

  spdlog::info("n_nearest_neighbor = {:d}", n_nearest_neighbor_);

  spdlog::info("examine_ground_truth = {:d}", examine_ground_truth_);

  spdlog::info("require_neurofilter = {:b}", require_neurofilter_);
  spdlog::info("filter_dim_latent = {:d}", filter_dim_latent_);
  spdlog::info("filter_leaky_relu_negative_slope = {:.3f}", filter_leaky_relu_negative_slope_);
  spdlog::info("filter_train_dropout_p_ = {:.3f}", filter_train_dropout_p_);

  spdlog::info("filter_train_is_gpu_ = {:b}", filter_train_is_gpu_);
  spdlog::info("filter_infer_is_gpu = {:b}", filter_infer_is_gpu_);
  spdlog::info("filter_device_id = {:d}", filter_device_id_);

  spdlog::info("filter_train_nexample = {:d}", filter_train_nexample_);
  spdlog::info("filter_train_batchsize = {:d}", filter_train_batchsize_);
  spdlog::info("filter_train_nepoch = {:d}", filter_train_nepoch_);
  spdlog::info("filter_train_learning_rate = {:.3f}", filter_train_learning_rate_);
  spdlog::info("filter_train_min_lr = {:.7f}", filter_train_min_lr_);

  spdlog::info("filter_train_clip_grad = {:b}", filter_train_clip_grad_);
  spdlog::info("filter_train_clip_grad_norm_type = {:.1f}", filter_train_clip_grad_norm_type_);
  spdlog::info("filter_train_clip_grad_max_norm = {:.3f}", filter_train_clip_grad_max_norm_);

  spdlog::info("filter_query_filepath = {:s}", filter_query_filepath_);

  spdlog::info("filter_train_is_mthread = {:b}", filter_train_is_mthread_);
  spdlog::info("filter_collect_is_mthread = {:b}", filter_collect_is_mthread_);
  spdlog::info("filter_collect_nthread = {:d}", filter_collect_nthread_);
  spdlog::info("filter_train_nthread = {:d}", filter_train_nthread_);

  spdlog::info("filter_remove_square = {:b}", filter_remove_square_);

  spdlog::info("filter_train_val_split = {:.3f}", filter_train_val_split_);

  spdlog::info("filter_query_id_filename = {:s}", filter_query_id_filename_);
  spdlog::info("filter_query_filename = {:s}", filter_query_filename_);

  spdlog::info("filter_query_noise_level = {:.3f}", filter_query_noise_level_);

  spdlog::info("to_dump_index = {:b}", to_dump_index_);
  spdlog::info("model_dump_file_postfix = {:s}", model_dump_file_postfix_);
  spdlog::info("dump_node_info_folderpath = {:s}", dump_node_info_folderpath_);
  spdlog::info("dump_filters_folderpath = {:s}", dump_filters_folderpath_);
  spdlog::info("dump_data_folderpath = {:s}", dump_data_folderpath_);

  spdlog::info("to_load_index = {:b}", to_load_index_);
  spdlog::info("index_load_folderpath = {:s}", index_load_folderpath_);
  spdlog::info("dump_node_info_folderpath = {:s}", load_node_info_folderpath_);
  spdlog::info("load_filters_folderpath = {:s}", load_filters_folderpath_);
  spdlog::info("load_data_folderpath = {:s}", load_data_folderpath_);

  spdlog::info("filter_is_conformal = {:b}", filter_is_conformal_);
  spdlog::info("filter_conformal_core_type = {:s}", filter_conformal_core_type_);
  spdlog::info("filter_conformal_confidence = {:.3f}", filter_conformal_confidence_);
  spdlog::info("filter_conformal_default_confidence = {:.3f}", filter_conformal_default_confidence_);
//  spdlog::info("filter_conformal_train_val_split = {:.3f}", filter_conformal_train_val_split_);

  spdlog::info("filter_max_gpu_memory_mb = {:.1f}", filter_max_gpu_memory_mb_);
  spdlog::info("filter_model_setting_str = {:s}", filter_model_setting_str_);
  spdlog::info("filter_candidate_settings_filepath = {:s}", filter_candidate_settings_filepath_);
  spdlog::info("filter_allocate_is_gain = {:b}", filter_allocate_is_gain_);

  spdlog::info("filter_conformal_adjust_confidence_by_recall = {:b}", filter_conformal_adjust_confidence_by_recall_);
  spdlog::info("filter_conformal_recall = {:.6f}", filter_conformal_recall_);

  spdlog::info("filter_conformal_gamma = {:.6f}", filter_conformal_gamma_);

  spdlog::info("filter_conformal_coverage = {:.6f}", filter_conformal_coverage_);

  spdlog::info("filter_conformal_is_smoothen = {:b}", filter_conformal_is_smoothen_);
  spdlog::info("filter_conformal_smoothen_method = {:s}", filter_conformal_smoothen_method_);
  spdlog::info("filter_conformal_smoothen_core = {:s}", filter_conformal_smoothen_core_);

  spdlog::info("filter_trial_confidence_level = {:.6f}", filter_trial_confidence_level_);
  spdlog::info("filter_trial_iterations = {:d}", filter_trial_iterations_);
  spdlog::info("filter_trial_nnode = {:d}", filter_trial_nnode_);
  spdlog::info("filter_default_node_size_threshold = {:d}", filter_default_node_size_threshold_);
  spdlog::info("allocator_cpu_trial_iterations = {:d}", allocator_cpu_trial_iterations_);

  spdlog::info("filter_retrain = {:b}", filter_retrain_);
  spdlog::info("filter_reallocate_single = {:b}", filter_reallocate_single_);
  spdlog::info("filter_reallocate_multi = {:b}", filter_reallocate_multi_);

  spdlog::info("navigator_is_learned = {:b}", navigator_is_learned_);
  spdlog::info("navigator_train_k_nearest_neighbor = {:d}", navigator_train_k_nearest_neighbor_);
  spdlog::info("navigator_is_combined = {:b}", navigator_is_combined_);
  spdlog::info("navigator_combined_lambda = {:.6f}", navigator_combined_lambda_);
  spdlog::info("navigator_is_gpu = {:b}", navigator_is_gpu_);
  spdlog::info("navigator_train_val_split = {:.3f}", navigator_train_val_split_);

  spdlog::info("filter_num_synthetic_query_per_filter = {:d}", filter_num_synthetic_query_per_filter_);
  spdlog::info("filter_train_num_global_example = {:d}", filter_train_num_global_example_);
  spdlog::info("filter_train_num_local_example = {:d}", filter_train_num_local_example_);
  spdlog::info("filter_query_min_noise = {:.3f}", filter_query_min_noise_);
  spdlog::info("filter_query_max_noise = {:.3f}", filter_query_max_noise_);
  spdlog::info("dump_query_folderpath = {:s}", dump_query_folderpath_);

  spdlog::info("to_profile_search = {:b}", to_profile_search_);
  spdlog::info("to_profile_search_exhausting = {:b}", to_profile_search_exhausting_);
  spdlog::info("to_profile_filters = {:b}", to_profile_filters_);

  spdlog::info("filter_fixed_node_size_threshold = {:d}", filter_fixed_node_size_threshold_);

  spdlog::info("filter_cnn_num_channel = {:d}", filter_cnn_num_channel_);
  spdlog::info("filter_cnn_kernel_size = {:d}", filter_cnn_kernel_size_);
  spdlog::info("filter_lr_adjust_factor = {:.3f}", filter_lr_adjust_factor_);

  spdlog::info("filter_rnn_hidden_dim = {:d}", filter_rnn_hidden_dim_);
}
