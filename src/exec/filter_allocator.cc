//
// Created by Qitong Wang on 2023/2/22.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "filter_allocator.h"

#include <random>
#include <immintrin.h>
#include <queue>
#include <stack>
#include <utility>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include <spdlog/spdlog.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <boost/filesystem.hpp>

#include "vec.h"
#include "distance.h"
#include "answer.h"
#include "index.h"
namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

// 添加必要的命名空间引用
namespace fs = boost::filesystem;

dstree::Allocator::Allocator(dstree::Config &config, 
                            //  const std::vector<dstree::Answers>& train_answers, // 新增参数
                             ID_TYPE nfilters) :
    config_(config),
    // train_answers_(train_answers),
    is_recall_calculated_(false),
    node_size_threshold_(0),
    min_validation_recall_(1) {

  if (config_.get().filter_infer_is_gpu_) {
    if (torch::cuda::is_available()) {
      cudaSetDevice(config_.get().filter_device_id_);

      size_t gpu_free_bytes_, gpu_total_bytes_;
      cudaMemGetInfo(&gpu_free_bytes_, &gpu_total_bytes_);
      VALUE_TYPE gpu_free_mb = static_cast<VALUE_TYPE>(gpu_free_bytes_) / 1024 / 1024;

      if (gpu_free_mb < config.filter_max_gpu_memory_mb_) {
        if (gpu_free_mb > 1) {
          spdlog::error("allocator required {:.3f}mb is not available; down to all free {:.3f}mb",
                        config.filter_max_gpu_memory_mb_, gpu_free_mb);

          available_gpu_memory_mb_ = gpu_free_mb;
        } else {
          spdlog::error("allocator only {:.3f}mb gpu memory is free; exit", gpu_free_mb);
          spdlog::shutdown();
          exit(FAILURE);
        }
      } else {
        spdlog::info("allocator requested {:.3f}mb; {:.3f}mb available",
                     config.filter_max_gpu_memory_mb_, gpu_free_mb);

        available_gpu_memory_mb_ = config.filter_max_gpu_memory_mb_;
      }
    } else {
      spdlog::error("allocator gpu unavailable");
      spdlog::shutdown();
      exit(FAILURE);
    }
  }

  // TODO support model setting list
  candidate_model_settings_.emplace_back(config.filter_model_setting_str_);

  if (nfilters > 0) {
    filter_infos_.reserve(nfilters);
  }

  measure_cpu();
  assert(config_.get().filter_infer_is_gpu_);
  measure_gpu();

  node_size_threshold_ = constant::MAX_ID;
  for (ID_TYPE candidate_model_i = 0; candidate_model_i < candidate_model_settings_.size(); ++candidate_model_i) {
    ID_TYPE current_node_size_threshold =
        candidate_model_settings_[candidate_model_i].gpu_ms_per_query / cpu_ms_per_series_;
    if (current_node_size_threshold < node_size_threshold_) {
      node_size_threshold_ = current_node_size_threshold;
    }
  }
  //QYL
  // void dstree::Allocator::set_training_data(
  //   const std::vector<dstree::Answers>& train_answers) {
  //     train_answers_ = train_answers;
  // }

#ifdef DEBUG
  spdlog::info("allocator node size threshold measured {:d}; default {:d}",
               node_size_threshold_, config_.get().filter_default_node_size_threshold_);
#endif

  if (node_size_threshold_ < config_.get().filter_default_node_size_threshold_) {
#ifdef DEBUG
    spdlog::info("allocator node size threshold measured {:d}; rectified to default {:d}",
                 node_size_threshold_, config_.get().filter_default_node_size_threshold_);
#endif

    node_size_threshold_ = config_.get().filter_default_node_size_threshold_;
  }
}

RESPONSE dstree::Allocator::push_filter_info(const FilterInfo &filter_info) {
  filter_infos_.push_back(filter_info);

  return SUCCESS;
}

struct TrialCache {
  TrialCache(dstree::Config &config,
             ID_TYPE thread_id,
             at::cuda::CUDAStream stream,
             std::vector<upcite::MODEL_SETTING> &candidate_model_settings,
             std::vector<dstree::FilterInfo> &filter_infos,
             ID_TYPE trial_nnode,
             ID_TYPE trial_nmodel,
             std::vector<ID_TYPE> &sampled_filter_idx,
             std::vector<VALUE_TYPE> &filter_pruning_ratios,
             ID_TYPE *trial_sample_i_ptr,
             pthread_mutex_t *sample_idx_mutex) :
      config_(config),
      thread_id_(thread_id),
      stream_(stream),
      candidate_model_settings_ref_(candidate_model_settings),
      filter_infos_ref_(filter_infos),
      trial_nnode_(trial_nnode),
      trial_nmodel_(trial_nmodel),
      sampled_filter_idx_ref_(sampled_filter_idx),
      filter_pruning_ratios_ref_(filter_pruning_ratios),
      trial_sample_i_ptr_(trial_sample_i_ptr),
      sample_idx_mutex_(sample_idx_mutex) {}
  ~TrialCache() = default;

  std::reference_wrapper<dstree::Config> config_;

  ID_TYPE thread_id_;

  at::cuda::CUDAStream stream_;

  std::reference_wrapper<std::vector<upcite::MODEL_SETTING>> candidate_model_settings_ref_;
  std::reference_wrapper<std::vector<dstree::FilterInfo>> filter_infos_ref_;

  ID_TYPE trial_nnode_;
  ID_TYPE trial_nmodel_;

  std::reference_wrapper<std::vector<ID_TYPE>> sampled_filter_idx_ref_;
  std::reference_wrapper<std::vector<VALUE_TYPE>> filter_pruning_ratios_ref_;

  ID_TYPE *trial_sample_i_ptr_;
  pthread_mutex_t *sample_idx_mutex_;
};

void trial_thread_F(TrialCache &trial_cache) {
  at::cuda::setCurrentCUDAStream(trial_cache.stream_);
  at::cuda::CUDAStreamGuard guard(trial_cache.stream_); // compiles with libtorch-gpu

#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("allocator thread {:d}", trial_cache.thread_id_);
  spdlog::debug("allocator candidate_model_settings_ref_.get().size() {:d}",
                trial_cache.candidate_model_settings_ref_.get().size());
  spdlog::debug("allocator filter_infos_ref_.get().size() {:d}", trial_cache.filter_infos_ref_.get().size());
  spdlog::debug("allocator trial_nnode_ {:d}", trial_cache.trial_nnode_);
  spdlog::debug("allocator trial_nmodel_ {:d}", trial_cache.trial_nmodel_);
  spdlog::debug("allocator sampled_filter_idx_ref_.get().size() {:d}",
                trial_cache.sampled_filter_idx_ref_.get().size());
  spdlog::debug("allocator filter_pruning_ratios_ref_.get().size() {:d}",
                trial_cache.filter_pruning_ratios_ref_.get().size());
  spdlog::debug("allocator trial_sample_i_ {:d}", *trial_cache.trial_sample_i_ptr_);
#endif
#endif

  while (true) {
    pthread_mutex_lock(trial_cache.sample_idx_mutex_);

#ifdef DEBUG
#ifndef DEBUGGED
    spdlog::debug("allocator thread {:d} locked, *trial_cache.trial_sample_i_ptr_ = {:d}",
                  trial_cache.thread_id_,
                  *trial_cache.trial_sample_i_ptr_);
#endif
#endif

    if ((*trial_cache.trial_sample_i_ptr_) >= trial_cache.trial_nnode_) {
      
      pthread_mutex_unlock(trial_cache.sample_idx_mutex_);

      break;
    } else {
      // iterate over nodes (check all models for this node)
      // TODO iterate over sampled [node, model] pairs
      ID_TYPE trial_sample_i = *trial_cache.trial_sample_i_ptr_;
      *trial_cache.trial_sample_i_ptr_ = trial_sample_i + 1;

#ifdef DEBUG
#ifndef DEBUGGED
      spdlog::debug("allocator thread {:d} to unlock; trial_sample_i = {:d}, *trial_cache.trial_sample_i_ptr_ = {:d}",
                    trial_cache.thread_id_,
                    trial_sample_i,
                    *trial_cache.trial_sample_i_ptr_);
#endif
#endif

      pthread_mutex_unlock(trial_cache.sample_idx_mutex_);

      ID_TYPE filter_sample_pos = trial_cache.sampled_filter_idx_ref_.get()[trial_sample_i];

#ifdef DEBUG
#ifndef DEBUGGED
      spdlog::debug("allocator thread {:d} sampled_filter_id = {:d}",
                    trial_cache.thread_id_, filter_sample_pos);
#endif
#endif

      std::reference_wrapper<dstree::FilterInfo> filter_info = trial_cache.filter_infos_ref_.get()[filter_sample_pos];
      auto filter_ref = filter_info.get().node_.get().get_filter();

#ifdef DEBUG
#ifndef DEBUGGED
      spdlog::debug("allocator thread {:d} check node {:d}",
                    trial_cache.thread_id_,
                    filter_ref.get_id()
      );
#endif
#endif

      for (ID_TYPE model_i = 0; model_i < trial_cache.trial_nmodel_; ++model_i) {
        auto &candidate_model_setting = trial_cache.candidate_model_settings_ref_.get()[model_i];

#ifdef DEBUG
#ifndef DEBUGGED
        spdlog::debug("allocator thread {:d} check model {:s} on node {:d}",
                      trial_cache.thread_id_,
                      candidate_model_setting.model_setting_str,
                      filter_ref.get_id()
        );
#endif
#endif

        filter_ref.get().trigger_trial(candidate_model_setting);
        filter_ref.get().train(true);

        // 2-d array of [no. models, no. nodes]
        trial_cache.filter_pruning_ratios_ref_.get()[trial_cache.trial_nnode_ * model_i + trial_sample_i] =
            filter_ref.get().get_val_pruning_ratio();

#ifdef DEBUG
#ifndef DEBUGGED
        spdlog::debug("allocator thread {:d} node {:d} model {:d} pruning ratio = {:.3f}",
                      trial_cache.thread_id_,
                      trial_sample_i,
                      model_i,
                      trial_cache.filter_pruning_ratios_ref_.get()[trial_cache.trial_nnode_ * model_i + trial_sample_i]
        );
#endif
#endif
      }
    }
  }
}




RESPONSE dstree::Allocator::trial_collect_mthread() {
  // printf("[DEBUG] Sorting filter_infos_ by node size (descending)\n");
  std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

  ID_TYPE end_i_exclusive = filter_infos_.size();
  // printf("[DEBUG] Initial end_i_exclusive: %ld\n", end_i_exclusive);

  while (end_i_exclusive > 1 && filter_infos_[end_i_exclusive - 1].node_.get().get_size()
      < config_.get().filter_default_node_size_threshold_) {
    end_i_exclusive -= 1;
  }
  // printf("[DEBUG] Adjusted end_i_exclusive: %ld\n", end_i_exclusive);
  ID_TYPE offset = 0;
  //----------------------filter_trial_nnode_=32(小数据集可能不适配), end_i_exclusive =4
  // end_i_exclusive表示通过筛选之后的叶子节点
  ID_TYPE step = end_i_exclusive / config_.get().filter_trial_nnode_;
  // printf("[DEBUG] Step size for sampling: %ld\n", step);

  auto sampled_filter_idx = upcite::make_reserved<ID_TYPE>(config_.get().filter_trial_nnode_);
  for (ID_TYPE sample_i = 0; sample_i < config_.get().filter_trial_nnode_; ++sample_i) {
    sampled_filter_idx.push_back(offset + sample_i * step);
    // printf("[DEBUG] Sampled filter index %ld: %ld\n", sample_i, offset + sample_i * step);
  }

  // 2-d array of [no. models, no. nodes]
  auto filter_pruning_ratios = upcite::make_reserved<VALUE_TYPE>(
      config_.get().filter_trial_nnode_ * candidate_model_settings_.size());
  // printf("[DEBUG] Initializing filter_pruning_ratios with size: %ld\n", filter_pruning_ratios.size());

  for (ID_TYPE i = 0; i < config_.get().filter_trial_nnode_ * candidate_model_settings_.size(); ++i) {
    filter_pruning_ratios.push_back(0);
  }

  std::vector<std::unique_ptr<TrialCache>> trial_caches;
  std::unique_ptr<pthread_mutex_t> sample_idx_mutex = std::make_unique<pthread_mutex_t>();
  ID_TYPE trial_sample_i = 0;

  // printf("[DEBUG] Initializing trial caches and threads\n");
  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    at::cuda::CUDAStream new_stream = at::cuda::getStreamFromPool(false, config_.get().filter_device_id_);

    spdlog::info("trial thread {:d} stream id = {:d}, query = {:d}, priority = {:d}",
                 thread_id,
                 static_cast<ID_TYPE>(new_stream.id()),
                 static_cast<ID_TYPE>(new_stream.query()),
                 static_cast<ID_TYPE>(new_stream.priority())); // compiles with libtorch-gpu
    // printf("[DEBUG] Creating trial cache for thread %ld\n", thread_id);

    trial_caches.emplace_back(std::make_unique<TrialCache>(config_,
                                                           thread_id,
                                                           std::move(new_stream),
                                                           std::ref(candidate_model_settings_),
                                                           std::ref(filter_infos_),
                                                           config_.get().filter_trial_nnode_,
                                                           candidate_model_settings_.size(),
                                                           std::ref(sampled_filter_idx),
                                                           std::ref(filter_pruning_ratios),
                                                           &trial_sample_i,
                                                           sample_idx_mutex.get()));
  }

  std::vector<std::thread> threads;
  // printf("[DEBUG] Launching threads\n");

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    // printf("[DEBUG] Starting thread %ld\n", thread_id);
    threads.emplace_back(trial_thread_F, std::ref(*trial_caches[thread_id]));
  }

  // Print sizes for debugging
  // printf("[DEBUG] config_.get().filter_train_nthread_: %ld\n", config_.get().filter_train_nthread_);
  // printf("[DEBUG] trial_caches.size(): %ld\n", trial_caches.size());
  // Join threads
  // printf("[DEBUG] Joining threads\n");

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    threads[thread_id].join();
  }

#ifdef DEBUG
  auto sampled_filter_ids = upcite::make_reserved<ID_TYPE>(config_.get().filter_trial_nnode_);
  for (ID_TYPE filter_i = 0; filter_i < config_.get().filter_trial_nnode_; ++filter_i) {
    sampled_filter_ids.push_back(filter_infos_[sampled_filter_idx[filter_i]].node_.get().get_id());
  }

  spdlog::info("allocator sampled node ids = {:s}",
    upcite::array2str(sampled_filter_ids.data(), sampled_filter_ids.size()));
  // printf("[DEBUG] Sampled node IDs: %s\n", upcite::array2str(sampled_filter_ids.data(), sampled_filter_ids.size()).c_str());

#endif

#ifdef DEBUG
  spdlog::info("allocator trial pruning ratios = {:s}",
               upcite::array2str(filter_pruning_ratios.data(), filter_pruning_ratios.size()));
  // printf("[DEBUG] Trial pruning ratios: %s\n", upcite::array2str(filter_pruning_ratios.data(), filter_pruning_ratios.size()).c_str());
#endif


  // printf("[DEBUG] Calculating pruning probabilities for each model\n");
  for (ID_TYPE model_i = 0; model_i < candidate_model_settings_.size(); ++model_i) {
    VALUE_TYPE mean = 0;
    for (ID_TYPE sample_i = 0; sample_i < sampled_filter_idx.size(); ++sample_i) {
      // 2-d array of [no. models, no. nodes]
      mean += filter_pruning_ratios[sampled_filter_idx.size() * model_i + sample_i];
    }
    candidate_model_settings_[model_i].pruning_prob = mean / sampled_filter_idx.size();

#ifdef DEBUG
    spdlog::info("allocator model {:s} pruning ratio = {:.3f}",
                 candidate_model_settings_[model_i].model_setting_str,
                 candidate_model_settings_[model_i].pruning_prob);
    // printf("[DEBUG] Model %s pruning ratio: %.3f\n",
    //               candidate_model_settings_[model_i].model_setting_str.c_str(),
    //               candidate_model_settings_[model_i].pruning_prob);
#endif
  }
  // printf("[DEBUG] Exiting trial_collect_mthread() function\n");
  return SUCCESS;
}

RESPONSE dstree::Allocator::measure_cpu() {
  // test cpu_ms_per_series_
  auto batch_nbytes = static_cast<ID_TYPE>(
      sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().leaf_max_nseries_;
  auto trial_batch = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), batch_nbytes));

  auto distances = make_reserved<VALUE_TYPE>(config_.get().leaf_max_nseries_);

  if (config_.get().on_disk_) {
    // credit to https://stackoverflow.com/a/19728404
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<ID_TYPE> uni_i_d(0, config_.get().db_nseries_ - config_.get().leaf_max_nseries_);

    auto start = std::chrono::high_resolution_clock::now();

    for (ID_TYPE trial_i = 0; trial_i < config_.get().allocator_cpu_trial_iterations_; ++trial_i) {
      std::ifstream db_fin;
      db_fin.open(config_.get().db_filepath_, std::ios::in | std::ios::binary);

      ID_TYPE batch_bytes_offset = static_cast<ID_TYPE>(
          sizeof(VALUE_TYPE)) * config_.get().series_length_ * uni_i_d(rng);

      db_fin.seekg(batch_bytes_offset);
      db_fin.read(reinterpret_cast<char *>(trial_batch), batch_nbytes);

      for (ID_TYPE series_i = 0; series_i < config_.get().leaf_max_nseries_; ++series_i) {
        VALUE_TYPE distance = upcite::cal_EDsquare(trial_batch,
                                                   trial_batch + series_i * config_.get().series_length_,
                                                   config_.get().series_length_);
        distances.push_back(distance);
      }

      db_fin.close();
      distances.clear();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cpu_ms_per_series_ = duration.count() / static_cast<double_t>(
        config_.get().allocator_cpu_trial_iterations_ * config_.get().leaf_max_nseries_);
  } else {
    std::ifstream db_fin;
    db_fin.open(config_.get().db_filepath_, std::ios::in | std::ios::binary);
    db_fin.read(reinterpret_cast<char *>(trial_batch), batch_nbytes);

    auto start = std::chrono::high_resolution_clock::now();

    for (ID_TYPE trial_i = 0; trial_i < config_.get().allocator_cpu_trial_iterations_; ++trial_i) {
      distances.clear();

      for (ID_TYPE series_i = 0; series_i < config_.get().leaf_max_nseries_; ++series_i) {
        VALUE_TYPE distance = upcite::cal_EDsquare(trial_batch,
                                                   trial_batch + series_i * config_.get().series_length_,
                                                   config_.get().series_length_);
        distances.push_back(distance);
      }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    db_fin.close();

    cpu_ms_per_series_ = duration.count() / static_cast<double_t>(
        config_.get().allocator_cpu_trial_iterations_ * config_.get().leaf_max_nseries_);
  }

#ifdef DEBUG
  spdlog::info("allocator trial cpu time = {:.6f}mus", cpu_ms_per_series_);
#endif

  free(trial_batch);
  return SUCCESS;
}

RESPONSE dstree::Allocator::measure_gpu() {
  if (torch::cuda::is_available()) {
    bool measure_required = false;
    for (auto const &model_setting_ref : candidate_model_settings_) {
      if (model_setting_ref.gpu_mem_mb <= constant::EPSILON) {
        measure_required = true;
      }
    }

    if (!measure_required) {
      return SUCCESS;
    }

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);
    ID_TYPE query_nbytes = sizeof(VALUE_TYPE) * config_.get().series_length_;
    auto random_input = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_nbytes));

    for (ID_TYPE i = 0; i < config_.get().series_length_; ++i) {
      random_input[i] = dist(e2);
    }

    auto input_tsr_ = torch::from_blob(random_input,
                                       {1, config_.get().series_length_},
                                       torch::TensorOptions().dtype(TORCH_VALUE_TYPE));

    std::unique_ptr<torch::Device> device = nullptr;
    if (config_.get().filter_infer_is_gpu_) {
      device = std::make_unique<torch::Device>(torch::kCUDA,
                                               static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device = std::make_unique<torch::Device>(torch::kCPU);
    }
    input_tsr_ = input_tsr_.to(*device);

    auto trial_filter = std::make_unique<dstree::Filter>(config_, -1, input_tsr_);

    for (auto &model_setting_ref : candidate_model_settings_) {
      if (model_setting_ref.gpu_mem_mb <= constant::EPSILON) {
        // printf("Allocator::measure_gpu()调用collect_running_info \n");
        trial_filter->collect_running_info(model_setting_ref);
      }
    }

    free(random_input);
  }

  return SUCCESS;
}

RESPONSE dstree::Allocator::evaluate() {
  // test candidate_model_setting_.pruning_prob
  // printf("[DEBUG] Entering evaluate() function\n");
  // printf("[DEBUG] Calling trial_collect_mthread()\n");
  trial_collect_mthread();

  // calculate gain for [node_i, model_i]
  // printf("[DEBUG] Preparing filter_ids_, gains_matrix_, and mem_matrix_\n");
  // printf("[DEBUG] filter_infos_.size(): %ld\n", filter_infos_.size());
  // printf("[DEBUG] candidate_model_settings_.size(): %ld\n", candidate_model_settings_.size());

  filter_ids_.reserve(filter_infos_.size());
  gains_matrix_.reserve(filter_infos_.size() * candidate_model_settings_.size());
  mem_matrix_.reserve(filter_infos_.size() * candidate_model_settings_.size());

  for (ID_TYPE filter_i = 0; filter_i < filter_infos_.size(); ++filter_i) {
    auto &filter_info = filter_infos_[filter_i];
    filter_info.score = 0;
    // printf("[DEBUG] Processing filter %ld, node ID: %ld\n", filter_i, filter_info.node_.get().get_id());

    filter_ids_.push_back(filter_info.node_.get().get_id());

    for (ID_TYPE model_i = 0; model_i < candidate_model_settings_.size(); ++model_i) {
      auto &candidate_model_setting_ = candidate_model_settings_[model_i];
      // printf("[DEBUG] Processing model %ld: %s\n", model_i, candidate_model_setting_.model_setting_str.c_str());

      // TODO support model in cpu
      double_t amortized_gpu_sps = static_cast<double_t>(candidate_model_setting_.gpu_ms_per_query)
          / static_cast<double_t>(filter_info.node_.get().get_size());

      // printf("[DEBUG] amortized_gpu_sps: %f, cpu_ms_per_series_: %f\n", amortized_gpu_sps, cpu_ms_per_series_);
    
      if (amortized_gpu_sps > cpu_ms_per_series_) {
        spdlog::error("allocator model {:s} slower than cpu: {:f} > {:f}",
                      candidate_model_setting_.model_setting_str, amortized_gpu_sps, cpu_ms_per_series_);
        // printf("[ERROR] Model %s is slower than CPU\n", candidate_model_setting_.model_setting_str.c_str());

      }

      auto gain = static_cast<VALUE_TYPE>(static_cast<double_t>(filter_info.node_.get().get_size())
          * static_cast<double_t>((1 - filter_info.external_pruning_probability_)
              * candidate_model_setting_.pruning_prob)
          * (cpu_ms_per_series_ - amortized_gpu_sps));
      // printf("[DEBUG] Calculated gain for filter %ld, model %ld: %f\n", filter_i, model_i, gain);

      if (gain < 0) {
        // forbid harmful plans
        // printf("[DEBUG] Gain is negative, setting gain to 0 and memory to %f\n", available_gpu_memory_mb_ + 1);

        gains_matrix_.push_back(0);
        mem_matrix_.push_back(available_gpu_memory_mb_ + 1);
      } else {
        // printf("[DEBUG] Gain is positive, adding gain: %f and memory: %f\n", gain, candidate_model_setting_.gpu_mem_mb);

        gains_matrix_.push_back(gain);
        mem_matrix_.push_back(candidate_model_setting_.gpu_mem_mb);
      }

      if (gain > filter_info.score) {
        // printf("[DEBUG] Updating filter %ld score to %f\n", filter_i, gain);

        filter_info.score = gain;
        filter_info.model_setting = candidate_model_setting_;
      }
    }
  }
  // printf("[DEBUG] Exiting evaluate() function\n");

  return SUCCESS;
}






//这个函数是选择合适的叶子节点插入filter，具体选择是通过gain方法，bi = (1 − p^lb)× (p^F × t^S ×|Ni|− t^F)
// bi > 0 ⇒ |Ni| > a*(t^F/t^S)
RESPONSE dstree::Allocator::assign() {
  VALUE_TYPE allocated_gpu_memory_mb = 0;
  ID_TYPE allocated_filters_count = 0;
  // printf("[DEBUG] Entering assign() function\n");

  // printf("[DEBUG] config_.get().filter_allocate_is_gain_ = %d\n", config_.get().filter_allocate_is_gain_);

  if (config_.get().filter_allocate_is_gain_) {
    //-------------2. 增益优先分配策略 (filter_allocate_is_gain_ = true)
    // printf("[DEBUG] Using gain-based allocation\n");

    evaluate();

    // -------------情况 2.1: 仅一个候选模型配置 
    if (candidate_model_settings_.size() == 1) {
      // printf("[DEBUG] Sorting filter_infos_ by score (descending)\n");
      // 按增益分数降序排序
      std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterScore);
      
      // --------2.1.1：处理无有效增益的情况，此时回退到基于节点大小的分配
      if (filter_infos_[0].score <= 0) {
        spdlog::error("allocator gain-based allocation failed; revert to size-based allocation");
        // printf("[DEBUG] Gain-based allocation failed, reverting to size-based allocation\n");
        for (auto &filter_info : filter_infos_) {
           //检查节点大小和内存限制
          if ((filter_info.node_.get().get_size() > config_.get().filter_node_size_threshold_
              || config_.get().to_profile_filters_)
              && allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb <= available_gpu_memory_mb_) {
            //激活过滤器    
            // printf("[DEBUG] Attempting to activate filter for node ID: %ld\n", filter_info.node_.get().get_id());
            if (filter_info.node_.get().activate_filter(candidate_model_settings_[0]) == SUCCESS) {
              allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
              allocated_filters_count += 1;
              // printf("[DEBUG] Successfully activated filter for node ID: %ld\n", filter_info.node_.get().get_id());
            }
          } else {
            // printf("[DEBUG] Skipping filter for node ID: %ld (size or memory constraint)\n", filter_info.node_.get().get_id());
            break; // 内存不足时终止循环
          }
        }
      } else {
        //-------------2.1.2: 正常增益分配流程
        // printf("[DEBUG] Proceeding with gain-based allocation\n");

        for (auto &filter_info : filter_infos_) {
          if (allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb > available_gpu_memory_mb_) {
            break;
          }
          // 根据配置或增益分数激活
          if (config_.get().to_profile_filters_ || filter_info.score > 0) {
            // printf("[DEBUG] Attempting to activate filter for node ID: %ld\n", filter_info.node_.get().get_id());

            if (filter_info.node_.get().activate_filter(filter_info.model_setting) == SUCCESS) {
              allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
              allocated_filters_count += 1;
              // printf("[DEBUG] Successfully activated filter for node ID: %ld\n", filter_info.node_.get().get_id());

            }
          } else if (filter_info.score <= 0) {
            // printf("[DEBUG] Failed to activate filter for node ID: %ld\n", filter_info.node_.get().get_id());
            break; // 增益不足时终止
          }
        }
      }
    } else {
      // ----------情况 2.2: 多候选模型配置（待实现）
      // printf("[DEBUG] Multiple candidate model settings found (TODO: knapsack solver)\n");
      // TODO knapsack solver
    }
  } else { // default: implant the default model to all leaf nodes
    printf("[DEBUG] Using size-based allocation\n");

    std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

    if (config_.get().filter_model_setting_str_.empty() || candidate_model_settings_.empty()) {
      spdlog::error("allocator default model setting does not exist (set by --filter_model_setting=)");
      return FAILURE;
    } else {
      if (candidate_model_settings_.size() > 1) {
        spdlog::warn("allocator > 1 default model settings found; use the first {:s}",
                     candidate_model_settings_[0].model_setting_str);
      }

      ID_TYPE filter_node_size_threshold = config_.get().filter_fixed_node_size_threshold_;
      if (filter_node_size_threshold < 0) {
        filter_node_size_threshold = node_size_threshold_;
      }
      spdlog::info("allocator assign filter_node_size_threshold {:d}, measured {:d} fixed {:d}",
                   filter_node_size_threshold, node_size_threshold_, config_.get().filter_fixed_node_size_threshold_);

      for (auto &filter_info : filter_infos_) {
        if ((filter_info.node_.get().get_size() >= filter_node_size_threshold
            || config_.get().to_profile_filters_)
            && allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb <= available_gpu_memory_mb_) {
          if (filter_info.node_.get().activate_filter(candidate_model_settings_[0]) == SUCCESS) {
            allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
            allocated_filters_count += 1;
          }
        } else {
          filter_info.node_.get().deactivate_filter();
        }
      }
    }
  }

  spdlog::info("allocator assigned {:d} models of {:.3f}mb/{:.3f}mb gpu memory",
               allocated_filters_count, allocated_gpu_memory_mb, available_gpu_memory_mb_);
  return SUCCESS;
}

RESPONSE dstree::Allocator::reassign() {
  if (candidate_model_settings_.size() != 1) {
    spdlog::error("allocator reallocation only supports single candidate");
    return FAILURE;
  }

  VALUE_TYPE allocated_gpu_memory_mb = 0;
  ID_TYPE allocated_filters_count = 0;

  if (config_.get().filter_allocate_is_gain_) {
    if (candidate_model_settings_.size() == 1) {
      for (auto &filter_info : filter_infos_) {
        filter_info.model_setting = candidate_model_settings_[0];
      }

      auto min_nseries = static_cast<ID_TYPE>(candidate_model_settings_[0].gpu_ms_per_query / cpu_ms_per_series_);
      spdlog::info("allocator re-assign (single), derived min_nseries = {:d}", min_nseries);

      std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

      for (auto &filter_info : filter_infos_) {
        if (allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb > available_gpu_memory_mb_
            || filter_info.node_.get().get_size() < min_nseries) {
          break;
        } else {
          assert(filter_info.model_setting.get().gpu_mem_mb > 0);
          assert(filter_info.model_setting.get().gpu_ms_per_query > 0);
        }

        if (filter_info.node_.get().activate_filter(filter_info.model_setting) == SUCCESS) {
          allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
          allocated_filters_count += 1;
        }
      }
    } else {
      // TODO is reassignment possible for multi models?
    }
  } else {
    // default: implant the default model to all leaf nodes
    std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

    if (config_.get().filter_model_setting_str_.empty() || candidate_model_settings_.empty()) {
      spdlog::error("allocator default model setting does not exist (set by --filter_model_setting=)");
      return FAILURE;
    } else {
      if (candidate_model_settings_.size() > 1) {
        spdlog::warn("allocator > 1 default model settings found; use the first {:s}",
                     candidate_model_settings_[0].model_setting_str);
      }

      ID_TYPE filter_node_size_threshold = config_.get().filter_fixed_node_size_threshold_;
      if (filter_node_size_threshold < 0) {
        filter_node_size_threshold = node_size_threshold_;
      }
      spdlog::info("allocator reassign filter_node_size_threshold {:d}, measured {:d} fixed {:d}",
                   filter_node_size_threshold, node_size_threshold_, config_.get().filter_fixed_node_size_threshold_);

      for (auto &filter_info : filter_infos_) {
//        spdlog::debug("allocator reassign node_.get_size {:d}, has_trained_filter {:d}",
//                      filter_info.node_.get().get_size(),
//                      filter_info.node_.get().has_trained_filter());

        if (filter_info.node_.get().get_size() >= filter_node_size_threshold
            && filter_info.node_.get().has_trained_filter()
            && allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb <= available_gpu_memory_mb_) {
          if (filter_info.node_.get().activate_filter(candidate_model_settings_[0]) == SUCCESS) {
            allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
            allocated_filters_count += 1;
          }
        } else {
          filter_info.node_.get().deactivate_filter();
        }
      }
    }
  }

  spdlog::info("allocator reassigned {:d} models of {:.1f}/{:.1f}mb gpu memory",
               allocated_filters_count, allocated_gpu_memory_mb, available_gpu_memory_mb_);
  return SUCCESS;
}



/*
Revised by Yanlin Qi
1. add KNN search process to compute recall
2. add multiple batches 
*/ 
RESPONSE dstree::Allocator::set_confidence_from_recall(const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes){
  printf("allocator filter_infos_.size() = %d\n", filter_infos_.size());

  // 创建节点ID到filter_infos_索引的映射
  std::unordered_map<ID_TYPE, size_t> node_id_to_index;
  for (size_t i = 0; i < filter_infos_.size(); ++i) {
    node_id_to_index[filter_infos_[i].node_.get().get_id()] = i;
  }

  // 打印映射表（调试用）
  printf("节点ID到索引映射表：\n");
  for (const auto& [node_id, index] : node_id_to_index) {
    printf("  节点ID %d -> 索引 %zu\n", node_id, index);
  }

  if (!is_recall_calculated_) { //初始为false，执行内部代码（计算Conformal校准所需的示例数量）
    ID_TYPE num_conformal_examples, num_train_examples;

    if (config_.get().filter_train_num_global_example_ > 0) {
      num_train_examples = static_cast<ID_TYPE>(config_.get().filter_train_num_global_example_ * config_.get().filter_train_val_split_);
      ID_TYPE num_global_valid_examples = config_.get().filter_train_num_global_example_ - num_train_examples;
      num_conformal_examples = num_global_valid_examples;

    } else { // 仅全局数据的场景 only contains global examples
      num_train_examples = static_cast<ID_TYPE>(config_.get().filter_train_nexample_ * config_.get().filter_train_val_split_);
      ID_TYPE num_valid_examples = config_.get().filter_train_nexample_ - num_train_examples;
      num_conformal_examples = num_valid_examples;  
    }

    // 2.2 初始化存储最近邻距离和对应过滤器ID的数组
    const ID_TYPE K = config_.get().n_nearest_neighbor_; 
    for (ID_TYPE sorted_error_i = 0; sorted_error_i < num_conformal_examples + 2; ++sorted_error_i) {
      
      ID_TYPE total_hit_count = 0; //存储所有query的找到的真实KNN数量
      const ID_TYPE batch_total_knn = K * num_conformal_examples;

      printf("\n----- sorted_error_i = %d -----\n", sorted_error_i);  // 打印当前误差区间索引

      for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
        // 原始index.cc的filter_collect()函数中，收集的global_1nn_distances,global_bsf_distances,都是包含了训练集和校准集全部的query
        // 这里需要区分训练集和校准集，训练集的query_id从0到num_train_examples-1, 校准集的query_id从num_train_examples到num_train_examples+num_conformal_examples-1
        ID_TYPE current_query_id = num_train_examples + query_i;

        printf("\n[Query %d] current_query_id = %d\n", query_i, current_query_id);
        
        // 获取当前查询的节点分布
        auto it = query_knn_nodes.find(current_query_id);
        if (it == query_knn_nodes.end()) {
          fprintf(stderr, "Query %d not found\n", current_query_id);
          continue;
        }       
        // node_counts 是 std::unordered_map<ID_TYPE, ID_TYPE>, 表示 <节点ID, 该节点下的真实KNN数量>
        const auto& node_counts = it->second;  
        ID_TYPE hit_count = 0;
        ID_TYPE wrong_pruned_count = 0; // 统计错误减枝的节点数
        ID_TYPE no_filter_count = 0;    // 统计没有使用过滤器的节点数
        
        for (const auto& [node_id, count_in_node] : node_counts) {
            printf("[node_id %d] KNN series = %d\n", node_id, count_in_node);
            
            // 使用映射查找节点，而不是直接用node_id作为索引
            auto map_it = node_id_to_index.find(node_id);
            if (map_it == node_id_to_index.end()) {
                printf("警告：节点ID %d 未找到对应的索引\n", node_id);
                continue;  // 跳过这个节点
            }
            
            // 找到节点，使用正确的索引访问
            size_t node_index = map_it->second;
            if (node_index >= filter_infos_.size()) {
                printf("错误：节点索引 %zu 超出 filter_infos_ 范围 (size=%zu)\n", 
                       node_index, filter_infos_.size());
                continue;
            }
            
            auto& filter_info = filter_infos_[node_index];
            auto& target_node = filter_info.node_;

            if (target_node.get().has_active_filter()) {
              printf("-----target_node has_active_filter -----\n"); 
              VALUE_TYPE abs_error = target_node.get().get_filter_abs_error_interval_by_pos(sorted_error_i);
              printf("abs_error: %.3f\n", abs_error);
              VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(current_query_id);
              VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(current_query_id);
              // 判断是否覆盖该节点所有KNN
              if (pred_distance - abs_error <= bsf_distance) {
                hit_count += count_in_node; // 累加该节点的贡献量
                printf("COVERED (hit += %d)", count_in_node);
              } 

            } else {
              printf("---------- target_node not active_filter ----------- \n"); 
              hit_count += count_in_node; // 无过滤器则全部命中
              printf("no_filter -> fully COVERED (hit += %d)", count_in_node);
            }
            printf("\n");
          }
          
        total_hit_count += hit_count; // 累加所有query的hit_count
        printf("[Query %d] hit_count = %d\n", query_i, hit_count);
      } 
        // end query loop
        // [10] 打印最终统计结果
      printf("\n===== sorted_error_i = %d =====\n", sorted_error_i);
      printf("Total hit count: %d\n", total_hit_count);
      printf("Batch total knn: %d\n", batch_total_knn);
      printf("avg Recall: %.2f%%\n", (total_hit_count * 100.0) / batch_total_knn);
      
      // 存储当前error分位数对应的平均召回率
      ERROR_TYPE recall = static_cast<ERROR_TYPE>(total_hit_count) / batch_total_knn;
      if (!is_recall_calculated_) {
        // 首次计算时，存储召回率
        validation_recalls_.push_back(recall);
        
        // 更新最小验证召回率
        if (min_validation_recall_ > recall) {
          min_validation_recall_ = recall;
        }
      }
    }
    
    if (!is_recall_calculated_) {
      // 确保validation_recalls_数组的值单调递减
      if (validation_recalls_.size() > 0) {
        validation_recalls_[validation_recalls_.size() - 1] = 1 - constant::EPSILON_GAP;
        printf("constant::EPSILON_GAP = %.10f\n", constant::EPSILON_GAP);
        for (ID_TYPE backtrace_i = validation_recalls_.size() - 2; backtrace_i >= 0; --backtrace_i) {
          if (validation_recalls_[backtrace_i] > validation_recalls_[backtrace_i + 1] - constant::EPSILON_GAP) {
            validation_recalls_[backtrace_i] = validation_recalls_[backtrace_i + 1] - constant::EPSILON_GAP;
          }
        }
      }
      
      // 打印修改后的validation_recalls_数组
      for (size_t i = 0; i < validation_recalls_.size(); ++i) {
        printf("validation_recalls_[%zu] = %.5f\n", i, validation_recalls_[i]);
      }
      
      is_recall_calculated_ = true;
      
      // 如果使用样条连续方法，为每个filter拟合recall到误差分位数的映射
      if (config_.get().filter_conformal_is_smoothen_) {
        printf("filter_infos_.size() = %ld\n", static_cast<long>(filter_infos_.size()));
        for (auto &filter_info : filter_infos_) {
          printf("Processing node with ID: %ld\n", static_cast<long>(filter_info.node_.get().get_id()));
          if (filter_info.node_.get().has_active_filter()) {
            filter_info.node_.get().fit_filter_conformal_spline(validation_recalls_);
        }
      }
      }
    }
  }
  
  // 根据用户指定的召回率阈值，动态调整误差分位数
  printf("---------------此时根据召回率阈值利用样条插值和离散方法来动态调整误差分位数----------------\n");
  printf("min_validation_recall_ = %.5f, filter_conformal_recall_ = %.5f\n", 
         min_validation_recall_, config_.get().filter_conformal_recall_);
  
  // 1. 如果最小验证召回率已经满足用户要求，不需要额外调整
  if (min_validation_recall_ > config_.get().filter_conformal_recall_) {
    printf("-------当前校准集的最小avgRecall都已经大于用户定义的filter_conformal_recall_, 此时不需要额外调整-------\n");
    
    spdlog::info("allocator requested recall {:.3f} out of trained min {:.3f}; do NOT adjust",
                 config_.get().filter_conformal_recall_, min_validation_recall_);
    // 一个校准集会产生一个filter_infos_
    for (auto &filter_info : filter_infos_) {
      if (filter_info.node_.get().has_active_filter()) {
        if (filter_info.node_.get().set_filter_abs_error_interval(0) == FAILURE) {
          spdlog::error("allocator failed to get node {:d} conformed by 0",
                        filter_info.node_.get().get_id());
        }
        spdlog::info("allocator node {:d} abs_error {:.3f} at min {:.3f}; requested {:.3f}",
                     filter_info.node_.get().get_id(),
                     filter_info.node_.get().get_filter_abs_error_interval(),
                     min_validation_recall_, config_.get().filter_conformal_recall_);
      }
    }


  } else { 
    printf("--------2. 这里表示当前并不是所有的分位数下的avgrecall都能满足用户阈值, 那么此时需要计算合适的误差分位数-----\n");
    // 2.1 使用样条函数预测用户要求的recall的误差分位数
    if (config_.get().filter_conformal_is_smoothen_) {
      printf("------2.1: using spline regression to compute oi(delta) when given recall-------\n");
      // 定义计数器
      int total_filters = 0;      // 总过滤器数量
      int active_filters = 0;     // 激活的过滤器数量
      //一个校准集会产生一个filter_infos_，这里需要遍历filter_infos_来计算每个filter的oi(delta)
      for (auto &filter_info : filter_infos_) {
        total_filters++; // 每次循环递增总过滤器数量
        
        if (filter_info.node_.get().has_active_filter()) {
          active_filters++; // 如果过滤器激活，递增激活过滤器数量
          printf("filter node ID: %ld\n", static_cast<long>(filter_info.node_.get().get_id()));
          // 根据用户给定的recall阈值，通过插值计算对应的alpha值
          if (filter_info.node_.get().set_filter_abs_error_interval_by_recall(config_.get().filter_conformal_recall_) == FAILURE) {
            spdlog::error("allocator failed to get node {:d} conformed at recall {:.3f}",
                          filter_info.node_.get().get_id(), config_.get().filter_conformal_recall_);
            printf("Failed to set filter abs error interval for node with ID: %ld\n", 
                   static_cast<long>(filter_info.node_.get().get_id()));
          }
        }
      }
      // 打印总过滤器数量和激活的过滤器数量
      printf("\n !!!!!!!!!!!!!!!!!!  Total filters: %d, Active filters: %d\n", total_filters, active_filters);
    } else {
      printf("------2.2: using discrete to compute oi(delta) when given recall-------\n");
      // 使用离散方法
      ID_TYPE last_recall_i = validation_recalls_.size() - 1;
      // 倒序访问validation_recalls_中的recall，找到满足用户要求的最小分位数
      for (ID_TYPE recall_i = validation_recalls_.size() - 2; recall_i >= 0; --recall_i) {
        if (validation_recalls_[recall_i] < config_.get().filter_conformal_recall_) {
          last_recall_i = recall_i + 1;
          printf("allocator reached recall %.5f with error_i %.5f (%d/%zu, 2 sentries included)\n",
                 validation_recalls_[last_recall_i],
                 static_cast<VALUE_TYPE>(last_recall_i) / validation_recalls_.size(),
                 last_recall_i,
                 validation_recalls_.size());
          break;
        }
      }
      // 根据找到的分位数设置每个过滤器的误差区间
      for (auto &filter_info : filter_infos_) {
        if (filter_info.node_.get().has_active_filter()) {
          if (filter_info.node_.get().set_filter_abs_error_interval_by_pos(last_recall_i) == FAILURE) {
            printf("allocator failed to get node %ld conformed with abs %.5f at pos %ld\n",
                   filter_info.node_.get().get_id(),
                   filter_info.node_.get().get_filter_abs_error_interval_by_pos(last_recall_i),
                   last_recall_i);
          }
        }
      }
    }
  }
  return SUCCESS;
}





// 实现多批次校准集的置信区间计算
RESPONSE dstree::Allocator::set_batch_confidence_from_recall(const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes) {
  printf("\n ------Allocator::set_batch_confidence_from_recall ------ \n");
  // 最终设置的满足覆盖率和召回率的min_satisfying_error_i为-1表示未找到
  ID_TYPE max_satisfying_error_i = -1;  // 初始化为-1表示未找到

  // 1. 创建节点ID到filter_infos_在数组中索引位置的映射（方便查找）
  std::unordered_map<ID_TYPE, size_t> node_id_to_index;
  for (size_t i = 0; i < filter_infos_.size(); ++i) {
    node_id_to_index[filter_infos_[i].node_.get().get_id()] = i;
  }

  const ID_TYPE K = config_.get().n_nearest_neighbor_; 
  
  // 3. 查询校准集批次信息
  ID_TYPE num_batches = 0;
  ID_TYPE examples_per_batch = 0;
  std::vector<std::vector<ID_TYPE>> batch_query_ids;
  
  bool found_calibration_info = false;
  printf("Allocator filter_infos_.size() = %zu\n", filter_infos_.size());

  for (size_t i = 0; i < filter_infos_.size() && !found_calibration_info; ++i) {

    auto& filter_info = filter_infos_[i];
    if (filter_info.node_.get().has_active_filter()) {
      auto& filter = filter_info.node_.get().get_filter().get();
      if (!filter.get_batch_calib_query_ids().empty()) {
        //在训练过程中，每个过滤器都会调用generate_calibration_batches，生成并存储batch_calib_query_ids_。
        //这个检查确保找到的过滤器已经完成了这一步骤。
        batch_query_ids = filter.get_batch_calib_query_ids();
        num_batches = batch_query_ids.size();
        examples_per_batch = batch_query_ids[0].size(); // 假设所有批次大小相似
        found_calibration_info = true;
        printf("从节点 %ld 获取到校准批次信息: %ld 批次, 每批约 %ld 样本\n", 
               filter_info.node_.get().get_id(), num_batches, examples_per_batch);
      }
    }
  }
  
  if (!found_calibration_info) {
    printf("错误: 未找到任何有效的校准批次信息\n");
    return FAILURE;
  }
  
  // 4. 确定误差分位数的数量
  ID_TYPE num_error_quantiles = examples_per_batch + 2; // 加2是为了添加哨兵值
  printf("误差分位数数量: %ld\n", num_error_quantiles);
  
  // 5. 存储每个误差分位数下满足召回率要求的批次数量
  std::vector<ID_TYPE> satisfying_batches(num_error_quantiles, 0);

  if (!is_recall_calculated_) { // 初始为false，执行内部代码
  
    // 初始化多批次校准集的召回率存储结构
    batch_validation_recalls_.clear();
    batch_validation_recalls_.resize(num_batches);
    
    // 6. 外层循环: 遍历误差分位数
    for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
      // 7. 中层循环: 遍历校准集批次
      for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
        const std::vector<ID_TYPE>& current_batch_query_ids = batch_query_ids[batch_i];
        ID_TYPE batch_size = current_batch_query_ids.size();
        
        ID_TYPE total_hit_count = 0; // 存储当前批次所有query找到的真实KNN数量
        const ID_TYPE batch_total_knn = K * batch_size;
        
        // 8. 内层循环: 遍历当前批次的每个查询
        for (ID_TYPE query_idx = 0; query_idx < batch_size; ++query_idx) {
          ID_TYPE current_query_id = current_batch_query_ids[query_idx];
          
          // 获取当前查询的节点分布
          auto it = query_knn_nodes.find(current_query_id);
          if (it == query_knn_nodes.end()) {
            // 如果找不到查询，输出警告并继续
            fprintf(stderr, "查询 %d 未找到\n", current_query_id);
            continue;
          }
          
          // node_counts: <节点ID, 该节点下的真实KNN数量>
          const auto& node_counts = it->second;
          ID_TYPE hit_count = 0;
          ID_TYPE wrong_pruned_count = 0; // 统计错误减枝的节点数
          ID_TYPE no_filter_count = 0;    // 统计没有使用过滤器的节点数
          // 遍历当前查询的所有相关节点
          for (const auto& [node_id, count_in_node] : node_counts) {
            // 使用映射查找节点
            // map_it: <节点ID, 该节点在filter_infos_中的索引位置>
            auto map_it = node_id_to_index.find(node_id);
            if (map_it == node_id_to_index.end()) {
              printf("警告:节点ID %d 未找到对应的索引\n", node_id);
              continue;
            }
            // 找到节点，使用正确的索引访问
            size_t node_index = map_it->second;
            if (node_index >= filter_infos_.size()) {
              printf("错误：节点索引 %zu 超出 filter_infos_ 范围 (size=%zu)\n", 
                     node_index, filter_infos_.size());
              continue;
            }
            auto& filter_info = filter_infos_[node_index];
            auto& target_node = filter_info.node_;
            if (target_node.get().has_active_filter()) {
              // 获取当前批次、当前误差分位数下的对应filter的误差，所以这里需要传入batch_i和error_i，但是不需要传入current_query_id，
              // 因为要计算当前batch下的所有query在同一个误差分位数下的recall。
              VALUE_TYPE abs_error = target_node.get().get_filter_batch_abs_error_interval_by_pos(batch_i, error_i);
              VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(current_query_id);
              VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(current_query_id);
              
              // 判断是否覆盖该节点所有KNN
              if (pred_distance - abs_error <= bsf_distance) {
                hit_count += count_in_node; // 累加该节点的贡献量
              } else {
                // 当前query在当前batch_i下，当前error_i下被错误减枝 
                wrong_pruned_count += 1; // 统计错误减枝的次数
              }
            } else {
              hit_count += count_in_node; // 无过滤器则全部命中
              // 当前query在当前batch_i下，当前error_i下没有使用filter，完全访问真实knn_nodes，所以全部命中
              no_filter_count += 1; // 统计没有使用过滤器的次数
            }
          }
          total_hit_count += hit_count; // 累加当前batch下的所有query的hit_count
          // 输出统计信息
          // if (error_i == 0 || error_i == num_error_quantiles-1) { // 只在第一个和最后一个误差分位数时输出，避免信息过多
          //   printf("查询 %d (batch %ld): 错误减枝节点数: %d, 无过滤器节点数: %d\n", 
          //          current_query_id, batch_i, wrong_pruned_count, no_filter_count);
          // }
        }
        // 内层循环结束 (查询遍历完成)
        
        // 计算当前批次、当前误差分位数下的平均召回率
        ERROR_TYPE recall = static_cast<ERROR_TYPE>(total_hit_count) / batch_total_knn;
        batch_validation_recalls_[batch_i].push_back(recall);
      }
      // 中层循环结束 (批次遍历完成)
    }
    // 外层循环结束 (误差分位数遍历完成)

    // 打印各批次在不同误差分位数下的召回率
    spdlog::info("\n各批次在不同误差分位数下的召回率:");
    for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
      std::string recall_str = fmt::format("批次 {}: ", batch_i);
      for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
        recall_str += fmt::format(" {:.4f}", batch_validation_recalls_[batch_i][error_i]);
      }
      spdlog::info(recall_str);
    }
    printf("\n");
    printf("各批次在不同误差分位数下的召回率:\n");
    for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
      std::string recall_str = fmt::format("批次 {}: ", batch_i);
      for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
        recall_str += fmt::format(" {:.2f}", batch_validation_recalls_[batch_i][error_i]);
      }
      printf("%s\n", recall_str.c_str());
    }
    // 计算和应用(recall, coverage)对
    printf("\n计算和应用(recall, coverage)对...\n");
    
    if (calculate_recall_coverage_pairs() != SUCCESS) {
      printf("警告: 计算(recall, coverage)对失败\n");
    }
    is_recall_calculated_ = true;
  }

  printf("处理完毕，准备安全清理资源\n");
  
  // 安全释放临时变量，避免析构时的问题
  node_id_to_index.clear();
  batch_query_ids.clear();
  satisfying_batches.clear();
  
  printf("set_batch_confidence_from_recall函数执行完成，准备返回\n");
  return SUCCESS;  // 添加明确的返回语句
}




// 计算每个误差分位数下的(recall, coverage)对并用于filter拟合
RESPONSE dstree::Allocator::calculate_recall_coverage_pairs() {
  if (batch_validation_recalls_.empty()) {
    printf("错误: 未找到批次召回率数据\n");
    return FAILURE;
  }
  
  ID_TYPE num_batches = batch_validation_recalls_.size();
  ID_TYPE num_error_quantiles = batch_validation_recalls_[0].size();
  // printf("开始计算召回率-覆盖率对, %ld批次, %ld误差分位数\n", num_batches, num_error_quantiles);
  spdlog::info("开始计算召回率-覆盖率对, {}批次, {}误差分位数", num_batches, num_error_quantiles);  
  // 存储每个误差分位数下的(recall, coverage)对
  std::vector<std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>>> error_recall_cov_pairs(num_error_quantiles);
  
  // 遍历每个误差分位数
  for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
    // printf("\n=== 误差分位数 %ld ===\n", error_i);
    spdlog::info("=== 误差分位数 {} ===", error_i);
    // 收集该误差分位数下所有批次的召回率， recalls是一列的内容
    std::vector<ERROR_TYPE> recalls;
    for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
      recalls.push_back(batch_validation_recalls_[batch_i][error_i]);
    }
    
    // 对每个排序后的召回率计算覆盖率 (好像还没排序呢？)
    for (ID_TYPE j = 0; j < recalls.size(); ++j) {
      ERROR_TYPE min_recall = recalls[j];
      ID_TYPE satisfying_batches = 0;
      
      // 计算达到min_recall的批次数量
      for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
        if (batch_validation_recalls_[batch_i][error_i] >= min_recall) {
            satisfying_batches++;
        }
      }
      
      // 计算覆盖率
      ERROR_TYPE coverage = static_cast<ERROR_TYPE>(satisfying_batches) / num_batches;
      // 存储(recall, coverage)对
      error_recall_cov_pairs[error_i].emplace_back(min_recall, coverage);
      // printf("Min Recall = %.4f, Coverage = %.2f (%ld/%ld批次)\n", 
      //        min_recall, coverage, satisfying_batches, num_batches);
      spdlog::info("Min Recall = {:.4f}, Coverage = {:.2f} ({}/{}) 批次", 
                   min_recall, coverage, satisfying_batches, num_batches);  
    }
  }
  // 保存(recall, coverage)对到CSV文件
  // save_recall_coverage_pairs(error_recall_cov_pairs);
  
  // 新增：保存带有实际误差值的三元组  
  // 关键存储：error_recall_cov_pairs
  save_recall_coverage_error_pairs(error_recall_cov_pairs);
  
  // 为每个filter训练统一的二元回归模型
  for (auto& filter_info : filter_infos_) {
    if (filter_info.node_.get().has_active_filter()) {
      auto& filter = filter_info.node_.get().get_filter().get();
      printf("\n为节点 %ld 训练统一的二元回归模型", filter_info.node_.get().get_id());
      spdlog::info("为节点 {} 训练统一的二元回归模型", filter_info.node_.get().get_id());
      // 收集所有误差分位数下的所有(recall, coverage)对和对应的误差位置
      std::vector<ERROR_TYPE> all_recalls;
      std::vector<ERROR_TYPE> all_coverages;
      std::vector<ID_TYPE> all_error_indices;
      
      // 从所有误差分位数收集数据
      for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
        for (const auto& [recall, coverage] : error_recall_cov_pairs[error_i]) {
          all_recalls.push_back(recall);
          all_coverages.push_back(coverage);
          all_error_indices.push_back(error_i);
        }
      }
      // printf("收集了 %zu 个训练样本点\n", all_recalls.size());
      // 训练单个统一的回归模型 - 修改调用方式
      if (filter_info.node_.get().train_regression_model_for_recall_coverage(
              all_recalls, all_coverages, all_error_indices, filter_info.node_.get().get_id()) != SUCCESS) {
        printf("警告: 节点 %ld 训练统一的二元回归模型失败\n", filter_info.node_.get().get_id());
        continue;
      }
      
      // 使用训练好的模型进行预测和设置
      ERROR_TYPE target_recall = config_.get().filter_conformal_recall_;
      ERROR_TYPE target_coverage = config_.get().filter_conformal_coverage_;
      // printf("使用目标值: recall=%.4f, coverage=%.4f\n", target_recall, target_coverage);
      // 使用训练好的模型直接预测最合适的误差值
      RESPONSE result = filter_info.node_.get().set_filter_abs_error_interval_by_recall_and_coverage(
          target_recall, target_coverage);
      spdlog::info("设置结果: {}", (result == SUCCESS) ? "成功" : "失败");
      printf("设置结果: %s\n", (result == SUCCESS) ? "成功" : "失败");
    }
  }
  // printf("回归模型训练完成，验证数据结构完整性\n");
  return SUCCESS;
}




// 保存(recall, coverage)对到CSV文件
RESPONSE dstree::Allocator::save_recall_coverage_pairs(
    const std::vector<std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>>>& error_recall_cov_pairs) {
    
    printf("\n开始保存(recall, coverage)对到CSV文件\n");
    
    // 检查输入数据是否为空
    if (error_recall_cov_pairs.empty()) {
        printf("错误: 没有可保存的(recall, coverage)对数据\n");
        return FAILURE;
    }
    
    ID_TYPE num_error_quantiles = error_recall_cov_pairs.size();
    
    // 确定文件名（可以根据需要调整）
    std::string save_path = config_.get().results_path_;
    
    // 确保目录存在
    if (!save_path.empty()) {
      namespace fs = boost::filesystem;
      if (!fs::exists(save_path)) {
        printf("创建结果保存目录: %s\n", save_path.c_str());
        fs::create_directories(save_path);
      }
    }
    
    std::string csv_filename = save_path + "/recall_coverage_pairs.csv";

    // 打开CSV文件
    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        printf("错误: 无法创建CSV文件 %s\n", csv_filename.c_str());
        return FAILURE;
    }
    
    // 创建排序后的数据副本
    std::vector<std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>>> sorted_pairs = error_recall_cov_pairs;
    
    // 对每个误差分位数的(recall, coverage)对按照recall升序排序
    for (auto& pairs : sorted_pairs) {
        std::sort(pairs.begin(), pairs.end(), 
                 [](const std::pair<ERROR_TYPE, ERROR_TYPE>& a, 
                    const std::pair<ERROR_TYPE, ERROR_TYPE>& b) {
                     return a.first < b.first;  // 按照recall升序排序
                 });
    }
    
    // 找出所有误差分位数中最大的(recall, coverage)对数量
    size_t max_pairs = 0;
    for (const auto& pairs : sorted_pairs) {
        max_pairs = std::max(max_pairs, pairs.size());
    }
    
    // 写入CSV标题行
    csv_file << "对序号";
    for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
        csv_file << ",error_" << error_i << "_recall_cov,";
    }
    csv_file << "\n";
    
    // 对于每个行索引
    for (size_t row = 0; row < max_pairs; ++row) {
        csv_file << row;
        
        // 遍历每个误差分位数
        for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
            const auto& pairs = sorted_pairs[error_i];
            
            // 如果这个误差分位数有足够多的对
            if (row < pairs.size()) {
                // 获取对应的(recall, coverage)对
                ERROR_TYPE recall = pairs[row].first;
                ERROR_TYPE coverage = pairs[row].second;
                
                // 写入召回率和覆盖率，保留4位小数
                csv_file << "," << std::fixed << std::setprecision(4) << recall
                         << "," << std::fixed << std::setprecision(4) << coverage;
            } else {
                // 如果这个误差分位数没有足够多的对，则填充空值
                csv_file << ",,";
            }
        }
        csv_file << "\n";
    }
    
    // 关闭CSV文件
    csv_file.close();
    
    // 打印成功信息
    printf("已成功保存排序后的(recall, coverage)对到 %s\n", csv_filename.c_str());
    
    return SUCCESS;
}




// 辅助函数：保存批次召回率结果到CSV文件
RESPONSE dstree::Allocator::save_batch_recall_results(
    const std::vector<std::vector<ID_TYPE>>& batch_query_ids,
    ID_TYPE num_batches,
    ID_TYPE num_error_quantiles,
    const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes) {
  
  // 定义结果保存路径
  const std::string results_dir = config_.get().results_path_;

  // 确保目录存在
  if (!results_dir.empty()) {
    namespace fs = boost::filesystem;
    if (!fs::exists(results_dir)) {
      printf("创建结果保存目录: %s\n", results_dir.c_str());
      if (!fs::create_directories(results_dir)) {
        printf("错误: 无法创建结果目录 %s\n", results_dir.c_str());
        return FAILURE;
      }
    }
  }
  
  // 获取K值和节点映射
  const ID_TYPE K = config_.get().n_nearest_neighbor_; 
  std::unordered_map<ID_TYPE, size_t> node_id_to_index;
  for (size_t i = 0; i < filter_infos_.size(); ++i) {
    node_id_to_index[filter_infos_[i].node_.get().get_id()] = i;
  }
  
  // 1. 为批次平均召回率创建CSV文件
  std::string batch_avg_file_path = results_dir + "/batch_avg_recalls.csv";
  std::ofstream batch_avg_recall_file(batch_avg_file_path);
  if (!batch_avg_recall_file.is_open()) {
    printf("错误: 无法创建批次平均召回率文件 %s\n", batch_avg_file_path.c_str());
    return FAILURE;
  }

  // 设置输出精度
  batch_avg_recall_file << std::fixed << std::setprecision(5);

  // 写入表头
  batch_avg_recall_file << "batch_id";
  for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
    batch_avg_recall_file << ",error_" << error_i;
  }
  batch_avg_recall_file << std::endl;

  // 写入每个批次的平均召回率
  for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
    batch_avg_recall_file << batch_i;
    for (ID_TYPE error_i = 0; error_i < batch_validation_recalls_[batch_i].size(); ++error_i) {
      batch_avg_recall_file << "," << batch_validation_recalls_[batch_i][error_i];
    }
    batch_avg_recall_file << std::endl;
  }
  batch_avg_recall_file.close();

  // 为批次平均错误减枝率和无过滤器率创建新的CSV文件
  std::string pruning_stats_path = results_dir + "/batch_pruning_stats.csv";
  std::ofstream pruning_stats_file(pruning_stats_path);
  if (!pruning_stats_file.is_open()) {
    printf("错误: 无法创建批次减枝统计文件 %s\n", pruning_stats_path.c_str());
    return FAILURE;
  }
  
  // 设置输出精度
  pruning_stats_file << std::fixed << std::setprecision(5);
  
  // 写入表头
  pruning_stats_file << "batch_id,error_quantile,avg_wrong_pruned_ratio,avg_no_filter_ratio,total_nodes_count" << std::endl;
  
  // 计算和写入每个批次在每个误差分位数下的平均错误减枝率和无过滤器率
    for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
    const std::vector<ID_TYPE>& current_batch_query_ids = batch_query_ids[batch_i];
    ID_TYPE batch_size = current_batch_query_ids.size();
    
    for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
      // 统计变量
      ID_TYPE total_nodes = 0;
      ID_TYPE total_wrong_pruned = 0;
      ID_TYPE total_no_filter = 0;
      
      // 遍历当前批次的所有查询
      for (ID_TYPE query_idx = 0; query_idx < batch_size; ++query_idx) {
        ID_TYPE current_query_id = current_batch_query_ids[query_idx];
        
        // 获取knn_nodes信息
        auto it = query_knn_nodes.find(current_query_id);
        if (it == query_knn_nodes.end()) continue;
        
        const auto& node_counts = it->second;
        
        // 遍历查询的所有相关节点
        for (const auto& [node_id, count_in_node] : node_counts) {
          auto map_it = node_id_to_index.find(node_id);
          if (map_it == node_id_to_index.end()) continue;
          
          size_t node_index = map_it->second;
          if (node_index >= filter_infos_.size()) continue;
          
          auto& filter_info = filter_infos_[node_index];
          auto& target_node = filter_info.node_;
          
          // 每个节点计数为1（统计节点数量而非K近邻数量）
          total_nodes++;
          
          if (target_node.get().has_active_filter()) {
            VALUE_TYPE abs_error = target_node.get().get_filter_batch_abs_error_interval_by_pos(batch_i, error_i);
            VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(current_query_id);
            VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(current_query_id);
            
            if (pred_distance - abs_error > bsf_distance) {
              // 错误减枝
              total_wrong_pruned++;
            }
          } else {
            // 无过滤器
            total_no_filter++;
          }
        }
      }
      
      // 计算平均比率
      float avg_wrong_pruned_ratio = total_nodes > 0 ? 
          static_cast<float>(total_wrong_pruned) / total_nodes : 0.0f;
      float avg_no_filter_ratio = total_nodes > 0 ? 
          static_cast<float>(total_no_filter) / total_nodes : 0.0f;
      
      // 写入统计结果
      pruning_stats_file << batch_i << "," << error_i << ","
                        << avg_wrong_pruned_ratio << "," 
                        << avg_no_filter_ratio << ","
                        << total_nodes << std::endl;
    }
  }
  
  pruning_stats_file.close();
  printf("已保存批次减枝统计数据到 %s\n", pruning_stats_path.c_str());

  // 2. 为每个批次创建单独的查询召回率文件
      for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
    std::string filename = results_dir + "/batch_" + std::to_string(batch_i) + "_query_recalls.csv";
    std::ofstream query_file(filename);
    if (!query_file.is_open()) {
      printf("警告: 无法创建批次 %ld 的查询召回率文件 %s\n", batch_i, filename.c_str());
      continue;
    }
    
    // 设置输出精度
    query_file << std::fixed << std::setprecision(5);
    
    // 写入表头
    query_file << "query_idx,original_query_id";
    for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
      query_file << ",error_" << error_i;
    }
    query_file << std::endl;
    
    // 获取当前批次的查询ID
    const std::vector<ID_TYPE>& current_batch_query_ids = batch_query_ids[batch_i];
    ID_TYPE batch_size = current_batch_query_ids.size();
    
    // 遍历每个查询，计算并写入其召回率
    for (ID_TYPE query_idx = 0; query_idx < batch_size; ++query_idx) {
      ID_TYPE current_query_id = current_batch_query_ids[query_idx];
      query_file << query_idx << "," << current_query_id;
      
      // 遍历每个误差分位数
      for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
        // 获取knn_nodes信息
          auto it = query_knn_nodes.find(current_query_id);
          if (it == query_knn_nodes.end()) {
          query_file << ",0.00000"; // 查询不存在，设置为0
            continue;
          }
        
        // 计算当前查询在当前误差分位数下的召回率
          const auto& node_counts = it->second;
          ID_TYPE hit_count = 0;
        
        // 遍历查询的所有相关节点
          for (const auto& [node_id, count_in_node] : node_counts) {
            auto map_it = node_id_to_index.find(node_id);
          if (map_it == node_id_to_index.end()) continue;
          
            size_t node_index = map_it->second;
          if (node_index >= filter_infos_.size()) continue;
            
            auto& filter_info = filter_infos_[node_index];
            auto& target_node = filter_info.node_;
            
            if (target_node.get().has_active_filter()) {
            VALUE_TYPE abs_error = target_node.get().get_filter_batch_abs_error_interval_by_pos(batch_i, error_i);
              VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(current_query_id);
              VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(current_query_id);
            
              if (pred_distance - abs_error <= bsf_distance) {
              hit_count += count_in_node;
            }else{
              // 当前query在当前batch_i下，当前error_i下被错误减枝 
              }
            } else {
            hit_count += count_in_node;
            // 当前query在当前batch_i下，当前error_i下没有使用filter，完全访问真实knn_nodes，所以全部命中
          }
        }
        
        // 计算并写入召回率
        float query_recall = static_cast<float>(hit_count) / K;
        query_file << "," << query_recall;
      }
      query_file << std::endl;
    }
    query_file.close();
  }

  printf("已保存批次平均召回率到 %s\n", batch_avg_file_path.c_str());
  printf("已保存各批次查询召回率到 %s/batch_X_query_recalls.csv 文件\n", results_dir.c_str());
  
  return SUCCESS;
}



// 新增：保存带批次召回率的置信区间计算函数
RESPONSE dstree::Allocator::set_batch_confidence_from_recall_save(const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes) {
  printf("\n ------Allocator::set_batch_confidence_from_recall_save ------ \n");
  
  // 首先执行标准的批次置信区间计算
  RESPONSE result = set_batch_confidence_from_recall(query_knn_nodes);
  if (result == FAILURE) {
    printf("错误: 执行 set_batch_confidence_from_recall 失败，无法保存结果\n");
    return FAILURE;
  }
  
  // 查询校准集批次信息
  ID_TYPE num_batches = 0;
  ID_TYPE examples_per_batch = 0;
  std::vector<std::vector<ID_TYPE>> batch_query_ids;
  bool found_calibration_info = false;
  
  // 从任意一个有效过滤器中获取校准批次信息
  for (size_t i = 0; i < filter_infos_.size() && !found_calibration_info; ++i) {
    auto& filter_info = filter_infos_[i];
    if (filter_info.node_.get().has_active_filter()) {
      auto& filter = filter_info.node_.get().get_filter().get();

      if (!filter.get_batch_calib_query_ids().empty()) {
        batch_query_ids = filter.get_batch_calib_query_ids();
        num_batches = batch_query_ids.size();
        examples_per_batch = batch_query_ids[0].size();
        found_calibration_info = true;

        printf("从节点 %ld 获取到校准批次信息: %ld 批次, 每批约 %ld 样本\n", 
               filter_info.node_.get().get_id(), num_batches, examples_per_batch);
      }
    }
  }
  
  if (!found_calibration_info) {
    printf("错误: 未找到任何有效的校准批次信息，无法保存结果\n");
    return FAILURE;
  }
  
  // 确定误差分位数的数量
  ID_TYPE num_error_quantiles = examples_per_batch + 2; // 加2是为了添加哨兵值
  printf("误差分位数数量: %ld\n", num_error_quantiles);
  
  // 保存召回率结果到文件
  RESPONSE recall_result = save_batch_recall_results(batch_query_ids, num_batches, num_error_quantiles, query_knn_nodes);

  // 保存节点访问统计结果到文件
  RESPONSE access_result = save_node_access_stats(batch_query_ids, num_batches, num_error_quantiles, query_knn_nodes);
  
  return (recall_result == SUCCESS && access_result == SUCCESS) ? SUCCESS : FAILURE;
}




// 新增：保存节点访问和减枝统计数据的方法实现
RESPONSE dstree::Allocator::save_node_access_stats(
    const std::vector<std::vector<ID_TYPE>>& batch_query_ids,
    ID_TYPE num_batches,
    ID_TYPE num_error_quantiles,
    const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes) {
    
  // 定义结果保存路径
  const std::string results_dir = config_.get().results_path_;

  // 确保目录存在
  if (!results_dir.empty()) {
    if (!fs::exists(results_dir)) {
      printf("创建结果保存目录: %s\n", results_dir.c_str());
      if (!fs::create_directories(results_dir)) {
        printf("错误: 无法创建结果目录 %s\n", results_dir.c_str());
        return FAILURE;
      }
    }
  }
  
  // 获取节点映射
  std::unordered_map<ID_TYPE, size_t> node_id_to_index;
  for (size_t i = 0; i < filter_infos_.size(); ++i) {
    node_id_to_index[filter_infos_[i].node_.get().get_id()] = i;
  }
  
  // 创建访问统计文件
  std::string access_stats_file_path = results_dir + "/node_access_stats.csv";
  std::ofstream access_stats_file(access_stats_file_path);
  if (!access_stats_file.is_open()) {
    printf("错误: 无法创建节点访问统计文件 %s\n", access_stats_file_path.c_str());
    return FAILURE;
  }
  
  // 写入表头
  access_stats_file << "batch_id,error_quantile,query_id,fully_accessed_ratio,wrong_pruned_ratio,total_node_count" << std::endl;
  
  // 遍历批次、误差分位数和查询
    for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
    const std::vector<ID_TYPE>& current_batch_query_ids = batch_query_ids[batch_i];
    ID_TYPE batch_size = current_batch_query_ids.size();
    
      for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
      for (ID_TYPE query_idx = 0; query_idx < batch_size; ++query_idx) {
        ID_TYPE current_query_id = current_batch_query_ids[query_idx];
        
        // 获取当前查询的节点分布
        auto it = query_knn_nodes.find(current_query_id);
        if (it == query_knn_nodes.end()) continue;
        
        const auto& node_counts = it->second;
        ID_TYPE total_nodes = 0;           // 总节点数
        ID_TYPE fully_accessed_count = 0;  // 完全访问的节点数
        ID_TYPE wrong_pruned_count = 0;    // 被错误减枝的节点数
        
        // 遍历查询相关的所有节点
        for (const auto& [node_id, count_in_node] : node_counts) {
          auto map_it = node_id_to_index.find(node_id);
          if (map_it == node_id_to_index.end()) continue;
          
          size_t node_index = map_it->second;
          if (node_index >= filter_infos_.size()) continue;
          
          auto& filter_info = filter_infos_[node_index];
          auto& target_node = filter_info.node_;
          
          // 统计总节点数
          total_nodes += count_in_node;
          
          if (target_node.get().has_active_filter()) {
            VALUE_TYPE abs_error = target_node.get().get_filter_batch_abs_error_interval_by_pos(batch_i, error_i);
            VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(current_query_id);
            VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(current_query_id);
            
            if (pred_distance - abs_error <= bsf_distance) {
              // 正确访问
              fully_accessed_count += count_in_node;
            } else {
              // 错误减枝
              wrong_pruned_count += count_in_node;
            }
          } else {
            // 无过滤器，完全访问
            fully_accessed_count += count_in_node;
          }
        }
        
        // 计算比例
        float fully_accessed_ratio = total_nodes > 0 ? 
            static_cast<float>(fully_accessed_count) / total_nodes : 0.0f;
        float wrong_pruned_ratio = total_nodes > 0 ? 
            static_cast<float>(wrong_pruned_count) / total_nodes : 0.0f;
        
        // 写入统计结果
        access_stats_file << batch_i << "," << error_i << "," << current_query_id << ","
                         << std::fixed << std::setprecision(5) << fully_accessed_ratio << ","
                         << wrong_pruned_ratio << "," << total_nodes << std::endl;
      }
    }
  }
  
  access_stats_file.close();
  printf("已保存节点访问统计数据到 %s\n", access_stats_file_path.c_str());
  
  // 创建批次汇总统计文件
  std::string batch_summary_path = results_dir + "/batch_access_summary.csv";
  std::ofstream batch_summary_file(batch_summary_path);
  if (!batch_summary_file.is_open()) {
    printf("错误: 无法创建批次访问汇总文件 %s\n", batch_summary_path.c_str());
    return FAILURE;
  }
  
  // 写入表头
  batch_summary_file << "batch_id,error_quantile,avg_fully_accessed_ratio,avg_wrong_pruned_ratio" << std::endl;
  
  // 计算每个批次在每个误差分位数下的平均访问统计
  for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
    const std::vector<ID_TYPE>& current_batch_query_ids = batch_query_ids[batch_i];
    ID_TYPE batch_size = current_batch_query_ids.size();
    
    for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
      float sum_fully_accessed_ratio = 0.0f;
      float sum_wrong_pruned_ratio = 0.0f;
      ID_TYPE valid_query_count = 0;
      
      for (ID_TYPE query_idx = 0; query_idx < batch_size; ++query_idx) {
        ID_TYPE current_query_id = current_batch_query_ids[query_idx];
        
        // 获取当前查询的节点分布
        auto it = query_knn_nodes.find(current_query_id);
        if (it == query_knn_nodes.end()) continue;
        
        const auto& node_counts = it->second;
        ID_TYPE total_nodes = 0;
        ID_TYPE fully_accessed_count = 0;
        ID_TYPE wrong_pruned_count = 0;
        
        // 遍历查询相关的所有节点
        for (const auto& [node_id, count_in_node] : node_counts) {
          auto map_it = node_id_to_index.find(node_id);
          if (map_it == node_id_to_index.end()) continue;
          
          size_t node_index = map_it->second;
          if (node_index >= filter_infos_.size()) continue;
          
          auto& filter_info = filter_infos_[node_index];
          auto& target_node = filter_info.node_;
          
          // 统计总节点数
          total_nodes += count_in_node;
          
          if (target_node.get().has_active_filter()) {
            VALUE_TYPE abs_error = target_node.get().get_filter_batch_abs_error_interval_by_pos(batch_i, error_i);
            VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(current_query_id);
            VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(current_query_id);
            
            if (pred_distance - abs_error <= bsf_distance) {
              // 正确访问
              fully_accessed_count += count_in_node;
            } else {
              // 错误减枝
              wrong_pruned_count += count_in_node;
            }
    } else {
            // 无过滤器，完全访问
            fully_accessed_count += count_in_node;
          }
        }
        
        if (total_nodes > 0) {
          sum_fully_accessed_ratio += static_cast<float>(fully_accessed_count) / node_counts.size();
          sum_wrong_pruned_ratio += static_cast<float>(wrong_pruned_count) / node_counts.size();
          valid_query_count++;
        }
      }
      
      // 计算当前批次当前误差分位数下的平均比例
      float avg_fully_accessed_ratio = valid_query_count > 0 ? 
          sum_fully_accessed_ratio / valid_query_count : 0.0f;
      float avg_wrong_pruned_ratio = valid_query_count > 0 ? 
          sum_wrong_pruned_ratio / valid_query_count : 0.0f;
      
      // 写入批次汇总结果
      batch_summary_file << batch_i << "," << error_i << ","
                         << std::fixed << std::setprecision(5) << avg_fully_accessed_ratio << ","
                         << avg_wrong_pruned_ratio << std::endl;
    }
  }
  
  batch_summary_file.close();
  printf("已保存批次访问汇总数据到 %s\n", batch_summary_path.c_str());
  
  return SUCCESS;
}



// 新增：保存(recall, coverage, error)三元组到CSV文件，针对每个filter的每个batch
RESPONSE dstree::Allocator::save_recall_coverage_error_pairs(
    const std::vector<std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>>>& error_recall_cov_pairs) {
    
    printf("\n开始保存(recall, coverage, error)三元组汇总表格\n");
    // 检查输入数据是否为空
    if (error_recall_cov_pairs.empty()) {
        printf("错误: 没有可保存的(recall, coverage)对数据\n");
        return FAILURE;
    }
    
    ID_TYPE num_error_quantiles = error_recall_cov_pairs.size();
    // 确定保存路径
    std::string save_path = config_.get().results_path_;
    
    // 确保目录存在
    if (!save_path.empty()) {
      namespace fs = boost::filesystem;
      if (!fs::exists(save_path)) {
        printf("创建结果保存目录: %s\n", save_path.c_str());
        fs::create_directories(save_path);
      }
    }
    
    // 查询校准集批次信息和filter信息
    ID_TYPE num_batches = 0;
    ID_TYPE examples_per_batch = 0;
    std::vector<std::vector<ID_TYPE>> batch_query_ids;
    bool found_calibration_info = false;
    
    // 从任意一个有效过滤器中获取校准批次信息
    for (size_t i = 0; i < filter_infos_.size() && !found_calibration_info; ++i) {
        auto& filter_info = filter_infos_[i];
        if (filter_info.node_.get().has_active_filter()) {
            auto& filter = filter_info.node_.get().get_filter().get();
            if (!filter.get_batch_calib_query_ids().empty()) {
                batch_query_ids = filter.get_batch_calib_query_ids();
                num_batches = batch_query_ids.size();
                examples_per_batch = batch_query_ids[0].size();
                found_calibration_info = true;
                printf("从节点 %ld 获取到校准批次信息: %ld 批次, 每批约 %ld 样本\n", 
                        filter_info.node_.get().get_id(), num_batches, examples_per_batch);
            }
         }
      }
    
    if (!found_calibration_info) {
        printf("错误: 未找到任何有效的校准批次信息，无法生成三元组\n");
        return FAILURE;
    }
    
    // 为每个过滤器创建一个文件夹来保存汇总表格
    for (size_t filter_idx = 0; filter_idx < filter_infos_.size(); ++filter_idx) {
        auto& filter_info = filter_infos_[filter_idx];
        if (!filter_info.node_.get().has_active_filter()) {
            continue;  // 跳过没有激活过滤器的节点
        }
        
        ID_TYPE filter_id = filter_info.node_.get().get_id();
        std::string filter_dir = save_path + "/filter_" + std::to_string(filter_id);
        
        // 创建filter目录
        namespace fs = boost::filesystem;
        if (!fs::exists(filter_dir)) {
            printf("创建过滤器 %ld 的结果目录: %s\n", (long)filter_id, filter_dir.c_str());
            fs::create_directories(filter_dir);
        }
        
        // 获取此过滤器的batch_alphas
        const auto& filter = filter_info.node_.get().get_filter().get();
        const auto& batch_alphas = filter.get_batch_alphas();
        
        if (batch_alphas.empty()) {
            printf("警告: 过滤器 %ld 的batch_alphas为空，跳过\n", (long)filter_id);
            continue;
        }
        
        // 表格1：每个batch为一行，每个error_quantile为列组，子列为(recall,cov,error)
        std::string table1_path = filter_dir + "/batch_error_summary.csv";
        std::ofstream table1_file(table1_path);
        if (!table1_file.is_open()) {
            printf("错误: 无法创建表格1文件 %s\n", table1_path.c_str());
            continue;
        }
        
        // 写入表格1表头
        table1_file << "batch_id_sorted";
        for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
            table1_file << ",recall,cov,actual error";
        }
        table1_file << std::endl;
        
        // 为每个batch收集所有误差分位数下的recall, coverage对
        for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
            // 为每个误差分位数收集该batch下的recall值
            std::vector<ERROR_TYPE> batch_recalls(num_error_quantiles);
            
            // 从batch_validation_recalls_获取该batch下所有误差分位数的recall值
            for (ID_TYPE error_i = 0; error_i < num_error_quantiles && error_i < batch_validation_recalls_[batch_i].size(); ++error_i) {
                batch_recalls[error_i] = batch_validation_recalls_[batch_i][error_i];
            }
            
            // 为每个batch写入一行数据
            table1_file << batch_i;  // batch_id
            
            // 对于每个error_quantile写入(recall,coverage,error)
            for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
                if (error_i >= batch_alphas[batch_i].size()) {
                    // 没有此误差分位数的数据
                    table1_file << ",,,";
                    continue;
                }
                
                // 获取此batch在此error_quantile下的recall值
                ERROR_TYPE recall = batch_recalls[error_i];
                
                // 计算coverage - 有多少批次达到这个recall值
                ID_TYPE satisfying_batches = 0;
                for (ID_TYPE other_batch_i = 0; other_batch_i < num_batches; ++other_batch_i) {
                    if (error_i < batch_validation_recalls_[other_batch_i].size() && 
                        batch_validation_recalls_[other_batch_i][error_i] >= recall) {
                        satisfying_batches++;
                    }
                }
                ERROR_TYPE coverage = static_cast<ERROR_TYPE>(satisfying_batches) / num_batches;
                
                // 获取误差值
                VALUE_TYPE error_value = batch_alphas[batch_i][error_i];
                
                // 写入(recall,coverage,error)三元组
                table1_file << "," << std::fixed << std::setprecision(4) << recall
                           << "," << std::fixed << std::setprecision(4) << coverage
                           << "," << std::fixed << std::setprecision(6) << error_value;
            }
            
            table1_file << std::endl;
        }
        
        table1_file.close();
        printf("已保存过滤器 %ld 的表格1: %s\n", (long)filter_id, table1_path.c_str());
        
        // 表格2：行按照error_quantile组织，显示每个batch的误差值
        std::string table2_path = filter_dir + "/recall_coverage_error_summary.csv";
        std::ofstream table2_file(table2_path);
        if (!table2_file.is_open()) {
            printf("错误: 无法创建表格2文件 %s\n", table2_path.c_str());
            continue;
        }
        
        // 写入表格2表头
        table2_file << "error_quantile,recall,coverage";
        for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
            table2_file << ",batch_" << batch_i << "_error";
        }
        table2_file << std::endl;
        
        // 按误差分位数组织数据
        for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
            // 检查是否至少有一个batch在这个误差分位数下有数据
            bool has_valid_data = false;
            for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
                if (error_i < batch_validation_recalls_[batch_i].size()) {
                    has_valid_data = true;
                    break;
                }
            }
            
            if (!has_valid_data) continue;
            
            // 为每个误差分位数创建一行，使用所有batch的recall值
            for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
                if (error_i >= batch_validation_recalls_[batch_i].size()) continue;
                
                ERROR_TYPE recall = batch_validation_recalls_[batch_i][error_i];
                
                // 计算coverage - 有多少批次达到这个recall值
                ID_TYPE satisfying_batches = 0;
                for (ID_TYPE other_batch_i = 0; other_batch_i < num_batches; ++other_batch_i) {
                    if (error_i < batch_validation_recalls_[other_batch_i].size() && 
                        batch_validation_recalls_[other_batch_i][error_i] >= recall) {
                        satisfying_batches++;
                    }
                }
                ERROR_TYPE coverage = static_cast<ERROR_TYPE>(satisfying_batches) / num_batches;
                
                // 写入基本信息
                table2_file << error_i << ","
                           << std::fixed << std::setprecision(4) << recall << ","
                           << std::fixed << std::setprecision(4) << coverage;
                
                // 写入每个batch的误差值
                for (ID_TYPE write_batch_i = 0; write_batch_i < num_batches; ++write_batch_i) {
                    if (error_i < batch_alphas[write_batch_i].size()) {
                        table2_file << "," << std::fixed << std::setprecision(6) << batch_alphas[write_batch_i][error_i];
                    } else {
                        table2_file << ",";  // 空值
                    }
                }
                
                table2_file << std::endl;
            }
        }
        
        table2_file.close();
        printf("已保存过滤器 %ld 的表格2: %s\n", (long)filter_id, table2_path.c_str());
        
        // 表格3：添加预测误差的汇总表
        for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
            // 为当前batch训练回归模型
            std::vector<ERROR_TYPE> all_recalls;
            std::vector<ERROR_TYPE> all_coverages;
            std::vector<ID_TYPE> all_error_indices;
            
            // 收集数据
            for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
                if (error_i >= batch_alphas[batch_i].size()) continue;
                
                for (const auto& [recall, coverage] : error_recall_cov_pairs[error_i]) {
                    // 移除过滤条件，使用所有coverage值
                    
                    all_recalls.push_back(recall);
                    all_coverages.push_back(coverage);
                    all_error_indices.push_back(error_i);
                }
            }
            
            if (all_recalls.empty()) continue;
            
            // 训练模型
            RESPONSE model_result = filter_info.node_.get().train_regression_model_for_recall_coverage_actual_error(
                all_recalls, all_coverages, all_error_indices, batch_i, filter_id);
                
            if (model_result != SUCCESS) {
                printf("警告: 过滤器 %ld 的批次 %ld 训练模型失败\n", (long)filter_id, (long)batch_i);
                continue;
            }
            
            // 创建表格3
            std::string table3_path = filter_dir + "/batch_" + std::to_string(batch_i) + "_with_prediction.csv";
            std::ofstream table3_file(table3_path);
            if (!table3_file.is_open()) {
                printf("错误: 无法创建表格3文件 %s\n", table3_path.c_str());
                continue;
            }
            
            // 写入表头
            table3_file << "error_quantile,recall,coverage,error_value,predicted_error" << std::endl;
            
            // 写入数据
            for (size_t i = 0; i < all_recalls.size(); ++i) {
                ERROR_TYPE recall = all_recalls[i];
                ERROR_TYPE coverage = all_coverages[i];
                ID_TYPE error_i = all_error_indices[i];
                VALUE_TYPE actual_error = batch_alphas[batch_i][error_i];
                double predicted_error = filter_info.node_.get().get_filter().get().predict_error_value(recall, coverage);
                
                table3_file << error_i << ","
                           << std::fixed << std::setprecision(4) << recall << ","
                           << std::fixed << std::setprecision(4) << coverage << ","
                           << std::fixed << std::setprecision(6) << actual_error << ","
                           << std::fixed << std::setprecision(6) << predicted_error << std::endl;
            }
            
            table3_file.close();
            printf("已保存过滤器 %ld 批次 %ld 的表格3: %s\n", (long)filter_id, (long)batch_i, table3_path.c_str());
        }
    }
    
    printf("已完成所有过滤器的数据表格保存\n");
    return SUCCESS;
}


