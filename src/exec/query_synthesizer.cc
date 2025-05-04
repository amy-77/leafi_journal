//
// Created by Qitong Wang on 2024/3/26.
// Copyright (c) 2024 Université Paris Cité. All rights reserved.
//

#include "query_synthesizer.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <immintrin.h>

#include "stat.h"
#include "sort.h"

namespace dstree = upcite::dstree;

dstree::Synthesizer::Synthesizer(dstree::Config &config,
                                 ID_TYPE num_leaves) :
    config_(config),
    num_leaves_(num_leaves) {

  if (num_leaves_ > 0) {
    leaves_.reserve(num_leaves_);
    accumulated_leaf_sizes_.reserve(num_leaves_);
  }
}

RESPONSE dstree::Synthesizer::push_node(dstree::Node &leaf_node) {
  leaves_.emplace_back(leaf_node);
  accumulated_leaf_sizes_.push_back(accumulated_leaf_sizes_.back() + leaf_node.get_size());

  return SUCCESS;
}


RESPONSE dstree::Synthesizer::generate_global_data(VALUE_TYPE *generated_queries) {
  printf("----------进入 generate_global_data ----------\n");
  ID_TYPE num_global_examples = config_.get().filter_train_num_global_example_;
  printf("num_global_examples: %d\n", num_global_examples);
  ID_TYPE num_series_within_filters = accumulated_leaf_sizes_.back();
  printf("num_series_within_filters: %d\n", num_series_within_filters);

  // provided by ChatGPT
  const gsl_rng_type *T;
  gsl_rng *r;
  // Create a generator chosen by the environment variable GSL_RNG_TYPE
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  gsl_rng_set(r, (unsigned long) time(nullptr));

  for (ID_TYPE query_i = 0; query_i < num_global_examples; ++query_i) {
    // 随机选择一个叶子结点
    auto random_i = static_cast<ID_TYPE>(gsl_rng_uniform_int(r, num_series_within_filters));
    ID_TYPE leaf_i = upcite::bSearchFloorID(random_i, accumulated_leaf_sizes_.data(), 0, accumulated_leaf_sizes_.size() - 1);
    // 随机选择当前叶子结点中的一条时间序列
    auto series_i = static_cast<ID_TYPE>(gsl_rng_uniform_int(r, leaves_[leaf_i].get().get_size()));
    VALUE_TYPE const *series_ptr = leaves_[leaf_i].get().get_series_ptr_by_id(series_i);
    // 随机生成噪声
    VALUE_TYPE noise_level = config_.get().filter_query_min_noise_ + gsl_rng_uniform(r) * (
        config_.get().filter_query_max_noise_ - config_.get().filter_query_min_noise_);
    // 生成新的时间序列
    VALUE_TYPE *series_to_generate = generated_queries + config_.get().series_length_ * query_i;
    for (ID_TYPE value_i = 0; value_i < config_.get().series_length_; ++value_i) {
      series_to_generate[value_i] = series_ptr[value_i] + static_cast<VALUE_TYPE>(gsl_ran_gaussian(r, noise_level));
    }
    // 对生成的序列进行Z-score标准化
    RESPONSE return_code = upcite::znormalize(series_to_generate, config_.get().series_length_);
    if (return_code == FAILURE) {
      spdlog::error("node {:d} failed to znorm series {:d} +noise {:.3f}; regenerate",
                    leaves_[leaf_i].get().get_id(), series_i, noise_level);
      query_i -= 1;
    }
  }
  gsl_rng_free(r);
  return SUCCESS;
}



struct LocalGenerationCache {
  LocalGenerationCache(dstree::Config &config,
                       ID_TYPE thread_id,
                       std::vector<std::reference_wrapper<dstree::Node>> &leaves,
                       ID_TYPE *leaf_i,
                       pthread_mutex_t *leaves_mutex) :
      config_(config),
      thread_id_(thread_id),
      leaves_(leaves),
      leaf_i_(leaf_i),
      leaves_mutex_(leaves_mutex) {}

  ~LocalGenerationCache() = default;

  std::reference_wrapper<dstree::Config> config_;
  ID_TYPE thread_id_;
  //leaves_的大小就是叶子节点的数量
  std::vector<std::reference_wrapper<dstree::Node>> leaves_;
  ID_TYPE *leaf_i_;
  pthread_mutex_t *leaves_mutex_;
};


void generation_thread_F(LocalGenerationCache &generation_cache) {
  ID_TYPE num_local_queries = generation_cache.config_.get().filter_train_num_local_example_;
  ID_TYPE series_length = generation_cache.config_.get().series_length_;

  VALUE_TYPE max_noise = generation_cache.config_.get().filter_query_max_noise_;
  VALUE_TYPE min_noise = generation_cache.config_.get().filter_query_min_noise_;

  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), sizeof(VALUE_TYPE) * 8));
  auto generated_series_ptr = static_cast<VALUE_TYPE *>(aligned_alloc(
      sizeof(__m256), sizeof(VALUE_TYPE) * series_length));

  // provided by ChatGPT
  const gsl_rng_type *T;
  gsl_rng *r;
  // Create a generator chosen by the environment variable GSL_RNG_TYPE
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  gsl_rng_set(r, (unsigned long) time(nullptr));

  // 获取叶子节点总数
  ID_TYPE num_leaves = generation_cache.leaves_.size();
  // printf(" 叶子节点总数 num_leaves: %d\n", num_leaves);
  // 使用互斥锁获取下一个要处理的叶子节点索引
  pthread_mutex_lock(generation_cache.leaves_mutex_);
  ID_TYPE local_leaf_i = *generation_cache.leaf_i_;
  // printf(" 当前线程要处理的叶子节点索引 local_leaf_i: %d\n", local_leaf_i);
  *generation_cache.leaf_i_ += 1; // 增加共享索引，为下一个线程准备
  pthread_mutex_unlock(generation_cache.leaves_mutex_);
  
  //遍历每个叶子节点，给每个叶子节点生成num_local_queries个查询
  while (local_leaf_i < num_leaves) {

    dstree::Node &current_node = generation_cache.leaves_[local_leaf_i];
    ID_TYPE node_size = current_node.get_size();
  

    // 所有global queries到当前节点的均值和标准差，确定local queries到当前节点的距离上限
    VALUE_TYPE mean, std;
    std::tie(mean, std) = current_node.get_filter_global_lnn_mean_std();
    VALUE_TYPE max_legal_lnn_distance = mean - std;

    if (max_legal_lnn_distance <= 0) {
      spdlog::error("thread {:d} node {:d} broken global nn dist stats, mean = {:.3f} std = {:.3f}",
                    generation_cache.thread_id_, current_node.get_id(), mean, std);
      spdlog::shutdown();
      exit(FAILURE);
    }
    // printf("节点ID: %d, %d 个时间序列, 距离上限: %.3f\n", current_node.get_id(), node_size, max_legal_lnn_distance);
    spdlog::info("节点ID: {:d}, {:d} 个时间序列, local距离上限: {:.3f}", current_node.get_id(), node_size, max_legal_lnn_distance);
    // 给每个叶子节点生成num_local_queries个查询
    for (ID_TYPE query_i = 0; query_i < num_local_queries; ++query_i) {
      // GSL库的函数，生成一个范围在[0, current_node.get_size()-1]的随机整数
      auto series_i = static_cast<ID_TYPE>(gsl_rng_uniform_int(r, current_node.get_size()));
      // 从当前叶子结点中随机选择一个时间序列
      VALUE_TYPE const *series_ptr = current_node.get_series_ptr_by_id(series_i);
      VALUE_TYPE noise_level = min_noise + gsl_rng_uniform(r) * (max_noise - min_noise);

      // 为一条原始序列的每个值添加高斯噪声
      for (ID_TYPE value_i = 0; value_i < series_length; ++value_i) {
        generated_series_ptr[value_i] = series_ptr[value_i] + static_cast<VALUE_TYPE>(gsl_ran_gaussian(r, noise_level));
      }

      // 对生成的序列进行Z-score标准化
      RESPONSE return_code = upcite::znormalize(generated_series_ptr, series_length);
      if (return_code == FAILURE) {
        // 如果标准化失败，重新生成序列
        spdlog::error("thread {:d} node {:d} failed to znorm series {:d} +noise {:.3f}; regenerate",
                      generation_cache.thread_id_, current_node.get_id(), series_i, noise_level);
        query_i -= 1;

      } else {
        // TODO check and prune if the generated series falling in the same leaf node
        // does it work for the EAPCA envelop of dstree?
        // QYL: 改 创建一个临时answer对象
        auto temp_answer = std::make_shared<dstree::Answers>(generation_cache.config_.get().n_nearest_neighbor_, query_i);
        // 传入answer参数
        VALUE_TYPE local_nn_distance = current_node.search(generated_series_ptr, query_i, m256_fetch_cache, temp_answer.get());
        // 检查生成的序列是否仍在局部邻域内
        if (local_nn_distance > max_legal_lnn_distance) {
           // 如果距离过大，重试当前样本
          spdlog::error(
              "thread {:d} node {:d} series {:d} +noise {:.3f} escaped the local neighbourhood ({:.3f} > {:.3f}); regenerate",
              generation_cache.thread_id_,
              current_node.get_id(),
              series_i,
              noise_level,
              local_nn_distance,
              max_legal_lnn_distance);

          query_i -= 1;
        } else {
          // 如果距离在范围内，保存生成的序列
          current_node.push_local_example(generated_series_ptr, local_nn_distance);
          spdlog::info("thread {:d} node {:d} series {:d} +noise {:.3f}, lnn = {:.3f} <= {:.3f}",
                       generation_cache.thread_id_, current_node.get_id(), series_i, noise_level,
                       local_nn_distance, max_legal_lnn_distance);
        }
      }
    }

    // 计算并打印本地查询距离统计
    auto [local_mean, local_std] = current_node.get_filter_local_lnn_mean_std();
    // printf("节点ID: %d 生成的本地查询距离统计 - 均值: %.6f, 标准差: %.6f\n", 
    //        current_node.get_id(), local_mean, local_std);
    
    // 比较全局查询和本地查询的距离分布
    auto [global_mean, global_std] = current_node.get_filter_global_lnn_mean_std();
    // printf("节点ID: %d 对比 - 全局查询均值: %.6f±%.6f vs 本地查询均值: %.6f±%.6f\n",
    //        current_node.get_id(), global_mean, global_std, local_mean, local_std);

    // 将生成的本地样本保存到磁盘 
    current_node.dump_local_example();
    // 使用互斥锁获取下一个要处理的叶子节点
    pthread_mutex_lock(generation_cache.leaves_mutex_);
    local_leaf_i = *generation_cache.leaf_i_;
    *generation_cache.leaf_i_ += 1;
    pthread_mutex_unlock(generation_cache.leaves_mutex_);
  }

  gsl_rng_free(r);
  free(m256_fetch_cache);
  free(generated_series_ptr);
}


/*
从每个叶子节点中随机选择时间序列作为基础
添加随机高斯噪声生成新的时间序列
在当前节点中搜索生成序列的最近邻
确保生成的查询与现有数据足够相似（距离不超过阈值）
*/
RESPONSE dstree::Synthesizer::generate_local_data() {
  // printf("----------进入 generate_local_data ----------\n");
  // printf("开始生成本地查询, filter_train_is_mthread_=%d, nthread=%ld\n", 
  //        config_.get().filter_train_is_mthread_, (long)config_.get().filter_train_nthread_);
         
  if (!config_.get().filter_train_is_mthread_) {
    // 单线程版本的实现
    printf("使用单线程生成本地查询数据\n");
    
    // 为单线程生成设置必要的资源
    ID_TYPE num_local_queries = config_.get().filter_train_num_local_example_;
    ID_TYPE series_length = config_.get().series_length_;
    ID_TYPE num_leaves = leaves_.size();
    
    // printf("叶子节点数量: %ld, 每个节点生成查询数: %ld\n", 
    //        (long)num_leaves, (long)num_local_queries);

    VALUE_TYPE max_noise = config_.get().filter_query_max_noise_;
    VALUE_TYPE min_noise = config_.get().filter_query_min_noise_;

    auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), sizeof(VALUE_TYPE) * 8));
    auto generated_series_ptr = static_cast<VALUE_TYPE *>(aligned_alloc(
        sizeof(__m256), sizeof(VALUE_TYPE) * series_length));

    // 初始化随机数生成器
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, (unsigned long) time(nullptr));

    // 依次处理每个叶子节点, 给每个叶子节点生成num_local_queries个查询
    for (ID_TYPE leaf_i = 0; leaf_i < num_leaves; ++leaf_i) {
      dstree::Node &current_node = leaves_[leaf_i];
      
      VALUE_TYPE mean, std; 
      // 获取节点统计信息，用于计算距离上限
      std::tie(mean, std) = current_node.get_filter_global_lnn_mean_std();
      VALUE_TYPE max_legal_lnn_distance = mean - std;
      
      if (max_legal_lnn_distance <= 0) {
        printf("警告: 节点%ld的全局nn距离统计异常, 均值=%.3f 标准差=%.3f\n",
               (long)current_node.get_id(), (double)mean, (double)std);
        continue; // 跳过这个节点，继续处理下一个
      }
      
      // printf("处理节点%ld, 最大合法距离阈值: %.3f\n", (long)current_node.get_id(), (double)max_legal_lnn_distance);
      
      // 为当前节点生成查询
      ID_TYPE successful_queries = 0;
      ID_TYPE attempts = 0;
      const ID_TYPE MAX_ATTEMPTS = num_local_queries * 3; // 允许的最大尝试次数
      
      while (successful_queries < num_local_queries && attempts < MAX_ATTEMPTS) {
        attempts++;
        
        // 从节点数据中随机选择一个序列作为基础
        auto series_i = static_cast<ID_TYPE>(gsl_rng_uniform_int(r, current_node.get_size()));
        VALUE_TYPE const *series_ptr = current_node.get_series_ptr_by_id(series_i);
        
        // 添加随机噪声
        VALUE_TYPE noise_level = min_noise + gsl_rng_uniform(r) * (max_noise - min_noise);
        for (ID_TYPE value_i = 0; value_i < series_length; ++value_i) {
          generated_series_ptr[value_i] = series_ptr[value_i] + static_cast<VALUE_TYPE>(gsl_ran_gaussian(r, noise_level));
        }
        
        // 对生成的序列进行Z-score标准化
        RESPONSE return_code = upcite::znormalize(generated_series_ptr, series_length);
        if (return_code == FAILURE) {
          if (attempts % 100 == 0) { // 只打印部分失败信息，避免日志过多
            printf("节点%ld: 标准化序列%ld (噪声%.3f)失败，重试\n",
                   (long)current_node.get_id(), (long)series_i, (double)noise_level);
          }
          continue; // 重试
        }
        
        // 在当前节点中搜索生成序列的最近邻
        // 创建临时answer对象，解决answer为空的问题
        auto temp_answer = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, successful_queries);
        VALUE_TYPE local_nn_distance = current_node.search(generated_series_ptr, successful_queries, m256_fetch_cache, temp_answer.get());
        
        // 确保生成的查询与现有数据足够相似（距离不超过阈值）
        if (local_nn_distance > max_legal_lnn_distance) {
          if (attempts % 100 == 0) { // 只打印部分失败信息
            printf("节点%ld: 序列%ld (噪声%.3f)超出本地邻域 (%.3f > %.3f)，重试\n",
                   (long)current_node.get_id(), (long)series_i, (double)noise_level,
                   (double)local_nn_distance, (double)max_legal_lnn_distance);
          }
          continue; // 重试
        }
        
        // 保存有效的本地示例
        current_node.push_local_example(generated_series_ptr, local_nn_distance);
        successful_queries++;
        
        if (successful_queries % 100 == 0 || successful_queries == num_local_queries) {
          printf("节点%ld: 成功生成%ld/%ld个查询\n", (long)current_node.get_id(), (long)successful_queries, (long)num_local_queries);
        }
      }
      
      if (successful_queries < num_local_queries) {
        printf("警告: 节点%ld只生成了%ld/%ld个查询 (尝试次数:%ld)\n",
               (long)current_node.get_id(), (long)successful_queries, 
               (long)num_local_queries, (long)attempts);
      }
      
      // 计算并打印本地查询距离统计
      auto [local_mean, local_std] = current_node.get_filter_local_lnn_mean_std();
      // printf("节点ID: %d 生成的本地查询距离统计 - 均值: %.2f, 标准差: %.6f\n", 
      //        current_node.get_id(), local_mean, local_std);
      
      // 比较全局查询和本地查询的距离分布
      auto [global_mean, global_std] = current_node.get_filter_global_lnn_mean_std();
      // printf("节点ID: %d 对比 - 全局查询均值: %.2f±%.2f vs 本地查询均值: %.2f±%.2f\n",
      //        current_node.get_id(), global_mean, global_std, local_mean, local_std);
      // 保存到磁盘
      current_node.dump_local_example();
    }
    
    // 释放资源
    gsl_rng_free(r);
    free(m256_fetch_cache);
    free(generated_series_ptr);
    
    printf("单线程本地查询生成完成\n");
    return SUCCESS;
  }

  // 多线程版本实现
  // printf("使用多线程生成本地查询数据, 线程数: %ld\n", (long)config_.get().filter_train_nthread_);
  //generation_caches 是每个线程的生成器
  std::vector<std::unique_ptr<LocalGenerationCache>> generation_caches;
  ID_TYPE leaf_i = 0;
  std::unique_ptr<pthread_mutex_t> leaves_mutex = std::make_unique<pthread_mutex_t>();
  
  // 初始化互斥锁
  pthread_mutex_init(leaves_mutex.get(), nullptr);
  for (ID_TYPE thread_i = 0; thread_i < config_.get().filter_train_nthread_; ++thread_i) {
    generation_caches.emplace_back(std::make_unique<LocalGenerationCache>(config_, thread_i, std::ref(leaves_), &leaf_i, leaves_mutex.get()));
  }

  std::vector<std::thread> threads;
  for (ID_TYPE thread_i = 0; thread_i < config_.get().filter_train_nthread_; ++thread_i) {
    threads.emplace_back(generation_thread_F, std::ref(*generation_caches[thread_i]));
  }
  for (ID_TYPE thread_i = 0; thread_i < config_.get().filter_train_nthread_; ++thread_i) {
    threads[thread_i].join();
  }
  printf("多线程本地查询生成完成\n");
  return SUCCESS;
}
