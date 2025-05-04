//
// Created by Qitong Wang on 2022/10/6.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "index.h"

#include <tuple>
#include <memory>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <chrono>
#include <thread>

#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>
#include <functional> // 需要包含以使用 std::reference_wrapper
// #include <iterator>   // 确保迭代器支持（可选，但建议包含）

#include "vec.h"
#include "eapca.h"
#include "answer.h"
#include "query_synthesizer.h"

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Index::Index(Config &config) : config_(config),
                                       nnode_(0),
                                       nleaf_(0),
                                       filter_train_query_ptr_(nullptr),
                                       allocator_(nullptr),
                                       navigator_(nullptr)
{
  buffer_manager_ = std::make_unique<dstree::BufferManager>(config_);

  root_ = std::make_unique<dstree::Node>(config_, *buffer_manager_, 0, nnode_);
  nnode_ += 1, nleaf_ += 1;

  if (config_.get().filter_infer_is_gpu_){
    // TODO support multiple devices
    device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
  } else {
    device_ = std::make_unique<torch::Device>(torch::kCPU);
  }

  if (config_.get().require_neurofilter_){
    allocator_ = std::make_unique<dstree::Allocator>(config);
  }
}



dstree::Index::~Index(){
  if (filter_train_query_ptr_ != nullptr)
  {
    std::free(filter_train_query_ptr_);
    filter_train_query_ptr_ = nullptr;
  }
}




// 构建树索引，这里insert函数就是构建index的关键流程，把数据插入到树中
// 构建好index以后, 调用train函数训练filter (train函数很长，包含了collect training数据，训练filter等)
RESPONSE dstree::Index::build(){
  //-------------------1. build index-------------------
  while (buffer_manager_->load_batch() == SUCCESS){
    // printf("---------------[DEBUG] Loaded a batch of size: %d\n", buffer_manager_->load_buffer_size());
    for (ID_TYPE series_id = 0; series_id < buffer_manager_->load_buffer_size(); ++series_id){
      // 将当前数据点（由 series_id 标识）插入到树中。
      //  printf("[DEBUG] Inserting series_id: %d\n", series_id);
      insert(series_id);
    }
    // printf("[DEBUG] Inserting series_id finished \n");
    if (config_.get().on_disk_){
      buffer_manager_->flush();
    }
  }
  printf("叶子节点的数量: %d\n", nleaf_);

  // 新增：打印所有叶子节点的ID
  // auto leaf_ids = get_all_leaf_ids();
  // printf("所有叶子节点的ID: [");
  // for (size_t i = 0; i < leaf_ids.size(); ++i) {
  //     printf("%d", leaf_ids[i]);
  //     if (i < leaf_ids.size() - 1) printf(", ");
  // }
  // printf("]\n");

  if (!buffer_manager_->is_fully_loaded()){ // 检查是否所有数据都已成功加载
    printf("-------[ERROR] Failed to fully load data.\n");
    return FAILURE;
  }
  // leaf_min_heap_: 这是一个优先队列，存储lb<minBSF的节点
  leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist>(
      CompareDecrNodeDist(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));

  //------------------2. train filter 训练索引中的过滤器----------------
  // navigator 开头的参数都不用管; 这个是学一个模型用来改变叶节点的访问顺序的
  if (config_.get().require_neurofilter_ || config_.get().navigator_is_learned_){
    //**********************   train filter  ********************* */
    printf("----------[DEBUG] Training filters...\n");
    train();
    // QYL: 传递 Index 的 train_answers_ 给Allocator
    // allocator_ = std::make_unique<dstree::Allocator>(train_answers_);
  }
  printf("[DEBUG] Index build completed successfully.\n");
  return SUCCESS;
}



// 在 Index 类中添加
std::vector<ID_TYPE> upcite::dstree::Index::get_all_leaf_ids() const{
  // std::vector<std::reference_wrapper<Node>> leaves;
  std::vector<std::reference_wrapper<upcite::dstree::Node>> leaves; // 明确使用你的 Node 类
  root_->enqueue_leaf(leaves); // 收集所有叶子节点
  std::vector<ID_TYPE> leaf_ids;
  for (const auto &leaf_ref : leaves){
    leaf_ids.push_back(leaf_ref.get().get_id());
  }
  return leaf_ids;
}



//
RESPONSE dstree::Index::insert(ID_TYPE batch_series_id){
  if (config_.get().is_sketch_provided_){
    buffer_manager_->emplace_series_eapca(std::move(std::make_unique<dstree::EAPCA>(
        buffer_manager_->get_sketch_ptr(batch_series_id),
        config_.get().sketch_length_,
        config_.get().vertical_split_nsubsegment_)));
  } else {
    buffer_manager_->emplace_series_eapca(std::move(std::make_unique<dstree::EAPCA>(
        buffer_manager_->get_series_ptr(batch_series_id),
        config_.get().series_length_,
        config_.get().vertical_split_nsubsegment_)));
  }
  dstree::EAPCA &series_eapca = buffer_manager_->get_series_eapca(batch_series_id);
  std::reference_wrapper<dstree::Node> target_node = std::ref(*root_);

  while (!target_node.get().is_leaf()){
    target_node = target_node.get().route(series_eapca, true);
  }
  if (target_node.get().is_full()){
    target_node.get().split(*buffer_manager_, nnode_);
    nnode_ += config_.get().node_nchild_, nleaf_ += config_.get().node_nchild_ - 1;
    target_node = target_node.get().route(series_eapca, true);
  }
  return target_node.get().insert(batch_series_id, series_eapca);
}



// 递归函数，给所有叶子节点插入初始filter，并且所有filter存入filter_cache中
RESPONSE dstree::Index::filter_initialize(dstree::Node &node,
                                          ID_TYPE *filter_id){
  if (!filter_id) {
    fprintf(stderr, "[ERROR] filter_initialize: filter_id pointer is null\n");
    return FAILURE;
  }

  try {
    if (node.is_leaf()) {
      // 为叶节点添加过滤器
      if (!filter_train_query_tsr_.defined()) {
        fprintf(stderr, "[ERROR] filter_initialize: filter_train_query_tsr_ is not defined\n");
        return FAILURE;
      }
      
      // 使用节点ID而不是顺序ID作为过滤器ID，更好地支持故障恢复
      ID_TYPE node_id = node.get_id();
      RESPONSE result = node.add_filter(node_id, filter_train_query_tsr_);
      
      if (result != SUCCESS) {
        fprintf(stderr, "[ERROR] Failed to add filter to node ID: %ld\n", (long)node_id);
        return FAILURE;
      }
      
      // 检查是否成功获取过滤器引用
      try {
        auto filter_ref = node.get_filter();
        filter_cache_.push(filter_ref);
        
        // 追踪过滤器ID到过滤器的映射（方便后续查找）
        filter_id_to_filter_[node_id] = &filter_ref.get();
        
        // printf("[INFO] Successfully initialized filter for node ID: %ld\n", (long)node_id);
      } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Failed to get filter reference from node ID: %ld, error: %s\n", 
               (long)node_id, e.what());
        return FAILURE;
      }
      
      (*filter_id)++;
    } else {
      // 递归处理子节点
      bool child_success = true;
      for (auto child_node : node) {
        RESPONSE child_result = filter_initialize(child_node, filter_id);
        if (child_result != SUCCESS) {
          fprintf(stderr, "[WARNING] Failed to initialize filter for a child of node ID: %ld\n", 
                 (long)node.get_id());
          child_success = false;
        }
      }
      
      if (!child_success) {
        return FAILURE;
      }
    }
    
    return SUCCESS;
  } catch (const std::exception& e) {
    fprintf(stderr, "[ERROR] Exception in filter_initialize for node ID: %ld, error: %s\n", 
           (long)node.get_id(), e.what());
    return FAILURE;
  }
}

RESPONSE dstree::Index::filter_deactivate(dstree::Node &node){
  if (node.is_leaf()){
    if (node.has_active_filter()){
      node.deactivate_filter();
    }
  } else {
    for (auto child_node : node)
    {
      filter_deactivate(child_node);
    }
  }

  return SUCCESS;
}



// 通过遍历树结构，计算查询序列与树中节点的局部最近邻距离，并更新全局最佳距离（BSF）
RESPONSE dstree::Index::filter_collect(){
  // printf("\n ------------------ 进入filter_collect --------------\n");
  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), sizeof(VALUE_TYPE) * 8));
  if (!m256_fetch_cache){
    fprintf(stderr, "[ERROR] Failed to allocate aligned memory for m256_fetch_cache\n");
    return FAILURE;
  }
  // 这里收集的数据是global query所有的global_1nn_distances和global_bsf_distances (针对每个filter)
  for (ID_TYPE query_id = 0; query_id < config_.get().filter_train_nexample_; ++query_id){
    const VALUE_TYPE *series_ptr = filter_train_query_ptr_ + config_.get().series_length_ * query_id;
    ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
    ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;

    // 初始化answer
    auto answer = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
    if (!answer) {
      fprintf(stderr, "[ERROR] Failed to create answer object\n");
      return FAILURE;
    }
    std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
    //----------- 近似搜索：第一部分：找到最近的叶子节点并计算局部最近邻距离 -----------------
    // step1: 找到和查询序列series_ptr最近的叶子节点
    while (!resident_node.get().is_leaf()) {
      resident_node = resident_node.get().route(series_ptr);
    }
    // step2: 获取当前query(series_ptr)到叶节点resident_node下所有时间序列的最近local距离
    auto minbsf = answer->get_bsf();
    VALUE_TYPE local_nn_distance = resident_node.get().search(series_ptr, query_id, m256_fetch_cache, answer.get());
    if (std::isnan(local_nn_distance) || local_nn_distance < 0) {
      fprintf(stderr, "Invalid distance: %.3f\n", local_nn_distance);
      continue;
    }
    
    // 存储global query到当前叶节点的最近距离 在 global_nn_distance 和 global_bsf_distance中
    resident_node.get().push_global_example(minbsf, local_nn_distance, 0); // answer->get_bsf()应该存访问该节点之前的全局最近邻，
    visited_node_counter += 1;
    visited_series_counter += resident_node.get().get_size();

    // 每访问一个叶子节点，用当前节点的局部1NN距离来决定是否更新bsf_distances(minBSF)
    if (answer->is_bsf(local_nn_distance)) {
      // spdlog::info("filter query {:d} update bsf {:.3f} after node {:d} series {:d}", query_id, local_nn_distance, visited_node_counter, visited_series_counter);
      answer->push_bsf(local_nn_distance, resident_node.get().get_id());
    }
    leaf_min_heap_.push(std::make_tuple(std::ref(*root_), 0));


    //------------------------ 第二部分：遍历其他叶子节点并更新全局最佳距离 --------------------
    // 创建优先队列，利用lb，使用优先队列（leaf_min_heap_）遍历其他叶子节点，
    std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
    VALUE_TYPE node2visit_lbdistance;
    while (!leaf_min_heap_.empty()){
      if (leaf_min_heap_.empty()){
        printf("警告：堆为空但尝试访问元素\n");
        break;
      }

      // 每次从队列取出下界距离最小的节点进行处理
      std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();
      leaf_min_heap_.pop();

      if (node_to_visit.get().is_leaf()) {
        //确认不是已经访问过的叶子节点
        if (node_to_visit.get().get_id() != resident_node.get().get_id()) {
          auto minbsf = answer->get_bsf();
          local_nn_distance = node_to_visit.get().search(series_ptr, query_id, m256_fetch_cache, answer.get());
          //！！！把global query到当前叶节点的最近距离 在 global_nn_distance 和 global_bsf_distance中
          
          node_to_visit.get().push_global_example(minbsf, local_nn_distance, node2visit_lbdistance);
          visited_node_counter += 1;
          visited_series_counter += node_to_visit.get().get_size();
          if (answer->is_bsf(local_nn_distance)) {
            answer->push_bsf(local_nn_distance, node_to_visit.get().get_id());
          }
        }

      } else {
        // printf("-----search child node %d 's knn:  -----\n", node_to_visit.get().get_id());
        for (auto child_node : node_to_visit.get()) {
          VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(series_ptr);
          leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
        }
      }
    }

    // printf("-----遍历结束------\n");
    query_knn_nodes_[query_id].clear();
    while (!answer->empty()) {
      auto answer_i = answer->pop_answer();
      if (answer_i.node_id_ > 0){
        query_knn_nodes_[query_id][answer_i.node_id_]++;
      }
      nnn_to_return -= 1;
    }

    // 打印最终的统计结果（调试用）
    // printf("---- query_knn_nodes_[%d] 统计结果 ----\n", query_id);
    // for (const auto& [node_id, count] : query_knn_nodes_[query_id]) {
    //     printf("query_id=%d, node_id=%d, count=%d\n", query_id, node_id, count);
    // }

  }
  // 使用后确保释放内存
  std::free(m256_fetch_cache);
  return SUCCESS;
}




// // 通过遍历树结构，计算查询序列与树中节点的局部最近邻距离，并更新全局最佳距离（BSF）
// RESPONSE dstree::Index::filter_collect()
// {
//   printf("\n ------------------ 进入filter_collect --------------\n");

//   auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), sizeof(VALUE_TYPE) * 8));
//   if (!m256_fetch_cache)
//   {
//     fprintf(stderr, "[ERROR] Failed to allocate aligned memory for m256_fetch_cache\n");
//     return FAILURE;
//   }
//   // 这里收集的数据是训练集+测试集(校准集）所有的global_1nn_distances和global_bsf_distances (这是当前filter下的)，事实上所有的filter都会被收集）
//   for (ID_TYPE query_id = 0; query_id < config_.get().filter_train_nexample_; ++query_id)
//   {
//     const VALUE_TYPE *series_ptr = filter_train_query_ptr_ + config_.get().series_length_ * query_id;
//     ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
//     ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;

//     // 初始化answer
//     auto answer = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
//     if (!answer)
//     {
//       fprintf(stderr, "[ERROR] Failed to create answer object\n");
//       return FAILURE;
//     }
//     std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
//     //----------- 近似搜索：第一部分：找到最近的叶子节点并计算局部最近邻距离 -----------------
//     // 找到和查询序列series_ptr的特征值最近的叶子节点
//     while (!resident_node.get().is_leaf())
//     {
//       resident_node = resident_node.get().route(series_ptr);
//     }
//     // 获取当前query(series_ptr)到节点resident_node下所有时间序列的最近local距离
//     auto minbsf = answer->get_bsf();
//     VALUE_TYPE local_nn_distance = resident_node.get().search(series_ptr, query_id, m256_fetch_cache, answer.get());
//     if (std::isnan(local_nn_distance) || local_nn_distance < 0)
//     {
//       fprintf(stderr, "Invalid distance: %.3f\n", local_nn_distance);
//       continue;
//     }
//     // QYL: minbsf should be calculated before searching this node.
//     resident_node.get().push_global_example(minbsf, local_nn_distance, 0); // answer->get_bsf()应该存访问该节点之前的全局最近邻，
//     visited_node_counter += 1;
//     visited_series_counter += resident_node.get().get_size();

//     // 每访问一个叶子节点，用当前节点的局部1NN距离来决定是否更新minBSF
//     if (answer->is_bsf(local_nn_distance))
//     {
//       answer->push_bsf(local_nn_distance, resident_node.get().get_id());
//     }
//     leaf_min_heap_.push(std::make_tuple(std::ref(*root_), 0));



    
//     //------------------------ 第二部分：遍历其他叶子节点并更新全局最佳距离 --------------------
//     // 创建优先队列，利用lb，使用优先队列（leaf_min_heap_）遍历其他叶子节点，
//     std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
//     VALUE_TYPE node2visit_lbdistance;
//     // printf("Initial size of leaf_min_heap_: %zu\n", leaf_min_heap_.size());
//     while (!leaf_min_heap_.empty())
//     {
//       // printf("Current size of leaf_min_heap_: %zu\n", leaf_min_heap_.size());
//       // printf("-----遍历最小堆的其余叶子节点------\n");
//       if (leaf_min_heap_.empty())
//       {
//         printf("警告：堆为空但尝试访问元素\n");
//         break;
//       }

//       // 在处理pop()后的元素前做额外验证
//       std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();
//       leaf_min_heap_.pop();

//       if (node_to_visit.get().is_leaf())
//       {
//         // printf("-----search leaf node %d :  -----\n", node_to_visit.get().get_id());
//         if (node_to_visit.get().get_id() != resident_node.get().get_id())
//         {
//           // QYL
//           auto minbsf = answer->get_bsf();
//           // printf("节点ID: %d, 地址: %p\n", node_to_visit.get().get_id(), (void*)&node_to_visit.get());

//           local_nn_distance = node_to_visit.get().search(series_ptr, query_id, m256_fetch_cache, answer.get());
//           // printf("push leaf node %d 's knn  \n", node_to_visit.get().get_id());
//           node_to_visit.get().push_global_example(minbsf, local_nn_distance,
//                                                   node2visit_lbdistance);

//           visited_node_counter += 1;
//           visited_series_counter += node_to_visit.get().get_size();

//           if (answer->is_bsf(local_nn_distance))
//           {
//             // spdlog::info("filter query {:d} update bsf {:.3f} after node {:d} series {:d}",
//             //              query_id, local_nn_distance, visited_node_counter, visited_series_counter);
//             // printf("filter query %d update bsf %.3f after node %d series %d\n",
//             //               query_id, local_nn_distance, visited_node_counter, visited_series_counter);
//             answer->push_bsf(local_nn_distance, node_to_visit.get().get_id());
//           }
//         }
//       }
//       else
//       {
//         // printf("-----search child node %d 's knn:  -----\n", node_to_visit.get().get_id());
//         for (auto child_node : node_to_visit.get())
//         {
//           VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(series_ptr);
//           // printf("push child node \n");
//           leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
//         }
//       }
//     }
//     // printf("-----遍历结束------\n");
//     query_knn_nodes_[query_id].clear();
//     while (!answer->empty())
//     {
//       auto answer_i = answer->pop_answer();
//       if (answer_i.node_id_ > 0)
//       {
//         // printf("query %d nn %d = %.3f, node %d\n", query_id, nnn_to_return, answer_i.nn_dist_, answer_i.node_id_);
//         // 统计 node_id 出现的次数
//         query_knn_nodes_[query_id][answer_i.node_id_]++;
//       }
//       nnn_to_return -= 1;
//     }
//     // 打印最终的统计结果（调试用）
//     // printf("---- query_knn_nodes_[%d] 统计结果 ----\n", query_id);
//     // for (const auto& [node_id, count] : query_knn_nodes_[query_id]) {
//     //     printf("query_id=%d, node_id=%d, count=%d\n", query_id, node_id, count);
//     // }
//   }
//   // 使用后确保释放内存
//   std::free(m256_fetch_cache);
//   return SUCCESS;
// }

// auto& node_counts = query_knn_nodes_[query_id];
// auto knn_distribution = answer->get_knn_node_distribution(query_id);
// printf("----访问answer中的knn_distribution------\n");
// for (const auto& [node_id, stats] : knn_distribution) {
//   node_counts[node_id] = stats.first; // stats.first 是 count_in_node
// }

// std::vector<ID_TYPE> knn_nodes = answer->get_knn_node_ids();
// query_knn_nodes_[query_id] = knn_nodes;
// 往 train_answers_存入answer，这部分内容会导致程序崩溃Segmentation fault (core dumped)
// train_answers_.emplace_back(dstree::Answers(*answer));
// if (config_.get().navigator_is_learned_) {
//   nnn_to_return = config_.get().navigator_train_k_nearest_neighbor_;
// }
// //打印内容

struct SearchCache{
  SearchCache(ID_TYPE thread_id,
              VALUE_TYPE *m256_fetch_cache,
              dstree::Answers *answer,
              pthread_mutex_t *answer_mutex,
              std::reference_wrapper<std::priority_queue<dstree::NODE_DISTNCE,
                                                         std::vector<dstree::NODE_DISTNCE>,
                                                         dstree::CompareDecrNodeDist>>
                  leaf_min_heap,
              pthread_mutex_t *leaf_pq_mutex,
              ID_TYPE *visited_node_counter,
              ID_TYPE *visited_series_counter,
              pthread_mutex_t *log_mutex) : thread_id_(thread_id),
                                            query_id_(-1),
                                            query_series_ptr_(nullptr),
                                            m256_fetch_cache_(m256_fetch_cache),
                                            answer_(answer),
                                            answer_mutex_(answer_mutex),
                                            leaf_min_heap_(leaf_min_heap),
                                            leaf_pq_mutex_(leaf_pq_mutex),
                                            visited_node_counter_(visited_node_counter),
                                            visited_series_counter_(visited_series_counter),
                                            log_mutex_(log_mutex) {}

  ID_TYPE thread_id_;
  ID_TYPE query_id_;
  VALUE_TYPE *query_series_ptr_;
  VALUE_TYPE *m256_fetch_cache_;
  dstree::Answers *answer_;
  pthread_mutex_t *answer_mutex_;
  std::reference_wrapper<std::priority_queue<
      dstree::NODE_DISTNCE, std::vector<dstree::NODE_DISTNCE>, dstree::CompareDecrNodeDist>>
      leaf_min_heap_;
  pthread_mutex_t *leaf_pq_mutex_;
  ID_TYPE *visited_node_counter_;
  ID_TYPE *visited_series_counter_;
  pthread_mutex_t *log_mutex_;
};

//
void search_thread_F(const SearchCache &search_cache){
  // aligned_alloc within thread might cause a "corrupted size vs. prev_size" glibc error
  // https://stackoverflow.com/questions/49628615/understanding-corrupted-size-vs-prev-size-glibc-error
  //  auto m256_fetch_cache = std::unique_ptr<VALUE_TYPE>(static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), 8)));

  // WARN undefined behaviour
  std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
  VALUE_TYPE node2visit_lbdistance;

  while (true){
    pthread_mutex_lock(search_cache.leaf_pq_mutex_);
    if (search_cache.leaf_min_heap_.get().empty()){
      pthread_mutex_unlock(search_cache.leaf_pq_mutex_);
      break;
    }
    else
    {
      // 从队列顶部获取节点和下界距离
      std::tie(node_to_visit, node2visit_lbdistance) = search_cache.leaf_min_heap_.get().top();
      search_cache.leaf_min_heap_.get().pop();
      pthread_mutex_unlock(search_cache.leaf_pq_mutex_);
    }

    // 获取 global_bsf:  for a more precise bsf distance   这个没问题，访问节点之前先获取bsf
    pthread_mutex_lock(search_cache.answer_mutex_);
    VALUE_TYPE global_bsf = search_cache.answer_->get_bsf();
    pthread_mutex_unlock(search_cache.answer_mutex_);

    // 这个代码只统计了有过滤器的节点的最近邻，并且push_global_example
    if (node_to_visit.get().has_filter())
    {
      // 这里搜索的时候好像只找了当前节点下的最近邻？

      VALUE_TYPE local_nn_distance = node_to_visit.get().search_mt(
          search_cache.query_series_ptr_, search_cache.query_id_, *search_cache.answer_, search_cache.answer_mutex_);

      // 调用 push_global_example 并打印
      node_to_visit.get().push_global_example(global_bsf, local_nn_distance, node2visit_lbdistance);
      //  QYL 假设 local_knn_distances 是当前节点的 K 个最近邻距离
      // std::vector<VALUE_TYPE> local_knn_distances = node_to_visit.get().search_k_mt(search_cache.query_series_ptr, search_cache.answer, search_cache.answer_mutex, K);

      pthread_mutex_lock(search_cache.log_mutex_);
      *search_cache.visited_node_counter_ += 1;
      *search_cache.visited_series_counter_ += node_to_visit.get().get_size();
      pthread_mutex_unlock(search_cache.log_mutex_);
    }
    else if (node2visit_lbdistance <= global_bsf)
    {
      // 当前节点没有激活filter，并且lb无法减枝时，也需要访问当前节点，但是此时由于没有插入filter，所以不需要存储训练和校准数据。
      VALUE_TYPE local_nn_distance = node_to_visit.get().search_mt(
          search_cache.query_series_ptr_, search_cache.query_id_, *search_cache.answer_, search_cache.answer_mutex_);

      pthread_mutex_lock(search_cache.log_mutex_);
      *search_cache.visited_node_counter_ += 1;
      *search_cache.visited_series_counter_ += node_to_visit.get().get_size();
      pthread_mutex_unlock(search_cache.log_mutex_);
    }
  }
}

// 收集信息
RESPONSE dstree::Index::filter_collect_mthread()
{
  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(
      sizeof(__m256), 8 * config_.get().filter_collect_nthread_));

  ID_TYPE visited_node_counter = 0;
  ID_TYPE visited_series_counter = 0;

  std::unique_ptr<Answers> answer = nullptr;
  if (config_.get().navigator_is_learned_)
  { // TODO
    assert(!config_.get().require_neurofilter_);
    answer = std::make_unique<dstree::Answers>(config_.get().navigator_train_k_nearest_neighbor_, -1);
  }
  else
  {
    answer = std::make_unique<dstree::Answers>(config_.get().n_nearest_neighbor_, -1);
  }

  std::unique_ptr<pthread_mutex_t> answer_mutex = std::make_unique<pthread_mutex_t>();
  std::unique_ptr<pthread_mutex_t> leaf_pq_mutex = std::make_unique<pthread_mutex_t>();
  std::unique_ptr<pthread_mutex_t> log_mutex = std::make_unique<pthread_mutex_t>();

  pthread_mutex_init(answer_mutex.get(), nullptr);
  pthread_mutex_init(leaf_pq_mutex.get(), nullptr);
  pthread_mutex_init(log_mutex.get(), nullptr);

  std::vector<SearchCache> search_caches;
  std::stack<std::reference_wrapper<dstree::Node>> node_stack;

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id)
  {
    search_caches.emplace_back(thread_id,
                               m256_fetch_cache + 8 * thread_id,
                               answer.get(),
                               answer_mutex.get(),
                               std::ref(leaf_min_heap_),
                               leaf_pq_mutex.get(),
                               &visited_node_counter,
                               &visited_series_counter,
                               log_mutex.get()); // 传递当前 query_id
  }

  // QYL 修改 Answers 初始化逻辑，明确 K 值
  // ID_TYPE K = config_.get().n_nearest_neighbor_;
  // if (config_.get().navigator_is_learned_) {
  //     K = config_.get().navigator_train_k_nearest_neighbor_;
  // }
  // answer = std::make_unique<dstree::Answers>(target_k); // 传入 K 值

  // --------1. 近似搜索: 遍历每个query，找当前query最可能落在哪个叶子节点，就计算该叶子节点下的1NN-------
  for (ID_TYPE query_id = 0; query_id < config_.get().filter_train_nexample_; ++query_id)
  {
    VALUE_TYPE *series_ptr = filter_train_query_ptr_ + config_.get().series_length_ * query_id;

    visited_node_counter = 0;
    visited_series_counter = 0;
    answer->reset(query_id);
    // 从根节点开始遍历树
    std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
    // 找到当前查询序列距离最近的叶子节点
    while (!resident_node.get().is_leaf())
    {
      resident_node = resident_node.get().route(series_ptr);
    }

    for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id)
    {
      search_caches[thread_id].query_id_ = query_id;
      search_caches[thread_id].query_series_ptr_ = series_ptr;
    }

    // 获取全局最佳距离，并在当前叶子节点中搜索
    VALUE_TYPE global_bsf_distance = answer->get_bsf();
    VALUE_TYPE local_nn_distance = resident_node.get().search_mt(series_ptr, query_id, std::ref(*answer.get()), answer_mutex.get());

    // QYL  这个位置要改！！！！！！！！！当前节点的局部最近邻会被存到Filter.h的global_lnn_distances_中,在搜索阶段根据recall计算oi(delta)时用到
    resident_node.get().push_global_example(global_bsf_distance, local_nn_distance, 0);
    // QYL
    // resident_node.get().push_k_global_example(global_bsf_distance, local_knn_distances, 0);

    visited_node_counter += 1;
    visited_series_counter += resident_node.get().get_size();

    // 遍历整棵树的所有叶子节点（除了resident_node），计算其下界距离并存入leaf_min_heap_，为后续多线程搜索生成候选队列。
    assert(node_stack.empty() && leaf_min_heap_.empty());
    node_stack.push(std::ref(*root_));

    while (!node_stack.empty())
    {
      std::reference_wrapper<dstree::Node> node_to_visit = node_stack.top();
      node_stack.pop();

      if (node_to_visit.get().is_leaf())
      {
        if (node_to_visit.get().get_id() != resident_node.get().get_id())
        {
          leaf_min_heap_.push(std::make_tuple(node_to_visit, node_to_visit.get().cal_lower_bound_EDsquare(series_ptr)));
        }
      }
      else
      {
        for (auto child_node : node_to_visit.get())
        {
          node_stack.push(child_node);
        }
      }
    }

    // 其实这里 search_thread_F 才是开始精确搜索，当前查询会遍历所有lb无法减枝的叶子节点进行KNN搜索
    std::vector<std::thread> threads;
    for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id)
    {
      threads.emplace_back(search_thread_F, search_caches[thread_id]);
    }

    for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id)
    {
      threads[thread_id].join();
    }

    // 11111111111111111   收集每个查询的答案
    train_answers_.emplace_back(dstree::Answers(*answer));

    ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;
    if (config_.get().navigator_is_learned_)
    {
      nnn_to_return = config_.get().navigator_train_k_nearest_neighbor_;
    }

    while (!answer->empty())
    {
      auto answer_i = answer->pop_answer();
      if (answer_i.node_id_ > 0)
      {
        // printf("query %d nn %d = %.3f, node %d\n", query_id, nnn_to_return, answer_i.nn_dist_, answer_i.node_id_);
        // spdlog::info("query {:d} nn {:d} = {:.3f}, node {:d}",
        //                 query_id, nnn_to_return, answer_i.nn_dist_, answer_i.node_id_);
      }
      else
      {
        printf("query %d nn %d = %.3f\n", query_id, nnn_to_return, answer_i.nn_dist_);
        // spdlog::info("query {:d} nn {:d} = {:.3f}",
        //                 query_id, nnn_to_return, answer_i.nn_dist_);
      }
      nnn_to_return -= 1;
    }
  }

  std::free(m256_fetch_cache);
  return SUCCESS;
}



// 遍历树结构，收集叶子节点的剪枝信息，并根据配置触发过滤器的动态分配。
RESPONSE dstree::Index::filter_allocate(bool to_assign, bool reassign){
  printf("[DEBUG] Entering dstree::Index::filter_allocate()\n");
  printf("[DEBUG] to_assign = %d, reassign = %d\n", to_assign, reassign);
  // 这里to_assign = 1, reassign = 0
  std::stack<std::reference_wrapper<dstree::Node>> node_cache;
  node_cache.push(std::ref(*root_));
  // printf("[DEBUG] Pushed root node to stack.\n");
  // 1. 树遍历与剪枝信息收集
  while (!node_cache.empty()){
    // printf("[DEBUG] Processing next node in stack...\n");
    std::reference_wrapper<dstree::Node> node_to_visit = node_cache.top();
    node_cache.pop();
    /*
    遍历树结构中的所有节点，把所有的叶子节点都插入filter
    如果是叶子节点，记录减枝率并且把该节点的filter_info存到filter_infos_
    如果是内部节点，将其子节点推入node_cache
    */
    if (node_to_visit.get().is_leaf()){
      // printf("[DEBUG] Node is a leaf node.\n");
      FilterInfo filter_info(node_to_visit);
      // printf(" filter_id %d /n", filter_info.node_.get().get_id());
      // 利用lb减枝的比例, lb_pruned count/ lb_size
      filter_info.external_pruning_probability_ = node_to_visit.get().get_envelop_pruning_frequency();
      // printf("[DEBUG] Pushing filter info for leaf node ID: %d\n", node_to_visit.get().get_id());
      // printf("filter_info.external_pruning_probability_: %d\n", filter_info.external_pruning_probability_);
      allocator_->push_filter_info(filter_info);
    } else {
      // printf("[DEBUG] Node is an internal node. Pushing child nodes to stack.\n");
      for (auto child_node : node_to_visit.get()){ // 获取当前内部节点的子节点列表
        // printf("[DEBUG] Pushing child node with ID: %d\n", child_node.get().get_id());
        node_cache.push(child_node);
      }
    }
  }
  // 2. 过滤器分配决策
  //  这里是选择合适的叶子节点进行active激活
  if (to_assign){
    printf("[DEBUG] Calling allocator_->assign()...\n");
    // allocator_->assign()这个函数是选择合适(gain最大/节点size大)的叶子节点激活filter
    return allocator_->assign();
    printf("[DEBUG] finished allocator_->assign()...\n");
  }
  else if (reassign){
    printf("[DEBUG] Calling allocator_->reassign()...\n");
    return allocator_->reassign();
  }else{
    printf("[DEBUG] No assignment or reassignment required. Returning SUCCESS.\n");
    return SUCCESS;
  }
}



// QYL batch
RESPONSE dstree::Index::filter_train(){
  printf("开始训练过滤器，filter_cache_大小: %zu\n", filter_cache_.size());
  spdlog::info("开始训练过滤器，filter_cache_大小: {}", filter_cache_.size());
  // Added: Track processed filters to avoid duplicates in case of issues
  std::unordered_set<ID_TYPE> processed_filter_ids;
  size_t filter_count = filter_cache_.size();
  size_t active_filters_processed = 0;
  size_t inactive_filters_skipped = 0;
  
  // Safely process the filter cache
  while (!filter_cache_.empty()) {
    // Get the top filter reference
    std::reference_wrapper<Filter> filter = filter_cache_.top();
    filter_cache_.pop();
    // Get filter ID for tracking
    ID_TYPE filter_id = filter.get().get_id();
    // Skip already processed filters (shouldn't happen but added as safety)
    if (processed_filter_ids.find(filter_id) != processed_filter_ids.end()) {
      printf("警告: 过滤器ID: %ld 已经处理过，跳过\n", filter_id);
      continue;
    }
    // Process active filters
    if (filter.get().is_active()) {
      printf("正在训练节点ID: %ld 的过滤器\n", filter_id);
      
      try {
        RESPONSE result = filter.get().batch_train();
        if (result != SUCCESS) {
          printf("警告: 过滤器ID: %ld 训练失败\n", filter_id);
        } else {
          active_filters_processed++;
        }
      } catch (const std::exception& e) {
        printf("错误: 过滤器ID: %ld 训练时发生异常: %s\n", filter_id, e.what());
      }
    } else {
      inactive_filters_skipped++;
    }
    
    // Mark as processed
    processed_filter_ids.insert(filter_id);
  }
  
  // Summary report
  printf("过滤器训练完成: 总共 %zu 个, 处理了 %zu 个活跃过滤器, 跳过 %zu 个非活跃过滤器\n", 
         filter_count, active_filters_processed, inactive_filters_skipped);
  
  return SUCCESS;
}




struct TrainCache{
  TrainCache(ID_TYPE thread_id,
             at::cuda::CUDAStream stream,
             std::stack<std::reference_wrapper<dstree::Filter>> &filter_cache,
             pthread_mutex_t *filter_cache_mutex) : thread_id_(thread_id),
                                                    stream_(stream),
                                                    filter_cache_(filter_cache),
                                                    filter_cache_mutex_(filter_cache_mutex) {}

  ~TrainCache() = default;
  ID_TYPE thread_id_;
  at::cuda::CUDAStream stream_;
  // TODO remove ref to filter; use node instead
  std::stack<std::reference_wrapper<dstree::Filter>> &filter_cache_;
  pthread_mutex_t *filter_cache_mutex_;
};




// This is the training function used by each thread
void train_thread_F(TrainCache &train_cache) {
  at::cuda::setCurrentCUDAStream(train_cache.stream_);
  at::cuda::CUDAStreamGuard guard(train_cache.stream_); // compiles with libtorch-gpu
  
  // Added: Track processed filters for this thread
  std::unordered_set<ID_TYPE> processed_filters;
  size_t processed_count = 0;
  size_t active_count = 0;

  while (true) {
    // 添加互斥锁
    pthread_mutex_lock(train_cache.filter_cache_mutex_);
    // filter_cache_是一个包含多个Filter的栈，而每个线程可以获取多个过滤器，直到栈为空为止
    if (train_cache.filter_cache_.empty()) {
      // 解锁
      pthread_mutex_unlock(train_cache.filter_cache_mutex_);
      break;
    } else {
      // 主要进入这个分支
      std::reference_wrapper<dstree::Filter> filter = train_cache.filter_cache_.top(); // 获取栈顶元素
      train_cache.filter_cache_.pop();                                                 // 移除栈顶元素
      pthread_mutex_unlock(train_cache.filter_cache_mutex_);                           // 解锁
      // Get filter ID
      ID_TYPE filter_id = filter.get().get_id();
      // Check if this filter was already processed by this thread (shouldn't happen, but for safety)
      if (processed_filters.find(filter_id) != processed_filters.end()) {
        printf("Thread %d: 警告 - 过滤器 %ld 已被处理过，跳过\n", 
               train_cache.thread_id_, static_cast<long>(filter_id));
        continue;
      }
      
      // 如果当前这个叶子节点的filter被激活了，则进行Mi的训练
      if (filter.get().is_active()) {
        // printf("Thread %d: 开始训练过滤器 ID: %ld\n", train_cache.thread_id_, static_cast<long>(filter_id));
        try {
          // 这里调用的filter.train和单线程调用的是一样的，里面包含了Conformal Prediction的过程
          RESPONSE result = filter.get().batch_train();
          if (result == SUCCESS) {
            active_count++;
            // printf("Thread %d: 成功训练过滤器 ID: %ld\n", train_cache.thread_id_, static_cast<long>(filter_id));
          } else {
             printf("Thread %d: 警告 - 过滤器 ID: %ld 训练失败\n", train_cache.thread_id_, static_cast<long>(filter_id));
          }
        } catch (const std::exception& e) {
          printf("Thread %d: 错误 - 过滤器 ID: %ld 训练时发生异常: %s\n", 
                 train_cache.thread_id_, static_cast<long>(filter_id), e.what());
        }
      }
      
      // Track this filter as processed
      processed_filters.insert(filter_id);
      processed_count++;
    }
  }
  // printf("Thread %d: 完成任务，共处理 %zu 个过滤器，成功训练 %zu 个活跃过滤器\n", 
  //        train_cache.thread_id_, processed_count, active_count);
}




// 这个是多线程train filter的总函数，里面每个线程调用train_thread_F
RESPONSE dstree::Index::filter_train_mthread() {
  // 多线程训练需要GPU支持
  assert(config_.get().filter_train_is_gpu_);
  assert(torch::cuda::is_available());

  printf("\n------------ filter_train_mthread开始 -------------\n");
  printf("filter_train_nthread_: %ld, 激活的过滤器数量: %zu\n", 
         (long)config_.get().filter_train_nthread_, filter_cache_.size());
  
  if (filter_cache_.empty()) {
    printf("警告: filter_cache_ 为空，没有过滤器需要训练\n");
    return SUCCESS;
  }
  
  // 创建过滤器栈和互斥锁
  std::stack<std::reference_wrapper<dstree::Filter>> filters;
  std::unique_ptr<pthread_mutex_t> filter_stack_mutex = std::make_unique<pthread_mutex_t>();
  pthread_mutex_init(filter_stack_mutex.get(), nullptr);

  // 统计活跃过滤器数量
  ID_TYPE active_filter_count = 0;
  
  // 收集所有过滤器
  std::stack<std::reference_wrapper<dstree::Node>> node_stack;
  node_stack.push(std::ref(*root_));

  while (!node_stack.empty()) {
    std::reference_wrapper<dstree::Node> node_to_visit = node_stack.top();
    node_stack.pop();

    if (node_to_visit.get().is_leaf()) {
      if (node_to_visit.get().has_filter()) {
        filters.push(node_to_visit.get().get_filter());

        if (node_to_visit.get().has_active_filter()) {
          active_filter_count++;
        }

      }
    } else {
      for (auto child_node : node_to_visit.get()) {
        node_stack.push(child_node);
      }
    }
  }

  printf("收集到 %zu 个过滤器，其中 %ld 个激活\n", filters.size(), (long)active_filter_count);
  
  if (filters.empty()) {
    printf("警告: 没有找到过滤器，训练终止\n");
    return SUCCESS;
  }

#ifdef DEBUG
  spdlog::debug("indexing filters.size = {:d}", filters.size());
#endif

  // 创建训练缓存
  std::vector<std::unique_ptr<TrainCache>> train_caches;
  
  // 确定线程数量不超过过滤器数量
  ID_TYPE num_threads = std::min(
      config_.get().filter_train_nthread_, 
      static_cast<ID_TYPE>(filters.size())
  );
  
  if (num_threads < config_.get().filter_train_nthread_) {
    printf("注意: 调整线程数量从 %ld 到 %ld 以匹配过滤器数量\n", 
           (long)config_.get().filter_train_nthread_, (long)num_threads);
  }

  for (ID_TYPE thread_id = 0; thread_id < num_threads; ++thread_id) {
    at::cuda::CUDAStream new_stream = at::cuda::getStreamFromPool(false, config_.get().filter_device_id_);

    // printf("创建训练线程 %ld, CUDA流ID: %ld\n", 
    //        (long)thread_id, (long)static_cast<ID_TYPE>(new_stream.id()));

#ifdef DEBUG
    spdlog::info("train thread {:d} stream id = {:d}, query = {:d}, priority = {:d}",
                 thread_id,
                 static_cast<ID_TYPE>(new_stream.id()),
                 static_cast<ID_TYPE>(new_stream.query()),
                 static_cast<ID_TYPE>(new_stream.priority())); // compiles with libtorch-gpu
#endif

    train_caches.emplace_back(std::make_unique<TrainCache>(thread_id,                  // 线程 ID
                                                           std::move(new_stream),      // 独占的 CUDA 流
                                                           std::ref(filters),          // 所有线程共享的过滤器栈
                                                           filter_stack_mutex.get())); // 共享的互斥锁
  }

  // 创建并启动训练线程
  std::vector<std::thread> threads;
  printf("启动 %ld 个训练线程...\n", (long)num_threads);
  for (ID_TYPE thread_id = 0; thread_id < num_threads; ++thread_id) {
    //------------每个线程都调用一个train_thread_F，这个函数是filter_train_mthread的核心------------------
    // 每个线程通过 std::ref(*train_caches[thread_id]) 获取自己的 TrainCache 引用
    threads.emplace_back(train_thread_F, std::ref(*train_caches[thread_id]));
  }
  // 等待所有线程完成
  // printf("等待所有训练线程完成...\n");
  for (ID_TYPE thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads[thread_id].join();
  }
  // printf("------------ filter_train_mthread完成 -------------\n");
  return SUCCESS;
}





//  这个是train filter的过程，这个函数被总函数 dstree::Index::build() 调用
RESPONSE dstree::Index::train(bool is_retrain)
{
  printf("-----进入dstree::Index::train()--------\n");
  // ---------------------- 1. 生成或加载训练数据  ----------------ßß------
  // 创建查询生成器，用于生成训练过滤器的查询数据
  // local query generation is called after collecting global results
  dstree::Synthesizer query_synthesizer(config_, nleaf_);
  // filter_query_filepath_ 实际输入了，是用来训练filter的查询数据的文件路径
  if (!fs::exists(config_.get().filter_query_filepath_))
  {
    // 不走下面这个分支，而是else分支，else分支不重新train filter
    printf("[DEBUG] Filter query file does not exist. Generating synthetic queries...\n");
    assert(!is_retrain); // not applicable to loaded filters

    if (!config_.get().filter_query_filepath_.empty())
    {
      spdlog::error("filter train query filepath {:s} does not exist", config_.get().filter_query_filepath_);
      return FAILURE;
    }

    ID_TYPE query_set_nbytes = -1;
    // 根据CPU/GPU配置选择生成查询数据的模式
    ////针对每个过滤器生成特定数量的查询
    if (config_.get().filter_num_synthetic_query_per_filter_ > 0)
    { 
      printf("index::train() 进入分支 filter_num_synthetic_query_per_filter_: %d\n",
           config_.get().filter_num_synthetic_query_per_filter_);
      ID_TYPE num_synthetic_queries = root_->get_num_synthetic_queries(allocator_->get_node_size_threshold());

      query_set_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * num_synthetic_queries;
      filter_train_query_ptr_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_set_nbytes));

      ID_TYPE num_generated_queries = 0;
      root_->synthesize_query(filter_train_query_ptr_, num_generated_queries, allocator_->get_node_size_threshold());
      assert(num_generated_queries == num_synthetic_queries);

      config_.get().filter_train_nexample_ = num_synthetic_queries;

      spdlog::info("filter generated {:d} synthetic train queries", config_.get().filter_train_nexample_);
      
    }  else if (config_.get().filter_train_num_global_example_ > 0) {
      
      
      // 实际使用： 生成固定总数的全局查询样本用于所有过滤器
      printf("\n----------1.index::train() 进入分支 filter_train_num_global_example_: %d----------\n",
           config_.get().filter_train_num_global_example_);
      // 为全局查询(global queries)分配内存空间
      query_set_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().filter_train_num_global_example_;
      // filter_train_query_ptr_是一个指针，指向存储所有训练查询数据的内存块。
      filter_train_query_ptr_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_set_nbytes));
      // 搜索并收集符合阈值条件的叶子节点，存入query_synthesizer
      ID_TYPE leaf_size_threshold = config_.get().filter_default_node_size_threshold_;
      std::stack<std::reference_wrapper<dstree::Node>> node_cache;
      node_cache.push(std::ref(*root_));
      while (!node_cache.empty()){
        std::reference_wrapper<dstree::Node> node_to_visit = node_cache.top();
        node_cache.pop();
        if (node_to_visit.get().is_leaf()){
          if (node_to_visit.get().get_size() >= leaf_size_threshold || config_.get().to_profile_filters_){
            query_synthesizer.push_node(node_to_visit);
          }
        }
        else{
          for (auto child_node : node_to_visit.get()){
            node_cache.push(child_node);
          }
        }
      }

      // 生成全局查询样本，这里没有计算距离 global_nn_distance，后续会进入filter_collect()函数存储全局查询到各节点的最近距离
      printf("\n---------- 2. 生成全局查询样本 ----------\n");
      RESPONSE return_code = query_synthesizer.generate_global_data(filter_train_query_ptr_);
      
      config_.get().filter_train_nexample_ = config_.get().filter_train_num_global_example_;
      if (return_code == FAILURE){
        spdlog::error("failed to generate global queries");
        return FAILURE;
      }
      // printf("filter generated %d synthetic global queries\n", config_.get().filter_train_nexample_);
      spdlog::info("filter generated {:d} synthetic global queries", config_.get().filter_train_nexample_);
      // local query generation is called after collecting global results



    } else {
      spdlog::error("erroneous config for query generation");
      return FAILURE;
    }

    assert(query_set_nbytes > 0);
    std::string filter_query_filepath = config_.get().index_dump_folderpath_ + config_.get().filter_query_filename_;
    printf("[DEBUG] filter_query_filepath = %s\n", filter_query_filepath.c_str());
    std::ofstream query_fout(filter_query_filepath, std::ios::binary | std::ios_base::app);
    query_fout.write(reinterpret_cast<char *>(filter_train_query_ptr_), query_set_nbytes);
    query_fout.close();

    
  } else {

    // 目前没有用到
    // filter_query_filepath_不为空，此时不用生成合成数据，直接
    printf("\n-------- 1. index.train: Loading filter query data from file...\n");
    printf("-------  filter_query_filepath_ = %s\n", config_.get().filter_query_filepath_.c_str());

    auto query_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().filter_train_nexample_;
    filter_train_query_ptr_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_nbytes));
    std::ifstream query_fin(config_.get().filter_query_filepath_, std::ios::in | std::ios::binary);

    if (!query_fin.good()) {
      spdlog::error("filter train query filepath {:s} cannot open", config_.get().filter_query_filepath_);
      printf("[ERROR] filter train query filepath %s cannot open\n", config_.get().filter_query_filepath_.c_str());
    }

    query_fin.read(reinterpret_cast<char *>(filter_train_query_ptr_), query_nbytes);
    if (query_fin.fail()) {
      spdlog::error("cannot read {:d} bytes from {:s}", query_nbytes, config_.get().filter_query_filepath_);
      printf("[ERROR] cannot read %d bytes from %s\n", query_nbytes, config_.get().filter_query_filepath_.c_str());
      std::free(filter_train_query_ptr_);
      filter_train_query_ptr_ = nullptr;
    }
  }


  // ---------------------- 2. 配置训练设备（CPU/GPU） ----------------------
  // support difference devices for training and inference
  // printf("[DEBUG] 2. Configuring training device...\n");
  if (config_.get().require_neurofilter_){
    if (config_.get().filter_train_is_gpu_){
      // TODO support multiple devices
      device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
  }
  else if (config_.get().navigator_is_learned_){
    if (config_.get().navigator_is_gpu_) {
      device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
  }

  // ---------------------- 3. 将查询数据转换为PyTorch张量 ----------------------
  // printf("[DEBUG] 3. Converting query data to PyTorch tensor...\n");
  filter_train_query_tsr_ = torch::from_blob(filter_train_query_ptr_,
                                             {config_.get().filter_train_nexample_, config_.get().series_length_},
                                             torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
  filter_train_query_tsr_ = filter_train_query_tsr_.to(*device_);

  if (config_.get().require_neurofilter_) {
    if (is_retrain) {
      printf("[DEBUG] Deactivating filters for retraining...\n");
      filter_deactivate(*root_);
    } else {
      // initialize filters
      ID_TYPE filter_id = 0;
      // 给所有叶子节点插入初始filter，并且所有filter存入filter_cache中
      filter_initialize(*root_, &filter_id);
      spdlog::info("initialized {:d} filters", filter_id);
    }
  }

  if (!is_retrain){
    // printf("[DEBUG] Collecting filter training data...\n");
    // collect filter training data, i.e., the bsf distances, nn distances, low-bound distances
    // if (config_.get().filter_train_is_mthread_)
    if (config_.get().filter_collect_is_mthread_) {
      printf("\n---------- 5. multi-threaded filter collection.\n");
      // 收集train filter之前需要的局部真实最近邻数据，这里就获得了每个query到每个节点下的local_nn_distance
      filter_collect_mthread();
    } else {
      printf("\n----------5. single-threaded filter collection.------\n");
      filter_collect();
    }

    //生成合成查询作为local query
    // printf("\n---------- 6. 生成合成查询作为local query ----------\n");
    if (config_.get().filter_train_num_local_example_ > 0) {
      // generate and *search* local queries
      printf("\n---------- 6. Generating local queries... ----------\n");
      RESPONSE return_code = query_synthesizer.generate_local_data();

      if (return_code == FAILURE) { 
        printf("filter_train_num_local_example_ > 0, failed to generate local queries\n");
        spdlog::error("failed to generate local queries");
        return FAILURE;
      }
      printf("filter_train_num_local_example_ > 0, filter generated synthetic local queries: %d\n",config_.get().filter_train_num_local_example_);
      spdlog::info("filter generated {:d} synthetic local queries", config_.get().filter_train_num_local_example_);
    }
  }

  // ---------------------- index.train.filter_allocate 训练filter ----------------------
  if (config_.get().require_neurofilter_) {
    // allocate filters among nodes (and activate them)
    printf("\n-------6. Allocating filters---------\n");
    // 给所有叶子节点插入filter，并选择合适的叶子节点进行激活
    filter_allocate(true);
    // train all filter model
    // 打印所有过滤器的local_lnn_distances_和global_lnn_distances_大小
    printf("-----打印所有过滤器的训练数据统计-----\n");
    spdlog::info("-----打印所有过滤器的训练数据统计-----");
    // 使用STL来复制过滤器ID值并打印统计信息
    std::vector<ID_TYPE> filter_ids;
    for (const auto& filter_pair : filter_id_to_filter_) {
      ID_TYPE filter_id = filter_pair.first;
      auto& filter = filter_pair.second;
      filter_ids.push_back(filter_id);
      
      // 获取该过滤器的全局和本地距离向量大小
      size_t global_size = filter->get_global_lnn_distances_size();
      size_t local_size = filter->get_local_lnn_distances_size();
       // printf("过滤器 ID: %d\n", filter_id);
      spdlog::info("过滤器 ID: {}", filter_id);
      
      // 如果有数据，打印距离的均值和标准差
      if (global_size > 0) {
        auto [global_mean, global_std] = filter->get_global_lnn_mean_std();
        // printf("  全局距离 - 均值: %.2f,   标准差: %.2f\n", global_mean, global_std);
        spdlog::info("  全局距离 - 均值: {:.2f},   标准差: {:.2f}", global_mean, global_std);
      }
      
      if (local_size > 0) {
        auto [local_mean, local_std] = filter->get_filter_local_lnn_mean_std();
        // printf("  本地距离 - 均值: %.2f,   标准差: %.2f\n", local_mean, local_std);
        spdlog::info("  本地距离 - 均值: {:.2f},   标准差: {:.2f}", local_mean, local_std);
      }
    }
    spdlog::info("总共 {} 个过滤器", filter_ids.size());
    // spdlog::info("-----统计结束-----");

    printf("\n---------- 7. 训练filter ----------\n");
    if (config_.get().filter_train_is_mthread_){
      printf("[DEBUG] Using multi-threaded filter training, filter_train_is_mthread_ is: %d\n", config_.get().filter_train_is_mthread_);
      filter_train_mthread();
    } else {
      printf("[DEBUG] Using single-threaded filter training. filter_train_is_mthread_ is: %d\n", config_.get().filter_train_is_mthread_);
      filter_train();
    }

    // support difference devices for training and inference
    // 根据配置参数动态选择模型推理（inference）时使用的计算设备（CPU 或 GPU）
    if (config_.get().filter_infer_is_gpu_){
      printf("[DEBUG] Configuring inference device...\n");
      // TODO support multiple devices
      device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      printf("[DEBUG] Using CPU for inference.\n");
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
  }

  // ---------------------- 7. 导航器训练准备（收集叶子节点信息） ----------------------

  if (config_.get().navigator_is_learned_){
    leaf_nodes_.reserve(nleaf_);

    auto node_pos_to_id = make_reserved<ID_TYPE>(nleaf_);
    std::unordered_map<ID_TYPE, ID_TYPE> node_id_to_pos;
    node_id_to_pos.reserve(nleaf_ * 2);

    std::stack<std::reference_wrapper<dstree::Node>> node_cache;
    node_cache.push(std::ref(*root_));

    while (!node_cache.empty()){
      std::reference_wrapper<dstree::Node> node_to_visit = node_cache.top();
      node_cache.pop();

      if (node_to_visit.get().is_leaf()) {
        leaf_nodes_.push_back(node_to_visit);
        node_pos_to_id.push_back(node_to_visit.get().get_id());
        node_id_to_pos[node_to_visit.get().get_id()] = leaf_nodes_.size() - 1;
      } else {
        for (auto child_node : node_to_visit.get())
        {
          node_cache.push(child_node);
        }
      }
    }

    // 树结构遍历与叶子节点统计：代码遍历树结构，记录叶子节点，并统计每个训练样本的答案分布到叶子节点的频率（nn_residence_distributions）
    // 这本质上是计算每个查询对应的叶子节点被选中的概率分布。
    // 导航器训练：利用这些分布数据、查询特征和配置参数，训练一个导航器模型，可能用于预测查询在树结构中的路径或目标节点
    auto nn_residence_distributions = make_reserved<VALUE_TYPE>(config_.get().filter_train_nexample_ * nleaf_);
    for (ID_TYPE cell_i = 0; cell_i < config_.get().filter_train_nexample_ * nleaf_; ++cell_i){
      nn_residence_distributions.push_back(0);
    }

    for (ID_TYPE query_i = 0; query_i < config_.get().filter_train_nexample_; ++query_i){
      // use copy constructor to avoid destruct train_answers_
      Answers answers = Answers(train_answers_[query_i]);

      while (!answers.empty()){
        nn_residence_distributions[nleaf_ * query_i + node_id_to_pos[answers.pop_answer().node_id_]] += 1;
      }
    }

    for (ID_TYPE cell_i = 0; cell_i < config_.get().filter_train_nexample_ * nleaf_; ++cell_i){
      nn_residence_distributions[cell_i] /= config_.get().navigator_train_k_nearest_neighbor_;
    }

#ifdef DEBUG
    for (ID_TYPE query_i = 0; query_i < config_.get().filter_train_nexample_; ++query_i){
      spdlog::debug("navigator train query {:d} target = {:s}",
                    query_i, upcite::array2str(nn_residence_distributions.data() + nleaf_ * query_i, nleaf_));
    }
#endif

    navigator_ = std::make_unique<dstree::Navigator>(config_,
                                                     node_pos_to_id,
                                                     filter_train_query_tsr_,
                                                     nn_residence_distributions,
                                                     *device_);

    navigator_->train();
  }

  return SUCCESS;
}





RESPONSE dstree::Index::load(){
  ID_TYPE ifs_buf_size = sizeof(ID_TYPE) * config_.get().leaf_max_nseries_ * 2; // 2x expanded for safety
  ID_TYPE max_num_local_bytes = config_.get().filter_train_num_local_example_;
  if (config_.get().filter_train_num_global_example_ > max_num_local_bytes){
    max_num_local_bytes = config_.get().filter_train_num_global_example_;
  }
  max_num_local_bytes *= sizeof(VALUE_TYPE) * config_.get().series_length_;
  if (max_num_local_bytes > ifs_buf_size){
    ifs_buf_size = max_num_local_bytes;
  }

  void *ifs_buf = std::malloc(ifs_buf_size);
  nnode_ = 0;
  nleaf_ = 0;
  RESPONSE status = root_->load(ifs_buf, std::ref(*buffer_manager_), nnode_, nleaf_);
  std::free(ifs_buf);
  if (status == FAILURE){
    spdlog::info("failed to load index");
    return FAILURE;
  }
  // TODO in-memory only; supports on-disk
  if (!config_.get().on_disk_){
    buffer_manager_->load_batch();
  }
  leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist>(
      CompareDecrNodeDist(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));

  // 这里应该是加载model
  printf("require_neurofilter_= %d\n", config_.get().require_neurofilter_);
  if (config_.get().require_neurofilter_){
    if (!config_.get().to_load_filters_){
      train();
    } else {
      if (config_.get().filter_retrain_){
        train(true);
      } else if (config_.get().filter_reallocate_multi_){
        // TODO
        filter_allocate(false, true);
      } else if (config_.get().filter_reallocate_single_){
        filter_allocate(false, true);
      } else {
        // initialize allocator for setting conformal intervals
        printf("---------进入load中的filter_allocate(false)分支----------\n");
        filter_allocate(false);
        // 在这里添加调试信息
        // printf("Loading filters...\n");
        // 这里需要一个函数来获取激活的过滤器数量
        printf("Number of active filters: %d\n", get_active_filter_count());
        // printf("Filter query filepath: %s\n", config_.get().filter_query_filepath_.c_str());
        printf("Filter load folder: %s\n", config_.get().index_load_folderpath_.c_str());

        // 添加：加载query_knn_nodes.bin文件
        printf("尝试加载查询KNN节点数据...\n");
        std::string knn_nodes_filepath = config_.get().index_load_folderpath_ + "query_knn_nodes.bin";
        
        if (fs::exists(knn_nodes_filepath)) {
            printf("找到查询KNN节点数据文件: %s\n", knn_nodes_filepath.c_str());
            std::ifstream knn_nodes_fin(knn_nodes_filepath, std::ios::binary);
            
            if (knn_nodes_fin.good()) {
                // 清空现有数据
                query_knn_nodes_.clear();
                
                // 读取查询数量
                ID_TYPE num_queries = 0;
                knn_nodes_fin.read(reinterpret_cast<char*>(&num_queries), sizeof(ID_TYPE));
                printf("文件中包含 %ld 个查询的KNN节点信息\n", static_cast<long>(num_queries));
                
                // 读取每个查询的节点信息
                for (ID_TYPE i = 0; i < num_queries; ++i) {
                    // 读取查询ID
                    ID_TYPE query_id = 0;
                    knn_nodes_fin.read(reinterpret_cast<char*>(&query_id), sizeof(ID_TYPE));
                    
                    // 读取该查询的节点映射大小
                    ID_TYPE num_nodes = 0;
                    knn_nodes_fin.read(reinterpret_cast<char*>(&num_nodes), sizeof(ID_TYPE));
                    
                    // 初始化该查询的节点映射
                    std::unordered_map<ID_TYPE, ID_TYPE> node_counts;
                    // 读取每个节点ID和计数
                    for (ID_TYPE j = 0; j < num_nodes; ++j){
                        ID_TYPE node_id = 0;
                        ID_TYPE count = 0;
                        knn_nodes_fin.read(reinterpret_cast<char*>(&node_id), sizeof(ID_TYPE));
                        knn_nodes_fin.read(reinterpret_cast<char*>(&count), sizeof(ID_TYPE));
                        
                        node_counts[node_id] = count;
                    }
                    // 存储到query_knn_nodes_中
                    query_knn_nodes_[query_id] = std::move(node_counts);
                }
                
                printf("成功加载查询KNN节点数据\n");
            } else {
                printf("无法打开查询KNN节点数据文件\n");
            }
        } else {
            printf("未找到查询KNN节点数据文件: %s\n", knn_nodes_filepath.c_str());
        }
        
        // 加载每个过滤器的batch_alphas.bin文件
        printf("尝试加载过滤器批处理alpha值...\n");
        
        // 遍历所有节点，查找激活的过滤器并加载其alpha值
        std::stack<std::reference_wrapper<dstree::Node>> node_stack;
        node_stack.push(std::ref(*root_));
        int loaded_filter_count = 0;
        
        while (!node_stack.empty()) {
            auto node = node_stack.top();
            node_stack.pop();
            
            if (node.get().is_leaf() && node.get().has_active_filter()) {
                ID_TYPE node_id = node.get().get_id();
                std::string alphas_filepath = config_.get().index_load_folderpath_ + 
                                             "filter_" + std::to_string(node_id) + "_alphas.bin";
                
                if (fs::exists(alphas_filepath)) {
                    printf("加载过滤器 %ld 的批处理alpha值: %s\n", 
                           static_cast<long>(node_id), alphas_filepath.c_str());
                    
                    // 简化加载逻辑，不使用try-catch和额外的资源清理
                    if (node.get().load_filter_batch_alphas(alphas_filepath) == SUCCESS) {
                        loaded_filter_count++;
                        printf("成功加载过滤器 %ld 的批处理alpha值\n", static_cast<long>(node_id));
                    } else {
                        printf("加载过滤器 %ld 的批处理alpha值失败\n", static_cast<long>(node_id));
                    }
                } else {
                    printf("未找到过滤器 %ld 的批处理alpha文件: %s\n", 
                           static_cast<long>(node_id), alphas_filepath.c_str());
                }
            }
            
            if (!node.get().is_leaf()) {
                for (auto child_node : node.get()) {
                    node_stack.push(child_node);
                }
            }
        }
        printf("成功加载 %d 个过滤器的批处理alpha值\n", loaded_filter_count);
      }
      // support difference devices for training and inference
      if (config_.get().filter_infer_is_gpu_){
        // TODO support multiple devices
        device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                                  static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
      } else {
        device_ = std::make_unique<torch::Device>(torch::kCPU);
      }
    }
  }

  if (config_.get().navigator_is_learned_)
  {
    train();
  }
  return SUCCESS;
}



// 保存model
RESPONSE dstree::Index::dump() const{
  ID_TYPE ofs_buf_size = config_.get().filter_train_nexample_;
  if (config_.get().filter_train_num_global_example_ > ofs_buf_size){
    ofs_buf_size = config_.get().filter_train_num_global_example_;
  }
  if (config_.get().filter_train_num_local_example_ > ofs_buf_size){
    ofs_buf_size = config_.get().filter_train_num_local_example_;
  }
  if (ofs_buf_size < 1){
    ofs_buf_size = 128;
  }
  ofs_buf_size *= sizeof(ID_TYPE) * config_.get().series_length_;
  void *ofs_buf = std::malloc(ofs_buf_size);

  root_->dump(ofs_buf);
  // 保存 query_knn_nodes_ 数据
  if (config_.get().require_neurofilter_ && config_.get().filter_is_conformal_){
    std::string knn_nodes_filepath = config_.get().index_dump_folderpath_ + "query_knn_nodes.bin";
    std::ofstream knn_nodes_fout(knn_nodes_filepath, std::ios::binary);

    if (knn_nodes_fout.good()){
      // 保存 query_knn_nodes_ 的大小
      ID_TYPE num_queries = query_knn_nodes_.size();
      knn_nodes_fout.write(reinterpret_cast<const char *>(&num_queries), sizeof(ID_TYPE));

      // 保存每个查询的节点映射
      for (const auto &[query_id, node_counts] : query_knn_nodes_){
        // 保存查询ID
        knn_nodes_fout.write(reinterpret_cast<const char *>(&query_id), sizeof(ID_TYPE));

        // 保存该查询的节点映射大小
        ID_TYPE num_nodes = node_counts.size();
        knn_nodes_fout.write(reinterpret_cast<const char *>(&num_nodes), sizeof(ID_TYPE));

        // 保存每个节点ID和计数
        for (const auto &[node_id, count] : node_counts){
          knn_nodes_fout.write(reinterpret_cast<const char *>(&node_id), sizeof(ID_TYPE));
          knn_nodes_fout.write(reinterpret_cast<const char *>(&count), sizeof(ID_TYPE));
        }
      }

      // printf("保存查询KNN节点数据到 %s 成功\n", knn_nodes_filepath.c_str());
    } else {
      spdlog::error("无法打开文件保存查询KNN节点数据: {}", knn_nodes_filepath);
    }

    // 保存每个过滤器的 batch_alphas_ 数据
    // 遍历所有叶节点
    std::stack<std::reference_wrapper<dstree::Node>> node_stack;
    node_stack.push(std::ref(*root_));

    while (!node_stack.empty()){
      auto node = node_stack.top();
      node_stack.pop();

      if (node.get().is_leaf() && node.get().has_active_filter()){
        ID_TYPE node_id = node.get().get_id();
        std::string alphas_filepath = config_.get().index_dump_folderpath_ +
                                      "filter_" + std::to_string(node_id) + "_alphas.bin";

        // 获取过滤器的 conformal_predictor_ 并保存 batch_alphas_
        node.get().save_filter_batch_alphas(alphas_filepath);
      }

      if (!node.get().is_leaf()){
        for (auto child_node : node.get()){
          node_stack.push(child_node);
        }
      }
    }
  }

  std::free(ofs_buf);
  return SUCCESS;
}



// query 搜索阶段
RESPONSE dstree::Index::search(bool is_profile){
  // ==================== 1. 前置检查模块 ====================
  // 检查查询文件是否存在
  if (!fs::exists(config_.get().query_filepath_)){
    spdlog::error("query filepath {:s} does not exist", config_.get().query_filepath_);
    return FAILURE;
  }
  printf(" ---------- Index::search ---------- \n");
  printf(" query_filepath_ opened successfully: %s\n", config_.get().query_filepath_.c_str());
  // 尝试打开查询文件
  std::ifstream query_fin(config_.get().query_filepath_, std::ios::in | std::ios::binary);
  if (!query_fin.good()){
    spdlog::error("query filepath {:s} cannot open", config_.get().query_filepath_);
    return FAILURE;
  }
  // ==================== 2. 读取查询数据模块 ====================
  // 计算需要读取的字节数并分配内存
  auto query_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().query_nseries_;
  auto query_buffer = static_cast<VALUE_TYPE *>(std::malloc(query_nbytes));
  // 从文件读取原始查询数据
  query_fin.read(reinterpret_cast<char *>(query_buffer), query_nbytes);

  if (query_fin.fail()){
    spdlog::error("cannot read {:d} bytes from {:s}", query_nbytes, config_.get().query_filepath_);
    return FAILURE;
  }

  // ==================== 3. 读取草图数据模块（没用到）====================
  VALUE_TYPE *query_sketch_buffer = nullptr;
  if (config_.get().is_sketch_provided_){
    // 检查草图文件是否存在  PAA=sketch
    if (!fs::exists(config_.get().query_sketch_filepath_)){
      spdlog::error("query sketch filepath {:s} does not exist", config_.get().query_sketch_filepath_);
      return FAILURE;
    }

    // 打开并读取草图数据
    std::ifstream query_sketch_fin(config_.get().query_sketch_filepath_, std::ios::in | std::ios::binary);
    if (!query_fin.good()){
      spdlog::error("query sketch filepath {:s} cannot open", config_.get().query_sketch_filepath_);
      return FAILURE;
    }

    auto query_sketch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().sketch_length_ * config_.get().query_nseries_;
    query_sketch_buffer = static_cast<VALUE_TYPE *>(std::malloc(query_sketch_nbytes));
    query_sketch_fin.read(reinterpret_cast<char *>(query_sketch_buffer), query_sketch_nbytes);

    // 错误处理
    if (query_sketch_fin.fail()){
      spdlog::error("cannot read {:d} bytes from {:s}", query_nbytes, config_.get().query_filepath_);
      return FAILURE;
    }
  }
  // ==================== 4. 算法核心调整模块 ====================
  // 如果启用神经过滤器和保形预测校准，动态调整置信区间
  //!!!!!!!!!!!!!!!!!!!!!!!!  利用用户指定得recall计算出合适的oi(delta)  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // get confidence intervals based on the required recall, during search

  printf("config_.get().filter_conformal_adjust_confidence_by_recall_ = %d\n",
         config_.get().filter_conformal_adjust_confidence_by_recall_);

  if (config_.get().require_neurofilter_ && config_.get().filter_is_conformal_ && config_.get().filter_conformal_adjust_confidence_by_recall_){
    allocator_.get()->set_batch_confidence_from_recall(this->get_query_knn_nodes());
  }
  
  printf("\n---------执行搜索和计算Recall模块------------\n");
  // ==================== 5. 执行搜索和计算Recall模块 ====================
  if (!is_profile){ // 当不是profile模式时，执行recall计算
    ground_truth_answers_.clear();
    actual_answers_.clear();
    
    // 初始化统计变量
    std::vector<ID_TYPE> gt_visited_nodes(config_.get().query_nseries_, 0);     // 暴力搜索每个查询访问的节点数
    std::vector<ID_TYPE> gt_visited_series(config_.get().query_nseries_, 0);    // 暴力搜索每个查询访问的序列数
    std::vector<ID_TYPE> filter_visited_nodes(config_.get().query_nseries_, 0); // 过滤器搜索每个查询访问的节点数
    std::vector<ID_TYPE> filter_visited_series(config_.get().query_nseries_, 0); // 过滤器搜索每个查询访问的序列数
    std::vector<ID_TYPE> filter_pruned_nodes(config_.get().query_nseries_, 0);  // 过滤器剪枝的节点数
    std::vector<ID_TYPE> filter_pruned_series(config_.get().query_nseries_, 0); // 过滤器剪枝的序列数
    
    // 总计统计变量
    ID_TYPE total_gt_nodes = 0;
    ID_TYPE total_gt_series = 0;
    ID_TYPE total_filter_nodes = 0;
    ID_TYPE total_filter_series = 0;
    ID_TYPE total_pruned_nodes = 0;
    ID_TYPE total_pruned_series = 0;

    printf("=== Phase 1: 执行暴力搜索获取ground truth ===\n");
    spdlog::info("=== Phase 1: 执行暴力搜索获取ground truth ===");
    for (ID_TYPE query_id = 0; query_id < config_.get().query_nseries_; ++query_id){
      // 遍历所有的query, 计算每个query的knn结果，存放在gt_answers

      VALUE_TYPE *sketch_ptr = nullptr;
      if (config_.get().is_sketch_provided_){
        sketch_ptr = query_sketch_buffer + config_.get().sketch_length_ * query_id;
      }

      // 创建答案对象并执行暴力搜索
      VALUE_TYPE *series_ptr = query_buffer + config_.get().series_length_ * query_id;
      auto gt_answers = new dstree::Answers(config_.get().n_nearest_neighbor_, query_id);
      ID_TYPE current_visited_nodes = 0, current_visited_series = 0;
      profile(query_id, series_ptr, sketch_ptr, gt_answers, current_visited_nodes, current_visited_series);
      ground_truth_answers_.push_back(std::shared_ptr<dstree::Answers>(gt_answers));
      
      // 记录此次查询的统计信息
      gt_visited_nodes[query_id] = current_visited_nodes;
      gt_visited_series[query_id] = current_visited_series;
      total_gt_nodes += current_visited_nodes;
      total_gt_series += current_visited_series;
      
      // QYL 查看1nn的真实最近邻结果
      if (query_id < 200) {
        // printf("\n===== 查询 %ld 的Ground Truth结果 =====\n", query_id);
        spdlog::info("===== 查询 {:d} 的Ground Truth结果 =====", query_id);
        auto gt_topk = gt_answers->get_current_topk();
        for (size_t i = 0; i < gt_topk.size(); ++i) {
          // printf("query_id=%ld, K=%zu: distance=%.4f, node ID=%ld, series id=%ld\n",
          //       query_id, i+1, gt_topk[i].nn_dist_, gt_topk[i].node_id_, gt_topk[i].global_offset_);
          spdlog::info("query_id={:d}, K={:d}: distance={:.4f}, node ID={:d}, series id={:d}",
                      query_id, i+1, gt_topk[i].nn_dist_, gt_topk[i].node_id_, gt_topk[i].global_offset_);
        }
      }
    }

    printf("===== Phase 2: 执行带过滤器的实际搜索 =====\n");
    spdlog::info("=== Phase 2: 执行带过滤器的实际搜索 ===");
    for (ID_TYPE query_id = 0; query_id < config_.get().query_nseries_; ++query_id){
      // 遍历所有的query, 计算每个query的knn结果，存放在actual_answers     
      VALUE_TYPE *sketch_ptr = nullptr;
      if (config_.get().is_sketch_provided_){
        sketch_ptr = query_sketch_buffer + config_.get().sketch_length_ * query_id;
      }

      // 创建答案对象并执行实际搜索
      VALUE_TYPE *series_ptr = query_buffer + config_.get().series_length_ * query_id;
      auto actual_answers = new dstree::Answers(config_.get().n_nearest_neighbor_, query_id);
      ID_TYPE current_visited_nodes = 0, current_visited_series = 0;
      ID_TYPE current_pruned_nodes = 0, current_pruned_series = 0;
      search(query_id, series_ptr, sketch_ptr, actual_answers, 
             current_visited_nodes, current_visited_series, 
             current_pruned_nodes, current_pruned_series);
      actual_answers_.push_back(std::shared_ptr<dstree::Answers>(actual_answers));
      
      // 记录此次查询的统计信息
      filter_visited_nodes[query_id] = current_visited_nodes;
      filter_visited_series[query_id] = current_visited_series;
      filter_pruned_nodes[query_id] = current_pruned_nodes;
      filter_pruned_series[query_id] = current_pruned_series;
      total_filter_nodes += current_visited_nodes;
      total_filter_series += current_visited_series;
      total_pruned_nodes += current_pruned_nodes;
      total_pruned_series += current_pruned_series;

      // QYL 记录 1nn的实际最近邻结果  
      if (query_id < 200) {
        // printf("\n===== 查询 %ld 的带filter的实际结果 =====\n", query_id);
        spdlog::info("===== 查询 {} 的实际结果 =====", query_id);
        auto actual_topk = actual_answers->get_current_topk();
        for (size_t i = 0; i < actual_topk.size(); ++i) {
          // printf("query_id = %ld, K=%zu: 距离=%.4f, 节点ID=%ld, series的全局偏移量=%ld\n", 
                // query_id, i+1, actual_topk[i].nn_dist_, actual_topk[i].node_id_, actual_topk[i].global_offset_);
          spdlog::info("query_id = {}, K={}: 距离={:.4f}, 节点ID={}, series的全局偏移量={}", 
                      query_id, i+1, actual_topk[i].nn_dist_, actual_topk[i].node_id_, actual_topk[i].global_offset_);
        }
      }
    }

    // 将查询结果写入文件，便于对比
    std::string results_path = config_.get().results_path_;
    // 确保结果目录存在
    if (!results_path.empty()) {
      namespace fs = boost::filesystem;
      if (!fs::exists(results_path)) {
        printf("创建结果保存目录: %s\n", results_path.c_str());
        fs::create_directories(results_path);
      }
    }
    
    // 更改为CSV格式以便于Excel打开
    std::string comparison_file = results_path + "/query_results_comparison.csv";
    
    std::ofstream result_file(comparison_file);
    if (result_file.is_open()) {
      // CSV表头
      result_file << "query_id,K,暴力距离,过滤器距离,暴力节点ID,过滤器节点ID,暴力offset,过滤器offset,是否匹配\n";
      
      // 遍历所有查询
      for (ID_TYPE query_id = 0; query_id < config_.get().query_nseries_; ++query_id) {
        auto& gt_answers = ground_truth_answers_[query_id];
        auto& actual_answers = actual_answers_[query_id];
        
        auto gt_topk = gt_answers->get_current_topk();
        auto actual_topk = actual_answers->get_current_topk();
        
        ID_TYPE k_max = std::min(gt_topk.size(), actual_topk.size());
        
        for (ID_TYPE k = 0; k < k_max; ++k) {
          bool is_match = (gt_topk[k].global_offset_ == actual_topk[k].global_offset_);
          result_file << query_id << "," << (k+1) << ","
                     << gt_topk[k].nn_dist_ << "," << actual_topk[k].nn_dist_ << ","
                     << gt_topk[k].node_id_ << "," << actual_topk[k].node_id_ << ","
                     << gt_topk[k].global_offset_ << "," << actual_topk[k].global_offset_ << ","
                     << (is_match ? "是" : "否") << "\n";
        }
      }
      
      result_file.close();
      printf("查询结果对比已保存到: %s\n", comparison_file.c_str());
      spdlog::info("查询结果对比已保存到: {}", comparison_file);
    } else {
      printf("错误：无法创建结果对比文件 %s\n", comparison_file.c_str());
      spdlog::error("无法创建结果对比文件 {}", comparison_file);
    }
    
    // 创建节点访问统计文件
    std::string stats_file = results_path + "/nodes_statistics.csv";
    std::ofstream stats_result(stats_file);
    
    if (stats_result.is_open()) {
      // CSV表头
      stats_result << "query_id,GT_visited_nodes,GT_visited_series,Filter_visited_nodes,Filter_visited_series,Pruned_nodes,Pruned_series,Pruning_ratio\n";
      
      // 按照查询ID写入单个查询的统计信息
      for (ID_TYPE query_id = 0; query_id < config_.get().query_nseries_; ++query_id) {
        // 计算剪枝比例
        double pruning_ratio = gt_visited_nodes[query_id] > 0 ? 
                            static_cast<double>(filter_pruned_nodes[query_id]) / gt_visited_nodes[query_id] : 0.0;
        
        stats_result << query_id << ","
                    << gt_visited_nodes[query_id] << ","
                    << gt_visited_series[query_id] << ","
                    << filter_visited_nodes[query_id] << ","
                    << filter_visited_series[query_id] << ","
                    << filter_pruned_nodes[query_id] << ","
                    << filter_pruned_series[query_id] << ","
                    << pruning_ratio << "\n";
      }
      
      // 添加总计行
      double total_pruning_ratio = total_gt_nodes > 0 ? 
                                  static_cast<double>(total_pruned_nodes) / total_gt_nodes : 0.0;
                                  
      stats_result << "总计,"
                  << total_gt_nodes << ","
                  << total_gt_series << ","
                  << total_filter_nodes << ","
                  << total_filter_series << ","
                  << total_pruned_nodes << ","
                  << total_pruned_series << ","
                  << total_pruning_ratio << "\n";
      
      stats_result.close();
      printf("节点访问统计已保存到: %s\n", stats_file.c_str());
      spdlog::info("节点访问统计已保存到: {}", stats_file);
      
      // 打印总体统计信息
      printf("\n===== 节点访问统计 =====\n");
      printf("暴力搜索总访问节点数: %ld, 总访问序列数: %ld\n", total_gt_nodes, total_gt_series);
      printf("过滤器搜索总访问节点数: %ld, 总访问序列数: %ld\n", total_filter_nodes, total_filter_series);
      printf("总剪枝节点数: %ld, 总剪枝序列数: %ld\n", total_pruned_nodes, total_pruned_series);
      printf("总体剪枝比例: %.4f\n", total_pruning_ratio);
      
      spdlog::info("暴力搜索总访问节点数: {}, 总访问序列数: {}", total_gt_nodes, total_gt_series);
      spdlog::info("过滤器搜索总访问节点数: {}, 总访问序列数: {}", total_filter_nodes, total_filter_series);
      spdlog::info("总剪枝节点数: {}, 总剪枝序列数: {}", total_pruned_nodes, total_pruned_series);
      spdlog::info("总体剪枝比例: {:.4f}", total_pruning_ratio);
    } else {
      printf("错误：无法创建节点统计文件 %s\n", stats_file.c_str());
      spdlog::error("无法创建节点统计文件 {}", stats_file);
    }
    
    // 计算并输出recall
    // calculate_recall();
    calculate_batch_recall();
  } else {

    // 原有的搜索逻辑（不计算recall）
    VALUE_TYPE *series_ptr, *sketch_ptr = nullptr;
    for (ID_TYPE query_id = 0; query_id < config_.get().query_nseries_; ++query_id){
      series_ptr = query_buffer + config_.get().series_length_ * query_id;

      if (config_.get().is_sketch_provided_){
        sketch_ptr = query_sketch_buffer + config_.get().sketch_length_ * query_id;
      }
      profile(query_id, series_ptr, sketch_ptr);
    }
  }
  // 释放内存
  std::free(query_buffer);
  if (query_sketch_buffer != nullptr){
    std::free(query_sketch_buffer);
  }

  printf("[DEBUG] === recall计算正常结束 ===\n");
  return SUCCESS;
}





// 计算Recall的实现(分批次)
void dstree::Index::calculate_batch_recall(){
  // 确保结果保存路径存在
  std::string save_path = config_.get().results_path_;
  if (!save_path.empty()) {
    namespace fs = boost::filesystem;
    if (!fs::exists(save_path)) {
      printf("创建结果保存目录: %s\n", save_path.c_str());
      fs::create_directories(save_path);
    }
  }

  const int num_batches = 100;  // 分成100个批次
  ID_TYPE queries_per_batch = (ground_truth_answers_.size() + num_batches - 1) / num_batches; // 向上取整

  printf("\n===== 按批次计算Recall（共%d个批次）=====\n", num_batches);
  spdlog::info("===== 按批次计算Recall（共{}个批次）=====", num_batches);
  std::vector<double> batch_recalls(num_batches, 0.0);
  std::vector<int> batch_query_counts(num_batches, 0);
  double total_recall = 0.0;

  // 获取目标recall和coverage配置
  double target_recall = config_.get().filter_conformal_recall_;
  double target_coverage = config_.get().filter_conformal_coverage_;
  
  printf("目标Recall: %.2f, 目标覆盖率: %.2f\n", target_recall, target_coverage);
  spdlog::info("目标Recall: {:.2f}, 目标覆盖率: {:.2f}", target_recall, target_coverage);
  for (ID_TYPE query_id = 0; query_id < ground_truth_answers_.size(); ++query_id) {
    int batch_id = query_id / queries_per_batch; // 确定当前查询属于哪个批次
    if (batch_id >= num_batches)
      batch_id = num_batches - 1; // 安全检查

    auto &gt_result = ground_truth_answers_[query_id];
    auto &actual_result = actual_answers_[query_id];

    // 提取ground truth的global_offset集合
    std::set<ID_TYPE> gt_offsets;
    auto gt_copy = *gt_result; // 创建副本以避免修改原始对象

    while (!gt_copy.empty()) {
      auto answer = gt_copy.pop_answer();
      gt_offsets.insert(answer.global_offset_); // 使用global_offset作为唯一标识符
    }

    // 检查实际结果中有多少匹配的offset
    int hits = 0;
    auto actual_copy = *actual_result; // 创建副本

    while (!actual_copy.empty()) {
      auto answer = actual_copy.pop_answer();
      if (gt_offsets.find(answer.global_offset_) != gt_offsets.end()) {
        hits++;
      }
    }

    // 计算当前查询的recall
    double recall = static_cast<double>(hits) / config_.get().n_nearest_neighbor_;
    total_recall += recall;

    // 累加到对应批次
    batch_recalls[batch_id] += recall;
    batch_query_counts[batch_id]++;
  }

  // 打印每个批次的平均recall
  printf("\n===== 各批次平均Recall =====\n");
  int batches_above_target = 0;
  int valid_batches = 0;
  
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    if (batch_query_counts[batch_id] > 0) {
      valid_batches++;
      double batch_avg_recall = batch_recalls[batch_id] / batch_query_counts[batch_id];
      
      // 检查是否达到目标recall
      bool above_target = (batch_avg_recall >= target_recall);
      if (above_target) {
        batches_above_target++;
      }
      
      printf("批次 %d (查询 %d-%d): 平均Recall = %.4f %s\n",
             batch_id + 1,
             batch_id * queries_per_batch,
             std::min((batch_id + 1) * queries_per_batch - 1, (ID_TYPE)ground_truth_answers_.size() - 1),
             batch_avg_recall,
             above_target ? "[达标]" : "[未达标]");

      spdlog::info("批次 {} (查询 {} - {}): 平均Recall = {:.4f} {}",
                   batch_id + 1,
                   batch_id * queries_per_batch,
                   std::min((batch_id + 1) * queries_per_batch - 1, (ID_TYPE)ground_truth_answers_.size() - 1),
                   batch_avg_recall,
                   above_target ? "[达标]" : "[未达标]");   
    }
  }

  // 计算总体平均recall
  double avg_recall = total_recall / ground_truth_answers_.size();
  
  // 计算达标批次的比例
  double achieved_coverage = static_cast<double>(batches_above_target) / valid_batches;
  
  printf("\n总体平均Recall: %.4f\n", avg_recall);
  spdlog::info("总体平均Recall: {:.4f}", avg_recall);
  printf("达到目标Recall(%.4f)的批次比例: %.4f (要求: %.4f)\n", 
         target_recall, achieved_coverage, target_coverage);
  spdlog::info("达到目标Recall(%.4f)的批次比例: %.4f (要求: %.4f)", 
         target_recall, achieved_coverage, target_coverage);
         
  if (achieved_coverage >= target_coverage) {
    printf("结论: 【满足】覆盖率要求 ✓\n");
    spdlog::info("结论: 【满足】覆盖率要求 ✓");
  } else {
    printf("结论: 【未满足】覆盖率要求 ✗\n");
    spdlog::error("结论: 【未满足】覆盖率要求 ✗");
  }
}




// 计算Recall的实现
void dstree::Index::calculate_recall(){
  double total_recall = 0.0;

  for (ID_TYPE query_id = 0; query_id < ground_truth_answers_.size(); ++query_id){
    auto &gt_result = ground_truth_answers_[query_id];
    auto &actual_result = actual_answers_[query_id];

    // 提取ground truth的global_offset集合
    std::set<ID_TYPE> gt_offsets;
    auto gt_copy = *gt_result; // 创建副本以避免修改原始对象

    while (!gt_copy.empty()){
      auto answer = gt_copy.pop_answer();
      gt_offsets.insert(answer.global_offset_); // 使用global_offset作为唯一标识符
    }

    // 检查实际结果中有多少匹配的offset
    int hits = 0;
    auto actual_copy = *actual_result; // 创建副本

    while (!actual_copy.empty()){
      auto answer = actual_copy.pop_answer();
      if (gt_offsets.find(answer.global_offset_) != gt_offsets.end())
      {
        hits++;
      }
    }

    // 计算当前查询的recall
    double recall = static_cast<double>(hits) / config_.get().n_nearest_neighbor_;
    total_recall += recall;

    // printf("Query %ld Recall: %.4f (hits: %d/%ld)\n",
    //        query_id, recall, hits, config_.get().n_nearest_neighbor_);
  }

  // 计算平均recall
  double avg_recall = total_recall / ground_truth_answers_.size(); // 使用正确的变量名
  printf("\n总体平均Recall: %.4f\n", avg_recall);
}




// 执行暴力搜索 - 带访问统计版本
RESPONSE dstree::Index::profile(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr,
                                dstree::Answers *results, 
                                ID_TYPE &visited_node_counter, ID_TYPE &visited_series_counter){
  VALUE_TYPE *route_ptr = query_ptr;
  if (config_.get().is_sketch_provided_){
    route_ptr = sketch_ptr;
  }

  // 初始化统计变量
  visited_node_counter = 0;
  visited_series_counter = 0;

  // 修改：使用传入的results对象或创建新的
  std::shared_ptr<dstree::Answers> answers;
  if (results != nullptr){
    //实际走的是这个分支
    answers = std::shared_ptr<dstree::Answers>(results, [](dstree::Answers *) {}); // 非拥有指针
  } else {
    answers = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
  }

  if (config_.get().require_neurofilter_){
    filter_query_tsr_ = torch::from_blob(query_ptr,
                                         {1, config_.get().series_length_},
                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
    filter_query_tsr_ = filter_query_tsr_.to(*device_);
  }


  //从根节点开始，逐步向下遍历树结构，直到找到包含目标查询序列的叶子节点。
  std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
  while (!resident_node.get().is_leaf()){
    //这步快速找到最可能包含相似序列的初始叶子节点
    resident_node = resident_node.get().route(route_ptr);
  }
  //阶段一： 近似搜索，查找当前节点resident_node下距离query_ptr最近距离，更新bsf
  resident_node.get().search(query_ptr, query_id, *answers, visited_node_counter, visited_series_counter);

  //阶段二：精确搜索：对于同一个query，使用优先队列(leaf_min_heap_)按下界距离从小到大检查其他节点
  leaf_min_heap_.push(std::make_tuple(std::ref(*root_), root_->cal_lower_bound_EDsquare(route_ptr)));
  std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
  VALUE_TYPE node2visit_lbdistance;

  while (!leaf_min_heap_.empty()){
    std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();
    leaf_min_heap_.pop();

    if (node_to_visit.get().is_leaf()){
      // if (visited_node_counter < config_.get().search_max_nnode_ && visited_series_counter < config_.get().search_max_nseries_){
        //is_bsf 用于判断lb_distance是否小于minbsf
        if (node_to_visit.get().get_id() != resident_node.get().get_id()){
            if (answers->is_bsf(node2visit_lbdistance)) {
              // printf("节点%d 搜索\n", node_to_visit.get().get_id());
              // spdlog::info("节点{} 搜索", node_to_visit.get().get_id());
              node_to_visit.get().search(query_ptr, query_id, *answers, visited_node_counter, visited_series_counter);
            }
        }
      // }
        visited_node_counter += 1;
        visited_series_counter += node_to_visit.get().get_size();
    } else {
      for (auto child_node : node_to_visit.get()){
        VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(route_ptr);
        leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
      }
    }
  }
  // printf("查询 %d: 访问节点数: %d, 访问序列数: %d\n", query_id, visited_node_counter, visited_series_counter);
  spdlog::info("查询 {}: 访问节点数: {} 个, 访问序列数: {} 个", query_id, visited_node_counter, visited_series_counter);
  // printf("----------------------------------\n");
  // 修改：只在未传入results时打印结果
  if (results == nullptr){
    ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;
    while (!answers->empty()){
      auto answer = answers->pop_answer();
      if (answer.node_id_ > 0){
        printf("query %d nn %d = %.3f, node %d, db_global_offset_ %d\n",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
        spdlog::info("query {} nn {} = {:.3f}, node {}, db_global_offset_ {}",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
      } else {
        printf("query %d nn %d = %.3f, db_global_offset_ %d\n",
               query_id, nnn_to_return, answer.nn_dist_, answer.global_offset_);
        spdlog::info("query {} nn {} = {:.3f}, node {}, db_global_offset_ {}",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
      }
      nnn_to_return -= 1;
    }
    if (nnn_to_return > 0){
      return FAILURE;
    }
  }

  return SUCCESS;
}




// 执行带过滤器的实际搜：带访问统计版本
RESPONSE dstree::Index::search(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr,
                              dstree::Answers *results,
                              ID_TYPE &visited_node_counter, ID_TYPE &visited_series_counter,
                              ID_TYPE &nfpruned_node_counter, ID_TYPE &nfpruned_series_counter){
  VALUE_TYPE *route_ptr = query_ptr;
  if (config_.get().is_sketch_provided_){
    route_ptr = sketch_ptr;
  }
  // printf("query_id: %d\n", query_id);
  fflush(stdout);
  spdlog::info("\n----------- query_id: {} ----------------", query_id);
  
  // 初始化计数器变量
  visited_node_counter = 0;
  visited_series_counter = 0;
  nfpruned_node_counter = 0;
  nfpruned_series_counter = 0;
  
  // 修改：使用传入的results对象或创建新的
  std::shared_ptr<dstree::Answers> answers;
  if (results != nullptr){
    answers = std::shared_ptr<dstree::Answers>(results, [](dstree::Answers *) {}); // 非拥有指针
  } else {
    answers = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
  }

  if (config_.get().require_neurofilter_){
    filter_query_tsr_ = torch::from_blob(query_ptr,
                                         {1, config_.get().series_length_},
                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
    filter_query_tsr_ = filter_query_tsr_.to(*device_);
  }

  //从根节点开始，逐步向下遍历树结构，直到找到包含目标查询序列的叶子节点。
  std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
  while (!resident_node.get().is_leaf()){
    //这步快速找到最可能包含相似序列的初始叶子节点
    resident_node = resident_node.get().route(route_ptr);
  }
  //阶段一： 近似搜索，查找当前节点resident_node下距离query_ptr最近距离，更新bsf
  resident_node.get().search(query_ptr, query_id, *answers, visited_node_counter, visited_series_counter);
  //  filter_collect: VALUE_TYPE local_nn_distance = resident_node.get().search(series_ptr, query_id, m256_fetch_cache, answer.get());

  //阶段二：精确搜索：对于同一个query，使用优先队列(leaf_min_heap_)按下界距离从小到大检查其他节点
  if (config_.get().is_exact_search_){

    leaf_min_heap_.push(std::make_tuple(std::ref(*root_), root_->cal_lower_bound_EDsquare(route_ptr)));
    std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
    VALUE_TYPE node2visit_lbdistance;
    
    while (!leaf_min_heap_.empty()){
      std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();
      leaf_min_heap_.pop();

      if (node_to_visit.get().is_leaf()){
        // if (visited_node_counter < config_.get().search_max_nnode_ && visited_series_counter < config_.get().search_max_nseries_){
          
          if (node_to_visit.get().get_id() != resident_node.get().get_id()){
            //  利用lbdistance判断：lb<minbsf 继续搜索？ 问：lb>minbsf, 此时true>lb>minbsf, 不需要继续搜
            if (answers->is_bsf(node2visit_lbdistance)) {
              
              if (node_to_visit.get().has_active_filter()){
                // printf("=============节点%d 有激活过滤器=============\n", node_to_visit.get().get_id());
                // spdlog::info("=============节点{} 有激活过滤器=============", node_to_visit.get().get_id());
                VALUE_TYPE predicted_nn_distance = node_to_visit.get().filter_infer_calibrated(filter_query_tsr_);
                if (predicted_nn_distance > answers->get_bsf()){
                  // 如果预测的nn距离大于bsf距离，则进行过滤
                  // printf("节点%d 剪枝: 校准后距离=%.1f > BSF=%.1f\n", node_to_visit.get().get_id(), predicted_nn_distance, answers->get_bsf());
                  spdlog::info("节点{} 剪枝: 预测距离={:.1f} > BSF={:.1f}", node_to_visit.get().get_id(), predicted_nn_distance, answers->get_bsf());

                  nfpruned_node_counter += 1;
                  nfpruned_series_counter += node_to_visit.get().get_size();
                } else {
                  // 如果pred-error < bsf，则不进行剪枝
                  // printf("节点%d 不剪枝: 校准后距离=%.1f <= BSF=%.1f\n", node_to_visit.get().get_id(), predicted_nn_distance, answers->get_bsf());
                  spdlog::info("节点{} 不剪枝: 预测距离={:.1f} <= BSF={:.1f}", node_to_visit.get().get_id(), predicted_nn_distance, answers->get_bsf());
                  node_to_visit.get().search(query_ptr, query_id, *answers, visited_node_counter,
                                             visited_series_counter);
                }

              } else {
                //对于没有激活过滤器的节点，直接进行搜索
                // printf("节点%d 没有激活过滤器,直接搜\n", node_to_visit.get().get_id());
                spdlog::info("节点{} 没有激活过滤器,直接搜\n", node_to_visit.get().get_id());
                node_to_visit.get().search(query_ptr, query_id, *answers, visited_node_counter,
                                           visited_series_counter);
              }
              //暂时修改用于检查
              // node_to_visit.get().search(query_ptr, query_id, *answers, visited_node_counter, visited_series_counter);
            }
          }
        //}
        visited_node_counter += 1;
        visited_series_counter += node_to_visit.get().get_size();

      } else {
        // 对于非叶子节点，将子节点加入优先队列
        for (auto child_node : node_to_visit.get()){
          VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(route_ptr);
          leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
        }
      }

    }
    // 打印被过滤器剪枝的节点数量
    // printf("查询 %d: 过滤器剪枝节点数: %d, 剪枝序列数: %d\n", query_id, nfpruned_node_counter, nfpruned_series_counter);
    // printf("查询 %d: 访问节点数: %d, 访问序列数: %d\n", query_id, visited_node_counter, visited_series_counter);
    spdlog::info("查询 {}: 过滤器剪枝了 {} 个节点, {} 个序列", query_id, nfpruned_node_counter, nfpruned_series_counter);
    spdlog::info("查询 {}: 访问节点数: {} 个, 访问序列数: {} 个", query_id, visited_node_counter, visited_series_counter);
    // printf("----------------------------------\n");
  }

  // 修改：只在未传入results时打印结果
  if (results == nullptr){
    printf("----------search:results == nullptr, 打印结果----------\n");
    ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;
    while (!answers->empty()) {
      auto answer = answers->pop_answer();
      if (answer.node_id_ > 0){
        printf("query %d nn %d = %.3f, node %d, db_global_offset_ %d\n",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
        spdlog::info("query {} nn {} = {:.3f}, node {}, db_global_offset_ {}",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
      } else {
        printf("query %d nn %d = %.3f, db_global_offset_ %d\n", query_id, nnn_to_return, answer.nn_dist_, answer.global_offset_);
        spdlog::info("query {} nn {} = {:.3f}, node {}, db_global_offset_ {}",
               query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
      }
      nnn_to_return -= 1;
    }
    if (nnn_to_return > 0){
      return FAILURE;
    }
  }

  return SUCCESS;
}






RESPONSE dstree::Index::search_navigated(ID_TYPE query_id, VALUE_TYPE *series_ptr, VALUE_TYPE *sketch_ptr)
{
  VALUE_TYPE *route_ptr = series_ptr;
  if (config_.get().is_sketch_provided_)
  {
    route_ptr = sketch_ptr;
  }

  auto answers = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);

  if (config_.get().require_neurofilter_ || config_.get().navigator_is_learned_)
  {
    filter_query_tsr_ = torch::from_blob(series_ptr,
                                         {1, config_.get().series_length_},
                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE))
                            .to(*device_);
  }

  std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);

  while (!resident_node.get().is_leaf())
  {
    resident_node = resident_node.get().route(route_ptr);
  }

  ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
  ID_TYPE nfpruned_node_counter = 0, nfpruned_series_counter = 0;

  resident_node.get().search(series_ptr, query_id, *answers, visited_node_counter, visited_series_counter);

  if (config_.get().is_exact_search_){
    auto node_prob = navigator_->infer(filter_query_tsr_);
    auto node_distances = make_reserved<VALUE_TYPE>(nleaf_);

    if (config_.get().navigator_is_combined_){
      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i){
        if (leaf_nodes_[leaf_i].get().get_id() == resident_node.get().get_id()){
          node_distances.push_back(constant::MAX_VALUE);
        }
        else
        {
          node_distances.push_back(leaf_nodes_[leaf_i].get().cal_lower_bound_EDsquare(route_ptr));
        }
      }

      VALUE_TYPE min_prob = constant::MAX_VALUE, max_prob = constant::MIN_VALUE;
      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i){
        if (node_prob[leaf_i] < min_prob)
        {
          min_prob = node_prob[leaf_i];
        }
        else if (node_prob[leaf_i] > max_prob)
        {
          max_prob = node_prob[leaf_i];
        }
      }

      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i){
        node_prob[leaf_i] = (node_prob[leaf_i] - min_prob) / (max_prob - min_prob);
      }

      VALUE_TYPE min_lb_dist = constant::MAX_VALUE, max_lb_dist = constant::MIN_VALUE;
      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i){
        if (node_distances[leaf_i] < min_lb_dist)
        {
          min_lb_dist = node_distances[leaf_i];
        }
        else if (node_distances[leaf_i] > max_lb_dist && node_distances[leaf_i] < constant::MAX_VALUE / 2)
        {
          max_lb_dist = node_distances[leaf_i];
        }
      }

      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i){
        node_prob[leaf_i] = config_.get().navigator_combined_lambda_ * node_prob[leaf_i] + (1 - config_.get().navigator_combined_lambda_) * (1 - (node_distances[leaf_i] - min_lb_dist) / (max_lb_dist - min_lb_dist));
      }
    }

    auto node_pos_probs = make_reserved<std::tuple<ID_TYPE, VALUE_TYPE>>(nleaf_);

    for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i)
    {
      if (leaf_nodes_[leaf_i].get().get_id() != resident_node.get().get_id())
      {
        node_pos_probs.push_back(std::tuple<ID_TYPE, VALUE_TYPE>(leaf_i, node_prob[leaf_i]));
      }
    }

#ifdef DEBUG
    // #ifndef DEBUGGED
    spdlog::debug("query {:d} node_distances = {:s}",
                  answers.get()->query_id_, upcite::array2str(node_distances.data(), nleaf_));

    spdlog::debug("query {:d} node_prob = {:s}",
                  answers.get()->query_id_, upcite::array2str(node_prob.data(), nleaf_));
// #endif
#endif

    std::sort(node_pos_probs.begin(), node_pos_probs.end(), dstree::compDecrProb);

    for (ID_TYPE prob_i = 0; prob_i < node_pos_probs.size(); ++prob_i)
    {
      ID_TYPE leaf_i = std::get<0>(node_pos_probs[prob_i]);
      auto node_to_visit = leaf_nodes_[leaf_i];

      // #ifdef DEBUG
      ////#ifndef DEBUGGED
      //      spdlog::debug("query {:d} leaf_i {:d} ({:d}) dist {:.3f} prob {:.3f} ({:.3f}) bsf {:.3f}",
      //                    answers.get()->query_id_,
      //                    leaf_i, navigator_->get_id_from_pos(leaf_i),
      //                    node_distances[leaf_i],
      //                    std::get<1>(node_pos_probs[prob_i]), node_prob[leaf_i],
      //                    answers->get_bsf());
      ////#endif
      // #endif

      if (visited_node_counter < config_.get().search_max_nnode_ &&
          visited_series_counter < config_.get().search_max_nseries_)
      {
        if (config_.get().examine_ground_truth_ || answers->is_bsf(node_distances[leaf_i]))
        {
          node_to_visit.get().search(series_ptr, query_id, *answers, visited_node_counter, visited_series_counter);
        }
      }
    }
  }

  spdlog::info("query {:d} visited {:d} nodes {:d} series",
               query_id, visited_node_counter, visited_series_counter);

  ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;

  while (!answers->empty())
  {
    auto answer = answers->pop_answer();

    if (answer.node_id_ > 0)
    {
      printf("query %d nn %d = %.3f, node %d, db_global_offset_ %d\n",
             query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
    }
    else
    {
      printf("query %d nn %d = %.3f, db_global_offset_ %d\n",
             query_id, nnn_to_return, answer.nn_dist_, answer.global_offset_);
    }
    nnn_to_return -= 1;
  }

  if (nnn_to_return > 0)
  {
    return FAILURE;
  }

  return SUCCESS;
}

// 获取激活的过滤器数量
int dstree::Index::get_active_filter_count() const
{
  int count = 0;
  if (root_)
  {
    count_active_filters(*root_, count);
  }
  return count;
}

// 递归计算激活的过滤器数量
void dstree::Index::count_active_filters(const Node &node, int &count) const
{
  if (node.has_active_filter())
  {
    count++;
  }

  if (!node.is_leaf())
  {
    // 不能直接用迭代器，因为 begin() 和 end() 不是 const 方法
    // 为了解决这个问题，我们可以使用 node 的 children_ 成员变量
    // 但由于这是私有成员，我们需要改用其他方法判断

    // 如果节点不是叶子节点，使用递归方式获取所有子节点
    // 创建一个临时的非const Node以便访问其子节点
    Node &non_const_node = const_cast<Node &>(node);
    for (auto &child : non_const_node)
    {
      count_active_filters(child.get(), count);
    }
  }
}



// 增强版加载函数，特别针对批处理alpha值和查询KNN节点数据
RESPONSE dstree::Index::load_enhanced()
{
  printf("===== 开始使用增强版加载函数 =====\n");
  
  // ==== 第一部分：基本缓冲区设置 ====
  // 计算所需的缓冲区大小
  ID_TYPE ifs_buf_size = sizeof(ID_TYPE) * config_.get().leaf_max_nseries_ * 2; // 基础大小
  ID_TYPE max_num_local_bytes = config_.get().filter_train_num_local_example_;
  
  // 确保缓冲区足够大以处理全局或本地示例
  if (config_.get().filter_train_num_global_example_ > max_num_local_bytes)
  {
    max_num_local_bytes = config_.get().filter_train_num_global_example_;
  }
  max_num_local_bytes *= sizeof(VALUE_TYPE) * config_.get().series_length_;
  
  // 使用更大的缓冲区
  if (max_num_local_bytes > ifs_buf_size)
  {
    ifs_buf_size = max_num_local_bytes;
  }

  printf("分配读取缓冲区: %ld 字节\n", (long)ifs_buf_size);
  void *ifs_buf = std::malloc(ifs_buf_size);
  if (ifs_buf == nullptr) {
    spdlog::error("内存分配失败，无法继续加载");
    return FAILURE;
  }

  // ==== 第二部分：加载树结构 ====
  printf("开始加载树结构...\n");
  nnode_ = 0;
  nleaf_ = 0;
  RESPONSE status = root_->load(ifs_buf, std::ref(*buffer_manager_), nnode_, nleaf_);
  
  if (status == FAILURE)
  {
    spdlog::error("加载索引失败");
    std::free(ifs_buf);
    return FAILURE;
  }
  
  printf("成功加载树结构: %ld 个节点, %ld 个叶子节点\n", (long)nnode_, (long)nleaf_);
  
  // ==== 第三部分：加载批处理数据 ====
  // 仅针对内存模式
  if (!config_.get().on_disk_)
  {
    printf("加载批处理数据到内存...\n");
    buffer_manager_->load_batch();
  }
  
  // 初始化叶节点最小堆
  leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist>(
      CompareDecrNodeDist(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));
  
  // ==== 第四部分：神经过滤器处理 ====
  printf("神经过滤器设置: require_neurofilter=%d, to_load_filters=%d\n", 
         config_.get().require_neurofilter_, config_.get().to_load_filters_);
  
  if (config_.get().require_neurofilter_)
  {
    if (!config_.get().to_load_filters_)
    {
      printf("不加载已有过滤器，执行新训练...\n");
      train();
    }
    else
    {
      if (config_.get().filter_retrain_)
      {
        printf("使用已保存的初始化执行重新训练...\n");
        train(true);
      }
      else if (config_.get().filter_reallocate_multi_ || config_.get().filter_reallocate_single_)
      {
        printf("执行过滤器重新分配...\n");
        filter_allocate(false, true);
      }
      else
      {
        // ==== 第五部分：加载过滤器及相关数据 ====
        printf("===== 加载过滤器相关数据 =====\n");
        printf("初始化过滤器分配器...\n");
        filter_allocate(false);
        
        // 获取激活的过滤器数量
        int active_filter_count = get_active_filter_count();
        printf("激活的过滤器数量: %d\n", active_filter_count);
        printf("加载目录: %s\n", config_.get().index_load_folderpath_.c_str());
        
        // 5.1 加载查询KNN节点数据
        printf("\n===== 加载查询KNN节点数据 =====\n");
        std::string knn_nodes_filepath = config_.get().index_load_folderpath_ + "query_knn_nodes.bin";
        bool knn_nodes_loaded = false;
        
        if (fs::exists(knn_nodes_filepath)) {
            printf("找到查询KNN节点数据文件: %s\n", knn_nodes_filepath.c_str());
            std::ifstream knn_nodes_fin(knn_nodes_filepath, std::ios::binary);
            
            if (knn_nodes_fin.good()) {
                // 清空现有数据
                query_knn_nodes_.clear();
                
                // 读取查询数量
                ID_TYPE num_queries = 0;
                knn_nodes_fin.read(reinterpret_cast<char*>(&num_queries), sizeof(ID_TYPE));
                printf("文件中包含 %ld 个查询的KNN节点信息\n", static_cast<long>(num_queries));
                
                // 读取每个查询的节点信息
                for (ID_TYPE i = 0; i < num_queries; ++i) {
                    // 读取查询ID
                    ID_TYPE query_id = 0;
                    knn_nodes_fin.read(reinterpret_cast<char*>(&query_id), sizeof(ID_TYPE));
                    
                    // 读取该查询的节点映射大小
                    ID_TYPE num_nodes = 0;
                    knn_nodes_fin.read(reinterpret_cast<char*>(&num_nodes), sizeof(ID_TYPE));
                    
                    // 初始化该查询的节点映射
                    std::unordered_map<ID_TYPE, ID_TYPE> node_counts;
                    // 读取每个节点ID和计数
                    for (ID_TYPE j = 0; j < num_nodes; ++j){
                        ID_TYPE node_id = 0;
                        ID_TYPE count = 0;
                        knn_nodes_fin.read(reinterpret_cast<char*>(&node_id), sizeof(ID_TYPE));
                        knn_nodes_fin.read(reinterpret_cast<char*>(&count), sizeof(ID_TYPE));
                        
                        node_counts[node_id] = count;
                    }
                    
                    // 存储到query_knn_nodes_中
                    query_knn_nodes_[query_id] = std::move(node_counts);
                }
                
                printf("成功加载 %zu 个查询的KNN节点数据\n", query_knn_nodes_.size());
                knn_nodes_loaded = true;
            } else {
                printf("无法打开查询KNN节点数据文件\n");
            }
        } else {
            printf("未找到查询KNN节点数据文件: %s\n", knn_nodes_filepath.c_str());
        }
        
        if (!knn_nodes_loaded) {
            printf("注意: 未能加载查询KNN节点数据，某些功能可能不可用\n");
        }
        
        // 5.2 加载过滤器的批处理alpha值
        printf("\n===== 加载过滤器批处理alpha值 =====\n");
        
        // 使用栈进行树的遍历
        std::stack<std::reference_wrapper<dstree::Node>> node_stack;
        node_stack.push(std::ref(*root_));
        int loaded_filter_count = 0;
        int attempted_load_count = 0;
        
        while (!node_stack.empty()) {
            auto node = node_stack.top();
            node_stack.pop();
            
            if (node.get().is_leaf() && node.get().has_active_filter()) {
                ID_TYPE node_id = node.get().get_id();
                attempted_load_count++;
                
                std::string alphas_filepath = config_.get().index_load_folderpath_ + 
                                           "filter_" + std::to_string(node_id) + "_alphas.bin";
                
                if (fs::exists(alphas_filepath)) {
                    printf("加载过滤器 %ld 的批处理alpha值: %s\n", 
                           static_cast<long>(node_id), alphas_filepath.c_str());
                    
                    // 加载alpha值
                    RESPONSE load_result = node.get().load_filter_batch_alphas(alphas_filepath);
                    if (load_result == SUCCESS) {
                        loaded_filter_count++;
                        printf("成功加载过滤器 %ld 的批处理alpha值\n", static_cast<long>(node_id));
                    } else {
                        printf("加载过滤器 %ld 的批处理alpha值失败\n", static_cast<long>(node_id));
                    }
                } else {
                    printf("未找到过滤器 %ld 的批处理alpha文件: %s\n", 
                           static_cast<long>(node_id), alphas_filepath.c_str());
                }
            }
            
            // 将非叶子节点的子节点加入栈中继续遍历
            if (!node.get().is_leaf()) {
                for (auto child_node : node.get()) {
                    node_stack.push(child_node);
                }
            }
        }
        
        printf("尝试加载 %d 个过滤器，成功加载 %d 个过滤器的批处理alpha值\n", 
               attempted_load_count, loaded_filter_count);
      }
      
      // ==== 第六部分：设置设备 ====
      // 针对训练和推理设置不同设备
      printf("\n===== 设置计算设备 =====\n");
      if (config_.get().filter_infer_is_gpu_){
        printf("为推理设置CUDA设备 %d\n", config_.get().filter_device_id_);
        device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                               static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
      } else {
        printf("为推理设置CPU设备\n");
        device_ = std::make_unique<torch::Device>(torch::kCPU);
      }
    }
  }

  // ==== 第七部分：学习导航器设置 ====
  if (config_.get().navigator_is_learned_)
  {
    printf("\n===== 训练导航器 =====\n");
    train();
  }
  
  // 释放缓冲区
  std::free(ifs_buf);
  printf("===== 增强版加载完成 =====\n");
  return SUCCESS;
}

// 执行暴力搜索，不带统计参数的版本
RESPONSE dstree::Index::profile(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr, dstree::Answers *results){
  // 临时变量存储统计数据
  ID_TYPE visited_node_counter = 0;
  ID_TYPE visited_series_counter = 0;
  
  // 调用带统计参数的版本
  return profile(query_id, query_ptr, sketch_ptr, results, visited_node_counter, visited_series_counter);
}

// 执行带过滤器的实际搜索，不带统计参数的版本
RESPONSE dstree::Index::search(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr, dstree::Answers *results){
  // 临时变量存储统计数据
  ID_TYPE visited_node_counter = 0;
  ID_TYPE visited_series_counter = 0;
  ID_TYPE nfpruned_node_counter = 0;
  ID_TYPE nfpruned_series_counter = 0;
  
  // 调用带统计参数的版本
  return search(query_id, query_ptr, sketch_ptr, results, 
               visited_node_counter, visited_series_counter,
               nfpruned_node_counter, nfpruned_series_counter);
}
