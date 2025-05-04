//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_INDEX_H
#define DSTREE_INDEX_H

#include <memory>
#include <vector>
#include <stack>
#include <unordered_map>

#include <torch/torch.h>

#include "global.h"
#include "config.h"
#include "buffer.h"
#include "node.h"
#include "filter.h"
#include "filter_allocator.h"
#include "navigator.h"

namespace upcite {
namespace dstree {

using NODE_DISTNCE = std::tuple<std::reference_wrapper<dstree::Node>, VALUE_TYPE>;

class CompareDecrNodeDist {
 public:
  bool operator()(const NODE_DISTNCE &a, const NODE_DISTNCE &b) {
    return std::get<1>(a) > std::get<1>(b);
  }
};

static bool compDecrProb(std::tuple<ID_TYPE, VALUE_TYPE> &a, std::tuple<ID_TYPE, VALUE_TYPE> &b) {
  return std::get<1>(a) > std::get<1>(b);
}

class Index {
 public:
  explicit Index(Config &config);
  ~Index();

  RESPONSE build();

  RESPONSE dump() const;
  RESPONSE load();
  RESPONSE load_enhanced();

  RESPONSE search(bool is_profile=false);
  RESPONSE search(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr = nullptr,
                 dstree::Answers *results = nullptr);
  RESPONSE search(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr,
                 dstree::Answers *results,
                 ID_TYPE &visited_node_counter, ID_TYPE &visited_series_counter,
                 ID_TYPE &nfpruned_node_counter, ID_TYPE &nfpruned_series_counter);
  RESPONSE search_navigated(ID_TYPE query_id, VALUE_TYPE *series_ptr, VALUE_TYPE *sketch_ptr = nullptr);
 
  //  新增 Getter 方法
  // const std::vector<Answers>& get_train_answers() const {
  //   return train_answers_;
  // }
  // std::vector<Answers> train_answers_;
  
  // 新增获取所有叶子节点ID的方法
  std::vector<ID_TYPE> get_all_leaf_ids() const; // 注意 const 修饰符
  void calculate_recall();
  void calculate_batch_recall();
  // 新增get_query_knn_nodes， 用于allocate recall
  const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& get_query_knn_nodes() const {
    return query_knn_nodes_;
  }
  ID_TYPE get_leaf_count() const {
    return nleaf_; 
  } 
  
  // 获取激活的过滤器数量
  int get_active_filter_count() const;
  
 private:
  // 递归计算激活的过滤器数量
  void count_active_filters(const Node& node, int& count) const;

  // QYL 在Index类中添加成员变量来存储ground truth和实际结果
  std::vector<std::shared_ptr<dstree::Answers>> ground_truth_answers_;
  std::vector<std::shared_ptr<dstree::Answers>> actual_answers_;

  RESPONSE insert(ID_TYPE batch_series_id);

  RESPONSE train(bool is_retrain = false);

  RESPONSE profile(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr = nullptr, 
                  dstree::Answers *results = nullptr);
  RESPONSE profile(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr,
                  dstree::Answers *results,
                  ID_TYPE &visited_node_counter, ID_TYPE &visited_series_counter);

  // initialize filter's member variables except the model
  RESPONSE filter_initialize(dstree::Node &node, ID_TYPE *filter_id);
  // to retrain
  RESPONSE filter_deactivate(dstree::Node &node);

  RESPONSE filter_collect();
  RESPONSE filter_collect_mthread();

  // assign model settings to filters and initialize their model variable
  RESPONSE filter_allocate(bool to_assign = true, bool reassign = false);

  RESPONSE filter_train();
  RESPONSE filter_train_mthread();
  //QYL: 存储每个query的KNN结果
  std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>> query_knn_nodes_;

  std::reference_wrapper<Config> config_;

  std::unique_ptr<BufferManager> buffer_manager_;

  std::unique_ptr<Node> root_;
  ID_TYPE nnode_, nleaf_;
  std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist> leaf_min_heap_;

  std::unique_ptr<Allocator> allocator_;

  VALUE_TYPE *filter_train_query_ptr_;
  torch::Tensor filter_train_query_tsr_;
  torch::Tensor filter_query_tsr_;
  std::unique_ptr<torch::Device> device_;
  std::stack<std::reference_wrapper<Filter>> filter_cache_;
  // Map filter IDs to filter references for easy lookup
  std::unordered_map<ID_TYPE, Filter*> filter_id_to_filter_;

  std::vector<Answers> train_answers_;
  std::unique_ptr<Navigator> navigator_;
  std::vector<std::reference_wrapper<Node>> leaf_nodes_;
};

}
}

#endif //DSTREE_INDEX_H
