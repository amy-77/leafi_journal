//
// Created by Qitong Wang on 2022/10/6.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//
#include "answer.h"
#include "vec.h"
namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

// ====================== 原有构造函数（兼容性扩展） ====================== 
dstree::Answers::Answers(ID_TYPE bsf_capacity, ID_TYPE query_id) :
    capacity_(bsf_capacity), //表示结果集的最大容量。
    query_id_(query_id), //表示当前查询的 ID。
    bsf_distance_(constant::MAX_VALUE),
    visited_node_count_(0),
    visited_series_count_(0) { 
    bsf_distances_ = std::priority_queue<Answer, std::vector<Answer>, compAnswerLess>(
      compAnswerLess(), make_reserved<Answer>(bsf_capacity + 1));
}


//将一个新的Answer对象(distance和node_id)插入优先队列 bsf_distances_中
RESPONSE dstree::Answers::push_bsf(VALUE_TYPE distance, ID_TYPE node_id, ID_TYPE global_offset, ID_TYPE query_id) {
  if (bsf_distances_.size() >= capacity_ && distance >= bsf_distances_.top().nn_dist_) {
    return SUCCESS; // 不插入
  }
  bsf_distances_.emplace(distance, node_id, global_offset, query_id); // QYL 传入 query_id
  if (bsf_distances_.size() > capacity_) {
    bsf_distances_.pop(); //移除队列中的最大元素
  }
  if (!bsf_distances_.empty()) {
    bsf_distance_ = bsf_distances_.top().nn_dist_;
  }
  // bsf_distance_ = bsf_distances_.top().nn_dist_; //更新 bsf_distance_ 为当前 k 个最小距离中的最大值
  return SUCCESS;
}


//检查当前距离 distance 是否比当前的最小距离 bsf_distance_更小。
RESPONSE dstree::Answers::check_push_bsf(VALUE_TYPE distance, ID_TYPE node_id, ID_TYPE global_offset, ID_TYPE query_id) {
  if (is_bsf(distance)) {
    push_bsf(distance, node_id, global_offset, query_id);
  }
  return SUCCESS;
}


//获取优先队列顶部的元素的最小距离 nn_dist_，并将其赋值给 bsf
VALUE_TYPE dstree::Answers::pop_bsf() {
  VALUE_TYPE bsf = bsf_distances_.top().nn_dist_;
  bsf_distances_.pop();
  bsf_distance_ = bsf_distances_.empty() ? constant::MAX_VALUE : bsf_distances_.top().nn_dist_;
  return bsf;
}

//获取优先队列顶部的元素（一个 Answer 对象）
upcite::Answer dstree::Answers::pop_answer() {
  auto answer = bsf_distances_.top();
  bsf_distances_.pop();
  bsf_distance_ = bsf_distances_.empty() ? constant::MAX_VALUE : bsf_distances_.top().nn_dist_;
  return answer;
}

// // 6. 新增的 get_knn_node_distribution 实现
// std::unordered_map<ID_TYPE, std::pair<ID_TYPE, VALUE_TYPE>> dstree::Answers::get_knn_node_distribution(ID_TYPE query_id) const {
//   printf("query  %d 进入 get_knn_node_distribution！\n", query_id);
//   std::unordered_map<ID_TYPE, std::pair<ID_TYPE, VALUE_TYPE>> node_stats;
//   // <node, <count, min_distance>>  相当于存储的query的KNN所在的节点，该节点对应的KNN series数量，以及这些series中的最小距离
//   if (bsf_distances_.empty()) {
//     printf("错误：查询 %d 的结果为空！\n", query_id);
//     return node_stats;
//   }
//   auto temp_queue = bsf_distances_;
//   while (!temp_queue.empty()) {  //循环处理临时队列，直到队列为空
//     const auto& answer = temp_queue.top();
//     if (answer.query_id_ == query_id) {
//       auto& stats = node_stats[answer.node_id_];
//       stats.first += 1;
//       if (stats.first == 1 || answer.nn_dist_ < stats.second) {
//         stats.second = answer.nn_dist_;  // 记录最小距离
//       } 
//     }
//     temp_queue.pop();
//   }
//   return node_stats;
// }