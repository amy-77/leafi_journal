//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_QUERYANSWER_H
#define DSTREE_QUERYANSWER_H

#include <queue>
#include <functional>

#include "global.h"

namespace constant = upcite::constant;

namespace upcite {

struct Answer {
 public:
  explicit Answer(VALUE_TYPE nn_dist, ID_TYPE node_id = -1,  ID_TYPE global_offset = -1,  ID_TYPE query_id = -1) : nn_dist_(nn_dist), node_id_(node_id), global_offset_(global_offset), query_id_(query_id) {};
  ~Answer() = default;
  VALUE_TYPE nn_dist_;
  ID_TYPE node_id_;
  ID_TYPE global_offset_; // 序列在原始数据集中的位置
  ID_TYPE query_id_;  // 新增字段：关联的查询ID
};


// 原先的代码是最小堆，
struct compAnswerLess {
 public:
  bool operator()(Answer &answer_1, Answer &answer_2) const { return answer_1.nn_dist_ < answer_2.nn_dist_; }
};

namespace dstree {

class Answers {
 public:
  // ====================== 原有构造函数 ====================== 
  Answers(ID_TYPE capacity, ID_TYPE query_id);
  Answers(ID_TYPE capacity, ID_TYPE query_id, VALUE_TYPE bsf_distance, std::priority_queue<Answer, std::vector<Answer>, compAnswerLess> &bsf_distances) :
      capacity_(capacity),
      query_id_(query_id),
      bsf_distance_(bsf_distance),
      bsf_distances_(std::priority_queue<Answer, std::vector<Answer>, compAnswerLess>(bsf_distances)),
      visited_node_count_(0),
      visited_series_count_(0) {};
  ~Answers() = default;

  Answers(const Answers &answers) {
    capacity_ = answers.capacity_;
    query_id_ = answers.query_id_;
    bsf_distance_ = answers.bsf_distance_;
    bsf_distances_ = std::priority_queue<Answer, std::vector<Answer>, compAnswerLess>(answers.bsf_distances_);
    visited_node_count_ = answers.visited_node_count_;
    visited_series_count_ = answers.visited_series_count_;
  }

  Answers &operator=(const Answers &answers) {
    capacity_ = answers.capacity_;
    query_id_ = answers.query_id_;
    bsf_distance_ = answers.bsf_distance_;
    bsf_distances_ = std::priority_queue<Answer, std::vector<Answer>, compAnswerLess>(answers.bsf_distances_);
    visited_node_count_ = answers.visited_node_count_;
    visited_series_count_ = answers.visited_series_count_;
    return *this;
  };

  bool is_bsf(VALUE_TYPE distance) const {
    if (bsf_distances_.size() < capacity_) {
      return true;
    } else {
      return distance < bsf_distances_.top().nn_dist_;
    }
  }
  
  // 新增方法：获取当前Top K结果的拷贝
  std::vector<Answer> get_current_topk() const {
    std::vector<Answer> result;
    auto tmp = bsf_distances_;  // 拷贝优先队列
    while (!tmp.empty()) {
        result.push_back(tmp.top());
        tmp.pop();
    }
    // 恢复为从小到大排序（优先队列默认是最大堆，弹出的顺序是从大到小）
    std::reverse(result.begin(), result.end());
    return result;
  }
  
  // 新增方法：更新访问节点和序列计数
  void update_visited_counts(ID_TYPE nodes, ID_TYPE series) {
    visited_node_count_ += nodes;
    visited_series_count_ += series;
  }
  
  // 新增方法：获取访问节点计数
  ID_TYPE get_visited_node_count() const {
    return visited_node_count_;
  }
  
  // 新增方法：获取访问序列计数
  ID_TYPE get_visited_series_count() const {
    return visited_series_count_;
  }

  RESPONSE push_bsf(VALUE_TYPE distance, ID_TYPE node_id = -1, ID_TYPE global_offset = -1, ID_TYPE query_id = -1);
  RESPONSE check_push_bsf(VALUE_TYPE distance, ID_TYPE node_id = -1, ID_TYPE global_offset = -1, ID_TYPE query_id = -1);
  

  VALUE_TYPE get_bsf() const {
     //    return bsf_distances_.top();
    // return bsf_distance_;
    // QYL, 
    return bsf_distances_.empty() ? constant::MAX_VALUE : bsf_distances_.top().nn_dist_;
  };

 

  VALUE_TYPE pop_bsf();
  Answer pop_answer();

  RESPONSE reset(ID_TYPE query_id) {
    query_id_ = query_id;
    while (!bsf_distances_.empty()) {
      bsf_distances_.pop();
    }
    bsf_distance_ = constant::MAX_VALUE;
    visited_node_count_ = 0;
    visited_series_count_ = 0;
    return SUCCESS;
  }

  bool empty() const { return bsf_distances_.empty(); }

  ID_TYPE query_id_;
  
  // QYL 新增：获取指定query_id的KNN结果，按node_id聚合
  // std::unordered_map<ID_TYPE, std::pair<ID_TYPE, VALUE_TYPE>> 
  // get_knn_node_distribution(ID_TYPE query_id) const;

 private:
  ID_TYPE capacity_;
  VALUE_TYPE bsf_distance_;
  std::priority_queue<Answer, std::vector<Answer>, compAnswerLess> bsf_distances_;
  ID_TYPE visited_node_count_ = 0;    // 存储访问的节点数量
  ID_TYPE visited_series_count_ = 0;  // 存储访问的序列数量
};

}
}

#endif //DSTREE_QUERYANSWER_H
