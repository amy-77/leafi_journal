//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_FILTER_H
#define DSTREE_FILTER_H

#include <memory>
#include <functional>
#include <fstream>

#include <torch/torch.h>
#include <spdlog/spdlog.h>

#include "global.h"
#include "config.h"
#include "conformal.h"
#include "models.h"
#include "stat.h"

namespace upcite {
namespace dstree {

class Filter {
 public:
  Filter(dstree::Config &config,
         ID_TYPE id,
         std::reference_wrapper<torch::Tensor> shared_train_queries);
  ~Filter() = default;

  // QYL
  size_t get_global_bsf_distances_size() const {
    return global_bsf_distances_.size();
  }
  size_t get_global_pred_distances_size() const {
    return global_pred_distances_.size();
  }


  
  // 根据训练好的模型预测误差值
  double predict_error_value(double recall, double coverage) const;
  
  // 添加回归模型训练函数声明
  RESPONSE train_regression_model_for_recall_coverage(
      const std::vector<ERROR_TYPE>& recalls,
      const std::vector<ERROR_TYPE>& coverages,
      const std::vector<ID_TYPE>& error_indices,
      ID_TYPE filter_id);
  
  // 添加新函数：使用实际批次误差而非最大误差
  RESPONSE train_regression_model_for_recall_coverage_actual_error(
      const std::vector<ERROR_TYPE>& recalls,
      const std::vector<ERROR_TYPE>& coverages,
      const std::vector<ID_TYPE>& error_indices,
      ID_TYPE batch_id,
      ID_TYPE filter_id);
  
  // 设置基于预测误差的函数 (修改已有函数)
  RESPONSE set_filter_abs_error_interval_by_recall_and_coverage(ERROR_TYPE recall, ERROR_TYPE coverage);

  // 添加获取批次alpha值的方法
  const std::vector<std::vector<VALUE_TYPE>>& get_batch_alphas() const {
    return conformal_predictor_->get_batch_alphas();
  }

  RESPONSE push_global_example(VALUE_TYPE bsf_distance, VALUE_TYPE nn_distance, VALUE_TYPE lb_distance) {
    global_bsf_distances_.push_back(bsf_distance);
    global_lnn_distances_.push_back(nn_distance);
    lb_distances_.push_back(lb_distance);
    global_data_size_ += 1;
    return SUCCESS;
  };
  
  VALUE_TYPE get_bsf_distance(ID_TYPE pos) const { return global_bsf_distances_[pos]; };
  VALUE_TYPE get_pred_distance(ID_TYPE pos) const { return global_pred_distances_[pos]; };



  RESPONSE push_local_example(VALUE_TYPE const *query, VALUE_TYPE nn_distance) {
    local_queries_.insert(local_queries_.end(), query, query + config_.get().series_length_);
    local_lnn_distances_.push_back(nn_distance);

    local_data_size_ += 1;

    return SUCCESS;
  };

  RESPONSE dump_local_example(const std::string &filter_query_filepath) {
    std::ofstream query_fout(filter_query_filepath, std::ios::binary | std::ios_base::app);
    query_fout.write(reinterpret_cast<char *>(local_queries_.data()), sizeof(VALUE_TYPE) * local_queries_.size());
    query_fout.close();
    return SUCCESS;
  };

  bool is_active() const { return is_active_; }
  bool is_trained() const { return is_trained_; }
  RESPONSE activate(MODEL_SETTING &model_setting) {
    if (model_setting.model_setting_str != model_setting_ref_.get().model_setting_str) {
      is_trained_ = false;
    }

    model_setting_ref_ = model_setting;
    is_active_ = true;

    return SUCCESS;
  }

  RESPONSE deactivate() {
    is_active_ = false;

    return SUCCESS;
  }

  RESPONSE trigger_trial(MODEL_SETTING &model_setting) {
    model_setting_ref_ = model_setting;
    is_active_ = false;

    return SUCCESS;
  }

  RESPONSE collect_running_info(MODEL_SETTING &model_setting);
  // QYL batch
  RESPONSE batch_train(bool is_trial = false);

  // 在 Filter 类声明中添加
  RESPONSE load_batch_alphas(const std::string& filepath);

  // 添加清理批处理alphas的方法
  RESPONSE clear_batch_alphas();

  // 添加保存批处理alphas的方法
  RESPONSE save_filter_batch_alphas(const std::string& filepath) const;

  RESPONSE train(bool is_trial = false);
  VALUE_TYPE infer(torch::Tensor &query_series) const;
  VALUE_TYPE infer_calibrated(torch::Tensor &query_series) const;
  VALUE_TYPE infer_raw(torch::Tensor &query_series) const;

  RESPONSE dump(std::ofstream &node_fos) const;
  RESPONSE load(std::ifstream &node_ifs, void *ifs_buf);

  ID_TYPE get_id() const { return id_; };
  VALUE_TYPE get_node_summarization_pruning_frequency() const;
  //QYL
  // const std::vector<VALUE_TYPE>& get_global_lnn_distances() const {return global_lnn_distances_;};

  VALUE_TYPE get_nn_distance(ID_TYPE pos) const { return global_lnn_distances_[pos]; };
  //  QYL,  返回单个样本的 K 个最近邻距离（一维向量）
  // const std::vector<VALUE_TYPE>& get_knn_distances(ID_TYPE pos) const { return global_knn_distances_[pos]; };
  

  std::tuple<VALUE_TYPE, VALUE_TYPE> get_global_lnn_mean_std() const {
    return upcite::cal_mean_std(global_lnn_distances_.data(), global_lnn_distances_.size());
  };

  std::tuple<VALUE_TYPE, VALUE_TYPE> get_filter_local_lnn_mean_std() const {
    return upcite::cal_mean_std(local_lnn_distances_.data(), local_lnn_distances_.size());
  };

  VALUE_TYPE get_val_pruning_ratio() const;

  VALUE_TYPE get_abs_error_interval() const { return conformal_predictor_->get_alpha(); };

  VALUE_TYPE get_abs_error_interval_by_pos(ID_TYPE pos) const { 
    printf("进入filter::get_abs_error_interval_by_pos \n");
    return conformal_predictor_->get_alpha_by_pos(pos); };
  
  VALUE_TYPE get_batch_abs_error_interval_by_pos(ID_TYPE batch_i, ID_TYPE pos) const {
    // printf("进入filter::get_batch_abs_error_interval_by_pos \n");
    // printf("Filter ID %ld: global_lnn_distances_大小: %zu, global_bsf_distances_大小: %zu\n", 
    //   id_, global_lnn_distances_.size(), global_bsf_distances_.size());
    return conformal_predictor_->get_batch_alpha_by_pos(batch_i, pos);
  };
  
  // 获取校准批次对应的查询ID
  const std::vector<std::vector<ID_TYPE>>& get_batch_calib_query_ids() const {
    return batch_calib_query_ids_;
  }

  RESPONSE set_abs_error_interval_by_pos(ID_TYPE pos) {
    return conformal_predictor_->set_alpha_by_pos(pos);
  };

  RESPONSE set_batch_abs_error_interval_by_pos(ID_TYPE pos) {
    return conformal_predictor_->set_batch_alpha_by_pos(pos);
  };  

  RESPONSE fit_filter_conformal_spline(std::vector<ERROR_TYPE> &recalls) {
    return conformal_predictor_->fit_spline(config_.get().filter_conformal_smoothen_core_, recalls);
  }

  RESPONSE fit_filter_batch_bivariate_regression(
      std::vector<ERROR_TYPE> &avg_recalls,
      std::vector<ID_TYPE> &satisfying_batches_counts,
      ID_TYPE total_batches) {
    return conformal_predictor_->fit_batch_bivariate_regression(avg_recalls,  satisfying_batches_counts, total_batches);
  }

  RESPONSE fit_filter_batch_conformal_spline(std::vector<ERROR_TYPE> &recalls) {
    return conformal_predictor_->fit_batch_spline(config_.get().filter_conformal_smoothen_core_, recalls);
  }

  RESPONSE set_abs_error_interval_by_recall(VALUE_TYPE recall) {
    return conformal_predictor_->set_alpha_by_recall(recall);
  };

  RESPONSE set_filter_abs_error_interval_by_recall_and_coverage(VALUE_TYPE recall, VALUE_TYPE coverage) {
    return conformal_predictor_->set_alpha_by_recall_and_coverage(recall, coverage);
  };

  RESPONSE set_abs_error_interval(VALUE_TYPE abs_error) {
    return conformal_predictor_->set_alpha(abs_error, false, true);
  };

  // 获取全局距离向量的大小
  size_t get_global_lnn_distances_size() const {
    return global_lnn_distances_.size();
  }
  
  // 获取本地距离向量的大小
  size_t get_local_lnn_distances_size() const {
    return local_lnn_distances_.size();
  }

  // 生成校准批次，使用组合方法 (n choose k)
  RESPONSE generate_calibration_batches(
      torch::Tensor& calib_data, 
      torch::Tensor& calib_targets,
      std::vector<torch::Tensor>& calib_data_batches,
      std::vector<torch::Tensor>& calib_target_batches,
      std::vector<std::vector<ID_TYPE>>& calib_query_ids);
      
  // 生成均匀划分的校准批次
  RESPONSE generate_uniform_calibration_batches(
      torch::Tensor& calib_data, 
      torch::Tensor& calib_targets,
      std::vector<torch::Tensor>& calib_data_batches,
      std::vector<torch::Tensor>& calib_target_batches,
      std::vector<std::vector<ID_TYPE>>& calib_query_ids);

  // 保存校准查询ID到文件
  RESPONSE save_calib_query_ids(const std::vector<std::vector<ID_TYPE>>& calib_query_ids, 
                               const std::string& filename);

 private:
  std::vector<VALUE_TYPE> global_bsf_distances_;
  std::vector<VALUE_TYPE> global_pred_distances_;
  
  RESPONSE fit_conformal_predictor(bool is_trial = false, bool collect_runtime_stat = false);
  /**
   * @brief 拟合批量保形预测器
   * @param is_trial 是否为试验模式
   * @param calib_batch_indices 校准批次索引列表
   * @param examples_per_batch 每批次样本数
   * @return 执行状态码
   */
  
  RESPONSE fit_batch_conformal_predictor(bool is_trial,
                                     ID_TYPE num_batches,
                                     const std::vector<torch::Tensor>& calib_data_batches,
                                     const std::vector<torch::Tensor>& calib_target_batches);

  std::reference_wrapper<dstree::Config> config_;

  ID_TYPE id_;

  bool is_trained_;
  bool is_active_;

  std::reference_wrapper<MODEL_SETTING> model_setting_ref_;
  // for filter loading only
  MODEL_SETTING model_setting_;

  // torch::save only takes shared_ptr
  std::shared_ptr<FilterModel> model_;
  std::unique_ptr<ConformalRegressor> conformal_predictor_;

  // TODO ref?
  // TODO support different device for training and inference
  std::unique_ptr<torch::Device> device_;

  ID_TYPE global_data_size_, local_data_size_;

  std::reference_wrapper<torch::Tensor> global_queries_;

  //original
  std::vector<VALUE_TYPE> global_lnn_distances_;  //存储当前filter下的所有query(global queries)到该节点的最近1NN距离
  //QYL
  // std::vector<std::vector<VALUE_TYPE>> global_knn_distances_; // 存储每个节点的 K 个最近邻距离

  std::vector<VALUE_TYPE> local_queries_;
  std::vector<VALUE_TYPE> local_lnn_distances_;

  // currently not being used
  std::vector<VALUE_TYPE> lb_distances_; // lower bounds
  std::vector<VALUE_TYPE> ub_distances_; // upper bounds

  // 存储每个校准批次对应的查询ID
  std::vector<std::vector<ID_TYPE>> batch_calib_query_ids_;

  bool is_distances_preprocessed_, is_distances_logged;

  // 存储训练集样本数量
  // ID_TYPE num_train_examples;
  ID_TYPE num_global_train_examples;

};

}
}

#endif //DSTREE_FILTER_H
