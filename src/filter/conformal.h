//
// Created by Qitong Wang on 2023/2/20.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
// credit: https://github.com/henrikbostrom/crepes
//

#ifndef DSTREE_CONFORMAL_H
#define DSTREE_CONFORMAL_H

#include <vector>
#include <memory>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>

#include "global.h"
#include "config.h"
#include "interval.h"
#include <Eigen/Dense>

// 条件包含dlib
#ifdef USE_DLIB_GAM
#include <dlib/matrix.h>
#include <dlib/svm.h>
#include <dlib/statistics.h>
#endif

namespace upcite {

enum CONFORMAL_CORE {
  DISCRETE = 0,
  SPLINE = 1 // smoothened
};

enum CONFIDENCE_LEVEL_EXTERNAL {
  EXT_DISCRETE = 2,
  EXT_SPLINE = 3
};

class ConformalPredictor {
 public:
  ConformalPredictor() : is_fitted_(false), is_trial_(false) {};
  ~ConformalPredictor() = default;
    // 添加这个方法
    // 添加新的public getter方法
  VALUE_TYPE get_alpha() const;
  RESPONSE set_alpha(VALUE_TYPE alpha, bool is_trial = true, bool to_rectify = false);

  VALUE_TYPE get_alpha_by_pos(ID_TYPE pos) const;

  VALUE_TYPE get_batch_alpha_by_pos(ID_TYPE batch_i, ID_TYPE pos) const;

  RESPONSE set_alpha_by_pos(ID_TYPE pos);

  RESPONSE set_batch_alpha_by_pos(ID_TYPE pos);

  // 添加获取批次alpha值的方法
  const std::vector<std::vector<VALUE_TYPE>>& get_batch_alphas() const {
    return batch_alphas_;
  }

  RESPONSE dump(std::ofstream &node_fos) const;
  RESPONSE load(std::ifstream &node_ifs, void *ifs_buf);
  RESPONSE load_batch_alphas(const std::string& filepath);
  
  // 添加保存批处理alphas的方法
  RESPONSE save_batch_alphas(const std::string& filepath) const;

  // 新增：清理批次数据
  void clear_batch_data();

  // 新增：在内存中初始化批处理alphas值
  RESPONSE initialize_batch_alphas();

  bool is_fitted() const {
    return is_fitted_;
  }

 protected:
  bool is_fitted_;
  bool is_trial_;
  CONFORMAL_CORE core_;

  VALUE_TYPE confidence_level_{};
  ID_TYPE abs_error_i_{};
  VALUE_TYPE alpha_{};

  std::vector<ERROR_TYPE> alphas_;
  std::vector<std::vector<VALUE_TYPE>> batch_alphas_; // 存储每个批次的alphas值

};

class ConformalRegressor : public ConformalPredictor {
 public:
  explicit ConformalRegressor(std::string core_type_str, VALUE_TYPE confidence);
  ~ConformalRegressor() = default;


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

  double predict_error_value(double recall, double coverage) const;

  // Getter方法
  const std::vector<std::vector<VALUE_TYPE>>& get_batch_alphas() const {
      return batch_alphas_;
  }

  RESPONSE set_alpha_by_recall(VALUE_TYPE recall);
  // QYL 从二元回归模型中预测分位数
  ID_TYPE predict_quantile_from_bivariate(ERROR_TYPE recall, ERROR_TYPE coverage) const;
  RESPONSE set_alpha_by_recall_and_coverage(ERROR_TYPE target_recall, ERROR_TYPE target_coverage);
  // batch
  RESPONSE fit_batch(const std::vector<std::vector<ERROR_TYPE>>& batch_residuals);

  RESPONSE fit(std::vector<ERROR_TYPE> &residuals);
//               std::vector<VALUE_TYPE> &sigmas, std::vector<ID_TYPE> &bins);

  RESPONSE fit_spline(std::string &spline_core, std::vector<ERROR_TYPE> &recalls);

  RESPONSE fit_batch_spline(std::string &spline_core, std::vector<ERROR_TYPE> &avg_recalls);

  RESPONSE fit_batch_bivariate_regression(std::vector<ERROR_TYPE> &avg_recalls,
  std::vector<ID_TYPE> &satisfying_batches_counts, ID_TYPE total_batches);

  INTERVAL predict(VALUE_TYPE y_hat,
                   VALUE_TYPE confidence_level = -1,
                   VALUE_TYPE y_max = constant::MAX_VALUE,
                   VALUE_TYPE y_min = constant::MIN_VALUE);

  // qyl
  VALUE_TYPE predict_calibrated(VALUE_TYPE y_hat) const;

  // 新增方法声明
  // 检查是否应该使用线性模型
  bool check_nonlinearity(const std::vector<double>& X1,
                          const std::vector<double>& X2,
                          const std::vector<double>& Y);
  
  // GAM模型拟合
  void fit_gam_model(const std::vector<double>& X1,
                     const std::vector<double>& X2,
                     const std::vector<double>& Y);
  
  // 基于回归模型预测alpha值
  VALUE_TYPE predict_alpha(ERROR_TYPE recall, ERROR_TYPE coverage) const;

 private:
  std::unique_ptr<gsl_interp_accel> gsl_accel_;
  std::unique_ptr<gsl_spline> gsl_spline_;
  // 二元回归模型权重
  double recall_weight_;
  double coverage_weight_;
  
  // 新增变量
  std::vector<double> regression_coeffs_;  // 回归系数
  // bool regression_model_fitted_ = false;   // 模型是否已拟合
  bool is_linear_ = true;                // 是否使用线性模型
  double gam_smoothness_ = 0.5;          // GAM模型平滑度
  
  // 可选：用于处理dlib库缺失的情况
  #ifdef USE_DLIB_GAM
  // 定义RVM模型类型 (替代不存在的GAM类型)
  typedef dlib::matrix<double,2,1> sample_type;
  typedef dlib::vector_normalizer<sample_type> normalizer_type;
  normalizer_type normalizer;  // 保存训练数据的归一化参数
  dlib::rvm_regression_trainer<dlib::radial_basis_kernel<sample_type>> rvm_trainer;
  typedef dlib::decision_function<dlib::radial_basis_kernel<sample_type>> rvm_function_type;
  rvm_function_type rvm_model;
  #endif

  // 添加配置对象
  // dstree::Config config_;
};

}

#endif //DSTREE_CONFORMAL_H
