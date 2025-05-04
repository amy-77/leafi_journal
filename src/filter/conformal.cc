//
// Created by Qitong Wang on 2023/2/20.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "conformal.h"

#include <algorithm>
#include <fstream>

#include "spdlog/spdlog.h"

#include "comp.h"
#include "vec.h"
#include "config.h"
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

namespace upcite {

upcite::ConformalRegressor::ConformalRegressor(std::string core_type_str,
                                               VALUE_TYPE confidence) :
    gsl_accel_(nullptr),
    gsl_spline_(nullptr){
  if (core_type_str == "discrete") {
    core_ = DISCRETE;
  } else if (core_type_str == "spline") {
    core_ = SPLINE;
  } else {
    spdlog::error("conformal core {:s} is not recognized; roll back to the default: discrete",
                  core_type_str);
    core_ = DISCRETE;
  }

  confidence_level_ = confidence;
}

RESPONSE upcite::ConformalRegressor::fit(std::vector<ERROR_TYPE> &residuals) {
  // printf("------------进入conformal_predictor_->fit(residuals)-----------\n");
  if (!alphas_.empty()) {
    spdlog::warn("conformal alphas have been set; clear");
    alphas_.clear();
  }
  // 将残差的绝对值存储到 alphas_ 中
  alphas_.assign(residuals.begin(), residuals.end());
  for (auto &alpha : alphas_) { alpha = alpha < 0 ? -alpha : alpha; }

  // 对alphas_进行排序（从小到大）
  std::sort(alphas_.begin(), alphas_.end()); // non-decreasing
  
  // printf("core_ = %d\n", core_);
  if (core_ == DISCRETE) {
    is_fitted_ = true;
    is_trial_ = false;

    printf("\n !!!!!!!!!!!!!!  [DEBUG] confidence_level_ = %.3f\n", static_cast<float>(confidence_level_));    
    abs_error_i_ = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence_level_);
    printf("is_fitted_ = true, core_ == DISCRETE");
    printf("\n !!!!!!!!!!!  [DEBUG] 检查分位数索引: abs_error_i_ = %ld, alphas_.size() = %ld\n", 
       static_cast<long>(abs_error_i_), 
       static_cast<long>(alphas_.size()));
    // 根据置信水平选择对应的分位数
    alpha_ = alphas_[abs_error_i_]; // suspicious for a segfault
    printf("\n !!!!!!!!!!!  [DEBUG] alpha_ = %.3f\n", static_cast<float>(alpha_));
  } else { // core_ == SPLINE
    // fit later with recalls as input
    // printf("is_fitted_ = false, core_ == Spline, fit later with recalls as input\n");
    is_fitted_ = false;
  }
  return SUCCESS;
}

RESPONSE upcite::ConformalRegressor::fit_batch(const std::vector<std::vector<ERROR_TYPE>>& batch_residuals) {
  // 清空之前的批次alphas
  batch_alphas_.clear();
  // 将残差的绝对值存储到 alphas_ 中
  ID_TYPE num_batches = batch_residuals.size();
  batch_alphas_.resize(num_batches);

  // printf("[DEBUG] ConformalRegressor::fit_batch: 处理 %d 个批次\n", num_batches);
  // printf("batch_alphas_外层大小（批次数）: %zu\n", batch_alphas_.size());

  // 为每个批次分别排序并计算alphas
  for (ID_TYPE batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    const auto& residuals = batch_residuals[batch_idx];
    if (residuals.size() < 3) {
      printf("[ERROR] 批次 %d 样本不足: %zu (需要至少3个)\n", batch_idx+1, residuals.size());
      return FAILURE;
    }
    // 复制并排序此批次的残差
    std::vector<ERROR_TYPE> sorted_residuals = residuals;
    
    // 排序残差值
    std::sort(sorted_residuals.begin(), sorted_residuals.end());
    
    // 确保所有值非负（传入的残差应该已经是正的，但为了安全起见）
    // for (auto& val : sorted_residuals) {
    //   if (val < 0) {
    //     val = 0;
    //   }
    // }
    
    // 计算此批次的alphas
    std::vector<VALUE_TYPE> batch_alpha_values;
    batch_alpha_values.reserve(sorted_residuals.size());
    
    for (ID_TYPE i = 0; i < sorted_residuals.size(); ++i) {
      batch_alpha_values.push_back(sorted_residuals[i]);
    }
    batch_alphas_[batch_idx] = std::move(batch_alpha_values);
    // printf("批次 %d 的alpha值数量: %zu\n", batch_idx, batch_alphas_[batch_idx].size());
    // 在所有批次处理完后，打印二维矩阵的大小
    // printf("batch_alphas_ 维度: [%zu, %zu] (批次数 x 每批次alpha值数量)\n", 
    //       batch_alphas_.size(),  // 批次数
    //       batch_alphas_.empty() ? 0 : batch_alphas_[0].size()  // 第一个批次的大小（假设所有批次大小相同）
    // );
    // QYL 没问题，每个batch的误差有原本batch size+2个
    // printf("[DEBUG] 批次 %d: 残差范围 [%.3f, %.3f], 样本数 %zu\n", 
    //        batch_idx+1, batch_alphas_[batch_idx].front(), batch_alphas_[batch_idx].back(), 
    //        batch_alphas_[batch_idx].size());
  }

  // 打印所有批次的alphas
  // printf("打印第1个批次的alphas:\n");
  // for (const auto& alpha : batch_alphas_[0]) {
  //   printf("%.3f ", static_cast<double>(alpha));
  // }
  // printf("\n");
  
  // // 设置默认alphas为第一个批次的值（可选）
  // if (!batch_alphas_.empty() && !batch_alphas_[0].empty()) {
  //   alphas_ = batch_alphas_[0];
  //   abs_error_i_ = 0;
  // }
  return SUCCESS;
}

RESPONSE upcite::ConformalRegressor::fit_spline(std::string &spline_core, std::vector<ERROR_TYPE> &recalls) {
  std::cout << "spline_core = " << spline_core << std::endl;

  // 打印 alphas_ 的大小
  printf("alphas_.size() = %ld\n", static_cast<long>(alphas_.size()));

  // 打印 alphas_ 的内容   
  // printf("alphas_ = [");
  // for (size_t i = 0; i < alphas_.size(); ++i) {
  //   printf("%.3f", static_cast<double>(alphas_[i]));
  //   if (i < alphas_.size() - 1) {
  //     printf(", ");
  //   }
  // }
  // printf("]\n");

  assert(recalls.size() == alphas_.size());
  gsl_accel_ = std::unique_ptr<gsl_interp_accel>(gsl_interp_accel_alloc());

  if (spline_core == "steffen") {  //选的是Steffen 样条
    gsl_spline_ = std::unique_ptr<gsl_spline>(gsl_spline_alloc(gsl_interp_steffen, recalls.size()));
  } else if (spline_core == "cubic") { //如果spline_core是"cubic"，则使用 三次样条（gsl_interp_cspline）
    gsl_spline_ = std::unique_ptr<gsl_spline>(gsl_spline_alloc(gsl_interp_cspline, recalls.size()));
  } else {
    spdlog::error("conformal spline core {:s} is not recognized; roll back to the default: steffen", spline_core);

    gsl_spline_ = std::unique_ptr<gsl_spline>(gsl_spline_alloc(gsl_interp_steffen, recalls.size()));
  }
  //初始化样条曲线
  gsl_spline_init(gsl_spline_.get(), recalls.data(), alphas_.data(), recalls.size());
  
  is_fitted_ = true; //表示样条曲线已成功拟合。
  is_trial_ = false; //表示当前不是试验模式

  return SUCCESS;
}




RESPONSE upcite::ConformalRegressor::fit_batch_bivariate_regression(
    std::vector<ERROR_TYPE> &avg_recalls, 
    std::vector<ID_TYPE> &satisfying_batches_counts,
    ID_TYPE total_batches) {
    
  printf("\n----------开始执行fit_batch_bivariate_regression----------\n");
  printf("参数大小: avg_recalls=%zu, satisfying_batches_counts=%zu, total_batches=%ld\n", 
          avg_recalls.size(), satisfying_batches_counts.size(), (long)total_batches);

  // 检查数据有效性
  printf("检查batch_alphas_: 地址=%p, 是否为空=%s\n", 
          (void*)&batch_alphas_, batch_alphas_.empty() ? "是" : "否");
  
  if (batch_alphas_.empty()) {
    spdlog::error("Cannot fit regression without batch alpha data");
    printf("Cannot fit regression without batch alpha data\n");
    return FAILURE;
  }
  
  printf("batch_alphas_大小(批次数)=%zu\n", batch_alphas_.size());
  for (size_t i = 0; i < batch_alphas_.size(); i++) {
    printf("批次%zu: 大小=%zu\n", i, batch_alphas_[i].size());
  }
  
  // 确保输入数据长度一致
  printf("开始检查首个批次...\n");
  if (batch_alphas_.size() > 0) {
    printf("首个批次非空: batch_alphas_[0]大小=%zu\n", batch_alphas_[0].size());
  } else {
    printf("警告: batch_alphas_长度为0但之前检查不为空!\n");
    return FAILURE;
  }
  
  const size_t num_quantiles = batch_alphas_[0].size();
  printf("批次0中alpha值数量: num_quantiles=%zu\n", num_quantiles);
  
  if (avg_recalls.size() != num_quantiles || satisfying_batches_counts.size() != num_quantiles) {
    spdlog::error("Input data size mismatch for bivariate regression");
    printf("Input data size mismatch for bivariate regression\n");
    return FAILURE;
  }
  
  // 打印基本信息
  printf("\n开始构建二元回归模型...\n");
  printf("误差分位数\t平均召回率\t覆盖率\t\tAlpha值\n");
  
  // 准备回归数据
  std::vector<double> X_recall(num_quantiles);       // 第一个输入变量：平均recall
  std::vector<double> X_coverage(num_quantiles);     // 第二个输入变量：覆盖率
  std::vector<double> Y_quantile(num_quantiles);     // 输出变量：误差分位数
  std::vector<double> Y_alpha(num_quantiles);        // 对应的alpha值
  
  // 准备训练数据
  for (size_t i = 0; i < num_quantiles; i++) {
    X_recall[i] = avg_recalls[i];
    X_coverage[i] = static_cast<double>(satisfying_batches_counts[i]) / total_batches;
    Y_quantile[i] = static_cast<double>(i);
    
    // 计算该分位数的最大alpha值
    double max_alpha = 0.0; // 初始化最大alpha值
    
    // printf("处理分位数%zu: ", i);
    for (size_t batch_idx = 0; batch_idx < batch_alphas_.size(); batch_idx++) {
      // printf("检查批次%zu (大小=%zu): ", batch_idx, batch_alphas_[batch_idx].size());
              // 添加严格的边界检查
      if (batch_alphas_[batch_idx].empty()) {
          printf("批次%zu为空, 跳过\n", batch_idx);
          continue;  // 跳过空批次
      }
      
      if (i < batch_alphas_[batch_idx].size()) {
        // printf("访问索引%zu, 值=%.6f", i, batch_alphas_[batch_idx][i]);
        if (batch_alphas_[batch_idx][i] > max_alpha) {
          max_alpha = batch_alphas_[batch_idx][i];
          // printf(" (新最大值)");
        }
      } else {
        printf("索引%zu超出范围", i);
      }
      // printf("\n");
    }
    // 不再使用平均值，而是使用最大值
    Y_alpha[i] = max_alpha;
    
    // printf("%zu\t%.4f\t%.4f\t%.4f\n", 
    //        i, X_recall[i], X_coverage[i], Y_alpha[i]);
  }

  // === 3. 线性诊断 ===
  is_linear_ = check_nonlinearity(X_recall, X_coverage, Y_alpha);
  printf("线性诊断结果: is_linear_ = %s\n", is_linear_ ? "true (线性)" : "false (非线性)");

  // === 4. 模型拟合 ===
  if (is_linear_) {
    // 多元线性回归
    Eigen::MatrixXd X(num_quantiles, 2);
    Eigen::VectorXd Y(num_quantiles);
    for (size_t i = 0; i < num_quantiles; ++i) {
        X(i, 0) = X_recall[i];
        X(i, 1) = X_coverage[i];
        Y(i) = Y_alpha[i];
    }
    
    // 添加截距项
    Eigen::MatrixXd X_with_intercept = Eigen::MatrixXd::Ones(num_quantiles, 3);
    X_with_intercept.block(0, 1, num_quantiles, 2) = X;
    
    // 求解回归系数
    Eigen::VectorXd beta = (X_with_intercept.transpose() * X_with_intercept)
                          .ldlt().solve(X_with_intercept.transpose() * Y);
    
    // 保存系数 [β0, β1, β2]
    regression_coeffs_ = {beta(0), beta(1), beta(2)};
    
    spdlog::info("Linear model fitted: Y = {:.3f} + {:.3f}*Recall + {:.3f}*Coverage",
                regression_coeffs_[0], regression_coeffs_[1], regression_coeffs_[2]);
  } else {
    #ifdef USE_DLIB_GAM
        fit_gam_model(X_recall, X_coverage, Y_alpha);
    #else
        // 退化为二次多项式
        Eigen::MatrixXd X_poly(num_quantiles, 5);
        for (size_t i = 0; i < num_quantiles; ++i) {
            X_poly(i, 0) = 1.0; // 截距
            X_poly(i, 1) = X_recall[i];
            X_poly(i, 2) = X_coverage[i];
            X_poly(i, 3) = X_recall[i] * X_recall[i];
            X_poly(i, 4) = X_coverage[i] * X_coverage[i];
        }
        Eigen::VectorXd Y_eigen = Eigen::Map<Eigen::VectorXd>(Y_alpha.data(), Y_alpha.size());
        Eigen::VectorXd beta_poly = (X_poly.transpose() * X_poly)
                                  .ldlt().solve(X_poly.transpose() * Y_eigen);
        
        // 保存系数 [β0, β1, β2, β3, β4]
        regression_coeffs_.resize(5);
        for (int i = 0; i < 5; ++i) {
            regression_coeffs_[i] = beta_poly(i);
        }
        
        spdlog::warn("Using quadratic model due to missing GAM support");
    #endif
  }

  is_fitted_ = true;
  printf("训练完成: model_fitted=%d, coeffs_size=%zu\n", 
          is_fitted_, regression_coeffs_.size());
  return SUCCESS;

  
  // 使用GSL多项式拟合进行二元回归
  // 注意：GSL没有直接的二元回归函数，这里使用多项式拟合的方法
  
  // // 1. 使用recall和coverage的加权和作为模型输入
  // const double recall_weight = 0.6;    // 召回率权重
  // const double coverage_weight = 0.4;  // 覆盖率权重
  // std::vector<double> X_combined(num_quantiles);
  
  // for (size_t i = 0; i < num_quantiles; i++) {
  //   // 将两个特征组合为一个特征
  //   X_combined[i] = recall_weight * X_recall[i] + coverage_weight * X_coverage[i];
  // }
  
  // // 2. 拟合曲线：组合特征 -> 误差分位数
  // gsl_accel_.reset(gsl_interp_accel_alloc());
  // gsl_spline_.reset(gsl_spline_alloc(gsl_interp_steffen, num_quantiles));
  
  // // 对X进行排序并相应调整Y
  // std::vector<size_t> indices(num_quantiles);
  // for (size_t i = 0; i < num_quantiles; i++) {
  //   indices[i] = i;
  // }
  
  // // 按X_combined排序
  // std::sort(indices.begin(), indices.end(),
  //           [&X_combined](size_t a, size_t b) { return X_combined[a] < X_combined[b]; });
  
  // std::vector<double> X_sorted(num_quantiles);
  // std::vector<double> Y_sorted(num_quantiles);
  
  // for (size_t i = 0; i < num_quantiles; i++) {
  //   X_sorted[i] = X_combined[indices[i]];
  //   Y_sorted[i] = Y_quantile[indices[i]];
  // }
  
  // // 初始化样条
  // gsl_spline_init(gsl_spline_.get(), X_sorted.data(), Y_sorted.data(), num_quantiles);
  
  // printf("\n二元回归模型构建完成。\n");
  
  // // 保存模型参数以便后续使用
  // recall_weight_ = recall_weight;
  // coverage_weight_ = coverage_weight;
  
  // is_fitted_ = true;
  // is_trial_ = false;
  
  // return SUCCESS;
}





bool upcite::ConformalRegressor::check_nonlinearity(
    const std::vector<double>& X1,
    const std::vector<double>& X2,
    const std::vector<double>& Y) {
    
    const size_t n = X1.size();
    if (n < 10) return true; // 数据量小默认线性

    // 1. 计算线性模型残差
    Eigen::MatrixXd X_linear(n, 2);
    Eigen::VectorXd Y_vec(n);
    for (size_t i = 0; i < n; ++i) {
        X_linear(i, 0) = X1[i];
        X_linear(i, 1) = X2[i];
        Y_vec(i) = Y[i];
    }
    
    // 添加截距项
    Eigen::MatrixXd X_with_intercept = Eigen::MatrixXd::Ones(n, 3);
    X_with_intercept.block(0, 1, n, 2) = X_linear;
    
    // 求解线性回归系数
    Eigen::VectorXd beta = (X_with_intercept.transpose() * X_with_intercept)
                           .ldlt().solve(X_with_intercept.transpose() * Y_vec);
    
    // 计算残差
    Eigen::VectorXd predictions = X_with_intercept * beta;
    Eigen::VectorXd residuals = Y_vec - predictions;

    // 2. 残差自相关检验（Durbin-Watson）
    double dw_stat = 0.0;
    for (size_t i = 1; i < n; ++i) {
        dw_stat += pow(residuals(i) - residuals(i-1), 2);
    }
    dw_stat /= residuals.squaredNorm();
    
    // 3. 判断标准
    if (dw_stat < 1.0 || dw_stat > 3.0) {
        spdlog::debug("Nonlinear detected (Durbin-Watson={:.3f})", dw_stat);
        printf("Nonlinear detected (Durbin-Watson=%.3f)\n", dw_stat);

        return false; // 非线性关系
    }
    return true; // 线性关系
}


// USE_DLIB_GAM
void upcite::ConformalRegressor::fit_gam_model(
    const std::vector<double>& X1,
    const std::vector<double>& X2,
    const std::vector<double>& Y) {
    
    #ifdef USE_DLIB_GAM
    // 仅当定义了USE_DLIB_GAM时才能使用dlib库
    const size_t num_samples = X1.size();
    
    // 正确初始化样本矩阵
    std::vector<sample_type> samples(num_samples);
    std::vector<double> targets(Y.begin(), Y.end());
    
    for (size_t i = 0; i < num_samples; ++i) {
        // 每个样本是2x1的矩阵（列向量）
        samples[i](0) = X1[i];
        samples[i](1) = X2[i];
    }

    // 设置RVM训练器参数
    rvm_trainer.set_kernel(dlib::radial_basis_kernel<sample_type>(0.1));
    
    // 训练RVM模型
    normalizer.train(samples);
    // 修复：对样本集合应用正规化处理
    std::vector<sample_type> normalized_samples;
    normalized_samples.reserve(samples.size());
    for (const auto& sample : samples) {
        normalized_samples.push_back(normalizer(sample));
    }
    rvm_model = rvm_trainer.train(normalized_samples, targets);
    
    // 输出使用的相关向量数量
    spdlog::info("RVM model fitted successfully with {} relevance vectors", 
                 rvm_model.basis_vectors.size());
    #else
    // 当没有dlib库时，回退到多项式拟合
    spdlog::warn("Dlib RVM not available, falling back to polynomial regression");
    
    // 创建多项式特征
    Eigen::MatrixXd X_poly(X1.size(), 5);
    for (size_t i = 0; i < X1.size(); ++i) {
        X_poly(i, 0) = 1.0; // 截距
        X_poly(i, 1) = X1[i];
        X_poly(i, 2) = X2[i];
        X_poly(i, 3) = X1[i] * X1[i];
        X_poly(i, 4) = X2[i] * X2[i];
    }
    
    Eigen::VectorXd Y_vec = Eigen::Map<const Eigen::VectorXd>(Y.data(), Y.size());
    
    // 求解多项式回归系数
    Eigen::VectorXd beta_poly = (X_poly.transpose() * X_poly)
                              .ldlt().solve(X_poly.transpose() * Y_vec);
    
    // 保存系数
    regression_coeffs_.resize(5);
    for (int i = 0; i < 5; ++i) {
        regression_coeffs_[i] = beta_poly(i);
    }
    #endif
}

VALUE_TYPE upcite::ConformalRegressor::predict_alpha(
    ERROR_TYPE recall, 
    ERROR_TYPE coverage) const {
    
    if (!is_fitted_) return constant::MAX_VALUE;

    if (is_linear_) {
        // 线性模型预测
        if (regression_coeffs_.size() >= 3) {
            return regression_coeffs_[0] + 
                   regression_coeffs_[1] * recall + 
                   regression_coeffs_[2] * coverage;
        } else {
            spdlog::error("Linear model coefficients not available");
            return constant::MAX_VALUE;
        }
    } else {
        // RVM/多项式预测
        #ifdef USE_DLIB_GAM
        // 仅当定义了USE_DLIB_GAM时才能使用dlib库
        sample_type sample;
        sample(0) = recall;
        sample(1) = coverage;
        
        // 使用训练好的RVM模型进行预测
        return rvm_model(normalizer(sample));
        #else
        // 二次多项式预测
        if (regression_coeffs_.size() >= 5) {
            return regression_coeffs_[0] + 
                   regression_coeffs_[1] * recall +
                   regression_coeffs_[2] * coverage +
                   regression_coeffs_[3] * recall * recall +
                   regression_coeffs_[4] * coverage * coverage;
        } else {
            spdlog::error("Polynomial model coefficients not available");
            return constant::MAX_VALUE;
        }
        #endif
    }
}

RESPONSE upcite::ConformalRegressor::fit_batch_spline(std::string &spline_core, std::vector<ERROR_TYPE> &avg_recalls) {
  // ================== 新增多批次处理逻辑 ================== //
  // 检查批次数据有效性
  if (batch_alphas_.empty()) {
    spdlog::error("Cannot fit spline without batch alpha data");
    return FAILURE;
  }

  // 验证所有批次的alpha数量一致
  const size_t num_quantiles = batch_alphas_[0].size();
  for (const auto& batch : batch_alphas_) {
    if (batch.size() != num_quantiles) {
      spdlog::error("Inconsistent quantile count across batches (expected {} got {})", 
                   num_quantiles, batch.size());
      return FAILURE;
    }
  }

  // 计算多批次平均alpha值
  std::vector<ERROR_TYPE> avg_alphas(num_quantiles, 0);
  for (size_t q = 0; q < num_quantiles; ++q) {
    double max_alpha = 0; // 初始化最大alpha值
    for (const auto& batch : batch_alphas_) { // 修正：使用batch_alphas_
      // 寻找最大的alpha值
      if (batch[q] > max_alpha) {
        max_alpha = batch[q];
      }
    }
    // 不再使用平均值，而是使用最大值
    avg_alphas[q] = max_alpha;
  }

  // 验证recall数据匹配
  if (avg_recalls.size() != num_quantiles) {
    spdlog::error("Recalls size mismatch (expected {} got {})", 
                 num_quantiles, avg_recalls.size());
    return FAILURE;
  }
  // ================== 修改结束 ================== //

  // 打印调试信息
  spdlog::debug("Fitting {} spline with {} quantiles", spline_core, num_quantiles);
  // std::cout << "spline_core = " << spline_core << std::endl;

  // 重建加速器和样条
  gsl_accel_.reset(gsl_interp_accel_alloc());
  
  // 选择样条类型
  const gsl_interp_type* spline_type = gsl_interp_steffen; // 默认
  if (spline_core == "cubic") {
    spline_type = gsl_interp_cspline;
  } else if (spline_core != "steffen") {
    spdlog::warn("Unsupported spline type: {}, using Steffen", spline_core);
  }

  // 创建样条对象
  gsl_spline_.reset(gsl_spline_alloc(spline_type, num_quantiles));

  // 初始化样条（使用平均后的数据）
  gsl_spline_init(gsl_spline_.get(), 
                 avg_recalls.data(), 
                 avg_alphas.data(),  // 使用计算的平均alpha
                 num_quantiles);

  // 打印拟合数据
  // #ifdef DEBUG
  // printf("[DEBUG] Spline fitting data:\n");
  // for (size_t i = 0; i < num_quantiles; ++i) {
  //   printf("Quantile %zu | Recall: %.4f | Alpha: %.4f\n", 
  //         i, avg_recalls[i], avg_alphas[i]);
  // }
  // #endif

  // 更新状态
  is_fitted_ = true;
  is_trial_ = false;

  return SUCCESS;
}


// 在conformal.cc中实现新方法
VALUE_TYPE upcite::ConformalRegressor::predict_calibrated(VALUE_TYPE y_hat) const {
  // 确保模型已训练
  if (!is_fitted_ && !is_trial_) {
    return y_hat; // 无法校准，直接返回原值
  }
  // 获取当前的alpha值
  VALUE_TYPE current_alpha = alpha_;
  // 计算校准后的预测值（预测 - alpha）
  VALUE_TYPE calibrated = y_hat - current_alpha;
  // 不允许负值
  calibrated = std::max(static_cast<VALUE_TYPE>(0.0), calibrated);
  // printf("predict_calibrated: 预测距离=%.3f, 预分配误差alpha=%.3f, 校准后距离=%.3f\n", y_hat, current_alpha, calibrated);
  spdlog::info("predict_calibrated: 预测距离={:.3f}, 预分配误差alpha={:.3f}, 校准后距离={:.3f}",  y_hat, current_alpha, calibrated);
  return calibrated;
}


upcite::INTERVAL upcite::ConformalRegressor::predict(VALUE_TYPE y_hat,
                                                     VALUE_TYPE confidence_level,
                                                     VALUE_TYPE y_max,
                                                     VALUE_TYPE y_min) {

  if (is_fitted_) {
    // printf("is_fitted_ = %d\n", is_fitted_);
    // printf("confidence_level_ = %.3f\n", confidence_level_);
    // printf("alphas_.size() = %zu\n", alphas_.size());
    // confidence_level_ > 1 denotes it is set externally
    if (confidence_level_ <= 1 && !upcite::is_equal(confidence_level_, confidence_level)) {
      assert(confidence_level >= 0 && confidence_level <= 1);
      
      abs_error_i_ = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence_level);
      alpha_ = alphas_[abs_error_i_];

      confidence_level_ = confidence_level;
    }
    // printf("y_hat = %.3f, alpha_ = %.3f\n", y_hat, alpha_);
    // spdlog::info("y_hat = {:.3f}, alpha_ = {:.3f}", y_hat, alpha_);
    return {y_hat - alpha_, y_hat + alpha_};

  } else if (is_trial_) {
    // printf("is_trial_ = %d\n", is_trial_);
    // printf("alpha_ = %.3f\n", alpha_);
    // spdlog::info("is_trial_ = %d, alpha_ = %.3f\n", is_trial_, alpha_);
    return {y_hat - alpha_, y_hat + alpha_};
  } else {
    return {y_min, y_max};
  }
}


RESPONSE upcite::ConformalPredictor::dump(std::ofstream &node_fos) const {
  node_fos.write(reinterpret_cast<const char *>(&core_), sizeof(CONFORMAL_CORE));
  // alphas can be recalculated
//  ID_TYPE alphas_size = static_cast<ID_TYPE>(alphas_.size());
//  node_fos.write(reinterpret_cast<const char *>(&alphas_size), sizeof(ID_TYPE));
//  node_fos.write(reinterpret_cast<const char *>(alphas_.data()), sizeof(VALUE_TYPE) * alphas_.size());
  return SUCCESS;
}


RESPONSE upcite::ConformalPredictor::load(std::ifstream &node_ifs, void *ifs_buf) {
  auto ifs_core_buf = reinterpret_cast<CONFORMAL_CORE *>(ifs_buf);
//  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);
//  auto ifs_value_buf = reinterpret_cast<VALUE_TYPE *>(ifs_buf);

  ID_TYPE read_nbytes = sizeof(CONFORMAL_CORE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  core_ = ifs_core_buf[0];

//  read_nbytes = sizeof(ID_TYPE);
//  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
//  ID_TYPE alphas_size = ifs_id_buf[0];
//  alphas_.reserve(alphas_size);
//
//  read_nbytes = sizeof(VALUE_TYPE) * alphas_size;
//  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
//  alphas_.insert(alphas_.begin(), ifs_value_buf, ifs_value_buf + alphas_size);

  return SUCCESS;
}

VALUE_TYPE upcite::ConformalPredictor::get_alpha() const {
  if (is_fitted_) {
    return alpha_;
  } else if (is_trial_) {
    return alpha_;
  } else {
    return constant::MAX_VALUE;
  }
}


RESPONSE upcite::ConformalPredictor::set_alpha(VALUE_TYPE alpha, bool is_trial, bool to_rectify) {
  if (is_trial) {
    if (is_fitted_) {
      spdlog::error("conformal model is already fitted; cannot run trial");
      return FAILURE;
    } else {
      alpha_ = alpha;

      is_trial_ = true;
    }
  } else if (is_fitted_) {
    if (to_rectify) {
      alpha_ = alpha;
    } else {
      spdlog::error("conformal model is already fitted; cannot directly adjust alpha");
      return FAILURE;
    }
  } else {
    alpha_ = alpha;
  }

  printf("进入set_alpha: is_fitted=%d\n", is_fitted_);
  // 函数结尾
  printf("设置结果: alpha=%.3f\n", alpha_);

  return SUCCESS;
}

VALUE_TYPE upcite::ConformalPredictor::get_batch_alpha_by_pos(ID_TYPE batch_i, ID_TYPE pos) const {
  if (batch_i >= batch_alphas_.size() || batch_alphas_[batch_i].empty()) {
    return constant::MAX_VALUE;
  }
  
  if (pos >= batch_alphas_[batch_i].size()) {
    return constant::MAX_VALUE;
  }
  
  return batch_alphas_[batch_i][pos];
}


//它的作用是根据给定的位置 pos，从 alphas_ 数组中获取对应的值（alpha
VALUE_TYPE upcite::ConformalPredictor::get_alpha_by_pos(ID_TYPE pos) const {
  // 检查 alphas_ 是否为空
  if (alphas_.empty()) {
    spdlog::error("alphas_ is empty in ConformalPredictor");
    printf("错误：alphas_为空，返回MAX_VALUE\n");
    return constant::MAX_VALUE;
  }
  
  // 检查 pos 是否合法
  if (pos < 0) {
    printf("错误：pos=%ld 为负数，返回MAX_VALUE\n", pos);
    return constant::MAX_VALUE;
  }
  if (static_cast<size_t>(pos) >= alphas_.size()) {
    printf("错误：pos=%ld 超出范围 alphas_.size()=%zu，返回MAX_VALUE\n", pos, alphas_.size());
    return constant::MAX_VALUE;
  }
  
  // 检查值是否有效
  VALUE_TYPE result = alphas_[pos];
  if (std::isnan(result) || std::isinf(result)) {
    printf("错误：alphas_[%ld]=%f 不是有效数值，返回MAX_VALUE\n", pos, result);
    return constant::MAX_VALUE;
  }
  
  printf("成功获取 alphas_[%ld]=%f\n", pos, result);
  return result;
}


RESPONSE upcite::ConformalPredictor::set_alpha_by_pos(ID_TYPE pos) {
  if (pos >= 0 && pos < alphas_.size()) {
    alpha_ = alphas_[pos];

    // TODO design a better workflow for is_fitted_
    is_fitted_ = true;
    confidence_level_ = EXT_DISCRETE; 
    return SUCCESS;
  }
  return FAILURE;
}

// 离散方法，给定误差位置，对所有batch的误差取平均找到对应位置的误差
RESPONSE upcite::ConformalPredictor::set_batch_alpha_by_pos(ID_TYPE pos) {
  // 检查批次数据是否为空
  if (batch_alphas_.empty()) {
    spdlog::error("No batch alpha data available");
    return FAILURE;
  }
  // 检查所有批次在pos位置的有效性
  for (const auto& batch : batch_alphas_) {
    if (pos < 0 || pos >= static_cast<ID_TYPE>(batch.size())) {
      spdlog::error("Invalid pos {:d} for batch alpha data", pos);
      return FAILURE;
    }
  }
  
  // 查找所有批次中pos位置的最大alpha值
  ERROR_TYPE max_alpha = 0;
  for (const auto& batch : batch_alphas_) {
    if (batch[pos] > max_alpha) {
      max_alpha = batch[pos];
    }
  }
  
  // 使用最大值而不是平均值
  alpha_ = max_alpha;

  // 更新状态标志
  is_fitted_ = true;
  confidence_level_ = EXT_DISCRETE;

  // 打印调试信息
  spdlog::debug("Set alpha at pos {:d} to {:.2f} (max value across {:d} batches)", 
                pos, alpha_, batch_alphas_.size());
  printf("设置位置 %ld 的 alpha 值为 %.2f (在 %ld 个批次中的最大值)\n", 
         pos, alpha_, batch_alphas_.size());
  return SUCCESS;
}  

RESPONSE upcite::ConformalRegressor::set_alpha_by_recall(VALUE_TYPE recall) {
  assert(gsl_accel_ != nullptr && gsl_spline_ != nullptr);
  alpha_ = gsl_spline_eval(gsl_spline_.get(), recall, gsl_accel_.get());
  printf("recall = %.3f, alpha_ = %.3f\n", static_cast<double>(recall), static_cast<double>(alpha_));
  // TODO design a better workflow for is_fitted_
  is_fitted_ = true;
  confidence_level_ = EXT_SPLINE;

  return SUCCESS;
}


RESPONSE upcite::ConformalRegressor::set_alpha_by_recall_and_coverage(ERROR_TYPE target_recall, ERROR_TYPE target_coverage) {
  if (!is_fitted_) {
    spdlog::error("Cannot set alpha, model not fitted yet");
    printf("设置alpha失败: 模型未训练\n");
    return FAILURE;
  }
  alpha_ = predict_error_value(target_recall, target_coverage);
  spdlog::info("设置 alpha 为 {:.3f}", alpha_);
  printf("设置 alpha=%.3f\n",  alpha_);
  // 更新状态
  is_fitted_ = true;
  confidence_level_ = EXT_SPLINE; // 使用特殊标记表示外部设置的置信度
  // printf("训练完成: model_fitted=%d, coeffs_size=%zu\n", is_fitted_, regression_coeffs_.size());
  // printf("进入set_alpha: is_fitted=%d\n", is_fitted_);
  // 函数结尾
  // printf("设置结果: alpha=%.3f\n", alpha_);
  return SUCCESS;
}


//这个函数返回的是分位数索引（index），而不是实际的误差值
ID_TYPE upcite::ConformalRegressor::predict_quantile_from_bivariate(ERROR_TYPE target_recall, ERROR_TYPE target_coverage) const {
  if (!gsl_accel_ || !gsl_spline_) {
    spdlog::error("二元回归模型未初始化");
    return 0; // 返回默认值
  }
  // 使用权重组合特征
  double combined_feature = recall_weight_ * target_recall + coverage_weight_ * target_coverage;
  // 使用样条曲线预测分位数
  double predicted_quantile = gsl_spline_eval(gsl_spline_.get(), combined_feature, gsl_accel_.get());
  // 将预测值四舍五入到最近的整数
  ID_TYPE result = static_cast<ID_TYPE>(std::round(predicted_quantile));
  // 确保结果在有效范围内
  if (result < 0) {
    result = 0;
  }
  // 如果有batch_alphas_数据，则使用第一个批次的大小作为上限
  size_t max_quantile = 0;
  if (!batch_alphas_.empty() && !batch_alphas_[0].empty()) {
    max_quantile = batch_alphas_[0].size() - 1;
  } else if (!alphas_.empty()) {
    max_quantile = alphas_.size() - 1;
  }
  if (result > static_cast<ID_TYPE>(max_quantile)) {
    result = static_cast<ID_TYPE>(max_quantile);
  }
  spdlog::debug("预测分位数: 召回率={:.4f}, 覆盖率={:.4f} => 分位数={}", 
               target_recall, target_coverage, result);
  return result;
}






RESPONSE upcite::ConformalRegressor::train_regression_model_for_recall_coverage(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages,
    const std::vector<ID_TYPE>& error_indices,
    ID_TYPE filter_id) {
    
    // printf("开始训练二元回归模型，训练样本点数量：%zu\n", recalls.size());    
    if (recalls.size() != coverages.size() || recalls.size() != error_indices.size() || recalls.empty()) {
        printf("错误：输入数据数量不匹配或为空\n");
        return FAILURE;
    }
    
    // 获取当前filter下的cp类的batch_alphas，用于将位置转换为实际误差值
    const auto& batch_alphas = this->get_batch_alphas();
    if (batch_alphas.empty()) {
        printf("错误：batch_alphas为空，无法获取误差值\n");
        return FAILURE;
    }
    
    // printf("batch_alphas大小: %zu 批次, 每个批次 %zu 个误差值\n", batch_alphas.size(),  batch_alphas[0].size());
    spdlog::info("batch_alphas大小: {} 批次, 每个批次 {} 个误差值", batch_alphas.size(),  batch_alphas[0].size());
    
    // 准备训练数据，将error_i转换为对应的最大误差值
    std::vector<double> actual_errors;
    actual_errors.reserve(error_indices.size());

    for (ID_TYPE error_i : error_indices) {
        // printf("error_i %d\n", error_i);
        // 从所有批次中找出该位置对应的最大误差值
        double max_error = 0.0;
        // printf("\nbatch_alphas[%d]大小: %zu ", error_i, batch_alphas[error_i].size());
        //内层循环是为了找到error_i位置下的最大误差值（取同一个误差分位下的所有bacth的最大值）
        for (const auto& batch : batch_alphas) {
            // printf("当前batch大小: %zu\n", batch.size());  17
            // printf("batch[%d]=%.6f ", error_i, batch[error_i]);
            if (error_i < batch.size() && batch[error_i] > max_error) {
                max_error = batch[error_i];
            }
        }
        // printf("max_error %.2f\n", max_error);
        // printf("\n");
        // actual_errors 是每个误差位置下的实际误差值（取所有bacth的最大值）
        actual_errors.push_back(max_error);
        // printf("actual_errors[%d]=%.6f\n", error_i, max_error);
    }
    
    // 打印actual_errors的大小
    printf("过滤器 %ld 使用最大误差值的actual_errors大小: %zu\n", 
           (long)filter_id, actual_errors.size());
    spdlog::info("过滤器 {} 使用最大误差值的actual_errors大小: {}", 
                filter_id, actual_errors.size());
    // 打印所有actual_errors的取值
    // for (size_t i = 0; i < actual_errors.size(); i++) {
    //     printf("%.2f, ", i, actual_errors[i]);
    // }
    // printf("\n");
    // 打印batch_alphas的内容，以矩阵形式展示
    printf("打印batch_alphas的内容(行是batch，列是不同误差分位数):\n");
    for (size_t batch_idx = 0; batch_idx < batch_alphas.size(); batch_idx++) {
        printf("批次 %zu: ", batch_idx);
        for (size_t i = 0; i < batch_alphas[batch_idx].size(); i++) {
            printf("%.2f ", batch_alphas[batch_idx][i]);
        }
        printf("\n");
    }


    // 样本数和特征数
    size_t n = recalls.size();  // 样本数
    size_t p = 6;  // 特征数：常数项、recall、coverage、recall*coverage、recall^2、coverage^2
    
    // 使用GSL库进行多元回归
    gsl_matrix *X = gsl_matrix_alloc(n, p);
    gsl_vector *y = gsl_vector_alloc(n);
    gsl_vector *c = gsl_vector_alloc(p);  // 回归系数
    gsl_matrix *cov = gsl_matrix_alloc(p, p);  // 协方差矩阵
    double chisq;  // 拟合优度
    
    // 填充设计矩阵X和目标向量y（注意：y现在是实际误差值）
    for (size_t i = 0; i < n; i++) {
        // 设计矩阵：[1, recall, coverage, recall*coverage, recall^2, coverage^2]
        gsl_matrix_set(X, i, 0, 1.0);  // 常数项
        gsl_matrix_set(X, i, 1, recalls[i]);
        gsl_matrix_set(X, i, 2, coverages[i]);
        gsl_matrix_set(X, i, 3, recalls[i] * coverages[i]);  // 交互项
        gsl_matrix_set(X, i, 4, recalls[i] * recalls[i]);
        gsl_matrix_set(X, i, 5, coverages[i] * coverages[i]);
        
        // 目标向量 - 使用实际误差值
        gsl_vector_set(y, i, actual_errors[i]);
        // printf("样本 %zu: recall=%.4f, coverage=%.4f -> 位置=%ld, 误差=%.6f\n", 
        //            i, recalls[i], coverages[i], error_indices[i], actual_errors[i]);
        
    }
    
    // 执行回归计算
    gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(n, p);
    int ret = gsl_multifit_linear(X, y, c, cov, &chisq, work);
    
    if (ret != GSL_SUCCESS) {
        printf("错误：GSL回归计算失败，错误码：%d\n", ret);
        // 清理资源
        gsl_multifit_linear_free(work);
        gsl_matrix_free(X);
        gsl_vector_free(y);
        gsl_vector_free(c);
        gsl_matrix_free(cov);
        return FAILURE;
    }
    
    // 提取回归系数
    regression_coeffs_.resize(p);
    for (size_t i = 0; i < p; i++) {
        regression_coeffs_[i] = gsl_vector_get(c, i);
    }
    
    // 打印回归系数和拟合优度
    // printf("回归系数: [常数项, recall, coverage, recall*coverage, recall^2, coverage^2]\n");
    // printf("         [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
    //        regression_coeffs_[0], regression_coeffs_[1], regression_coeffs_[2],
    //        regression_coeffs_[3], regression_coeffs_[4], regression_coeffs_[5]);

    printf("拟合优度 (chi^2): %.1f\n", chisq);
    spdlog::info("拟合优度 (chi^2): {}", chisq);
    // 清理GSL资源
    gsl_multifit_linear_free(work);
    gsl_matrix_free(X);
    gsl_vector_free(y);
    gsl_vector_free(c);
    gsl_matrix_free(cov);
    
    // 更新模型状态
    is_fitted_ = true;
    return SUCCESS;
}





// 添加ConformalPredictor::save_batch_alphas方法实现
// RESPONSE upcite::ConformalPredictor::save_batch_alphas(const std::string& filepath) const {
//     // 检查是否有批次数据
//     if (batch_alphas_.empty()) {
//         spdlog::error("No batch alpha data to save");
//         return FAILURE;
//     }
    
//     // 创建输出文件流   用 
//     std::ofstream alphas_fout(filepath, std::ios::binary);
//     if (!alphas_fout.good()) {
//         spdlog::error("Failed to open file for writing: {}", filepath);
//         return FAILURE;
//     }
    
//     // 写入批次数量
//     ID_TYPE num_batches = static_cast<ID_TYPE>(batch_alphas_.size());
//     alphas_fout.write(reinterpret_cast<const char*>(&num_batches), sizeof(ID_TYPE));
    
//     // 对每个批次
//     int total_invalid_values = 0;
    
//     for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
//         // 写入此批次的alpha值数量
//         ID_TYPE batch_size = static_cast<ID_TYPE>(batch_alphas_[batch_i].size());
//         alphas_fout.write(reinterpret_cast<const char*>(&batch_size), sizeof(ID_TYPE));
        
//         // 创建临时缓冲区用于验证和修复数据
//         // std::vector<ERROR_TYPE> validated_data(batch_size);
//         int batch_invalid_values = 0;
        
//         // 验证并修复每个alpha值
//         for (size_t j = 0; j < batch_size; ++j) {
//             ERROR_TYPE val = batch_alphas_[batch_i][j];
            
//             // 检查数据有效性
//             if (std::isnan(val) || std::isinf(val) || val < 0 || val > 1e10) {
//                 // 无效值替换为0
//                 if (batch_invalid_values < 10) { // 只打印前10个无效值避免过多输出
//                     spdlog::warn("保存批次{}索引{}的值{:.6f}无效, 替换为0", 
//                                 batch_i, j, static_cast<double>(val));
//                     printf("保存批次%ld索引%zu的值%.6f无效，但保持不变\n", 
//                           (long)batch_i, j, static_cast<double>(val));
//                 } else if (batch_invalid_values == 10) {
//                     spdlog::warn("更多无效值...");
//                     printf("更多无效值...\n");
//                 }
//                 // validated_data[j] = 0.0;
//                 batch_invalid_values++;
//             } //else {
//             //     // 有效值保持不变
//             //     validated_data[j] = val;
//             // }
//         }
        
//         // 更新总的无效值计数
//         total_invalid_values += batch_invalid_values;
        
//         // 如果有无效值，记录警告
//         if (batch_invalid_values > 0) {
//             spdlog::warn("保存批次 {} 包含 {} 个无效值 (总大小: {})", 
//                         batch_i, batch_invalid_values, batch_size);
//             printf("保存批次 %ld 包含 %d 个无效值 (总大小: %ld)\n", 
//                   (long)batch_i, batch_invalid_values, (long)batch_size);
//         }

        
//         // 写入验证后的数据
//         alphas_fout.write(reinterpret_cast<const char*>(batch_alphas_[batch_i].data()), 
//                          sizeof(ERROR_TYPE) * batch_size);
//     }
    
//     if (total_invalid_values > 0) {
//         spdlog::warn("保存alpha值时共发现并修复了 {} 个无效值", total_invalid_values);
//         printf("保存alpha值时共发现 %d 个无效值，但未修改\n", total_invalid_values);
//     }
    
//     spdlog::info("Successfully saved batch alphas to {}", filepath);
//     printf("成功保存批次alpha值到: %s\n", filepath.c_str());


//     printf("保存时 batch_alphas_ 类型: ERROR_TYPE (sizeof = %zu 字节)\n", sizeof(ERROR_TYPE));
//     for (size_t i = 0; i < batch_alphas_.size(); i++) {
//         printf("保存批次 %zu: 大小 = %zu 元素, 内存占用 = %zu 字节\n", 
//               i, batch_alphas_[i].size(), batch_alphas_[i].size() * sizeof(ERROR_TYPE));
        
//         // 可选：打印部分值示例
//         if (!batch_alphas_[i].empty()) {
//             printf("  示例值: [0]=%.6g", (double)batch_alphas_[i][0]);
//             if (batch_alphas_[i].size() > 1)
//                 printf(", [%zu]=%.6g", batch_alphas_[i].size()-1, (double)batch_alphas_[i].back());
//             printf("\n");
//         }
//     }
//     return SUCCESS;
// }



// 添加清理批处理数据的方法
void upcite::ConformalPredictor::clear_batch_data() {
    // 只清空数据，不强制释放内存
    for (auto& batch : batch_alphas_) {
        batch.clear();
    }
    batch_alphas_.clear();
}

// 修改load_batch_alphas方法的实现，正确处理不同类型大小
// RESPONSE upcite::ConformalPredictor::load_batch_alphas(const std::string& filepath) {
//     // 完全重置数据，确保没有旧数据
//     batch_alphas_.clear();
    
//     // 读取整个文件到内存
//     std::ifstream file(filepath, std::ios::binary | std::ios::ate);
//     if (!file) {
//         printf("无法打开文件: %s\n", filepath.c_str());
//         return FAILURE;
//     }
//     // 获取文件大小
//     std::streamsize size = file.tellg();
//     if (size < sizeof(int64_t)) {
//         printf("文件太小，不可能包含有效数据: %s\n", filepath.c_str());
//         return FAILURE;
//     }
    
//     printf("加载alpha文件: %s, 文件大小: %ld 字节\n", filepath.c_str(), (long)size);
    
//     file.seekg(0, std::ios::beg);

//     // 一次性读取整个文件到内存缓冲区
//     std::vector<char> buffer(size);
//     if (!file.read(buffer.data(), size)) {
//         printf("读取文件失败: %s\n", filepath.c_str());
//         return FAILURE;
//     }

//     // printf("文件头部16字节十六进制表示: ");
//     // for (size_t i = 0; i < std::min(size_t(16), size); i++) {
//     //     printf("%02x ", static_cast<unsigned char>(buffer[i]));
//     // }
//     // printf("\n");
    
//     // 从内存缓冲区解析数据
//     size_t pos = 0; // 当前解析位置
    
//     // 1. 读取批次数量 - 使用8字节int64_t匹配保存格式
//     if (pos + sizeof(int64_t) > size) {
//         printf("文件格式错误: 无法读取批次数量\n");
//         return FAILURE;
//     }
    
//     int64_t num_batches_64bit = *reinterpret_cast<int64_t*>(buffer.data() + pos);
//     pos += sizeof(int64_t);
    
//     // 转换为当前系统的ID_TYPE
//     ID_TYPE num_batches = static_cast<ID_TYPE>(num_batches_64bit);
    
//     // 检查批次数量的合理性
//     if (num_batches <= 0 || num_batches > 100) {
//         printf("无效的批次数量: %ld in file %s\n", (long)num_batches, filepath.c_str());
//         return FAILURE;
//     }
    
//     printf("文件包含 %ld 个批次\n", (long)num_batches);
    
//     // 预分配空间
//     try {
//         batch_alphas_.resize(num_batches);
//     } catch (const std::exception& e) {
//         printf("分配内存失败: %s\n", e.what());
//         return FAILURE;
//     }
    
//     // 2. 解析每个批次的数据
//     for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
//         // 读取批次大小 - 使用8字节int64_t匹配保存格式
//         if (pos + sizeof(int64_t) > size) {
//             printf("文件格式错误: 批次%ld的大小信息缺失\n", (long)batch_i);
//             return FAILURE;
//         }
        
//         int64_t batch_size_64bit = *reinterpret_cast<int64_t*>(buffer.data() + pos);
//         pos += sizeof(int64_t);
        
//         // 转换为当前系统的ID_TYPE
//         ID_TYPE batch_size = static_cast<ID_TYPE>(batch_size_64bit);
        
//         // 检查批次大小的合理性
//         if (batch_size <= 0 || batch_size > 10000) {
//             printf("无效的批次大小: %ld for batch %ld in file %s\n", 
//                   (long)batch_size, (long)batch_i, filepath.c_str());
//             return FAILURE;
//         }
        
//         // 检查是否有足够的数据 - 使用8字节double匹配保存格式
//         size_t required_bytes = batch_size * sizeof(double);
//         if (pos + required_bytes > size) {
//             printf("文件格式错误: 批次%ld的数据不完整, 需要%ld字节但只有%ld字节\n", 
//                   (long)batch_i, (long)required_bytes, (long)(size - pos));
//             return FAILURE;
//         }
        
//         printf("加载批次 %ld, 大小为 %ld, 需要 %ld 字节\n", 
//                (long)batch_i, (long)batch_size, (long)required_bytes);
        
//         try {
//             // 使用临时double缓冲区读取数据
//             std::vector<double> temp_buffer(batch_size);
//             memcpy(temp_buffer.data(), buffer.data() + pos, required_bytes);

//             printf("批次%ld数据头部16字节: ", (long)batch_i);
//             size_t batch_start = pos - required_bytes;
//             for (size_t i = 0; i < std::min(size_t(16), required_bytes); i++) {
//                 printf("%02x ", static_cast<unsigned char>(buffer[batch_start + i]));
//             }
//             printf("\n");

//             pos += required_bytes;
            
//             // 转换为目标VALUE_TYPE类型并存储
//             batch_alphas_[batch_i].resize(batch_size);
            
//             // 逐个转换并验证数据
//             bool has_invalid_data = false;
//             for (size_t j = 0; j < batch_size; j++) {
//                 double val = temp_buffer[j];
                
//                 // 检查数据有效性
//                 if (std::isnan(val) || std::isinf(val) || val > 1e10) {
//                     // 无效值用0替代
//                     printf("批次%ld索引%ld的值%.6g无效, 替换为0\n", (long)batch_i, (long)j, val);
//                     has_invalid_data = true;
//                 } else {
//                     batch_alphas_[batch_i][j] = static_cast<VALUE_TYPE>(val);
//                 }
//             }
            
//             if (has_invalid_data) {
//                 printf("警告：批次 %ld 包含无效数据\n", (long)batch_i);
//             }
            
//         } catch (const std::exception& e) {
//             printf("处理批次 %ld 时出错: %s\n", (long)batch_i, e.what());
//             return FAILURE;
//         }
//     }
    
//     // 检查是否解析了整个文件
//     if (pos < size) {
//         printf("警告：文件包含额外数据: 已解析 %ld 字节，文件共 %ld 字节\n", (long)pos, (long)size);
//     }
    
//     // 输出加载结果摘要
//     for (size_t i = 0; i < batch_alphas_.size(); i++) {
//         printf("批次%zu最终大小: %zu\n", i, batch_alphas_[i].size());
//         if (!batch_alphas_[i].empty()) {
//             printf("  值范围: %.6f 到 %.6f\n", 
//                   batch_alphas_[i][0], 
//                   batch_alphas_[i][batch_alphas_[i].size()-1]);
//         }
//     }
    
//     // 完成解析，标记为已拟合
//     is_fitted_ = true;
    
//     printf("成功从 %s 加载批次alpha值\n", filepath.c_str());
//     return SUCCESS;
// }


// 保存函数修改
RESPONSE upcite::ConformalPredictor::save_batch_alphas(const std::string& filepath) const {
    // 检查是否有批次数据
    if (batch_alphas_.empty()) {
        spdlog::error("No batch alpha data to save");
        printf("没有批次alpha数据可保存\n");
        return FAILURE;
    }
    
    // 使用文本模式打开文件
    std::ofstream alphas_fout(filepath);  // 移除 std::ios::binary
    if (!alphas_fout.good()) {
        spdlog::error("Failed to open file for writing: {}", filepath);
        printf("无法打开文件进行写入: %s\n", filepath.c_str());
        return FAILURE;
    }
    
    // 写入类型信息和批次数量
    alphas_fout << "TYPE_SIZE=" << sizeof(ERROR_TYPE) << std::endl;
    alphas_fout << "NUM_BATCHES=" << batch_alphas_.size() << std::endl;
    
    // 对每个批次
    for (size_t batch_i = 0; batch_i < batch_alphas_.size(); ++batch_i) {
        // 写入批次大小
        alphas_fout << "BATCH_" << batch_i << "_SIZE=" << batch_alphas_[batch_i].size() << std::endl;
        
        // 逐个写入值，使用科学计数法确保精度
        for (size_t j = 0; j < batch_alphas_[batch_i].size(); ++j) {
            ERROR_TYPE val = batch_alphas_[batch_i][j];
            
            // 检查值的有效性
            if (std::isnan(val) || std::isinf(val) || val < 0 || val > 1e10) {
                printf("警告：保存批次%zu索引%zu的值%.6g无效\n", batch_i, j, (double)val);
            }
            
            // 使用科学计数法，保证15位有效数字
            alphas_fout << std::scientific << std::setprecision(15) << val;
            
            // 每行结束添加换行符
            alphas_fout << std::endl;
        }
    }
    
    // printf("成功保存批次alpha值到: %s (文本格式)\n", filepath.c_str());
    return SUCCESS;
}

// 加载函数修改
RESPONSE upcite::ConformalPredictor::load_batch_alphas(const std::string& filepath) {
    // 完全重置数据，确保没有旧数据
    batch_alphas_.clear();
    
    // 使用文本模式打开文件
    std::ifstream file(filepath);
    if (!file) {
        printf("无法打开文件: %s\n", filepath.c_str());
        return FAILURE;
    }
    
    printf("加载alpha文件: %s (文本格式)\n", filepath.c_str());
    
    // 读取类型大小和批次数量
    std::string line;
    size_t saved_type_size = 0;
    size_t num_batches = 0;
    
    // 读取类型大小行
    if (!std::getline(file, line) || sscanf(line.c_str(), "TYPE_SIZE=%zu", &saved_type_size) != 1) {
        printf("文件格式错误: 无法读取类型大小\n");
        return FAILURE;
    }
    
    // 读取批次数量行
    if (!std::getline(file, line) || sscanf(line.c_str(), "NUM_BATCHES=%zu", &num_batches) != 1) {
        printf("文件格式错误: 无法读取批次数量\n");
        return FAILURE;
    }
    
    printf("文件中保存的ERROR_TYPE大小: %zu 字节, 批次数量: %zu\n", 
           saved_type_size, num_batches);
           
    // 检查批次数量的合理性
    if (num_batches <= 0 || num_batches > 100) {
        printf("无效的批次数量: %zu\n", num_batches);
        return FAILURE;
    }
    
    // 预分配空间
    try {
        batch_alphas_.resize(num_batches);
    } catch (const std::exception& e) {
        printf("分配内存失败: %s\n", e.what());
        return FAILURE;
    }
    
    // 解析每个批次的数据
    for (size_t batch_i = 0; batch_i < num_batches; ++batch_i) {
        // 读取批次大小行
        size_t batch_size = 0;
        if (!std::getline(file, line) || 
            sscanf(line.c_str(), "BATCH_%zu_SIZE=%zu", &batch_i, &batch_size) != 2) {
            printf("文件格式错误: 无法读取批次%zu大小\n", batch_i);
            return FAILURE;
        }
        
        // 检查批次大小的合理性
        if (batch_size <= 0 || batch_size > 10000) {
            printf("无效的批次大小: %zu (批次 %zu)\n", batch_size, batch_i);
            return FAILURE;
        }
        
        printf("加载批次 %zu, 大小为 %zu\n", batch_i, batch_size);
        
        // 预分配批次数据空间
        batch_alphas_[batch_i].resize(batch_size);
        
        // 逐行读取值
        int invalid_count = 0;
        for (size_t j = 0; j < batch_size; ++j) {
            if (!std::getline(file, line)) {
                printf("文件格式错误: 批次%zu数据不完整\n", batch_i);
                return FAILURE;
            }
            
            // 将文本转换为double
            double val = std::stod(line);
            
            // 检查数据有效性
            if (std::isnan(val) || std::isinf(val) || val < 0 || val > 1e10) {
                // 无效值用0替代
                if (invalid_count < 10) {
                    printf("批次%zu索引%ld的值%.6g无效, 替换为0\n", batch_i, (long)j, val);
                } else if (invalid_count == 10) {
                    printf("更多无效值...\n");
                }
                batch_alphas_[batch_i][j] = 0.0;
                invalid_count++;
            } else {
                // 有效值正常保存
                batch_alphas_[batch_i][j] = static_cast<ERROR_TYPE>(val);
            }
        }
        
        if (invalid_count > 0) {
            printf("警告：批次 %zu 包含 %d 个无效数据\n", batch_i, invalid_count);
        }
    }
    
    // 完成解析，标记为已拟合
    is_fitted_ = true;
    
    printf("成功从 %s 加载批次alpha值\n", filepath.c_str());
    return SUCCESS;
}



double upcite::ConformalRegressor::predict_error_value(double recall, double coverage) const {
    if (!is_fitted_) {
        spdlog::warn("回归模型未训练");
        printf("回归模型未训练\n");
        return -1.0;
    }
    // 打印回归系数
    // printf("回归系数: [");
    // for (size_t i = 0; i < regression_coeffs_.size(); ++i) {
    //     printf("%.2f", regression_coeffs_[i]);
    //     if (i < regression_coeffs_.size() - 1) {
    //         printf(", ");
    //     }
    // }
    // printf("]\n");
    
    // // 添加spdlog日志
    std::string coeff_str = "[";
    for (size_t i = 0; i < regression_coeffs_.size(); ++i) {
        coeff_str += fmt::format("{:.2f}", regression_coeffs_[i]);
        if (i < regression_coeffs_.size() - 1) {
            coeff_str += ", ";
        }
    }
    coeff_str += "]";
    spdlog::info("回归系数: {}", coeff_str);
    
    // 确保回归系数数量正确
    if (regression_coeffs_.size() < 6) {
        spdlog::error("回归系数不完整，需要6个系数但只有{}个", regression_coeffs_.size());
        printf("回归系数不完整，需要6个系数但只有%zu个\n", regression_coeffs_.size());
        return -1.0;
    }
    
    double predicted_error = regression_coeffs_[0] +                      // 常数项
                            regression_coeffs_[1] * recall +             // 一次项
                            regression_coeffs_[2] * coverage +           // 一次项
                            regression_coeffs_[3] * recall * coverage +  // 交互项
                            regression_coeffs_[4] * recall * recall +    // 二次项
                            regression_coeffs_[5] * coverage * coverage; // 二次项
    
    // 确保预测误差为正值
    // printf("预测误差值: %.4f, recall=%.4f, coverage=%.4f\n", std::max(0.0, predicted_error), recall, coverage);
    spdlog::info("预测误差值: {:.4f}, recall={:.4f}, coverage={:.4f}", std::max(0.0, predicted_error), recall, coverage);
    return std::max(0.0, predicted_error);
}

// 新增函数：使用特定批次的实际误差值
RESPONSE upcite::ConformalRegressor::train_regression_model_for_recall_coverage_actual_error(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages,
    const std::vector<ID_TYPE>& error_indices,
    ID_TYPE batch_id,
    ID_TYPE filter_id) {
    
    if (recalls.size() != coverages.size() || recalls.size() != error_indices.size() || recalls.empty()) {
        printf("错误：输入数据数量不匹配或为空\n");
        return FAILURE;
    }
    
    // 获取当前filter下的cp类的batch_alphas，用于将位置转换为实际误差值
    const auto& batch_alphas = this->get_batch_alphas();
    if (batch_alphas.empty()) {
        printf("错误：batch_alphas为空，无法获取误差值\n");
        return FAILURE;
    }
    
    if (batch_id >= batch_alphas.size()) {
        printf("错误：指定的batch_id %ld 超出有效范围 [0, %zu)\n", (long)batch_id, batch_alphas.size());
        return FAILURE;
    }
    
    printf("使用批次 %ld 的实际误差值训练模型，批次总数: %zu\n", (long)batch_id, batch_alphas.size());
    spdlog::info("使用批次 {} 的实际误差值训练模型，批次总数: {}", batch_id, batch_alphas.size());
    
    // 准备训练数据，将error_i转换为指定批次的实际误差值
    std::vector<double> actual_errors;
    actual_errors.reserve(error_indices.size());
    
    for (ID_TYPE error_i : error_indices) {
        // 从指定批次中获取实际误差值
        if (error_i < batch_alphas[batch_id].size()) {
            double error_value = batch_alphas[batch_id][error_i];
            actual_errors.push_back(error_value);
        } else {
            printf("警告：误差位置 %ld 超出批次 %ld 的误差值范围 [0, %zu)\n", 
                   (long)error_i, (long)batch_id, batch_alphas[batch_id].size());
            return FAILURE;
        }
    }
    
    // 打印actual_errors的大小
    printf("过滤器 %ld 的批次 %ld 的actual_errors大小: %zu\n", 
           (long)filter_id, (long)batch_id, actual_errors.size());
    spdlog::info("过滤器 {} 的批次 {} 的actual_errors大小: {}", 
                filter_id, batch_id, actual_errors.size());
    
    // 样本数和特征数
    size_t n = recalls.size();  // 样本数
    size_t p = 6;  // 特征数：常数项、recall、coverage、recall*coverage、recall^2、coverage^2
    
    // 使用GSL库进行多元回归
    gsl_matrix *X = gsl_matrix_alloc(n, p);
    gsl_vector *y = gsl_vector_alloc(n);
    gsl_vector *c = gsl_vector_alloc(p);  // 回归系数
    gsl_matrix *cov = gsl_matrix_alloc(p, p);  // 协方差矩阵
    double chisq;  // 拟合优度
    
    // 填充设计矩阵X和目标向量y
    for (size_t i = 0; i < n; i++) {
        // 设计矩阵：[1, recall, coverage, recall*coverage, recall^2, coverage^2]
        gsl_matrix_set(X, i, 0, 1.0);  // 常数项
        gsl_matrix_set(X, i, 1, recalls[i]);
        gsl_matrix_set(X, i, 2, coverages[i]);
        gsl_matrix_set(X, i, 3, recalls[i] * coverages[i]);  // 交互项
        gsl_matrix_set(X, i, 4, recalls[i] * recalls[i]);
        gsl_matrix_set(X, i, 5, coverages[i] * coverages[i]);
        
        // 目标向量 - 使用实际误差值
        gsl_vector_set(y, i, actual_errors[i]);
    }
    
    // 执行回归计算
    gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(n, p);
    int ret = gsl_multifit_linear(X, y, c, cov, &chisq, work);
    
    if (ret != GSL_SUCCESS) {
        printf("错误：GSL回归计算失败，错误码：%d\n", ret);
        // 清理资源
        gsl_multifit_linear_free(work);
        gsl_matrix_free(X);
        gsl_vector_free(y);
        gsl_vector_free(c);
        gsl_matrix_free(cov);
        return FAILURE;
    }
    
    // 提取回归系数
    regression_coeffs_.resize(p);
    for (size_t i = 0; i < p; i++) {
        regression_coeffs_[i] = gsl_vector_get(c, i);
    }
    
    printf("批次 %ld 的拟合优度 (chi^2): %.1f\n", (long)batch_id, chisq);
    spdlog::info("批次 {} 的拟合优度 (chi^2): {}", batch_id, chisq);
    
    // 清理GSL资源
    gsl_multifit_linear_free(work);
    gsl_matrix_free(X);
    gsl_vector_free(y);
    gsl_vector_free(c);
    gsl_matrix_free(cov);
    
    // 更新模型状态
    is_fitted_ = true;
    return SUCCESS;
}
}