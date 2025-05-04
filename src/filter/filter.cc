//
// Created by Qitong Wang on 2022/10/11.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "filter.h"
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <csignal>
#include <fstream>

#include <boost/filesystem.hpp>
#include <torch/data/example.h>
#include <torch/data/datasets/base.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "spdlog/spdlog.h"

#include "str.h"
#include "vec.h"
#include "comp.h"
#include "interval.h"
#include "dataset.h"
#include "scheduler.h"

#include <gsl/gsl_multifit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Filter::Filter(dstree::Config &config,
                       ID_TYPE id,
                       std::reference_wrapper<torch::Tensor> shared_train_queries) :
    config_(config),
    id_(id),
    is_active_(false),
    global_queries_(shared_train_queries),
    is_trained_(false),
    is_distances_preprocessed_(false),
    is_distances_logged(false),
    global_data_size_(0),
    local_data_size_(0),
    model_setting_ref_(MODEL_SETTING_PLACEHOLDER_REF) {
  if (config.filter_train_is_gpu_) {
    // TODO support multiple devices
    device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                              static_cast<c10::DeviceIndex>(config.filter_device_id_));
  } else {
    device_ = std::make_unique<torch::Device>(torch::kCPU);
  }

  // delayed until allocated (either in trial or activation)
  model_ = nullptr;

  if (!config.to_load_index_ && config.filter_train_nexample_ > 0) {
    global_bsf_distances_.reserve(config.filter_train_nexample_);
    global_lnn_distances_.reserve(config.filter_train_nexample_);

    // QYL
    // global_knn_distances_.reserve(config.filter_train_nexample_);
    lb_distances_.reserve(config.filter_train_nexample_);
  }

  if (config.filter_is_conformal_) {
    conformal_predictor_ = std::make_unique<upcite::ConformalRegressor>(config.filter_conformal_core_type_,
                                                                        config.filter_conformal_confidence_);
  } else {
    conformal_predictor_ = nullptr;
  }
}


// ---------------------------conformal_predictor 的过程---------------------
RESPONSE dstree::Filter::fit_conformal_predictor(bool is_trial, bool collect_runtime_stat) {
  // ===================== 阶段1：参数打印与初始化 =====================
  
  // printf("[DEBUG] =========== Entering fit_conformal_predictor ===========\n");
  // printf("[DEBUG] 当前模式: is_trial=%d, collect_runtime_stat=%d\n", 
        //  static_cast<int>(is_trial), 
        //  static_cast<int>(collect_runtime_stat));
  ID_TYPE num_conformal_examples;
  // is_trial=0, collect_runtime_stat=0
  // ===================== 阶段2：确定符合预测样本数量 =====================
  // printf("[DEBUG] --- 进入样本数量计算阶段 ---\n");
  if (!collect_runtime_stat) {
    //进入这个分支
    printf("[DEBUG] 常规模式：使用全局数据划分验证集\n");
    ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
    ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
    
    num_conformal_examples = num_global_valid_examples;

    // printf("[DEBUG] 全局训练样本量=%ld, 验证样本量=%ld\n", 
      // num_global_train_examples, num_global_valid_examples);
  } else {
    printf("[DEBUG] 运行时统计模式：动态生成样本\n");

    if (config_.get().filter_train_num_global_example_ > 0 && config_.get().filter_train_num_local_example_ >= 0) {
      printf("[DEBUG] 使用 filter_train_num_global_example_ 配置\n");

      ID_TYPE num_global_train_examples =
          config_.get().filter_train_num_global_example_ * config_.get().filter_train_val_split_;
      ID_TYPE num_global_valid_examples = config_.get().filter_train_num_global_example_ - num_global_train_examples;

      num_conformal_examples = num_global_valid_examples;
    } else if (config_.get().filter_train_nexample_ > 0) {
      ID_TYPE num_global_train_examples = config_.get().filter_train_nexample_ * config_.get().filter_train_val_split_;
      ID_TYPE num_global_valid_examples = config_.get().filter_train_nexample_ - num_global_train_examples;

      num_conformal_examples = num_global_valid_examples;
    } else {
      num_conformal_examples = 8192;
    }
  }

  auto residuals = upcite::make_reserved<ERROR_TYPE>(num_conformal_examples + 2);

  if (collect_runtime_stat) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);

    // include two sentry diffs
    for (ID_TYPE i = 0; i < num_conformal_examples + 2; ++i) {
      residuals.push_back(dist(e2));
    }
  } else {

    // printf("[DEBUG] 基于真实数据计算残差\n");
    //训练集的样本数量
    ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
    //验证集的样本数量
    ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
    VALUE_TYPE max_diff = constant::MIN_VALUE, mean_diff = 0, std_diff = 0;
    ID_TYPE num_diff = 0;
    // printf(" 不明白为啥，又算了一遍全局训练样本量=%ld, 验证样本量=%ld\n", 
    //   num_global_train_examples, num_global_valid_examples);


    // ===================== 添加调试打印 =====================
    // printf("\n[DEBUG] 数组大小检查:\n");
    // printf("global_pred_distances_.size() = %zu\n", global_pred_distances_.size());
    // printf("global_lnn_distances_.size() = %zu\n", global_lnn_distances_.size());
    // printf("num_global_train_examples = %d\n", num_global_train_examples);
    // printf("num_conformal_examples = %d\n", num_conformal_examples);
    // printf("global_data_size_ = %d\n", global_data_size_);

    // 边界检查断言
    assert(num_global_train_examples + num_conformal_examples <= global_pred_distances_.size());
    assert(num_global_train_examples + num_conformal_examples <= global_lnn_distances_.size());

    // 遍历验证集，求出真实最近邻距离和预测距离的差值
    // printf("[DEBUG] 遍历验证集 (共%ld样本)\n", num_global_valid_examples);

    // ===================== 原有残差计算逻辑 =====================
    for (ID_TYPE conformal_i = 0; conformal_i < num_conformal_examples; ++conformal_i) {
      // TODO torch::Tensor to ptr is not stable
      ID_TYPE idx = num_global_train_examples + conformal_i;
      // printf("[DEBUG] 处理样本 %d (全局索引 %d): ", conformal_i, idx);
      // printf("pred=%.3f, lnn=%.3f\n", global_pred_distances_[idx], global_lnn_distances_[idx]); 
      
      if (global_pred_distances_[idx] > constant::MIN_VALUE && global_pred_distances_[idx] < constant::MAX_VALUE &&
          !upcite::equals_zero(global_pred_distances_[idx])) {
        // TODO not necessary for global symmetrical confidence intervals
        VALUE_TYPE diff = abs(global_pred_distances_[idx] - global_lnn_distances_[idx]);
        if (diff > max_diff) {
          max_diff = diff;
        }
        mean_diff += diff;
        num_diff += 1;
        residuals.emplace_back(diff);
      }
    }

    // ===================== 残差统计结果 =====================
    // printf("\n[DEBUG] 残差统计结果:\n");
    printf("有效残差数量: %d (预期: %d)\n", num_diff, num_conformal_examples);   
    // printf("[DEBUG] 最大残差=%.3f\n", max_diff);

    if (num_diff < num_conformal_examples) {
      spdlog::error("adjuster {:d} {:s} collected {:d} pred diff; expected {:d}",
                    id_, model_setting_ref_.get().model_setting_str, num_diff, num_conformal_examples);
    }
    //计算误差的均值
    mean_diff /= num_diff;
    //计算误差的方差
    for (ID_TYPE diff_i = 0; diff_i < num_diff; ++diff_i) {
      std_diff += (residuals[diff_i] - mean_diff) * (residuals[diff_i] - mean_diff);
    }
    //误差标准差
    std_diff = sqrt(std_diff / num_diff);
    VALUE_TYPE max_normal_value = mean_diff + 3 * std_diff + constant::EPSILON_GAP;
    max_diff += constant::EPSILON_GAP;
    if (max_normal_value < max_diff) {
      max_normal_value = max_diff;
    }
    // printf("[DEBUG] 最终残差边界: max_normal_value=%.3f\n", max_normal_value);

    // //向residuals容器中添加一个值为0的哨兵值（sentry value）, 0表示残差的最小可能值，max_normal_value表示最小可能值   add the first of two sentries: 0
    residuals.push_back(0); 
    // add the second of two sentries: max range upper boundary previously using the max pred value
    residuals.push_back(max_normal_value);
    printf("最终residuals大小: %zu (应等于num_diff+2=%d)\n", 
       residuals.size(), num_diff + 2);
  }
  



  // ===================== 阶段4：核心逻辑分支处理 =====================
  // printf("------CP: is_trial && !collect_runtime_stat -------\n");
  if (is_trial && !collect_runtime_stat) {  //现在不用这个
    printf("[DEBUG] 进入试验模式分支\n");
    //遍历residual容器中的每个元素，如果元素小于0，则取反
    for (auto &residual : residuals) { residual = residual < 0 ? -residual : residual; }
    //对residuals容器中的残差进行升序排序。
    std::sort(residuals.begin(), residuals.end());
    printf("[DEBUG] 排序后残差范围: [%.3f ~ %.3f]\n", residuals.front(), residuals.back());

    //根据置信水平计算残差分位数的索引位置
    auto residual_i = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(residuals.size())
        * config_.get().filter_trial_confidence_level_);
    //设置置信水平，将分位数对应的残差值设置为CP的置信区间半径，true: 表示这是试验模式; false: 表示不进行修正。
    conformal_predictor_->set_alpha(residuals[residual_i], true, false);

#ifdef DEBUG
//#ifndef DEBUGGED
//id_ 是当前过滤器（Filter）的唯一标识符，
//model_setting_ref_.get().model_setting_str 是当前模型的设置字符串
//filter_trial_confidence_level 是试验模式下的置信度
    spdlog::debug("trial {:d} {:s} error (half-)interval = {:.3f} @ {:.2f}",
                  id_, model_setting_ref_.get().model_setting_str,
                  get_abs_error_interval(), //通过调用 conformal_predictor_->get_alpha()，获取当前CP的置信区间半径
                  config_.get().filter_trial_confidence_level_);
//#endif
#endif
  } else if (!is_trial && collect_runtime_stat) {
    //这里fit还是计算残差分位数
    printf("[DEBUG] !is_trial && collect_runtime_stat \n");
    conformal_predictor_->fit(residuals);
    // printf("[DEBUG] 完成基础符合预测器拟合\n");
    if (config_.get().filter_conformal_is_smoothen_) {
      // printf("[DEBUG] 启用平滑处理\n");

      auto recalls = upcite::make_reserved<ERROR_TYPE>(num_conformal_examples + 2);

      std::random_device rd;
      std::mt19937 e2(rd());
      std::uniform_real_distribution<> dist(0, 1);
      for (ID_TYPE i = 0; i < num_conformal_examples + 2; ++i) {
        recalls.push_back(dist(e2));
      }
      std::sort(recalls.begin(), recalls.end()); //non-decreasing
      //调用fit_spline函数，拟合f:recall_i -> alpha_i
      fit_filter_conformal_spline(recalls);
    }

  } else if (!is_trial && !collect_runtime_stat) { // is_trail: 背包算法等    collect_runtime_stat：收集运行时统计信息，
    //  std::signal(SIGSEGV, sigfaultHandler);   
    // 主要用的是这个
    printf("[DEBUG] !is_trial && !collect_runtime_stat 进入常规生产模式分支\n");
    RESPONSE return_code = FAILURE;
    // 计算残差alphas
    //！！！！！！！！！！！！！！！！！！！
    return_code = conformal_predictor_->fit(residuals);

    if (return_code == FAILURE) {
      printf("[ERROR] 符合预测器拟合失败! residuals.size()=%ld\n", residuals.size());

      spdlog::error("trial {:d} {:s} failed to get made conformal (with {:d}/{:d} residuals); disable it",
                    id_, model_setting_ref_.get().model_setting_str, residuals.size(), num_conformal_examples);
      is_trained_ = false;
      is_active_ = false;
    }
  } else {
    printf("[ERROR] 非法参数组合! is_trial=%d, collect_runtime_stat=%d\n", 
      static_cast<int>(is_trial), static_cast<int>(collect_runtime_stat));
    spdlog::error("trial {:d} {:s} both trial and collect modes were triggered",
                  id_, model_setting_ref_.get().model_setting_str);
    return FAILURE;
  }
  // printf("[DEBUG] =========== 函数执行完成 ===========\n");
  return SUCCESS;
}




// 修改fit_batch_conformal_predictor函数
RESPONSE dstree::Filter::fit_batch_conformal_predictor(
    bool is_trial,
    ID_TYPE num_calib_batches,
    const std::vector<torch::Tensor>& calib_data_batches,
    const std::vector<torch::Tensor>& calib_target_batches) {
  
  // 创建多个校准集的残差容器
  std::vector<std::vector<ERROR_TYPE>> batch_residuals(num_calib_batches);
  // 对每个校准批次计算残差
  for (ID_TYPE batch_idx = 0; batch_idx < num_calib_batches; ++batch_idx) {
    torch::Tensor batch_data = calib_data_batches[batch_idx];
    torch::Tensor batch_targets = calib_target_batches[batch_idx];
    ID_TYPE batch_size = batch_data.size(0);
    
    // 使用模型预测当前批次
    c10::InferenceMode guard;
    model_->eval();
    torch::Tensor batch_predictions = model_->forward(batch_data).detach().cpu();
    
    VALUE_TYPE max_diff = constant::MIN_VALUE, mean_diff = 0, std_diff = 0;
    ID_TYPE num_diff = 0;
    
    // 计算此批次的残差
    for (ID_TYPE i = 0; i < batch_size; ++i) {
      VALUE_TYPE pred = batch_predictions[i].item<VALUE_TYPE>();
      VALUE_TYPE target = batch_targets[i].item<VALUE_TYPE>();
      
      if (pred > constant::MIN_VALUE && pred < constant::MAX_VALUE && 
          !upcite::equals_zero(pred)) {
        VALUE_TYPE diff = std::abs(pred - target);  // 取绝对值
        batch_residuals[batch_idx].push_back(diff);
        if (diff > max_diff) max_diff = diff;
        mean_diff += diff;
        num_diff += 1;
      }
    }
    
    // 处理统计和哨兵值逻辑
    if (num_diff > 0) {
      // 计算统计值
      mean_diff /= num_diff;
      for (const auto& diff : batch_residuals[batch_idx]) {
        std_diff += (diff - mean_diff) * (diff - mean_diff);
      }
      std_diff = sqrt(std_diff / num_diff);
      
      // 添加哨兵值
      VALUE_TYPE max_normal_value = mean_diff + 3 * std_diff + constant::EPSILON_GAP;
      max_diff += constant::EPSILON_GAP;
      if (max_normal_value < max_diff) max_normal_value = max_diff;
      
      batch_residuals[batch_idx].push_back(0); // 最小值哨兵
      batch_residuals[batch_idx].push_back(max_normal_value); // 最大值哨兵
    } else {
      printf("[ERROR] 校准批次 %d 无有效残差!\n", batch_idx+1);
      batch_residuals[batch_idx].push_back(0);
      batch_residuals[batch_idx].push_back(0.1);
    }
  }
  // 打印batch_residuals的大小信息
  // printf("\nbatch_residuals大小统计信息:\n");
  // printf("批次总数: %zu\n", batch_residuals.size());
  // for (ID_TYPE batch_idx = 0; batch_idx < batch_residuals.size(); ++batch_idx) {
  //   printf("批次 %d: 包含 %zu 个残差值\n", batch_idx + 1, batch_residuals[batch_idx].size());
  // }
  
  // 使用计算好的批次残差拟合保形预测器
  RESPONSE return_code = conformal_predictor_->fit_batch(batch_residuals);
  
  if (return_code == FAILURE) {
    printf("[ERROR] 符合预测器拟合失败!\n");
    spdlog::error("filter {:d} {:s} failed in batch conformal fitting; disabling",
                 id_, model_setting_ref_.get().model_setting_str);
    is_trained_ = false;
    is_active_ = false;
    return FAILURE;
  }
  
  return SUCCESS;
}








// 这只是针对当前的filter进行训练，
RESPONSE dstree::Filter::train(bool is_trial) {

//功能：训练一个用于距离预测的机器学习模型（如CNN或线性模型），并结合保形预测（Conformal Prediction）校准预测结果。
/*
核心流程：
预处理：处理距离数据（如平方根转换）。
数据划分：将数据分为训练集、验证集。
模型训练：通过反向传播优化模型参数。
模型选择：根据验证损失保存最佳模型。
预测与校准：生成预测结果并调用保形预测校准。
参数：is_trial 表示是否为试验模式（影响后续校准逻辑）。
*/

  // ========================= 1. 前置检查 ==============================
  // 检查是否已训练或需要加载预训练模型
  if (is_trained_ || config_.get().to_load_filters_) {
    return FAILURE;
  }
  // 检查过滤器激活状态与模式合法性
  if (!is_active_ && !is_trial) {
    spdlog::error("filter {:d} neither is_active nor is_trial; exit", id_);
    spdlog::shutdown();
    exit(FAILURE);
  }

   // =========================== 2. CUDA流初始化 ==========================
  // 初始化CUDA流，用于GPU并行计算
  ID_TYPE stream_id = -1;
  if (config_.get().filter_train_is_gpu_) {
        // 获取当前CUDA流的ID（GPU训练时使用）
    stream_id = at::cuda::getCurrentCUDAStream(config_.get().filter_device_id_).id(); // compiles with libtorch-gpu
  }

  // ============================= 3. 数据预处理 =============================
  // 若配置要求移除平方（filter_remove_square_）且未预处理过。 
  // printf("[DEBUG] 条件检查: filter_remove_square_ = %d, is_distances_preprocessed_ = %d\n", 
    // static_cast<int>(config_.get().filter_remove_square_),  // 布尔转 int
    // static_cast<int>(is_distances_preprocessed_);          // 布尔转 int
  if (config_.get().filter_remove_square_ && !is_distances_preprocessed_) {
    // 对全局最近邻距离和最佳搜索距离取平方根
    for (ID_TYPE i = 0; i < global_data_size_; ++i) {
      global_lnn_distances_[i] = sqrt(global_lnn_distances_[i]); //对全局最近邻距离取平方根
      global_bsf_distances_[i] = sqrt(global_bsf_distances_[i]); //对最佳搜索距离取平方根
    }
    // printf("global_lnn_distances_.size() = %zu\n", global_lnn_distances_.size());
    // printf("global_bsf_distances_.size() = %zu\n", global_bsf_distances_.size());
    // printf("global_data_size_ = %zu\n", global_data_size_);
    // 对下界距离取平方根（如果存在）
    if (!lb_distances_.empty()) {
      for (ID_TYPE i = 0; i < global_data_size_; ++i) {
        lb_distances_[i] = sqrt(lb_distances_[i]);
      }
    }
    // 对local最近邻距离取平方根（如果存在局部local数据）
    if (local_data_size_ > 0) {
      for (ID_TYPE i = 0; i < local_data_size_; ++i) {
        local_lnn_distances_[i] = sqrt(local_lnn_distances_[i]);
      }
    }
    is_distances_preprocessed_ = true; // 标记已预处理
  }


#ifdef DEBUG
//#ifndef DEBUGGED
  if (!is_distances_logged) {
    // 记录下界距离、最佳搜索距离等预处理后的数据
    if (!lb_distances_.empty()) {
      spdlog::debug("filter {:d} s{:d} lb{:s} = {:s}",
                    id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                    upcite::array2str(lb_distances_.data(), global_data_size_));
    }               //记录下界距离

    spdlog::debug("filter {:d} s{:d} bsf{:s} = {:s}",
                  id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                  upcite::array2str(global_bsf_distances_.data(), global_data_size_));
                  //记录全局最佳搜索距离

    spdlog::debug("filter {:d} s{:d} gnn{:s} = {:s}",
                  id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                  upcite::array2str(global_lnn_distances_.data(), global_data_size_));
                  //记录全局最近邻距离
    if (local_data_size_ > 0) {
      spdlog::debug("filter {:d} s{:d} lnn{:s} = {:s}",
                    id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                    upcite::array2str(local_lnn_distances_.data(), local_data_size_));
    }             //记录局部最近邻距离

    is_distances_logged = true;
  }
//#endif
#endif

  // ============================ 5. 数据划分 ====================================
  //划分训练集和验证机
  ID_TYPE num_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
  ID_TYPE num_valid_examples = global_data_size_ - num_train_examples;
 
  torch::Tensor train_data, valid_data;
  torch::Tensor train_targets, valid_targets;

  // printf("[DEBUG] global_data_size_ = %ld\n", static_cast<long>(global_data_size_));
  // printf("[DEBUG] local_data_size_ = %ld\n", static_cast<long>(local_data_size_));
  // -------------------5.1 存在局部数据local data时的处理----------------
  if (local_data_size_ > 0) {
    // -------------------5.1.1  获取训练集的全局和局部数据----------------
    assert(global_data_size_ == config_.get().filter_train_num_global_example_);
    
    assert(local_data_size_ == config_.get().filter_train_num_local_example_);
    //确定全局数据的训练样本数量：
    ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
    //确定局部数据的训练样本数量：
    ID_TYPE num_local_train_examples = local_data_size_ * config_.get().filter_train_val_split_;
    
    //从global_queries_中获取全局训练数据
    torch::Tensor global_train_data = global_queries_.get().index({torch::indexing::Slice(0, num_train_examples)}).clone();
    //从global_lnn_distances_中获取指定训练集数量的全局训练标签：1nn_distance
    torch::Tensor global_train_targets = torch::from_blob(global_lnn_distances_.data(),
                                                          num_global_train_examples,
                                                          torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    //从local_queries_中获取局部训练数据
    torch::Tensor local_train_data = torch::from_blob(local_queries_.data(),
                                                      {num_local_train_examples, config_.get().series_length_},
                                                      torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    //从local_lnn_distances_中获取指定训练集数量的局部训练标签：1nn_distance                                                  
    torch::Tensor local_train_targets = torch::from_blob(local_lnn_distances_.data(),
                                                         num_local_train_examples,
                                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    //合并loca和global数据，作为完整的训练数据(query)和训练标签(dij)
    train_data = torch::cat({global_train_data, local_train_data}, 0);
    train_targets = torch::cat({global_train_targets, local_train_targets}, 0);
    
    // --------------------------5.1.2  获取验证集的全局和局部数据----------------
    ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
    ID_TYPE num_local_valid_examples = local_data_size_ - num_local_train_examples;

    torch::Tensor global_valid_data = global_queries_.get().index(
        {torch::indexing::Slice(num_train_examples, global_data_size_)}).clone();
    torch::Tensor global_valid_targets = torch::from_blob(global_lnn_distances_.data() + num_global_train_examples,num_global_valid_examples,torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    torch::Tensor local_valid_data = torch::from_blob(
        local_queries_.data() + num_local_train_examples * config_.get().series_length_,
        {num_local_valid_examples, config_.get().series_length_},
        torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    torch::Tensor local_valid_targets = torch::from_blob(local_lnn_distances_.data() + num_local_train_examples,num_local_valid_examples, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    valid_data = torch::cat({global_valid_data, local_valid_data}, 0);
    valid_targets = torch::cat({global_valid_targets, local_valid_targets}, 0);

    num_train_examples = num_global_train_examples + num_local_train_examples;
    num_valid_examples = num_global_valid_examples + num_local_valid_examples;

    assert(train_data.size(0) == num_train_examples && train_targets.size(0) == num_train_examples);
    assert(valid_data.size(0) == num_valid_examples && valid_targets.size(0) == num_valid_examples);
  
  } else {

    //5.2 不存在局部数据local data时的处理,  仅全局数据时的处理
    assert(global_data_size_ == config_.get().filter_train_nexample_);
    //train_data是query，train_targets是全局1nn最近距离
    train_data = global_queries_.get().index({torch::indexing::Slice(0, num_train_examples)}).clone();
    train_targets = torch::from_blob(global_lnn_distances_.data(),
                                     num_train_examples,
                                     torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    valid_data = global_queries_.get().index({torch::indexing::Slice(
        num_train_examples, global_data_size_)}).clone();
    valid_targets = torch::from_blob(global_lnn_distances_.data() + num_train_examples,
                                     num_valid_examples,
                                     torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
  }


  // ============================= 6. 数据加载器初始化 ==============================
  auto train_dataset = upcite::SeriesDataset(train_data, train_targets);
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      train_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);

  // reuse validation examples as conformal examples
  //这里验证集的数据作为conformal的数据
  ID_TYPE num_conformal_examples = num_valid_examples;
  torch::Tensor conformal_data = valid_data;
  torch::Tensor conformal_targets = valid_targets; //dij, 

  // ==================================== 7. 模型初始化 ==============================
  // 根据配置创建模型（如CNN或线性模型）
  model_ = dstree::get_model(config_);
  model_->to(*device_);

  // ==================================== 8. 训练准备 ================================
  // 最佳模型状态跟踪，用于提前终止
  // for early termination
  std::unordered_map<std::string, torch::Tensor> best_model_state;
  VALUE_TYPE best_validation_loss = constant::MAX_VALUE;
  ID_TYPE best_validation_epoch = -1;
  // 优化器选择（CNN用Adam，其他用SGD）
  std::shared_ptr<torch::optim::Optimizer> optimizer = nullptr;
  if (model_->model_type_ == CNN) {
    optimizer = std::make_shared<torch::optim::Adam>(model_->parameters(), config_.get().filter_train_learning_rate_);
  } else {
    optimizer = std::make_shared<torch::optim::SGD>(model_->parameters(), config_.get().filter_train_learning_rate_);
  }
  ID_TYPE initial_cooldown_epochs = config_.get().filter_train_nepoch_ / 2;

  //  学习率调整策略（ReduceLROnPlateau）, 用于验证损失
  upcite::optim::ReduceLROnPlateau lr_scheduler = upcite::optim::ReduceLROnPlateau(
      *optimizer, initial_cooldown_epochs, optim::MIN, config_.get().filter_lr_adjust_factor_);
  // 损失函数（均方误差）
  torch::nn::MSELoss mse_loss(torch::nn::MSELossOptions().reduction(torch::kMean));

#ifdef DEBUG
  std::vector<float> train_losses, valid_losses, batch_train_losses;
  train_losses.reserve(config_.get().filter_train_nepoch_);
  batch_train_losses.reserve(num_train_examples / config_.get().filter_train_batchsize_ + 1);

  valid_losses.reserve(config_.get().filter_train_nepoch_);
#endif

  //================================= 9. 训练循环 ==============================
  torch::Tensor batch_data, batch_target;
  for (ID_TYPE epoch = 0; epoch < config_.get().filter_train_nepoch_; ++epoch) {
    model_->train(); // 切换至训练模式
   
    // 9.1 前向传播与反向传播
    for (auto &batch : *train_data_loader) {
      batch_data = batch.data;
      batch_target = batch.target;
      optimizer->zero_grad();
      torch::Tensor prediction = model_->forward(batch_data);
      torch::Tensor loss = mse_loss->forward(prediction, batch_target);
      loss.backward(); // 反向传播
      if (config_.get().filter_train_clip_grad_) {
        auto norm = torch::nn::utils::clip_grad_norm_(model_->parameters(),
                                                      config_.get().filter_train_clip_grad_max_norm_,
                                                      config_.get().filter_train_clip_grad_norm_type_);
      }
      optimizer->step();

#ifdef DEBUG
      batch_train_losses.push_back(loss.detach().item<float>());
#endif
    }

#ifdef DEBUG
    train_losses.push_back(std::accumulate(batch_train_losses.begin(), batch_train_losses.end(), 0.0) / static_cast<VALUE_TYPE>(batch_train_losses.size()));
    batch_train_losses.clear();
#endif

    // 9.2 验证阶段
    { // evaluate
      VALUE_TYPE valid_loss = 0;

      c10::InferenceMode guard;
      model_->eval();  // 切换至评估模式

      torch::Tensor prediction = model_->forward(valid_data);

      valid_loss = mse_loss->forward(prediction, valid_targets).detach().item<VALUE_TYPE>();

#ifdef DEBUG
      valid_losses.push_back(valid_loss);
#endif
      // 记录最佳模型状态
      if (epoch > initial_cooldown_epochs) {
        if (best_validation_loss > valid_loss) {
          best_validation_loss = valid_loss;
          best_validation_epoch = epoch;

          for (const auto &pair : model_->named_parameters()) {
            best_model_state[pair.key()] = pair.value().clone();
          }
        }
      }
      // 学习率调整与早停策略
      upcite::optim::LR_RETURN_CODE return_code = lr_scheduler.check_step(valid_loss);
      if (return_code == upcite::optim::EARLY_STOP) {
        epoch = config_.get().filter_train_nepoch_;
      }
    }
  }

#ifdef DEBUG
  spdlog::debug("filter {:d} s{:d} {:s} tloss = {:s}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                upcite::array2str(train_losses.data(), config_.get().filter_train_nepoch_));
  spdlog::debug("filter {:d} s{:d} {:s} vloss = {:s}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                upcite::array2str(valid_losses.data(), config_.get().filter_train_nepoch_));
#endif

  c10::InferenceMode guard;

// ============================ 10. 模型恢复与预测 =============================
// 恢复最佳模型状态
  if (best_validation_epoch > initial_cooldown_epochs) {
#ifdef DEBUG
    spdlog::debug("filter {:d} s{:d} {:s} restore from e{:d}, vloss {:.4f}",
                  id_, stream_id, model_setting_ref_.get().model_setting_str,
                  best_validation_epoch, best_validation_loss);
#endif

    for (auto &pair : best_model_state) {
      model_->named_parameters()[pair.first].detach_();
      model_->named_parameters()[pair.first].copy_(pair.second);
    }
  }
  //调用模型进行评估，对全局数据进行预测
  model_->eval();

  auto prediction = model_->forward(global_queries_).detach().cpu();
  assert(prediction.size(0) == global_data_size_);
  auto *predictions_array = prediction.detach().cpu().contiguous().data_ptr<VALUE_TYPE>();
  
  // !!!!!!!!!!!!!!!!!!1  存储模型预测结果到global_pred_distances_
  global_pred_distances_.insert(global_pred_distances_.end(), predictions_array, predictions_array + global_data_size_);
 
  // 打印 global_pred_distances_ 的 size
  // printf("Size of global_pred_distances_: %zu\n", global_pred_distances_.size());
  #ifdef DEBUG
  spdlog::info("filter {:d}{:s} s{:d} {:s} g_pred{:s} = {:s}",
               id_, is_trial ? " (trial)" : "",
               stream_id, model_setting_ref_.get().model_setting_str,
               config_.get().filter_remove_square_ ? "" : "_sq",
               upcite::array2str(predictions_array, global_data_size_));

#endif

  if (config_.get().filter_is_conformal_) {
    //---------------------------------Conformal Prediction---------------------------------
    //这里就是我们要的，对预测距离进行Conformal Prediction的校准
    // printf("---------------正式进入CP了,集中注意力---------------\n");
    fit_conformal_predictor(is_trial); 
    // fit_conformal_predictor_batch(is_trial);
    // printf("\n");
  }

 //  net->to(torch::Device(torch::kCPU));
  c10::cuda::CUDACachingAllocator::emptyCache();

  if (!is_trial) {
    is_trained_ = true;
  } else {
    // TODO should this work around be improved
    global_pred_distances_.clear();
  }

  return SUCCESS;
}




//针对每个filter去train一个小模型，包括处理收集数据，训练模型，预测距离，收集误差
// 这只是针对当前的filter进行训练，而不是针对所有filter进行训练
RESPONSE dstree::Filter::batch_train(bool is_trial) {
// printf("\n--------Filter::batch_train-------\n");
//# 实现多校准集的 batch_train 函数
  // ========================= 1. 前置检查 ==============================
  if (is_trained_ || config_.get().to_load_filters_) {
    return FAILURE;
  }
  if (!is_active_ && !is_trial) {
    spdlog::error("filter {:d} neither is_active nor is_trial; exit", id_);
    spdlog::shutdown();
    exit(FAILURE);
  }

   // =========================== 2. CUDA流初始化 ==========================
  ID_TYPE stream_id = -1;
  if (config_.get().filter_train_is_gpu_) {
    stream_id = at::cuda::getCurrentCUDAStream(config_.get().filter_device_id_).id(); // compiles with libtorch-gpu
  }

  // ============================= 3. 数据预处理 =============================
  if (config_.get().filter_remove_square_ && !is_distances_preprocessed_) {
    for (ID_TYPE i = 0; i < global_data_size_; ++i) {
      global_lnn_distances_[i] = sqrt(global_lnn_distances_[i]); //对全局最近邻距离取平方根
      global_bsf_distances_[i] = sqrt(global_bsf_distances_[i]); //对最佳搜索距离取平方根
    }
    // printf("global_data_size_ = %zu\n", global_data_size_);
    // 对下界距离取平方根（如果存在）
    if (!lb_distances_.empty()) {
      for (ID_TYPE i = 0; i < global_data_size_; ++i) {
        lb_distances_[i] = sqrt(lb_distances_[i]);
      }
    }
    // 对local最近邻距离取平方根（如果存在局部local数据）
    if (local_data_size_ > 0) {
      for (ID_TYPE i = 0; i < local_data_size_; ++i) {
        local_lnn_distances_[i] = sqrt(local_lnn_distances_[i]);
      }
    }
    is_distances_preprocessed_ = true; 
  }


  // ========== 关键修改点 1: 数据划分策略 ==========
  // 将数据划分为: 训练集、验证集和多个校准集
  ID_TYPE num_calib_batches = config_.get().filter_conformal_num_batches_;
  ID_TYPE num_examples_per_calib_batch = global_data_size_ / num_calib_batches;

  float train_ratio = config_.get().filter_train_val_split_; 
  ID_TYPE num_calib_examples;
  ID_TYPE num_batches;
  ID_TYPE remainder;

  // 当前计算方式适合local data=0, 如果local data>0, 第一个if语句中会修改它的值
  ID_TYPE num_train_examples = global_data_size_ * train_ratio; //local data>0时，num_train_examples会加上local_data_size_ * train_ratio
  ID_TYPE num_valid_examples = global_data_size_ - num_train_examples;
  num_global_train_examples = global_data_size_ * train_ratio;
  ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;

  // printf("全局训练集数量 num_global_train_examples = %ld\n", num_global_train_examples);

  // 创建数据张量
  torch::Tensor train_data, valid_data, calibration_data;
  torch::Tensor train_targets, valid_targets, calibration_targets;
  torch::Tensor global_valid_data, global_valid_targets;
  std::vector<torch::Tensor> calib_data_batches(num_calib_batches);
  std::vector<torch::Tensor> calib_target_batches(num_calib_batches);

  // 处理全局和局部数据
  if (local_data_size_ > 0) {
    // printf("对local data CP划分 local_data_size_ > 0\n");
    // printf("global_data_size_: %ld\n", static_cast<long>(global_data_size_));
    // printf("local_data_size_: %ld\n", static_cast<long>(local_data_size_));
 
    // 5.1.1 获取训练集的全局和局部数据 (保持不变)
    assert(global_data_size_ == config_.get().filter_train_num_global_example_);
    assert(local_data_size_ == config_.get().filter_train_num_local_example_);
    // 全局和局部训练集划分
    ID_TYPE num_local_train_examples = local_data_size_ * config_.get().filter_train_val_split_;
    // 全局训练数据处理
    torch::Tensor global_train_data = global_queries_.get().index({torch::indexing::Slice(0, num_global_train_examples)}).clone();
    torch::Tensor global_train_targets = torch::from_blob(global_lnn_distances_.data(), num_global_train_examples, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    // 局部训练数据处理
    torch::Tensor local_train_data = torch::from_blob(local_queries_.data(),{num_local_train_examples, config_.get().series_length_}, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    torch::Tensor local_train_targets = torch::from_blob(local_lnn_distances_.data(), num_local_train_examples,torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    // 合并训练数据
    train_data = torch::cat({global_train_data, local_train_data}, 0);
    train_targets = torch::cat({global_train_targets, local_train_targets}, 0);
    
    // 5.1.2 获取验证集的全局和局部数据
    ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
    ID_TYPE num_local_valid_examples = local_data_size_ - num_local_train_examples;
    
    // 获取全局验证数据：global data中出去global train data的部分
    global_valid_data = global_queries_.get().index({torch::indexing::Slice(num_global_train_examples, global_data_size_)}).clone();
    global_valid_targets = torch::from_blob(global_lnn_distances_.data() + num_global_train_examples, num_global_valid_examples, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    
    // 获取局部验证数据：local data中出去local train data的部分
    torch::Tensor local_valid_data = torch::from_blob(
        local_queries_.data() + num_local_train_examples * config_.get().series_length_,
        {num_local_valid_examples, config_.get().series_length_}, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    torch::Tensor local_valid_targets = torch::from_blob(local_lnn_distances_.data() + num_local_train_examples,num_local_valid_examples, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    
    // 合并验证数据
    valid_data = torch::cat({global_valid_data, local_valid_data}, 0);
    valid_targets = torch::cat({global_valid_targets, local_valid_targets}, 0);

    num_train_examples = num_global_train_examples + num_local_train_examples;
    num_valid_examples = num_global_valid_examples + num_local_valid_examples;

    assert(train_data.size(0) == num_train_examples && train_targets.size(0) == num_train_examples);
    assert(valid_data.size(0) == num_valid_examples && valid_targets.size(0) == num_valid_examples);
  
    // 修改：只使用global_valid_data作为校准数据，local_valid_data不参与校准
    // 将local_valid_data保留为验证集
    calibration_data = global_valid_data;
    calibration_targets = global_valid_targets;
    
    // 确定全局验证样本数量，只使用这部分数据作为校准集
    num_calib_examples = global_valid_data.size(0);
    // printf("校准数据总量: %d\n", num_calib_examples);
    
    // 如果启用组合方法生成校准批次
    if (config_.get().filter_conformal_use_combinatorial_) {
      std::vector<std::vector<ID_TYPE>> calib_query_ids;
      if (generate_calibration_batches(calibration_data, calibration_targets, 
                                       calib_data_batches, calib_target_batches,
                                       calib_query_ids) == FAILURE) {
        printf("生成组合校准批次失败\n");
        return FAILURE;
      }
      
      // 存储校准批次对应的查询ID，用于后续计算recall
      batch_calib_query_ids_ = calib_query_ids;
      // 更新批次数量
      num_batches = calib_data_batches.size();
      
      // 保存查询ID到文件
      // save_calib_query_ids(calib_query_ids, "filter_" + std::to_string(id_) + "_calib_query_ids");
      // printf("校准批次信息已保存到 filter_%d_calib_query_ids.txt\n", id_);
    } else {
      // 均匀划分校准集
      std::vector<std::vector<ID_TYPE>> calib_query_ids;
      if (generate_uniform_calibration_batches(calibration_data, calibration_targets,
                                              calib_data_batches, calib_target_batches,
                                              calib_query_ids) == FAILURE) {
        printf("生成均匀校准批次失败\n");
        return FAILURE;
      }
      
      // 存储校准批次对应的查询ID，用于后续计算recall
      batch_calib_query_ids_ = calib_query_ids;
      // 更新批次数量
      num_batches = calib_data_batches.size();
    }
    
    // 更新样本数量统计
    num_train_examples = train_data.size(0);
    num_valid_examples = calibration_data.size(0);
    // 验证数据维度
    assert(train_data.size(0) == num_train_examples && train_targets.size(0) == num_train_examples);
    assert(calibration_data.size(0) == num_valid_examples && calibration_targets.size(0) == num_valid_examples);
    // printf("训练集大小: %d, 验证集大小(本地): %d, 校准批次数(全局): %d\n", 
    //       num_train_examples, num_valid_examples, num_batches);
    // 打印每个校准集的大小和总数统计
    ID_TYPE total_calib_size = 0;
    // printf("\n校准集详细信息:\n");
    for (ID_TYPE i = 0; i < num_batches; ++i) {
        ID_TYPE batch_size = calib_data_batches[i].size(0);
        total_calib_size += batch_size;
        // printf("校准集 %d: %d 个样本\n", i + 1, batch_size);
    }
    // printf("校准集总数: %d 个\n校准集数量: %d 个\n", total_calib_size, num_batches);


  
  } else {

    printf("只使用全局数据时的多批次CP划分 local_data_size_ <= 0\n");
    // ========== 关键修改点 2: 只使用全局数据时的多批次划分 ==========
    // 训练数据    
    train_data = global_queries_.get().index({torch::indexing::Slice(0, num_train_examples)}).clone();
    train_targets = torch::from_blob(global_lnn_distances_.data(),
                                   num_train_examples,
                                   torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    // 验证数据
    valid_data = global_queries_.get().index({torch::indexing::Slice(
        num_train_examples, num_train_examples + num_valid_examples)}).clone();
    valid_targets = torch::from_blob(global_lnn_distances_.data() + num_train_examples,
                                   num_valid_examples,
                                   torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    
    // 修改：只使用global_valid_data作为校准数据，local_valid_data不参与校准
    // 将local_valid_data保留为验证集
    calibration_data = valid_data;
    calibration_targets = valid_targets;
    
    // 确定全局验证样本数量，只使用这部分数据作为校准集
    num_calib_examples = valid_targets.size(0);
    // printf("校准数据总量: %d\n", num_calib_examples);
    
    // 如果启用组合方法生成校准批次
    if (config_.get().filter_conformal_use_combinatorial_) {
      std::vector<std::vector<ID_TYPE>> calib_query_ids;
      // 生成组合校准批次和对应的查询ID
      if (generate_calibration_batches(calibration_data, calibration_targets, 
                                       calib_data_batches, calib_target_batches,
                                       calib_query_ids) == FAILURE) {
        printf("生成组合校准批次失败\n");
        return FAILURE;
      }
      // 存储校准批次对应的查询ID，用于后续计算recall
      batch_calib_query_ids_ = calib_query_ids;
      
      // 保存查询ID到文件
      // save_calib_query_ids(calib_query_ids, "filter_" + std::to_string(id_) + "_calib_query_ids");
      
      // 更新批次数量
      num_batches = calib_data_batches.size();

    } else {
      // 均匀划分校准集
      std::vector<std::vector<ID_TYPE>> calib_query_ids;
      if (generate_uniform_calibration_batches(calibration_data, calibration_targets,
                                              calib_data_batches, calib_target_batches,
                                              calib_query_ids) == FAILURE) {
        printf("生成均匀校准批次失败\n");
        return FAILURE;
      }
      
      // 存储校准批次对应的查询ID，用于后续计算recall
      batch_calib_query_ids_ = calib_query_ids;
      // 更新批次数量
      num_batches = calib_data_batches.size();
    }
    
    // 更新样本数量统计
    num_train_examples = train_data.size(0);
    num_valid_examples = calibration_data.size(0);
    
    // 验证数据维度
    assert(train_data.size(0) == num_train_examples && train_targets.size(0) == num_train_examples);
    assert(calibration_data.size(0) == num_valid_examples && calibration_targets.size(0) == num_valid_examples);
    
    // printf("训练集大小: %d, 验证集大小(本地): %d, 校准批次数(全局): %d\n", 
          // num_train_examples, num_valid_examples, num_batches);
    // 打印每个校准集的大小和总数统计
    // ID_TYPE total_calib_size = 0;
    // printf("\n校准集详细信息:\n");
    // for (ID_TYPE i = 0; i < num_batches; ++i) {
    //     ID_TYPE batch_size = calib_data_batches[i].size(0);
    //     total_calib_size += batch_size;
    //     // printf("校准集 %d: %d 个样本\n", i + 1, batch_size);
    // }
    // printf("校准集总数: %d 个\n校准集数量: %d 个\n", total_calib_size, num_batches);
  }


  // ============================= 6. 数据加载器初始化 ==============================
  auto train_dataset = upcite::SeriesDataset(train_data, train_targets);
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      train_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);
  model_ = dstree::get_model(config_);
  model_->to(*device_);

  // ==================================== 8. 训练准备 ================================
  // for early termination
  std::unordered_map<std::string, torch::Tensor> best_model_state;
  VALUE_TYPE best_validation_loss = constant::MAX_VALUE;
  ID_TYPE best_validation_epoch = -1;
  // 优化器选择（CNN用Adam，其他用SGD）
  std::shared_ptr<torch::optim::Optimizer> optimizer = nullptr;
  if (model_->model_type_ == CNN) {
    optimizer = std::make_shared<torch::optim::Adam>(model_->parameters(), config_.get().filter_train_learning_rate_);
  } else {
    optimizer = std::make_shared<torch::optim::SGD>(model_->parameters(), config_.get().filter_train_learning_rate_);
  }
  ID_TYPE initial_cooldown_epochs = config_.get().filter_train_nepoch_ / 2;

  //  学习率调整策略（ReduceLROnPlateau）, 用于验证损失
  upcite::optim::ReduceLROnPlateau lr_scheduler = upcite::optim::ReduceLROnPlateau(
      *optimizer, initial_cooldown_epochs, optim::MIN, config_.get().filter_lr_adjust_factor_);
  torch::nn::MSELoss mse_loss(torch::nn::MSELossOptions().reduction(torch::kMean));

#ifdef DEBUG
  std::vector<float> train_losses, valid_losses, batch_train_losses;
  train_losses.reserve(config_.get().filter_train_nepoch_);
  batch_train_losses.reserve(num_train_examples / config_.get().filter_train_batchsize_ + 1);
  valid_losses.reserve(config_.get().filter_train_nepoch_);
#endif

  //================================= 9. 训练循环 ==============================
  torch::Tensor batch_data, batch_target;
  for (ID_TYPE epoch = 0; epoch < config_.get().filter_train_nepoch_; ++epoch) {
    model_->train();   
    for (auto &batch : *train_data_loader) {
      batch_data = batch.data;
      batch_target = batch.target;
      optimizer->zero_grad();
      torch::Tensor prediction = model_->forward(batch_data);
      torch::Tensor loss = mse_loss->forward(prediction, batch_target);
      loss.backward(); // 反向传播
      // 梯度裁剪防止爆炸
      if (config_.get().filter_train_clip_grad_) {
        auto norm = torch::nn::utils::clip_grad_norm_(model_->parameters(),
                                                      config_.get().filter_train_clip_grad_max_norm_,
                                                      config_.get().filter_train_clip_grad_norm_type_);
      }
      optimizer->step();

#ifdef DEBUG
      batch_train_losses.push_back(loss.detach().item<float>());
#endif
    }

#ifdef DEBUG
    train_losses.push_back(std::accumulate(batch_train_losses.begin(), batch_train_losses.end(), 0.0)
                               / static_cast<VALUE_TYPE>(batch_train_losses.size()));
    batch_train_losses.clear();
#endif

    // 9.2 验证阶段
    { // evaluate
      VALUE_TYPE valid_loss = 0;
      c10::InferenceMode guard;
      model_->eval();  
      torch::Tensor prediction = model_->forward(valid_data);
      valid_loss = mse_loss->forward(prediction, valid_targets).detach().item<VALUE_TYPE>();

#ifdef DEBUG
      valid_losses.push_back(valid_loss);
#endif
      // 记录最佳模型状态
      if (epoch > initial_cooldown_epochs) {
        if (best_validation_loss > valid_loss) {
          best_validation_loss = valid_loss;
          best_validation_epoch = epoch;

          for (const auto &pair : model_->named_parameters()) {
            best_model_state[pair.key()] = pair.value().clone();
          }
        }
      }
      // 学习率调整与早停策略
      upcite::optim::LR_RETURN_CODE return_code = lr_scheduler.check_step(valid_loss);
      if (return_code == upcite::optim::EARLY_STOP) {
        epoch = config_.get().filter_train_nepoch_;
      }
    }
  }

#ifdef DEBUG
  spdlog::debug("filter {:d} s{:d} {:s} tloss = {:s}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                upcite::array2str(train_losses.data(), config_.get().filter_train_nepoch_));
  spdlog::debug("filter {:d} s{:d} {:s} vloss = {:s}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                upcite::array2str(valid_losses.data(), config_.get().filter_train_nepoch_));
#endif

  c10::InferenceMode guard;

// ============================ 10. 模型恢复与预测 =============================
// 恢复最佳模型状态
  if (best_validation_epoch > initial_cooldown_epochs) {
#ifdef DEBUG
    spdlog::debug("filter {:d} s{:d} {:s} restore from e{:d}, vloss {:.4f}",
                  id_, stream_id, model_setting_ref_.get().model_setting_str,
                  best_validation_epoch, best_validation_loss);
#endif

    for (auto &pair : best_model_state) {
      model_->named_parameters()[pair.first].detach_();
      model_->named_parameters()[pair.first].copy_(pair.second);
    }
  }
  // ========== 关键修改点 4: 对全局数据进行预测 ==========
  // 模型预测
  model_->eval();
  auto prediction = model_->forward(global_queries_).detach().cpu();
  assert(prediction.size(0) == global_data_size_);
  auto *predictions_array = prediction.detach().cpu().contiguous().data_ptr<VALUE_TYPE>();
  
  // !!!!!!!!!!!!!!!!!!1  存储模型预测结果到global_pred_distances_
  global_pred_distances_.insert(global_pred_distances_.end(), predictions_array, predictions_array + global_data_size_);
 
  // 打印 global_pred_distances_ 的 size
  // printf("Size of global_pred_distances_: %zu\n", global_pred_distances_.size());
  #ifdef DEBUG
  spdlog::info("filter {:d}{:s} s{:d} {:s} g_pred{:s} = {:s}",
               id_, is_trial ? " (trial)" : "",
               stream_id, model_setting_ref_.get().model_setting_str,
               config_.get().filter_remove_square_ ? "" : "_sq",
               upcite::array2str(predictions_array, global_data_size_));
#endif

  // ========== 关键修改点 5: 多校准集保形预测 ==========
  if (config_.get().filter_is_conformal_) {
    // 对预测距离进行Conformal Prediction的校准
    // printf("\n ---------------正式进入CP了,集中注意力---------------\n");
    fit_batch_conformal_predictor(is_trial, num_batches, calib_data_batches, calib_target_batches);
  }

 //  net->to(torch::Device(torch::kCPU));
  c10::cuda::CUDACachingAllocator::emptyCache();

  if (!is_trial) {
    is_trained_ = true;

  } else {
    // TODO should this work around be improved
    global_pred_distances_.clear();
  }

  return SUCCESS;
}

// 将校准查询ID保存到文件
RESPONSE dstree::Filter::save_calib_query_ids(const std::vector<std::vector<ID_TYPE>>& calib_query_ids, 
                                              const std::string& filename) {
    // 使用配置中的结果路径
    
    std::string save_path = config_.get().results_path_; // 从配置中获取路径
    
    // 确保路径以'/'结尾
    if (!save_path.empty() && save_path.back() != '/') {
        save_path += '/';
    }
    
    // 创建完整文件名
    std::string full_filename = save_path + filename;
    if (filename.find(".txt") == std::string::npos) {
        full_filename += ".txt";
    }
    
    // 打开文件进行写入
    std::ofstream file(full_filename);
    if (!file.is_open()) {
        printf("错误: 无法创建文件 %s\n", full_filename.c_str());
        return FAILURE;
    }
    
    // 写入总批次数
    // file << calib_query_ids.size() << std::endl;
    
    // 逐批次写入查询ID
    for (size_t batch_idx = 0; batch_idx < calib_query_ids.size(); ++batch_idx) {
        const auto& batch = calib_query_ids[batch_idx];
        
        // 写入当前批次的ID数量
        // file << batch.size() << std::endl;
        
        // 写入当前批次的所有ID，用空格分隔
        for (size_t i = 0; i < batch.size(); ++i) {
            file << batch[i];
            if (i < batch.size() - 1) {
                file << " ";
            }
        }
        file << std::endl;
    }
    
    file.close();
    printf("已成功保存校准查询ID到 %s (共%zu批次)\n", 
           full_filename.c_str(), calib_query_ids.size());
    
    return SUCCESS;
}
    


    

RESPONSE dstree::Filter::collect_running_info(MODEL_SETTING &model_setting) {
  model_setting_ref_ = model_setting;

  model_ = dstree::get_model(config_);
  model_->to(*device_);

  c10::InferenceMode guard;
  model_->eval();

  if (config_.get().filter_is_conformal_ && !conformal_predictor_->is_fitted()) {
    printf("-------collect_running_info 进入fit_conformal_predictor(false, true)------------\n");
    fit_conformal_predictor(false, true);
  }

  model_setting_ref_.get().gpu_mem_mb = get_memory_footprint(*model_);

  auto trial_query = global_queries_.get().index({torch::indexing::Slice(0, 1)}).clone();
  auto trial_predictions = make_reserved<VALUE_TYPE>(config_.get().filter_trial_iterations_);

  auto start = std::chrono::high_resolution_clock::now();

  for (ID_TYPE trial_i = 0; trial_i < config_.get().filter_trial_iterations_; ++trial_i) {
    auto pred = model_->forward(trial_query).item<VALUE_TYPE>();

    if (conformal_predictor_ != nullptr) {
      pred = conformal_predictor_->predict(pred).left_bound_;
    }

    if (config_.get().filter_remove_square_) {
      trial_predictions.push_back(pred * pred);
    } else {
      trial_predictions.push_back(pred);
    }
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  model_setting_ref_.get().gpu_ms_per_query =
      static_cast<double_t>(duration.count()) / static_cast<double_t>(config_.get().filter_trial_iterations_);

#ifdef DEBUG
  spdlog::info("trial {:s} gpu mem = {:.3f}MB, time = {:.6f}mus",
               model_setting_ref_.get().model_setting_str,
               model_setting_ref_.get().gpu_mem_mb,
               model_setting_ref_.get().gpu_ms_per_query);
#endif

  return SUCCESS;
}



VALUE_TYPE dstree::Filter::infer(torch::Tensor &query_series) const {
#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("filter {:d} {:b} device {:s}, requested {:b}:{:d}",
                id_, is_trained_,
                device_->str(),
                config_.get().filter_infer_is_gpu_, config_.get().filter_device_id_);
  spdlog::debug("filter {:d} {:b} query device {:s}, requested {:b}:{:d}",
                id_, is_trained_,
                query_series.device().str(),
                config_.get().filter_infer_is_gpu_, config_.get().filter_device_id_);

  auto paras = model_->parameters();
  for (ID_TYPE i = 0; i < paras.size(); ++i) {
    spdlog::debug("filter {:d} {:b} model_p_{:d} device {:s}, requested {:b}:{:d}",
                  id_, is_trained_,
                  i, paras[i].device().str(),
                  config_.get().filter_infer_is_gpu_, config_.get().filter_device_id_);
  }
#endif
#endif
  // printf("--------------进入测试阶段的infer函数--------------\n");
  if (is_trained_) {
    printf("is_trained_ = %d\n", is_trained_);
    c10::InferenceMode guard;
    VALUE_TYPE pred = model_->forward(query_series).item<VALUE_TYPE>();
    if (conformal_predictor_ != nullptr) {
      pred = conformal_predictor_->predict(pred).left_bound_;
      // printf("pred = %.3f, left_bound_ = %.3f\n", pred);
    }

    if (config_.get().filter_remove_square_) {
      return pred * pred;
    } else {
      return pred;
    }
  } else {
    return constant::MAX_VALUE;
  }
}


// 在Filter类中添加infer_calibrated方法的实现
VALUE_TYPE dstree::Filter::infer_calibrated(torch::Tensor &query_series) const {
  if (is_trained_) {
    c10::InferenceMode guard;
    // 获取模型原始预测
    VALUE_TYPE raw_pred = model_->forward(query_series).item<VALUE_TYPE>();
    // 获取校准值（原始预测 - alpha）
    VALUE_TYPE calibrated_pred;
    if (conformal_predictor_ != nullptr) {
      // 使用ConformalRegressor获取校准后的下界
      calibrated_pred = conformal_predictor_->predict_calibrated(raw_pred);
      // printf("Filter %d: 原始预测距离=%.3f, 校准后距离=%.3f\n", 
            //  id_, raw_pred, calibrated_pred);
      // spdlog::info("Filter {}: raw_pred={:.3f}, calibrated={:.3f}", 
            //  id_, raw_pred, calibrated_pred);
    } else {
      calibrated_pred = raw_pred; // 无校准
    }

    // 应用平方处理 
    if (config_.get().filter_remove_square_) {
      return calibrated_pred * calibrated_pred;
    } else {
      return calibrated_pred;
    }
  } else {
    return constant::MAX_VALUE;
  }
}



// 在Filter类中添加新方法，仅返回原始预测
VALUE_TYPE dstree::Filter::infer_raw(torch::Tensor &query_series) const {
    if (is_trained_) {
        c10::InferenceMode guard;
        VALUE_TYPE pred = model_->forward(query_series).item<VALUE_TYPE>();
        
        // 仅返回模型原始预测，不进行任何校准
        if (config_.get().filter_remove_square_) {
            return pred * pred;
        } else {
            return pred;
        }
    } else {
        return constant::MAX_VALUE;
    }
}




RESPONSE dstree::Filter::dump(std::ofstream &node_fos) const {
  node_fos.write(reinterpret_cast<const char *>(&global_data_size_), sizeof(ID_TYPE));

  assert(global_bsf_distances_.size() == global_data_size_);
  ID_TYPE size_placeholder = global_bsf_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!global_bsf_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(global_bsf_distances_.data()),
                   sizeof(VALUE_TYPE) * global_bsf_distances_.size());
  }

  assert(global_lnn_distances_.size() == global_data_size_);
  size_placeholder = global_lnn_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!global_lnn_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(global_lnn_distances_.data()),
                   sizeof(VALUE_TYPE) * global_lnn_distances_.size());
  }

  assert(lb_distances_.size() == global_data_size_);
  size_placeholder = lb_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!lb_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(lb_distances_.data()), sizeof(VALUE_TYPE) * lb_distances_.size());
  }

  // currently upper bounds are not being used
  assert(ub_distances_.size() == 0);
  size_placeholder = ub_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!ub_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(ub_distances_.data()), sizeof(VALUE_TYPE) * ub_distances_.size());
  }

#ifdef DEBUG
  spdlog::debug("filter {:d} (trained {:b} active {:b}) n_pred {:d} n_glob {:d} n_local {:d}",
                id_, is_trained_, is_active_,
                global_pred_distances_.size(), global_data_size_, local_data_size_);
#endif

  if (is_trained_) {
    assert(global_pred_distances_.size() == global_data_size_);
  } else {
    assert(global_pred_distances_.empty());
  }
  size_placeholder = global_pred_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!global_pred_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(global_pred_distances_.data()),
                   sizeof(VALUE_TYPE) * global_pred_distances_.size());
  }

  node_fos.write(reinterpret_cast<const char *>(&local_data_size_), sizeof(ID_TYPE));

  if (local_data_size_ > 0) {
    assert(config_.get().series_length_ * local_data_size_ == local_queries_.size());
    assert(local_lnn_distances_.size() == local_data_size_);

    size_placeholder = local_queries_.size();
    node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));

    if (!local_queries_.empty()) {
      node_fos.write(reinterpret_cast<const char *>(local_queries_.data()),
                     sizeof(VALUE_TYPE) * local_queries_.size());
    }

    size_placeholder = local_lnn_distances_.size();
    node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));

    if (!local_lnn_distances_.empty()) {
      node_fos.write(reinterpret_cast<const char *>(local_lnn_distances_.data()),
                     sizeof(VALUE_TYPE) * local_lnn_distances_.size());
    }
  }

//  spdlog::debug("dump filter {:d} global {:d} local {:d} active {:b} train {:b}",
//                id_, global_data_size_, local_data_size_, is_active_, is_trained_);

  if (is_active_) {
    size_placeholder = model_setting_ref_.get().model_setting_str.size();
  } else {
    size_placeholder = -1;
  }
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (is_active_) {
    node_fos.write(reinterpret_cast<const char *>(model_setting_ref_.get().model_setting_str.data()),
                   sizeof(model_setting_ref_.get().model_setting_str));
  }

  ID_TYPE is_trained_placeholder = 0;
  if (is_trained_) {
    is_trained_placeholder = 1;
  }
  node_fos.write(reinterpret_cast<const char *>(&is_trained_placeholder), sizeof(ID_TYPE));
  if (is_trained_) {
    std::string model_filepath = config_.get().dump_filters_folderpath_ + std::to_string(id_) +
        config_.get().model_dump_file_postfix_;

    torch::save(model_, model_filepath);
  }

  ID_TYPE is_conformal_placeholder = 0;
  if (config_.get().filter_is_conformal_) {
    is_conformal_placeholder = 1;
  }
  node_fos.write(reinterpret_cast<const char *>(&is_conformal_placeholder), sizeof(ID_TYPE));
  if (config_.get().filter_is_conformal_) {
    conformal_predictor_->dump(node_fos);
  }

  return SUCCESS;
}




// 在 Filter 类实现中添加
RESPONSE dstree::Filter::load_batch_alphas(const std::string& filepath) {
    if (!is_trained_ || !is_active_ || !config_.get().filter_is_conformal_) {
        return FAILURE;
    }
    
    return conformal_predictor_->load_batch_alphas(filepath);
}

// load 函数

RESPONSE dstree::Filter::load(std::ifstream &node_ifs, void *ifs_buf) {
  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);
  auto ifs_value_buf = reinterpret_cast<VALUE_TYPE *>(ifs_buf);

  // global_data_size_
  ID_TYPE read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  global_data_size_ = ifs_id_buf[0];

  // bsf_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  ID_TYPE size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    global_bsf_distances_.insert(global_bsf_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }
  assert(global_bsf_distances_.size() == global_data_size_);
//  assert(node_ifs.good());

  // nn_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    global_lnn_distances_.insert(global_lnn_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }
  assert(global_lnn_distances_.size() == global_data_size_);
//  assert(node_ifs.good());

  // lb_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    lb_distances_.insert(lb_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }
  assert(lb_distances_.size() == global_data_size_);
  assert(node_ifs.good());

  // ub_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    ub_distances_.insert(ub_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }
  assert(ub_distances_.size() == 0);
//  assert(node_ifs.good());

  // pred_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    global_pred_distances_.insert(global_pred_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
    assert(global_pred_distances_.size() == global_data_size_);
//    assert(node_ifs.good());
  }

  // local_data_size_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  local_data_size_ = ifs_id_buf[0];


  if (local_data_size_ > 0) {
    // local_queries_
    read_nbytes = sizeof(ID_TYPE);
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    size_indicator = ifs_id_buf[0];
    assert(size_indicator == config_.get().series_length_ * local_data_size_);

    if (size_indicator > 0) {
      local_queries_.reserve(size_indicator);
      read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
      node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
      local_queries_.insert(local_queries_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
    }

    // local_lnn_distances_
    read_nbytes = sizeof(ID_TYPE);
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    size_indicator = ifs_id_buf[0];
    assert(size_indicator == local_data_size_);

    if (size_indicator > 0) {
      local_lnn_distances_.reserve(size_indicator);
      read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
      node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
      local_lnn_distances_.insert(local_lnn_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
    }
  }
  assert(local_queries_.size() == config_.get().series_length_ * local_data_size_);
  assert(local_lnn_distances_.size() == local_data_size_);
//  assert(node_ifs.good());

  // model_setting_
  is_active_ = false;
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];
  if (size_indicator > 0) {
    std::string model_setting_str;
    model_setting_str.resize(size_indicator);
    node_ifs.read(const_cast<char *>(model_setting_str.data()), size_indicator);

    if (config_.get().to_load_filters_) {
      model_setting_ = MODEL_SETTING(model_setting_str);
      model_setting_ref_ = std::ref(model_setting_);
      is_active_ = true;
    }
  }
//  assert(node_ifs.good());
  // model_
  is_trained_ = false;
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator == 0 && is_active_) {
    spdlog::error("loading filter {:d} activated but marked untrained; workaround by setting is_trained_",
                  id_);
    size_indicator = 1;
  }

  if (size_indicator > 0) {
    std::string model_filepath = config_.get().load_filters_folderpath_ + std::to_string(id_) +
        config_.get().model_dump_file_postfix_;
    if (!fs::is_regular_file(model_filepath)) {
      spdlog::error("Empty model_filepath found: {:s}", model_filepath);
      return FAILURE;
    }

    if (config_.get().filter_infer_is_gpu_) {
      // TODO support multiple devices
      device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
    model_ = dstree::get_model(config_);
    // TODO check if the to-be-loaded model type matches the persisted model type
    torch::load(model_, model_filepath);
    model_->to(*device_);
    model_->eval();
//  net->to(torch::Device(torch::kCPU));
    c10::cuda::CUDACachingAllocator::emptyCache();
    if (config_.get().to_load_filters_) {
      is_trained_ = true;
    }
  }
//  assert(node_ifs.good());
  // conformal_predictor_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];
  if (size_indicator > 0) {
    conformal_predictor_->load(node_ifs, ifs_buf);
    if (is_active_ && is_trained_ && config_.get().filter_is_conformal_) {
      // TODO check compatibility between the loaded setting and the new setting
      // printf("~~~~~~~~~~~~~ 进入了load函数:fit_conformal_predictor(false, false) ~~~~~~~~~~~~~~~~~~\n");
      fit_conformal_predictor(false, false);
    }
  }
  spdlog::debug("load filter {:d} global {:d} local {:d} active {:b} trained {:b}",
                id_, global_data_size_, local_data_size_, is_active_, is_trained_);
  assert(node_ifs.good());
  return SUCCESS;
}




VALUE_TYPE dstree::Filter::get_node_summarization_pruning_frequency() const {
  if (lb_distances_.empty() || lb_distances_.size() != global_bsf_distances_.size()) {
    return 0;
  }

  ID_TYPE pruned_counter = 0;
  for (ID_TYPE i = 0; i < lb_distances_.size(); ++i) {
    if (lb_distances_[i] > global_bsf_distances_[i]) {
      pruned_counter += 1;
    }
  }

  return static_cast<VALUE_TYPE>(pruned_counter) / static_cast<VALUE_TYPE>(lb_distances_.size());
}

VALUE_TYPE upcite::dstree::Filter::get_val_pruning_ratio() const {
  ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;

  VALUE_TYPE abs_error_interval = get_abs_error_interval();
  ID_TYPE pruned_counter = 0;

  for (ID_TYPE example_i = num_global_train_examples; example_i < global_data_size_; ++example_i) {
    if (global_pred_distances_[example_i] - abs_error_interval > global_bsf_distances_[example_i]){
      pruned_counter += 1;
    }
  }

  ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
  return static_cast<VALUE_TYPE>(pruned_counter) / num_global_valid_examples;
}

// 添加实现save_filter_batch_alphas方法
RESPONSE upcite::dstree::Filter::save_filter_batch_alphas(const std::string& filepath) const {
  if (!conformal_predictor_) {
    spdlog::error("没有初始化conformal_predictor，无法保存alphas");
    return FAILURE;
  }  
  // 调用ConformalPredictor中的方法保存批处理alphas
  return conformal_predictor_->save_batch_alphas(filepath);
}

// 添加在文件末尾，其他方法之后

// 添加清理批处理alphas的方法
RESPONSE dstree::Filter::clear_batch_alphas() {
  if (!is_trained_ || !is_active_ || !config_.get().filter_is_conformal_ || !conformal_predictor_) {
    return SUCCESS; // 如果不适用批处理alpha或预测器不存在，不需要清理
  }
  // 委托给ConformalPredictor的清理方法
  conformal_predictor_->clear_batch_data();
  return SUCCESS;
}



// 生成校准批次，使用增强组合方法 (每次重新打乱索引)
RESPONSE dstree::Filter::generate_calibration_batches(
    torch::Tensor& calib_data, 
    torch::Tensor& calib_targets,
    std::vector<torch::Tensor>& calib_data_batches,
    std::vector<torch::Tensor>& calib_target_batches,
    std::vector<std::vector<ID_TYPE>>& calib_query_ids) {
    
  // printf("使用增强组合方法生成校准批次（每次生成批次前重新打乱索引，突破组合数量限制）\n");
  // 获取校准数据大小和配置参数
  ID_TYPE num_calib_examples = calib_data.size(0);
  ID_TYPE n_parts = config_.get().filter_conformal_n_parts_;
  ID_TYPE k_parts = config_.get().filter_conformal_k_parts_;
  // 验证参数
  if (n_parts <= 0) {
    printf("错误: n_parts必须大于0, 当前值: %d\n", n_parts);
    return FAILURE;
  }
  if (k_parts <= 0 || k_parts > n_parts) {
    printf("错误: k_parts必须大于0且不大于n_parts, 当前值: %d\n", k_parts);
    return FAILURE;
  }
  // printf("校准数据总量: %d, 划分为%d份, 每个批次使用%d份\n", num_calib_examples, n_parts, k_parts);

  // 每份的样本数量
  ID_TYPE examples_per_part = num_calib_examples / n_parts;
  ID_TYPE remainder = num_calib_examples % n_parts;
  
  if (examples_per_part == 0) {
    printf("错误: 校准样本数量不足以分成%d份\n", n_parts);
    return FAILURE;
  }
  
  // 确定批次数量，不再受组合数限制
  ID_TYPE num_batches = config_.get().filter_conformal_num_batches_;  
  // 预分配批次存储空间
  calib_data_batches.resize(num_batches);
  calib_target_batches.resize(num_batches);
  calib_query_ids.resize(num_batches);
  
  // 为随机数生成器初始化
  std::random_device rd;
  std::mt19937 g(rd());

  // 原始索引，用于后续随机打乱
  std::vector<ID_TYPE> indices(num_calib_examples);
  for (ID_TYPE i = 0; i < num_calib_examples; ++i) {
    indices[i] = i;
  }
  
  for (ID_TYPE batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::shuffle(indices.begin(), indices.end(), g);
    // printf("\n===== 生成批次 %ld =====\n", batch_idx);
    
    // 计算每个部分的范围
    std::vector<std::pair<ID_TYPE, ID_TYPE>> part_ranges(n_parts);
    ID_TYPE start_idx = 0;
    
    for (ID_TYPE part_idx = 0; part_idx < n_parts; ++part_idx) {
      // 为最后一个部分添加剩余样本
      ID_TYPE extra = (part_idx == n_parts - 1) ? remainder : 0;
      ID_TYPE end_idx = start_idx + examples_per_part + extra;
      // 确保不超出索引范围
      end_idx = std::min(end_idx, static_cast<ID_TYPE>(indices.size()));
      part_ranges[part_idx] = {start_idx, end_idx};
      
      // printf("部分 %ld: 索引范围 [%ld, %ld), 大小=%ld\n", 
      //        part_idx, start_idx, end_idx, end_idx - start_idx);
      start_idx = end_idx;
    }
    
    // 随机选择k个部分
    std::vector<ID_TYPE> selected_parts(n_parts);
    for (ID_TYPE i = 0; i < n_parts; ++i) {
      selected_parts[i] = i;
    }
    std::shuffle(selected_parts.begin(), selected_parts.end(), g);
    selected_parts.resize(k_parts);
    std::sort(selected_parts.begin(), selected_parts.end()); // 排序便于检查重复
    
    // printf("选择的部分: ");
    // for (ID_TYPE part_idx : selected_parts) {
    //   printf("%ld ", part_idx);
    // }
    // printf("\n");
    
    // 计算当前批次大小
    ID_TYPE batch_size = 0;
    for (ID_TYPE part_idx : selected_parts) {
      batch_size += part_ranges[part_idx].second - part_ranges[part_idx].first;
    }
    
    // 为当前批次分配内存
    calib_data_batches[batch_idx] = torch::empty({batch_size, calib_data.size(1)}, calib_data.options());
    calib_target_batches[batch_idx] = torch::empty(batch_size, calib_targets.options());
    calib_query_ids[batch_idx].reserve(batch_size);
    
    // 填充当前批次数据
    ID_TYPE batch_pos = 0;
    for (ID_TYPE part_idx : selected_parts) {
      ID_TYPE part_start = part_ranges[part_idx].first;
      ID_TYPE part_end = part_ranges[part_idx].second;      
      ID_TYPE samples_shown = 0;
      for (ID_TYPE j = part_start; j < part_end; ++j) {
        ID_TYPE idx = indices[j];
        calib_data_batches[batch_idx][batch_pos] = calib_data[idx];
        calib_target_batches[batch_idx][batch_pos] = calib_targets[idx];
        
        // 记录原始查询ID（假设校准数据是从num_global_train_examples开始的全局数据）
        ID_TYPE original_query_id = num_global_train_examples + idx;
        calib_query_ids[batch_idx].push_back(original_query_id);
        batch_pos++;
      }
    }
  }
  
  // 最终汇总
  ID_TYPE total_batch_size = 0;
  for (ID_TYPE i = 0; i < num_batches; ++i) {
    total_batch_size += calib_data_batches[i].size(0);
  }
  // printf("成功生成 %d 个校准批次, 总样本数: %d \n", num_batches, total_batch_size);
  return SUCCESS;
}





// 生成均匀划分的校准批次
RESPONSE dstree::Filter::generate_uniform_calibration_batches(
    torch::Tensor& calib_data, 
    torch::Tensor& calib_targets,
    std::vector<torch::Tensor>& calib_data_batches,
    std::vector<torch::Tensor>& calib_target_batches,
    std::vector<std::vector<ID_TYPE>>& calib_query_ids) {
    
  // printf("使用均匀划分生成校准批次\n");
  
  // 获取校准数据大小和配置参数
  ID_TYPE num_calib_examples = calib_data.size(0);
  ID_TYPE num_batches = config_.get().filter_conformal_num_batches_;
  
  // 确保每个批次至少有3个样本
  num_batches = std::min(
      num_batches,
      static_cast<ID_TYPE>(std::floor(num_calib_examples / 3.0))
  );
  
  if (num_batches < 1) num_batches = 1;
  // printf("校准批次数: %d\n", num_batches);
  
  // 计算每个校准批次的样本数
  ID_TYPE examples_per_batch = num_calib_examples / num_batches;
  ID_TYPE remainder = num_calib_examples % num_batches;
  
  // 创建随机索引进行数据打乱
  std::vector<ID_TYPE> indices(num_calib_examples);
  for (ID_TYPE i = 0; i < num_calib_examples; ++i) {
      indices[i] = i;
  }
  
  // 随机打乱索引
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);
  
  // 预分配批次存储空间
  calib_data_batches.resize(num_batches);
  calib_target_batches.resize(num_batches);
  calib_query_ids.resize(num_batches);
  
  // 将校准数据分配到各批次
  for (ID_TYPE batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      // 计算当前批次的起始索引和结束索引
      ID_TYPE start_idx = batch_idx * examples_per_batch;
      // 为最后一个批次添加剩余样本
      ID_TYPE extra = (batch_idx == num_batches - 1) ? remainder : 0;
      ID_TYPE end_idx = start_idx + examples_per_batch + extra;
      // 确保不超出索引范围
      end_idx = std::min(end_idx, static_cast<ID_TYPE>(indices.size()));
      // 计算当前批次样本数
      ID_TYPE batch_size = end_idx - start_idx;
      
      // 初始化当前批次的数据和目标张量
      calib_data_batches[batch_idx] = torch::empty({batch_size, calib_data.size(1)}, calib_data.options());
      calib_target_batches[batch_idx] = torch::empty(batch_size, calib_targets.options());
      calib_query_ids[batch_idx].reserve(batch_size);
      
      // 填充数据
      for (ID_TYPE i = 0; i < batch_size; ++i) {
          ID_TYPE idx = indices[start_idx + i];
          calib_data_batches[batch_idx][i] = calib_data[idx];
          calib_target_batches[batch_idx][i] = calib_targets[idx];
          
          // 记录原始查询ID（假设校准数据是从num_global_train_examples开始的全局数据）
          ID_TYPE original_query_id = num_global_train_examples + idx;
          calib_query_ids[batch_idx].push_back(original_query_id);
      }
      // printf("校准批次 %d: 样本数 = %d\n", batch_idx + 1, batch_size);
  }
  
  // 最终汇总
  ID_TYPE total_batch_size = 0;
  for (ID_TYPE i = 0; i < num_batches; ++i) {
      total_batch_size += calib_data_batches[i].size(0);
  }
  
  // printf("成功生成 %d 个校准批次, 总样本数: %d\n", num_batches, total_batch_size);
  return SUCCESS;
}




RESPONSE dstree::Filter::train_regression_model_for_recall_coverage(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages,
    const std::vector<ID_TYPE>& error_indices,
    ID_TYPE filter_id) {
    
    // 直接委托给conformal_predictor_
    return conformal_predictor_->train_regression_model_for_recall_coverage(
        recalls, coverages, error_indices, filter_id);
}

// 添加新函数的实现，使用实际批次误差
RESPONSE dstree::Filter::train_regression_model_for_recall_coverage_actual_error(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages,
    const std::vector<ID_TYPE>& error_indices,
    ID_TYPE batch_id,
    ID_TYPE filter_id) {
    
    // 直接委托给conformal_predictor_的新函数
    return conformal_predictor_->train_regression_model_for_recall_coverage_actual_error(
        recalls, coverages, error_indices, batch_id, filter_id);
}

// 预测函数实现
double dstree::Filter::predict_error_value(double recall, double coverage) const {
    return conformal_predictor_->predict_error_value(recall, coverage);
}

// 设置函数实现
RESPONSE dstree::Filter::set_filter_abs_error_interval_by_recall_and_coverage(
    ERROR_TYPE recall, ERROR_TYPE coverage) {
    return conformal_predictor_->set_alpha_by_recall_and_coverage(recall, coverage);
}