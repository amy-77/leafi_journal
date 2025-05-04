//
// Created by Qitong Wang on 2022/12/13.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_SCHEDULER_H_
#define DSTREE_SRC_EXEC_SCHEDULER_H_

#include "global.h"

#include <torch/optim/schedulers/lr_scheduler.h>

namespace upcite {
namespace optim {

enum METRICS_MODE {
  MIN = 0,
  MAX = 1
};

enum THRESHOLD_MODE {
  RELATIVE = 0,
  ABSOLUTE = 1
};

enum LR_RETURN_CODE {
  SAME = 0,
  REDUCED = 1,
  EARLY_STOP = 2
};

// TODO refactor to ReduceLRSigmoid
//void adjust_learning_rate(torch::optim::SGD &optimizer,
//                          VALUE_TYPE max_lr,
//                          VALUE_TYPE min_lr,
//                          ID_TYPE epoch,
//                          ID_TYPE max_epoch) {
////    float new_lr = max_lr - (max_lr - min_lr) * ((float) epoch / (float) max_epoch);
//
//  float boundary = 9;
//  float current_step = (epoch / max_epoch - 0.5f) * boundary;
//  float new_lr = min_lr + (max_lr - min_lr) / (1 + exp(current_step));
//
//  for (auto &group : optimizer.param_groups()) {
//    if (group.has_options()) {
//      group.options().set_lr(new_lr);
//    }
//  }
//}

class TORCH_API ReduceLROnPlateau : public torch::optim::LRScheduler {
 public:
  explicit ReduceLROnPlateau(torch::optim::Optimizer &optimizer,
                             ID_TYPE initial_cooldown_epochs = 0,
                             METRICS_MODE mode = MIN,
                             double factor = 0.1,
                             ID_TYPE patience = 10,
                             VALUE_TYPE threshold = 1e-4,
                             THRESHOLD_MODE threshold_mode = RELATIVE,
                             ID_TYPE cooldown = 0,
                             double min_lr = 1e-7,
                             double eps = 1e-7);

  LR_RETURN_CODE check_step(VALUE_TYPE metrics, ID_TYPE epoch = -1);

 private:
  std::vector<double> get_lrs() override;

  bool is_better(VALUE_TYPE a, VALUE_TYPE best) const;
  bool in_cooldown() const; // does not have internal effects

  ID_TYPE patience_;
  ID_TYPE num_bad_epochs_;

  ID_TYPE initial_cooldown_;
  ID_TYPE cooldown_;
  ID_TYPE cooldown_counter_;

  ID_TYPE last_epoch_;
  VALUE_TYPE best_, mode_worst_;

  METRICS_MODE mode_;
  THRESHOLD_MODE threshold_mode_;
  VALUE_TYPE threshold_;

  std::vector<double> min_lrs;

  double factor_;
  double eps_;
};

}
}
#endif //DSTREE_SRC_EXEC_SCHEDULER_H_
