//
// Created by Qitong Wang on 2022/12/13.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "scheduler.h"

#include <cmath>

#include "spdlog/spdlog.h"

namespace constant = upcite::constant;

upcite::optim::ReduceLROnPlateau::ReduceLROnPlateau(torch::optim::Optimizer &optimizer,
                                                    ID_TYPE initial_cooldown_epochs,
                                                    upcite::optim::METRICS_MODE mode,
                                                    double factor,
                                                    ID_TYPE patience,
                                                    VALUE_TYPE threshold,
                                                    upcite::optim::THRESHOLD_MODE threshold_mode,
                                                    ID_TYPE cooldown,
                                                    double min_lr,
                                                    double eps) :
    torch::optim::LRScheduler(optimizer),
    initial_cooldown_(initial_cooldown_epochs),
    mode_(mode),
    factor_(factor),
    patience_(patience),
    threshold_(threshold),
    threshold_mode_(threshold_mode),
    cooldown_(cooldown),
    eps_(eps) {
  for (ID_TYPE i = 0; i < optimizer.param_groups().size(); ++i) {
    min_lrs.push_back(min_lr);
  }

  // _init_is_better
  if (mode == upcite::optim::MIN) {
    mode_worst_ = constant::MAX_VALUE;
  } else {
    mode_worst_ = constant::MIN_VALUE;
  }

  // _reset
  best_ = mode_worst_;
  cooldown_counter_ = 0;
  num_bad_epochs_ = 0;
}

upcite::optim::LR_RETURN_CODE upcite::optim::ReduceLROnPlateau::check_step(VALUE_TYPE metrics, ID_TYPE epoch) {
  LR_RETURN_CODE return_code = SAME;
  VALUE_TYPE current = metrics;

  if (epoch < 0) {
    epoch = last_epoch_ + 1;
  }

  if (epoch < initial_cooldown_) {
    if (is_better(current, best_)) {
      best_ = current;
    }
  } else {
    if (is_better(current, best_)) {
      best_ = current;
      num_bad_epochs_ = 0;
    } else {
      num_bad_epochs_ += 1;
    }

    if (in_cooldown()) {
      cooldown_counter_ -= 1;
      num_bad_epochs_ = 0;
    }

    if (num_bad_epochs_ > patience_) {
      if (get_current_lrs()[0] <= min_lrs[0]) {
        return_code = EARLY_STOP;
      } else {
#ifdef DEBUG
#ifndef DEBUGGED
        double old_lr = get_current_lrs()[0];
#endif
#endif

        step();

#ifdef DEBUG
#ifndef DEBUGGED
        double new_lr = get_current_lrs()[0];
    spdlog::debug("filter lr = {:f} -> {:f}", old_lr, new_lr);
#endif
#endif

        cooldown_counter_ = cooldown_;
        num_bad_epochs_ = 0;

        return_code = REDUCED;
      }
    }
  }

  last_epoch_ = epoch;
  return return_code;
}

std::vector<double> upcite::optim::ReduceLROnPlateau::get_lrs() {
  std::vector<double> lrs = get_current_lrs();

  // _reduce_lr
  for (ID_TYPE i = 0; i < lrs.size(); ++i) {
    double new_lr = fmax(lrs[i] * factor_, min_lrs[i]);

    if (lrs[i] - new_lr > eps_) {
      lrs[i] = new_lr;
    }
  }

  return lrs;
}

bool upcite::optim::ReduceLROnPlateau::is_better(VALUE_TYPE a, VALUE_TYPE best) const {
  if (mode_ == upcite::optim::MIN && threshold_mode_ == upcite::optim::RELATIVE) {
    VALUE_TYPE rel_epsilon = 1 - threshold_;
    return a < best * rel_epsilon;
  } else if (mode_ == upcite::optim::MIN && threshold_mode_ == upcite::optim::ABSOLUTE) {
    return a < best - threshold_;
  } else if (mode_ == upcite::optim::MAX && threshold_mode_ == upcite::optim::RELATIVE) {
    VALUE_TYPE rel_epsilon = 1 + threshold_;
    return a > best * rel_epsilon;
  } else if (mode_ == upcite::optim::MAX && threshold_mode_ == upcite::optim::ABSOLUTE) {
    return a > best + threshold_;
  }

  // should not be reached
  return false;
}

bool upcite::optim::ReduceLROnPlateau::in_cooldown() const {
  return cooldown_counter_ > 0;
}
