//
// Created by Qitong Wang on 2024/3/27.
// Copyright (c) 2024 Université Paris Cité. All rights reserved.
//

#include "stat.h"

ID_TYPE upcite::get_random_int_in_range(ID_TYPE min, ID_TYPE max_exclusive) {
  // Random number engine (using Mersenne Twister 19937 algorithm)
  std::random_device rd;  // Obtain a random number from hardware
  std::mt19937 eng(rd()); // Seed the generator

  // Define the range
  std::uniform_int_distribution<ID_TYPE> distr(min, max_exclusive - 1);

  return distr(eng); // Generate and return the random number
}

RESPONSE upcite::znormalize(VALUE_TYPE *series_in_place, ID_TYPE series_length) {
  VALUE_TYPE mean = 0, std = 0, diff;

  for (ID_TYPE value_i = 0; value_i < series_length; ++value_i) {
    mean += series_in_place[value_i];
  }

  mean /= (VALUE_TYPE) series_length;

  for (ID_TYPE value_i = 0; value_i < series_length; ++value_i) {
    diff = series_in_place[value_i] - mean;
    std += diff * diff;
  }

  std = (VALUE_TYPE) sqrt(std / (VALUE_TYPE) series_length);

  if (std <= constant::EPSILON) {
    return FAILURE;
  }

  for (ID_TYPE value_i = 0; value_i < series_length; ++value_i) {
    series_in_place[value_i] = (series_in_place[value_i] - mean) / std;
  }

  return SUCCESS;
}