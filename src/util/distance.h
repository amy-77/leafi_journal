//
// Created by Qitong Wang on 2022/10/19.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_DISTANCE_H_
#define DSTREE_SRC_EXEC_DISTANCE_H_

#include <immintrin.h>

#include "global.h"

namespace upcite {

static inline VALUE_TYPE cal_EDsquare(const VALUE_TYPE *series_1_ptr,
                                      const VALUE_TYPE *series_2_ptr,
                                      ID_TYPE series_length) {

  // QYL 添加指针有效性检查
  if (series_1_ptr == nullptr) {
      fprintf(stderr, "Error: series_1_ptr is Null pointer in cal_EDsquare\n");
      return 1e10;
  }
  if (series_2_ptr == nullptr) {
      fprintf(stderr, "Error: series_2_ptr is Null pointer in cal_EDsquare\n");
      return 1e10;
  }

  VALUE_TYPE distance_square = 0;
  series_length -= 1;
  while (series_length >= 0) {
    distance_square += (series_1_ptr[series_length] - series_2_ptr[series_length])
        * (series_1_ptr[series_length] - series_2_ptr[series_length]);
    series_length--;
  }

  return distance_square;
}

static inline VALUE_TYPE cal_early_EDsquare(const VALUE_TYPE *series_1_ptr,
                                            const VALUE_TYPE *series_2_ptr,
                                            ID_TYPE series_length,
                                            VALUE_TYPE bsf_distance) {
  VALUE_TYPE distance_square = 0;

  series_length -= 1;
  while (series_length >= 0) {
    distance_square += (series_1_ptr[series_length] - series_2_ptr[series_length])
        * (series_1_ptr[series_length] - series_2_ptr[series_length]);

    series_length--;

    if (distance_square > bsf_distance) {
      return distance_square;
    }
  }

  return distance_square;
}

static inline VALUE_TYPE cal_EDsquare_SIMD_8(const VALUE_TYPE *series_1_ptr,
                                             const VALUE_TYPE *series_2_ptr,
                                             ID_TYPE series_length,
                                             VALUE_TYPE *cache) {
  __m256 m256_square_cumulated = _mm256_setzero_ps(), m256_diff, m256_sum, m256_1, m256_2;

  for (ID_TYPE i = 0; i < series_length; i += 8) {
    m256_1 = _mm256_load_ps(series_1_ptr + i);
    m256_2 = _mm256_load_ps(series_2_ptr + i);
    m256_diff = _mm256_sub_ps(m256_1, m256_2);
    m256_square_cumulated = _mm256_fmadd_ps(m256_diff, m256_diff, m256_square_cumulated);
  }

  m256_sum = _mm256_hadd_ps(m256_square_cumulated, m256_square_cumulated);
  _mm256_store_ps(cache, _mm256_hadd_ps(m256_sum, m256_sum));

  return cache[0] + cache[4];
}

VALUE_TYPE inline cal_early_EDsquare_SIMD_8(const VALUE_TYPE *series_1_ptr,
                                            const VALUE_TYPE *series_2_ptr,
                                            ID_TYPE series_length,
                                            VALUE_TYPE *cache,
                                            VALUE_TYPE bsf_distance) {
  VALUE_TYPE partial_EDsquare = 0;
  __m256 m256_square, m256_diff, m256_sum, m256_1, m256_2;

  for (unsigned int i = 0; i < series_length; i += 8) {
    m256_1 = _mm256_load_ps(series_1_ptr + i);
    m256_2 = _mm256_load_ps(series_2_ptr + i);

    m256_diff = _mm256_sub_ps(m256_1, m256_2);
    m256_square = _mm256_mul_ps(m256_diff, m256_diff);
    m256_sum = _mm256_hadd_ps(m256_square, m256_square);

    _mm256_store_ps(cache, _mm256_hadd_ps(m256_sum, m256_sum));
    partial_EDsquare += (cache[0] + cache[4]);

    if (partial_EDsquare > bsf_distance) {
      return partial_EDsquare;
    }
  }

  return partial_EDsquare;
}

}

#endif //DSTREE_SRC_EXEC_DISTANCE_H_
