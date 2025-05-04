//
// Created by Qitong Wang on 2024/3/26.
// Copyright (c) 2024 Université Paris Cité. All rights reserved.
//

#include "sort.h"


ID_TYPE upcite::bSearchFloorID(ID_TYPE target, ID_TYPE const *sorted, ID_TYPE first_inclusive, ID_TYPE last_inclusive) {
  while (first_inclusive + 1 < last_inclusive) {
    unsigned int mid = (first_inclusive + last_inclusive) >> 1u;

    if (target < sorted[mid]) {
      last_inclusive = mid;
    } else {
      first_inclusive = mid;
    }
  }

  return first_inclusive;
}