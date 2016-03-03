// Copyright 2016 Alexey Rozhkov

#ifndef TESTS_CORE_TESTS_BALLS_AND_BINS_DATA_H_
#define TESTS_CORE_TESTS_BALLS_AND_BINS_DATA_H_

#include <core/utils/ras_types.h>
#include <vector>


const std::vector<IndValT> expectedVals[3][3] = {
  {
    {1},

    {1, 0,
     0, 1},

    {1, 0, 0,
     0, 1, 0,
     0, 0, 1}
  },

  {
    {2},

    {2, 0,
     1, 1,
     0, 2},

    {2, 0, 0,
     1, 1, 0,
     1, 0, 1,
     0, 2, 0,
     0, 1, 1,
     0, 0, 2}
  },

  {
    {3},

    {3, 0,
     2, 1,
     1, 2,
     0, 3},

    {3, 0, 0,
     2, 1, 0,
     2, 0, 1,
     1, 2, 0,
     1, 1, 1,
     1, 0, 2,
     0, 3, 0,
     0, 2, 1,
     0, 1, 2,
     0, 0, 3}
  }
};

#endif  // TESTS_CORE_TESTS_BALLS_AND_BINS_DATA_H_
