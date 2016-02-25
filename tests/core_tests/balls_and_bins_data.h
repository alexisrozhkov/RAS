//
// Created by alexey on 26.02.2016.
//

#ifndef RAS_BALLS_AND_BINS_DATA_H
#define RAS_BALLS_AND_BINS_DATA_H

#include <vector>

const std::vector<int> expectedVals[3][3] = {
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

#endif //RAS_BALLS_AND_BINS_DATA_H
