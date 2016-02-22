#include <iostream>
#include <balls_and_bins.h>

using namespace std;


int main() {
  cout << "Hello World!" << endl;

  auto res = balls_and_bins(3, 3, 1);
  for(int i = 0; i < res.size(); i++) std::cout << "------------------\n" << res[i] << std::endl;


  return 0;
}

