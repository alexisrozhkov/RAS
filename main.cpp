#include <iostream>
#include <perspective_embedding.h>

using namespace std;

//data = [100, 400, 10, 0; 100, 500, 10, 50; 200, 600, 20, 60; 100, 700, 10, 70; 1, 1, 1, 1]

int main() {
  cv::Mat matches = cv::Mat::zeros(5, 4, CV_64F);
  matches.at<double>(0, 0) = 100;
  matches.at<double>(1, 0) = 100;
  matches.at<double>(2, 0) = 200;
  matches.at<double>(3, 0) = 100;
  matches.at<double>(4, 0) = 1;

  matches.at<double>(0, 1) = 400;
  matches.at<double>(1, 1) = 500;
  matches.at<double>(2, 1) = 600;
  matches.at<double>(3, 1) = 700;
  matches.at<double>(4, 1) = 1;

  matches.at<double>(0, 2) = 10;
  matches.at<double>(1, 2) = 10;
  matches.at<double>(2, 2) = 20;
  matches.at<double>(3, 2) = 10;
  matches.at<double>(4, 2) = 1;

  matches.at<double>(0, 3) = 0;
  matches.at<double>(1, 3) = 50;
  matches.at<double>(2, 3) = 60;
  matches.at<double>(3, 3) = 70;
  matches.at<double>(4, 3) = 1;

  auto V = std::get<0>(perspective_embedding(matches, 2, true, 3));

  for(int i = 0; i < V.size(); i++) {
    std::cout << V[i] << "\n------------------\n";
  }

  return 0;
}

