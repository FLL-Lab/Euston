#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>  // 用于字符串 trim 处理

using namespace std;

extern string dname;

string trim(const string& s);
vector<string> split(const string& line);
vector<vector<double>> loadtxt(const string& filename, int rows=-1);