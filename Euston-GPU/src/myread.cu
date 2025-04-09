#include "myread.cuh"
#include <filesystem>
#include <unistd.h>
#include <linux/limits.h>
namespace fs = std::filesystem;

std::string getExecutablePath() {
    char path[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
    return (count != -1) ? std::string(path, count) : "";
}
string getDataPath()
{
	fs::path exeDir = fs::path(getExecutablePath()).parent_path();

	// 计算相对路径（例如：../../Data）
	fs::path dataPath = exeDir.parent_path().parent_path().parent_path() / "Data";
	dataPath = fs::absolute(dataPath);
	if (!fs::exists(dataPath)) {
		std::cerr << "Data directory not found: " << dataPath << std::endl;
	}
	return dataPath.string();
}

string dname = getDataPath();
// 辅助函数：去除字符串首尾的空白字符
string trim(const string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    if (start == string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// 辅助函数：按空格分割字符串
vector<string> split(const string& line) {
    vector<string> tokens;
    istringstream iss(line);
    string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

/**
 * @brief 读取文本文件为二维 double 数组（类似 np.loadtxt）
 * @param filename 输入文件名
 * @return vector<vector<double>> 二维数据容器
 */
vector<vector<double>> loadtxt(const string& filename, int rows) {
    vector<vector<double>> data;
    ifstream file(filename);

    // 检查文件是否打开成功
    if (!file.is_open()) {
        cerr << "Error: Unable to open file '" << filename << "'" << endl;
        return data;  // 返回空容器
    }

    string line;
    while (getline(file, line)) {
        // 去除首尾空白字符
        line = trim(line);
        if (line.empty()) continue;  // 跳过空行

        // 分割行内容为字符串数组
        vector<string> tokens = split(line);
        vector<double> row;

        // 转换每个 token 为 double
        bool is_valid = true;
        for (const string& token : tokens) {
            try {
                row.push_back(stod(token));  // 字符串转 double
            } catch (const invalid_argument& e) {
                cerr << "Error: Invalid numeric value in line: " << line << endl;
                is_valid = false;
                break;
            } catch (const out_of_range& e) {
                cerr << "Error: Out-of-range value in line: " << line << endl;
                is_valid = false;
                break;
            }
        }

        // 仅保存有效行
        if (is_valid && !row.empty()) {
            data.push_back(row);
			if(rows != -1)
			{
				if(data.size() == rows)
				{
					break;
				}
			}
        }
    }

    file.close();
    return data;
}
