#include "producttextcleaner.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_set>

std::string ProductTextCleaner::DBC2SBC(const std::string& input)
{
    std::string result;
    result.reserve(input.size());

    for (size_t i = 0; i < input.size();) {
        // 检查 UTF-8 三字节全角字符 (EF 85 xx 或 EF 86 xx 范围)
        if (i + 2 < input.size() && (unsigned char)input[i] == 0xEF && (unsigned char)input[i + 1] == 0xBC) {

            unsigned char third = (unsigned char)input[i + 2];
            if (third >= 0x81 && third <= 0xBF) {
                // 全角 ASCII 符号/数字/字母 -> 半角 ASCII
                char sbc = third - 0x81 + '!';
                result.push_back(sbc);
                i += 3;
                continue;
            }
        }
        // 处理全角空格 (E3 80 80 -> ' ')
        if (i + 2 < input.size() && (unsigned char)input[i] == 0xE3 && (unsigned char)input[i + 1] == 0x80 && (unsigned char)input[i + 2] == 0x80) {
            result.push_back(' ');
            i += 3;
            continue;
        }

        result.push_back(input[i]);
        i++;
    }
    return result;
}

std::string ProductTextCleaner::removePunctuationAndSymbols(const std::string& input)
{
    std::string result;
    result.reserve(input.size());

    // 1. 需要剔除的 UTF-8 中文标点/符号集合（按完整 UTF-8 字符串存储）
    static const std::unordered_set<std::string> cn_puncts = {
        "【", "】", "（", "）", "！", "？", "￥", "：", "；", "“", "”", "‘", "’", "，", "。", "、"
    };

    size_t i = 0;
    while (i < input.size()) {
        unsigned char c = input[i];

        // --- A. 单字节 ASCII 字符处理 (0x00 - 0x7F) ---
        if (c < 0x80) {
            // 过滤换行符、制表符以及常见 ASCII 标点 (保留字母、数字、空格)
            if (c == '\r' || c == '\n' || c == '\t' || c == '!' || c == '?' || c == '$' || c == '&' || c == '*' || c == '=' || c == '_' || c == '+' || c == '-' || c == '|' || c == '\\' || c == '/' || c == ',' || c == '.' || c == ';' || c == ':' || c == '"' || c == '\'' || c == '[' || c == ']' || c == '(' || c == ')') {
                result.push_back(' '); // 替换为空格
            } else {
                result.push_back(c); // 保留普通字符（字母、数字、常规空格）
            }
            i += 1;
        }
        // --- B. 多字节 UTF-8 字符处理 (如中文、中文标点) ---
        else {
            // 获取当前 UTF-8 字符的字节长度
            size_t len = 1;
            if ((c & 0xE0) == 0xC0)
                len = 2;
            else if ((c & 0xF0) == 0xE0)
                len = 3; // 绝大多数汉字和中文标点是 3 字节
            else if ((c & 0xF8) == 0xF0)
                len = 4;

            if (i + len <= input.size()) {
                std::string sub = input.substr(i, len);
                // 检查是否属于中文标点
                if (cn_puncts.count(sub)) {
                    result.push_back(' '); // 中文标点替换为空格
                } else {
                    result.append(sub); // 正常汉字原样保留
                }
            }
            i += len;
        }
    }

    return result;
}

std::string ProductTextCleaner::removeStopWords(std::string text)
{
    for (const auto& word : stop_words_) {
        size_t pos = 0;
        while ((pos = text.find(word, pos)) != std::string::npos) {
            text.replace(pos, word.length(), " ");
        }
    }
    return text;
}

std::string ProductTextCleaner::normalizeUnitsAndCase(std::string text)
{
    // 转小写以方便统一正则匹配
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);

    // 规范存储单位 (如 256g/256gb -> 256gb)
    std::regex gb_regex(R"(\b(\d+)\s*g\b)");
    text = std::regex_replace(text, gb_regex, "$1gb");

    // 规范多余连续空格为单个空格
    std::regex multi_space(R"(\s+)");
    text = std::regex_replace(text, multi_space, " ");

    // Trim 首尾空格
    size_t start = text.find_first_not_of(" ");
    size_t end = text.find_last_not_of(" ");
    if (start == std::string::npos)
        return "";
    return text.substr(start, end - start + 1);
}

std::string ProductTextCleaner::clean(const std::string& raw_ocr_text)
{
    std::string text = DBC2SBC(raw_ocr_text); // 1. 全角转半角
    text = removePunctuationAndSymbols(text); // 2. 去除干扰标点
    text = removeStopWords(text); // 3. 去除营销词
    text = normalizeUnitsAndCase(text); // 4. 规格与格式标准化

    // 5. 按空格切分，并利用 unordered_set 进行顺过去重
    std::istringstream iss(text);
    std::string token;
    std::unordered_set<std::string> seen_tokens;
    std::string result;
    result.reserve(text.size());

    while (iss >> token) { // istringstream 会自动忽略多余的连续空格
        // 如果这个词是第一次出现，才拼接回结果字符串
        if (seen_tokens.find(token) == seen_tokens.end()) {
            seen_tokens.insert(token);
            if (!result.empty()) {
                result += "_"; // 词之间用单个空格隔开
            }
            result += token;
        }
    }

    return result;
}
