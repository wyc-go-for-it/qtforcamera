#ifndef PRODUCTTEXTCLEANER_H
#define PRODUCTTEXTCLEANER_H

#include <algorithm>
#include <regex>
#include <string>
#include <vector>

class ProductTextCleaner {
public:
    ProductTextCleaner()
    {
        // 初始化营销停用词库（支持扩展）
        stop_words_ = {
            "包邮", "现货", "正品", "特价", "限时", "秒杀", "买一送一",
            "官方", "专柜", "热卖", "爆款", "促销", "优惠", "直营"
        };
    }

private:
    // 1. 全角字符转半角字符 (ASCII 范围及全角空格)
    std::string DBC2SBC(const std::string& input);

    // 2. 去除不可见字符与常见修饰性标点符号
    std::string removePunctuationAndSymbols(const std::string& input);

    // 3. 剔除营销噪声词
    std::string removeStopWords(std::string text);

    // 4. 规格单位与品牌大小写标准化
    std::string normalizeUnitsAndCase(std::string text);

public:
    std::string clean(const std::string& raw_ocr_text);

private:
    std::vector<std::string> stop_words_;
};

#endif // PRODUCTTEXTCLEANER_H
