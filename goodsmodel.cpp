#include "goodsmodel.h"

#include "productfeatureengine.h"
#include <fstream>
#include <iostream>

void GoodsModel::loadData()
{
    std::ifstream ifs("./model/data.bat", std::ios::binary);
    if (!ifs.is_open())
        return;

    lst.clear();
    uint32_t count = 0;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));

    for (uint32_t i = 0; i < count; ++i) {
        ProductRecord item;
        ifs.read(reinterpret_cast<char*>(&item.id), sizeof(item.id));

        uint32_t nameLen = 0;
        ifs.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        item.name.resize(nameLen);
        ifs.read(&item.name[0], nameLen);

        uint32_t codeLen = 0;
        ifs.read(reinterpret_cast<char*>(&codeLen), sizeof(codeLen));
        item.barcode.resize(codeLen);
        ifs.read(&item.barcode[0], codeLen);

        item.feature.resize(ProductFeatureEngine::featureDim);
        ifs.read(reinterpret_cast<char*>(item.feature.data()), ProductFeatureEngine::featureDim * sizeof(float));

        if (!lst.contains(item)) {
            lst.push_back(item);
        }
    }
}
