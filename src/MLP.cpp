#include "MLP.h"

#include "glm/detail/type_half.hpp"

#define COOP_VECTOR_TYPE VK_COMPONENT_TYPE_FLOAT16_KHR

MLP::MLP(ptr<Device> pDevice, std::filesystem::path mlpPath, std::vector<SpecConst>& specializationConstants)
    : mpDevice(pDevice)
{
    setupCooperativeVector();
    loadFromFile(mlpPath, specializationConstants);
}

MLP::~MLP()
{
}

void MLP::loadFromFile(std::filesystem::path mlpPath, std::vector<SpecConst>& specializationConstants)
{
    const size_t matrixAlignment = 64;
    const size_t vectorAlignment = 16;

    mPath = mlpPath;

    std::ifstream is(mlpPath, std::ios::binary);
    if (!is.is_open()) {
        Log::Error("Unable to open {}", mlpPath.string());
        return;
    }

    // Read MLP parameters from file
    uint32_t version = 0;
    uint32_t triplaneResolution = 0;
    uint32_t triplaneFeatureDim = 0;
    std::vector<float> triplanes;
    uint32_t layerCount = 0;

    is.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&triplaneResolution), sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&triplaneFeatureDim), sizeof(uint32_t));

    uint32_t triplaneSize = triplaneResolution * triplaneResolution * triplaneFeatureDim;
    triplanes.resize(3 * triplaneSize);
    for (uint32_t i = 0; i < 3 * triplaneSize; i++) {
        float value = 0.0f;
        is.read(reinterpret_cast<char*>(&value), sizeof(float));
        triplanes[i] = value;
    }

    std::vector<uint32_t> triplaneOffsets = {0, triplaneSize, 2 * triplaneSize};

    is.read(reinterpret_cast<char*>(&layerCount), sizeof(uint32_t));

    std::vector<uint32_t> inputs(layerCount);
    std::vector<uint32_t> outputs(layerCount);
    std::vector<std::vector<float>> weights(layerCount);
    std::vector<std::vector<float>> biases(layerCount);
    std::vector<size_t> weightSizes(layerCount);
    std::vector<uint32_t> weightOffsets(layerCount);
    std::vector<size_t> biasSizes(layerCount);
    std::vector<uint32_t> biasOffsets(layerCount);
    size_t offset = 0;

    for (uint32_t i = 0; i < layerCount; i++) {
        is.read(reinterpret_cast<char*>(&outputs[i]), sizeof(uint32_t));
        is.read(reinterpret_cast<char*>(&inputs[i]), sizeof(uint32_t));

        uint32_t weightCount = inputs[i] * outputs[i];
        for (uint32_t w = 0; w < weightCount; w++) {
            float weight = 0.0f;
            is.read(reinterpret_cast<char*>(&weight), sizeof(float));
            weights[i].push_back(weight);
        }

        uint32_t biasCount;
        is.read(reinterpret_cast<char*>(&biasCount), sizeof(uint32_t));
        for (uint32_t b = 0; b < biasCount; b++) {
            float bias = 0.0f;
            is.read(reinterpret_cast<char*>(&bias), sizeof(float));
            biases[i].push_back(bias);
        }

        weightSizes[i] = coopVecQuerySize(outputs[i], inputs[i]);
        biasSizes[i] = outputs[i] * sizeof(float) / 2;

        offset = Helpers::alignTo(offset, matrixAlignment);
        weightOffsets[i] = static_cast<uint32_t>(offset);
        offset += weightSizes[i];

        offset = Helpers::alignTo(offset, vectorAlignment);
        biasOffsets[i] = static_cast<uint32_t>(offset);
        offset += biasSizes[i];
    }

    is.close();

    std::vector<uint8_t> params(offset, 0);

    for (uint32_t i = 0; i < layerCount; i++) {
        coopVecConvert(params.data() + weightOffsets[i], weightSizes[i], weights[i].data(),
                       weights[i].size() * sizeof(float), outputs[i], inputs[i]);
        std::vector<uint16_t> bias16(biases[i].size());
        std::transform(biases[i].begin(), biases[i].end(), bias16.begin(),
                       [](float v) { return glm::detail::toFloat16(v); });
        std::memcpy(params.data() + biasOffsets[i], bias16.data(), biasSizes[i]);
    }

    // Setup and transfer parameters
    mpParams = mpDevice->createBuffer(params.size(),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    mpParams->copyFromHost(params.data(), params.size());

    mpWeightOffsets = mpDevice->createBuffer(weightOffsets.size() * sizeof(uint32_t),
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    mpWeightOffsets->copyFromHost(weightOffsets.data(), weightOffsets.size() * sizeof(uint32_t));

    mpBiasOffsets = mpDevice->createBuffer(weightOffsets.size() * sizeof(uint32_t),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    mpBiasOffsets->copyFromHost(biasOffsets.data(), biasOffsets.size() * sizeof(uint32_t));

    // Setup and transfer location buffer
    mpLocations = mpDevice->createBuffer(sizeof(BufferAddressMLP),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    BufferAddressMLP loc = {
        .params = mpParams->getDeviceAddress(),
        .weightOffsets = mpWeightOffsets->getDeviceAddress(),
        .biasOffsets = mpBiasOffsets->getDeviceAddress(),
    };

    mpLocations->copyFromHost(&loc, sizeof(BufferAddressMLP));

    // Setup triplane textures
    mTriplaneTextures.clear();
    mTriplaneTextures.resize(3);
    for (uint32_t plane = 0; plane < 3; plane++) {
        for (uint32_t dim = 0; dim < triplaneFeatureDim; dim += 4) {
            std::vector<float> planeData(triplaneResolution * triplaneResolution * 4);
            for (uint32_t y = 0; y < triplaneResolution; y++) {
                for (uint32_t x = 0; x < triplaneResolution; x++) {
                    for (uint32_t c = 0; c < 4; c++) {
                        uint32_t featureIndex = dim + c;
                        if (featureIndex < triplaneFeatureDim) {
                            uint32_t index = triplaneOffsets[plane] +
                                             (y * triplaneResolution + x) * triplaneFeatureDim + featureIndex;
                            planeData[(y * triplaneResolution + x) * 4 + c] = triplanes[index];
                        } else {
                            planeData[(y * triplaneResolution + x) * 4 + c] = 0.0f;
                        }
                    }
                }
            }
            const uint32_t depth = 1;
            const uint32_t bytesPerPixel = 4 * sizeof(float);
            const bool mipmap = true;
            mTriplaneTextures[plane].push_back(mpDevice->createTextureFromBuffer(
                TextureType::Texture2D, VK_FORMAT_R32G32B32A32_SFLOAT, planeData.data(), triplaneResolution,
                triplaneResolution, depth, bytesPerPixel, mipmap));
            mTriplaneTextures[plane].back()->setAddressMode(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                                            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                                            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        }
    }

    // Set specialization constants for the MLP shader code (shader will need to be recompiled)
    specializationConstants[0].uintValue = inputs[0];               // Input of MLP
    specializationConstants[1].uintValue = outputs[layerCount - 1]; // Output of MLP
    specializationConstants[2].uintValue = layerCount;              // Number of layers in MLP
    specializationConstants[3].uintValue = inputs[1];               // Number of neurons in a hidden layer
    specializationConstants[4].uintValue = version;                 // Version
    specializationConstants[5].uintValue = triplaneResolution;      // Size of triplane
    specializationConstants[6].uintValue = triplaneFeatureDim;      // Number of features per triplane
}

size_t MLP::coopVecQuerySize(uint32_t rows, uint32_t columns)
{
    size_t dstSize = 0;

    VkConvertCooperativeVectorMatrixInfoNV info = {
        .sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV,
        .srcSize = sizeof(float) * rows * columns,
        .pDstSize = &dstSize,
        .srcComponentType = VK_COMPONENT_TYPE_FLOAT32_KHR,
        .dstComponentType = COOP_VECTOR_TYPE,
        .numRows = rows,
        .numColumns = columns,
        .srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV,
        .srcStride = sizeof(float) * columns,
        .dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV,
    };
    mpfnVkConvertCooperativeVectorMatrixNV(mpDevice->getDevice(), &info);

    return dstSize;
}

size_t MLP::coopVecConvert(void* dstData, size_t dstSize, void* srcData, size_t srcSize, uint32_t rows,
                           uint32_t columns)
{
    VkConvertCooperativeVectorMatrixInfoNV info = {
        .sType = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV,
        .srcSize = sizeof(float) * rows * columns,
        .srcData = {.hostAddress = srcData},
        .pDstSize = &dstSize,
        .dstData = {.hostAddress = dstData},
        .srcComponentType = VK_COMPONENT_TYPE_FLOAT32_KHR,
        .dstComponentType = COOP_VECTOR_TYPE,
        .numRows = rows,
        .numColumns = columns,
        .srcLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV,
        .srcStride = sizeof(float) * columns,
        .dstLayout = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV,
        .dstStride = 0,
    };

    mpfnVkConvertCooperativeVectorMatrixNV(mpDevice->getDevice(), &info);

    return dstSize;
}

void MLP::setupCooperativeVector(bool printDebug)
{
    PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV coopVecPropsFunc =
        (PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV)vkGetInstanceProcAddr(
            mpDevice->getInstance(), "vkGetPhysicalDeviceCooperativeVectorPropertiesNV");

    if (coopVecPropsFunc) {
        std::map<VkComponentTypeKHR, std::string> types = {
            {VK_COMPONENT_TYPE_FLOAT16_KHR, "VK_COMPONENT_TYPE_FLOAT16_KHR"},
            {VK_COMPONENT_TYPE_FLOAT32_KHR, "VK_COMPONENT_TYPE_FLOAT32_KHR"},
            {VK_COMPONENT_TYPE_FLOAT64_KHR, "VK_COMPONENT_TYPE_FLOAT64_KHR"},
            {VK_COMPONENT_TYPE_SINT8_KHR, "VK_COMPONENT_TYPE_SINT8_KHR"},
            {VK_COMPONENT_TYPE_SINT16_KHR, "VK_COMPONENT_TYPE_SINT16_KHR"},
            {VK_COMPONENT_TYPE_SINT32_KHR, "VK_COMPONENT_TYPE_SINT32_KHR"},
            {VK_COMPONENT_TYPE_SINT64_KHR, "VK_COMPONENT_TYPE_SINT64_KHR"},
            {VK_COMPONENT_TYPE_UINT8_KHR, "VK_COMPONENT_TYPE_UINT8_KHR"},
            {VK_COMPONENT_TYPE_UINT16_KHR, "VK_COMPONENT_TYPE_UINT16_KHR"},
            {VK_COMPONENT_TYPE_UINT32_KHR, "VK_COMPONENT_TYPE_UINT32_KHR"},
            {VK_COMPONENT_TYPE_UINT64_KHR, "VK_COMPONENT_TYPE_UINT64_KHR"},
            {VK_COMPONENT_TYPE_SINT8_PACKED_NV, "VK_COMPONENT_TYPE_SINT8_PACKED_NV"},
            {VK_COMPONENT_TYPE_UINT8_PACKED_NV, "VK_COMPONENT_TYPE_UINT8_PACKED_NV"},
            {VK_COMPONENT_TYPE_FLOAT_E4M3_NV, "VK_COMPONENT_TYPE_FLOAT_E4M3_NV"},
            {VK_COMPONENT_TYPE_FLOAT_E5M2_NV, "VK_COMPONENT_TYPE_FLOAT_E5M2_NV"},
        };

        if (printDebug) {
            Log::Debug("Cooperative vector supported with following types");
            uint32_t n;
            Check::Vk(coopVecPropsFunc(mpDevice->getPhysicalDevice(), &n, nullptr));
            std::vector<VkCooperativeVectorPropertiesNV> props(n);
            std::for_each(props.begin(), props.end(),
                          [](auto& prop) { prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV; });
            Check::Vk(coopVecPropsFunc(mpDevice->getPhysicalDevice(), &n, props.data()));
            for (uint32_t i = 0; i < count(props); i++) {
                auto& p = props[i];
                Log::Debug("Set {}", i);
                Log::Debug("\tinput type:            {}\n"
                           "\tinput interpretation:  {}\n"
                           "\tmatrix interpretation: {}\n"
                           "\tbias interpretation:   {}\n"
                           "\tresult type:           {}\n"
                           "\ttranspose:             {}",
                           types[p.inputType], types[p.inputInterpretation], types[p.matrixInterpretation],
                           types[p.biasInterpretation], types[p.resultType], p.transpose ? "true" : "false");
            }
        }
    } else {
        Log::Error("Cooperative vector not supported");
    }

    mpfnVkConvertCooperativeVectorMatrixNV = (PFN_vkConvertCooperativeVectorMatrixNV)vkGetInstanceProcAddr(
        mpDevice->getInstance(), "vkConvertCooperativeVectorMatrixNV");
}

void MLP::attachTo(ptr<Shader> pShader)
{
    mpShader = pShader;

    pShader->setResource("locMlp", mpLocations);
    pShader->setResource("triplaneXY", mTriplaneTextures[0]);
    pShader->setResource("triplaneXZ", mTriplaneTextures[1]);
    pShader->setResource("triplaneYZ", mTriplaneTextures[2]);
}

void MLP::bind(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint)
{
    if (!mpShader) {
        Log::Error("MLP::bind() - The MLP has not been attached to a shader");
        return;
    }

    // Every resource of the MLP shares one set, so any of them identifies which set to bind
    auto info = mpShader->getResourceInfo("locMlp");
    if (info) {
        mpShader->bindResources(cmd, bindPoint, info->set);
    }
}
