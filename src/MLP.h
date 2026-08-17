#pragma once

#include "Mandrill.h"

#include "SpecConst.h"

using namespace Mandrill;

struct BufferAddressMLP {
    VkDeviceAddress params;
    VkDeviceAddress weightOffsets;
    VkDeviceAddress biasOffsets;
};

class MLP
{
public:
    MANDRILL_NON_COPYABLE(MLP)

    MLP(ptr<Device> pDevice, std::filesystem::path mlpPath, std::vector<SpecConst>& specializationConstants);
    ~MLP();

    /// Attach the parameters and the triplanes to the shader that runs the inference. Which set and binding they end
    /// up in is taken from the shader, so only the names have to match.
    void attachTo(ptr<Shader> pShader);

    /// Bind the set the attached resources live in. Call after attachTo().
    void bind(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint);

    std::filesystem::path getPath() const
    {
        return mPath;
    }

private:
    void loadFromFile(std::filesystem::path mlpPath, std::vector<SpecConst>& specializationConstants);

    ptr<Device> mpDevice;

    std::filesystem::path mPath;

    size_t coopVecQuerySize(uint32_t rows, uint32_t columns);
    size_t coopVecConvert(void* dstData, size_t dstSize, void* srcData, size_t srcSize, uint32_t rows,
                          uint32_t columns);
    void setupCooperativeVector(bool printDebug = false);

    // Cooperative vector params
    PFN_vkConvertCooperativeVectorMatrixNV mpfnVkConvertCooperativeVectorMatrixNV;
    ptr<Buffer> mpParams;
    ptr<Buffer> mpWeightOffsets;
    ptr<Buffer> mpBiasOffsets;
    std::vector<std::vector<ptr<Texture>>> mTriplaneTextures;

    // Param buffer locations, and the shader they are attached to
    ptr<Buffer> mpLocations;
    ptr<Shader> mpShader;
};
