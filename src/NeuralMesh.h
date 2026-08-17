#pragma once

#include "Mandrill.h"

#include "MLP.h"

using namespace Mandrill;

class NeuralMesh
{
public:
    MANDRILL_NON_COPYABLE(NeuralMesh)
    NeuralMesh(ptr<Device> pDevice, ptr<Pass> pPass, ptr<Scene> pScene, const std::filesystem::path& modelPath);
    ~NeuralMesh();

    void bindMLP(VkCommandBuffer cmd);

    ptr<Pipeline> getPipeline() const
    {
        return mpPipeline;
    }

    const std::string& getModelName() const
    {
        return mModelName;
    }

    uint32_t getNodeIndex() const
    {
        return mNodeIndex;
    }

    ptr<MLP> getMLP() const
    {
        return mpMLP;
    }

    glm::vec3 getMinBounds() const
    {
        return mMinBounds;
    }

    glm::vec3 getMaxBounds() const
    {
        return mMaxBounds;
    }

private:
    ptr<Device> mpDevice;
    ptr<Pass> mpPass;
    ptr<Shader> mpShader;
    ptr<Pipeline> mpPipeline;
    ptr<Scene> mpScene;

    std::string mModelName;
    uint32_t mNodeIndex = 0;

    ptr<MLP> mpMLP;
    glm::vec3 mMinBounds;
    glm::vec3 mMaxBounds;

    std::vector<SpecConst> mSpecializationConstants;
    std::vector<VkSpecializationMapEntry> mSpecializationMapEntries;
    VkSpecializationInfo mSpecializationInfo;
};
