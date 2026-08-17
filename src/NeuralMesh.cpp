#include "NeuralMesh.h"

#include "SpecConst.h"

static const std::string kVertexShaderPath = "Clouds/Cloud.vert";
static const std::string kFragmentShaderPath = "Clouds/Cloud.frag";

static VkSpecializationInfo initializeSpecialisationConstants(std::vector<SpecConst>& specializationConstants,
                                                              std::vector<VkSpecializationMapEntry>& mapEntries)
{
    specializationConstants.emplace_back(SpecConst(8u));
    specializationConstants.emplace_back(SpecConst(4u));
    specializationConstants.emplace_back(SpecConst(4u));
    specializationConstants.emplace_back(SpecConst(32u));
    specializationConstants.emplace_back(SpecConst(0u));
    specializationConstants.emplace_back(SpecConst(512u));
    specializationConstants.emplace_back(SpecConst(8u));

    mapEntries.resize(specializationConstants.size());
    for (uint32_t i = 0; i < mapEntries.size(); i++) {
        mapEntries[i] = {
            .constantID = i,
            .offset = i * static_cast<uint32_t>(sizeof(SpecConst)),
            .size = sizeof(SpecConst),
        };
    }
    VkSpecializationInfo info = {
        .mapEntryCount = count(mapEntries),
        .pMapEntries = mapEntries.data(),
        .dataSize = specializationConstants.size() * sizeof(SpecConst),
        .pData = specializationConstants.data(),
    };
    return info;
}

NeuralMesh::NeuralMesh(ptr<Device> pDevice, ptr<Pass> pPass, ptr<Scene> pScene, const std::filesystem::path& modelPath)
    : mpDevice(pDevice), mpPass(pPass), mpScene(pScene)
{
    // Default values for specialization constants
    mSpecializationInfo = initializeSpecialisationConstants(mSpecializationConstants, mSpecializationMapEntries);

    // Create shader
    std::vector<ShaderDesc> shaderDesc;
    shaderDesc.emplace_back(kVertexShaderPath, "main", VK_SHADER_STAGE_VERTEX_BIT);
    shaderDesc.emplace_back(kFragmentShaderPath, "main", VK_SHADER_STAGE_FRAGMENT_BIT, &mSpecializationInfo);
    mpShader = mpDevice->createShader(shaderDesc);

    // Create pipeline
    PipelineDesc desc;
    desc.depthTestEnable = VK_TRUE;
    desc.depthWriteEnable = VK_FALSE;
    desc.depthCompareOp = VK_COMPARE_OP_EQUAL; // Only render front faces
    desc.colorBlendAttachmentStates[0].blendEnable = VK_TRUE;
    desc.colorBlendAttachmentStates[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

    mpPipeline = mpDevice->createPipeline(mpPass, mpShader, desc);
    mpPipeline->setFrontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE);
    mpPipeline->setCullMode(VK_CULL_MODE_NONE);

    mModelName = modelPath.stem().string();

    // Extract paths for MLP, OBJ and bounds
    std::ifstream is(modelPath, std::ios::binary);
    if (!is.is_open()) {
        Log::Error("Unable to open {}", modelPath.string());
        return;
    }

    auto trim = [](const std::string& s) -> std::string {
        auto wsfront = std::find_if_not(s.begin(), s.end(), [](int c) { return std::isspace(c); });
        auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c) { return std::isspace(c); }).base();
        return (wsback <= wsfront ? std::string() : std::string(wsfront, wsback));
    };

    std::string mlpPath;
    std::string objPath;
    std::string boundsPath;
    std::getline(is, mlpPath);
    std::getline(is, objPath);
    std::getline(is, boundsPath);
    is.close();

    mlpPath = trim(mlpPath);
    objPath = trim(objPath);
    boundsPath = trim(boundsPath);

    // If paths start with . prepend with model path
    auto prependIfRelative = [modelPath](std::string& path) {
        if (path[0] == '.' && modelPath.has_parent_path()) {
            path = (modelPath.parent_path() / path).string();
        }
    };

    prependIfRelative(mlpPath);
    prependIfRelative(objPath);
    prependIfRelative(boundsPath);

    mpMLP = make_ptr<MLP>(mpDevice, mlpPath, mSpecializationConstants);

    // Reload pipeline so specialization constants are updated for loaded MLP
    mpPipeline->recreate();

    // Attach after the reload, so the resources are resolved against the reflection of the recompiled shader
    mpMLP->attachTo(mpShader);

    // Load OBJ file and add to scene
    auto meshIndices = mpScene->addMeshFromFile(objPath);
    mNodeIndex = mpScene->addNode();
    auto& node = mpScene->getNode(mNodeIndex);
    node.setPipeline(mpPipeline);
    for (auto meshIndex : meshIndices) {
        node.addMesh(meshIndex);
    }

    // Read bounds from file
    is.clear();
    is.open(boundsPath);
    if (!is.is_open()) {
        Log::Error("Unable to open {}", modelPath.string());
        return;
    }

    // Skip first line since it contains the header
    is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    float minX, minY, minZ, maxX, maxY, maxZ;
    char comma;
    if (is >> minX >> comma >> minY >> comma >> minZ >> comma >> maxX >> comma >> maxY >> comma >> maxZ) {
        mMinBounds = glm::vec3(minX, minY, minZ);
        mMaxBounds = glm::vec3(maxX, maxY, maxZ);
    }
}

NeuralMesh::~NeuralMesh()
{
}

void NeuralMesh::bindMLP(VkCommandBuffer cmd)
{
    mpMLP->bind(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
}
