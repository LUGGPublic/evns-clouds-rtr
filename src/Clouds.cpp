#include "Mandrill.h"

#include "NeuralMesh.h"

#include <numbers>

using namespace Mandrill;

class Clouds : public App
{
public:
    const uint32_t SHADOW_MAP_RESOLUTION = 1024;

    enum {
        PIPELINE_SKY,
        PIPELINE_SHADOW,
        PIPELINE_CLOUD_BACK_POS,
        PIPELINE_CLOUD_FRONT_DEPTH,
        PIPELINE_RESOLVE,
        PIPELINE_COUNT,
    };

    enum {
        TIMESTAMP_CLOUD_BEGIN,
        TIMESTAMP_CLOUD_END,
        TIMESTAMP_COUNT,
    };

    struct PushConstant {
        glm::vec2 viewPort;
        float sunElevation;
        float sunRotation;
        glm::vec3 minBounds;
        int renderMode;
        glm::vec3 maxBounds;
        float exposure;
        float triplaneMipmap;
    };

    Clouds(uint32_t physicalDeviceIndex = std::numeric_limits<uint32_t>::max()) : App("Clouds", 1920, 1080)
    {
        // Create a Vulkan instance and device
        std::vector<const char*> extensions = {
            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
            VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME,
            VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME,
        };

        VkPhysicalDeviceVulkan11Features vk11Features = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .storageBuffer16BitAccess = VK_TRUE,
        };

        VkPhysicalDeviceVulkan12Features vk12Features = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .pNext = &vk11Features,
            .uniformAndStorageBuffer8BitAccess = VK_TRUE,
            .shaderFloat16 = VK_TRUE,
            .timelineSemaphore = VK_TRUE,
            .bufferDeviceAddress = VK_TRUE,
            .vulkanMemoryModel = VK_TRUE,
            .vulkanMemoryModelDeviceScope = VK_TRUE,
        };

        VkPhysicalDeviceVulkan13Features vk13Features = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &vk12Features,
            .synchronization2 = VK_TRUE,
            .dynamicRendering = VK_TRUE,
        };

        VkPhysicalDeviceVulkan14Features vk14Features = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
            .pNext = &vk13Features,
            .dynamicRenderingLocalRead = VK_TRUE,
            .pushDescriptor = VK_TRUE,
        };

        VkPhysicalDeviceCooperativeVectorFeaturesNV coopVecFeatures = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
            .pNext = &vk14Features,
            .cooperativeVector = VK_TRUE,
        };

        VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicatedCompositesFeatures = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT,
            .pNext = &coopVecFeatures,
            .shaderReplicatedComposites = VK_TRUE,
        };

        VkPhysicalDeviceFeatures2 features2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &replicatedCompositesFeatures,
            .features =
                {
                    .independentBlend = VK_TRUE,
                    .samplerAnisotropy = VK_TRUE,
                    .vertexPipelineStoresAndAtomics = VK_TRUE,
                    .fragmentStoresAndAtomics = VK_TRUE,
                    .shaderInt64 = VK_TRUE,
                },
        };

        mpDevice = make_ptr<Device>(mpWindow, extensions, &features2, physicalDeviceIndex);
        mDepthFormat = Helpers::findDepthFormat(mpDevice);

        // Create a swapchain
        mpSwapchain = mpDevice->createSwapchain();

        // Create render graph
        mpGraph = mpDevice->createRenderGraph();
        addGraphResources();

        mpGraph->addPass("Cloud back position", {}, {"cloudBackPosition", "cloudBackDepth"},
                         [this](VkCommandBuffer cmd) { cloudBackPositionPass(cmd); });
        mpGraph->addPass("Shadow map", {}, {"shadowColor", "shadowDepth"},
                         [this](VkCommandBuffer cmd) { shadowPass(cmd); });
        mpGraph->addPass("Cloud", {"cloudBackPosition", "shadowDepth"}, {"cloudDepth", "cloudColor"},
                         [this](VkCommandBuffer cmd) { cloudPass(cmd); });
        mpGraph->addPass("Resolve", {"cloudColor"}, {"output"},
                         [this](VkCommandBuffer cmd) { resolvePass(cmd); });

        mpGraph->compile();

        // Create the passes that render into the graph's resources
        createPasses();

        // Create shader modules
        std::vector<ShaderDesc> shaderDesc;
        shaderDesc.emplace_back("Clouds/Fullscreen.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        shaderDesc.emplace_back("Clouds/Sky.frag", "main", VK_SHADER_STAGE_FRAGMENT_BIT);
        auto pShaderSky = mpDevice->createShader(shaderDesc);

        shaderDesc.clear();
        shaderDesc.emplace_back("Clouds/Cloud.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        shaderDesc.emplace_back("Clouds/Shadow.frag", "main", VK_SHADER_STAGE_FRAGMENT_BIT);
        auto pShaderShadow = mpDevice->createShader(shaderDesc);

        shaderDesc.clear();
        shaderDesc.emplace_back("Clouds/Cloud.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        shaderDesc.emplace_back("Clouds/Position.frag", "main", VK_SHADER_STAGE_FRAGMENT_BIT);
        auto pShaderCloudBack = mpDevice->createShader(shaderDesc);

        shaderDesc.clear();
        shaderDesc.emplace_back("Clouds/Cloud.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        shaderDesc.emplace_back("Clouds/Depth.frag", "main", VK_SHADER_STAGE_FRAGMENT_BIT);
        auto pShaderCloudDepth = mpDevice->createShader(shaderDesc);

        shaderDesc.clear();
        shaderDesc.emplace_back("Clouds/Fullscreen.vert", "main", VK_SHADER_STAGE_VERTEX_BIT);
        shaderDesc.emplace_back("Clouds/Resolve.frag", "main", VK_SHADER_STAGE_FRAGMENT_BIT);
        auto pShaderResolve = mpDevice->createShader(shaderDesc);

        // Create pipelines
        std::vector<VkVertexInputBindingDescription> emptyBindingDescription;
        std::vector<VkVertexInputAttributeDescription> emptyAttributeDescription;

        mPipelines.resize(PIPELINE_COUNT);

        PipelineDesc pipelineDescSky = PipelineDesc(emptyBindingDescription, emptyAttributeDescription);
        pipelineDescSky.depthTestEnable = VK_FALSE;
        pipelineDescSky.depthWriteEnable = VK_FALSE;
        pipelineDescSky.colorBlendAttachmentStates[0].blendEnable = VK_FALSE;
        mPipelines[PIPELINE_SKY] = mpDevice->createPipeline(mpResolvePass, pShaderSky, pipelineDescSky);

        PipelineDesc pipelineDescShadow;
        mPipelines[PIPELINE_SHADOW] = mpDevice->createPipeline(mpShadowPass, pShaderShadow, pipelineDescShadow);

        // The back-most cloud surface is the one that survives a greater-than depth test
        PipelineDesc pipelineDescCloudBackPos;
        pipelineDescCloudBackPos.depthTestEnable = VK_TRUE;
        pipelineDescCloudBackPos.depthCompareOp = VK_COMPARE_OP_GREATER;
        mPipelines[PIPELINE_CLOUD_BACK_POS] =
            mpDevice->createPipeline(mpCloudBackPosPass, pShaderCloudBack, pipelineDescCloudBackPos);
        mPipelines[PIPELINE_CLOUD_BACK_POS]->setFrontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE);
        mPipelines[PIPELINE_CLOUD_BACK_POS]->setCullMode(VK_CULL_MODE_FRONT_BIT);

        PipelineDesc pipelineDescCloudDepth;
        pipelineDescCloudDepth.depthTestEnable = VK_TRUE;
        pipelineDescCloudDepth.depthWriteEnable = VK_TRUE;
        pipelineDescCloudDepth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        mPipelines[PIPELINE_CLOUD_FRONT_DEPTH] =
            mpDevice->createPipeline(mpCloudDepthPass, pShaderCloudDepth, pipelineDescCloudDepth);

        PipelineDesc pipelineDescResolve = PipelineDesc(emptyBindingDescription, emptyAttributeDescription);
        pipelineDescResolve.depthTestEnable = VK_FALSE;
        pipelineDescResolve.depthWriteEnable = VK_FALSE;
        pipelineDescResolve.colorBlendAttachmentStates[0].blendEnable = VK_TRUE;
        mPipelines[PIPELINE_RESOLVE] = mpDevice->createPipeline(mpResolvePass, pShaderResolve, pipelineDescResolve);

        // Create an empty scene
        mpScene = mpDevice->createScene();
        registerCloudPipelines();

        // Setup cameras
        mpCamera = mpDevice->createCamera();
        mpCamera->setNearFar(0.1f, 10000.0f);
        mpCamera->setPosition(glm::vec3(392.0f, 9.0f, 7.0f));
        mpCamera->setTarget(glm::vec3(0.0f, 0.0f, 0.0f));
        mpCamera->setFov(60.0f);
        mpCamera->setProjection(CAMERA_PROJECTION_PERSPECTIVE);
        mpCamera->setMoveSpeed(50.0f);

        mpShadowCamera = mpDevice->createCamera();
        mpShadowCamera->setProjection(CAMERA_PROJECTION_ORTHOGRAPHIC);
        mpShadowCamera->setNearFar(1.0f, 10000.0f);
        mpShadowCamera->setOrthoSize(1000.0f);
        mpShadowCamera->setAspectRatio(1.0f);

        // Hand the graph's images and the camera matrices to the shaders that read them
        setShaderResources();

        // Create query pool for timestamps
        createQueryPoolTimestamp();

        // Initialize GUI, which is drawn by the pass that produces the final image
        App::createGUI(mpDevice, mpResolvePass);
    }

    ~Clouds()
    {
        destroyQueryPoolTimestamp();

        App::destroyGUI(mpDevice);
    }

    void update(float delta) override
    {
        mpSwapchain->waitForFence();

        if (!keyboardCapturedByGUI() && !mouseCapturedByGUI()) {
            mpCamera->update(mpWindow, delta, getCursorDelta());
        }

        if (mCameraTrackAnimation < 1.0f) {
            if (mCameraTrackAnimateRadius) {
                // Decrease the radius over time, to create a zoom-in effect
                mCameraTrackRadius = 4500.0f * (1.0f - mCameraTrackAnimation) + 400.0f * mCameraTrackAnimation;
            }
            cameraTrack(mpCamera, mCameraTrackAnimation);
            mCameraTrackAnimation += mCameraTrackSpeed / 10000.0f * delta;
            mpCamera->update(mpWindow, delta, getCursorDelta());
        }

        // Move shadow camera to match sun position
        const float radius = 1000.0f;
        glm::vec3 position = glm::vec3(radius * cosf(mSunElevation) * cosf(mSunRotation), radius * sinf(mSunElevation),
                                       radius * cosf(mSunElevation) * sinf(mSunRotation));

        mpShadowCamera->setPosition(position);
        mpShadowCamera->setTarget(glm::vec3(0.0f, 0.0f, 0.0f));

        mpShadowCamera->update();
    }

    void render() override
    {
        // Check if camera and graph resources need to be updated
        if (mpSwapchain->recreated()) {
            mpCamera->setAspectRatio(mpSwapchain->getAspectRatio());

            // Adding the resources again at the new extent and compiling again recreates them
            addGraphResources();
            mpGraph->compile();

            updatePasses();
            setShaderResources();
        }

        // Acquire frame from swapchain
        VkCommandBuffer cmd = mpSwapchain->acquireNextImage();

        uint32_t frameInFlightIndex = mpDevice->getFrameInFlightIndex();

        // Reset timestamp counters
        vkCmdResetQueryPool(cmd, mQueryPool, TIMESTAMP_COUNT * frameInFlightIndex, TIMESTAMP_COUNT);

        // The passes are recorded by the graph, so the constants they push are prepared here
        mPushConstant = {
            .viewPort = glm::vec2(mpSwapchain->getExtent().width, mpSwapchain->getExtent().height),
            .sunElevation = mSunElevation,
            .sunRotation = mSunRotation,
            .minBounds = mpNeuralMesh ? mpNeuralMesh->getMinBounds() : glm::vec3(0.0f),
            .renderMode = mRenderMode,
            .maxBounds = mpNeuralMesh ? mpNeuralMesh->getMaxBounds() : glm::vec3(0.0f),
            .exposure = mExposure,
            .triplaneMipmap = mTriplaneMipmap,
        };

        // Record every pass of the graph, in the order the graph resolved
        mpGraph->execute(cmd);

        // Only the cloud passes write timestamps, so a frame without a cloud leaves the queries unavailable
        mTimestampsWritten[frameInFlightIndex] = mpNeuralMesh != nullptr;

        // The output resource was declared with a final layout of transfer source, so it is ready to be presented
        mpSwapchain->present(cmd, mpGraph->getResource("output"));

        // Read out timestamps (frame-in-flight index changed when presenting the swapchain)
        uint32_t previousInFlightIndex = mpSwapchain->getPreviousInFlightIndex();
        if (mTimestampsWritten[previousInFlightIndex]) {
            Check::Vk(vkGetQueryPoolResults(mpDevice->getDevice(), mQueryPool,
                                            TIMESTAMP_COUNT * previousInFlightIndex, TIMESTAMP_COUNT,
                                            sizeof(mTimestamps), &mTimestamps[0], sizeof(uint64_t),
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        }
    }

    void appGUI(ImGuiContext* pContext) override
    {
        ImGui::SetCurrentContext(pContext);

        App::baseGUI(mpDevice, mpSwapchain, allPipelines());

        if (ImGui::Begin("Clouds")) {
            if (ImGui::CollapsingHeader("Clouds", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::Button("+")) {
                    std::string path =
                        Mandrill::OpenFile(mpWindow, "Neural Mesh files (*.neumesh)\0*.neumesh\0All\0*.*\0");
                    if (!path.empty()) {
                        removeNeuralMesh();
                        mpNeuralMesh = make_ptr<NeuralMesh>(mpDevice, mpCloudColorPass, mpScene, path);

                        mpScene->compile();
                        mpScene->createDescriptors(mpCamera);
                        mpScene->syncToDevice();

                        setShaderResources();
                    }
                }

                ImGui::SameLine();
                if (ImGui::Button("-")) {
                    removeNeuralMesh();
                }

                if (mpNeuralMesh) {
                    ImGui::SameLine();
                    ImGui::Text("%s", mpNeuralMesh->getModelName().c_str());
                    float cloudRenderTime = (mTimestamps[TIMESTAMP_CLOUD_END] - mTimestamps[TIMESTAMP_CLOUD_BEGIN]) *
                                            mpDevice->getProperties().physicalDevice.limits.timestampPeriod * 1e-6f;
                    ImGui::Text("Render time: %10.3f ms", cloudRenderTime);

                    if (ImGui::Checkbox("Backface culling", &mCloudBackfaceCulling)) {
                        mpNeuralMesh->getPipeline()->setCullMode(mCloudBackfaceCulling ? VK_CULL_MODE_BACK_BIT
                                                                                       : VK_CULL_MODE_NONE);
                    }
                }

                ImGui::SliderFloat("Triplane mipmap", &mTriplaneMipmap, 0.0f, 8.0f);
            }

            if (ImGui::CollapsingHeader("Misc Settings")) {
                const char* renderModes[] = {"MLP", "Position", "Back", "View", "Thickness"};
                ImGui::Combo("Render mode", &mRenderMode, renderModes, IM_ARRAYSIZE(renderModes));

                ImGui::SliderFloat("Exposure", &mExposure, 0.1f, 5.0f);

                ImGui::SeparatorText("Sun controls");
                ImGui::SliderAngle("Sun Elevation", &mSunElevation, 3.0f, 89.0f);
                ImGui::SliderAngle("Sun Rotation Phi", &mSunRotation, 0.0f, 360.0f);
            }

            if (ImGui::CollapsingHeader("Camera Track Controls")) {
                ImGui::SliderFloat("Camera height", &mCameraTrackHeight, -1000.0f, 1000.0f);
                ImGui::SliderFloat("Camera radius", &mCameraTrackRadius, 300.0f, 2500.0f);
                ImGui::DragFloat3("Camera target", &mCameraTrackTarget.x, 0.1f);
                ImGui::SliderInt("Camera track speed", &mCameraTrackSpeed, -500, 500);
                ImGui::Checkbox("Animate radius", &mCameraTrackAnimateRadius);
                if (ImGui::Button("Animate camera")) {
                    mCameraTrackAnimation = 0.0f;
                    cameraTrack(mpCamera, mCameraTrackAnimation);
                }
            }
        }

        ImGui::End();
    }

    void appKeyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods) override
    {
        App::baseKeyCallback(pWindow, key, scancode, action, mods, mpDevice, mpSwapchain, allPipelines());

        if (action == GLFW_PRESS) {
            mCameraTrackAnimation = 2.0f;
        }
    }

    void appCursorPosCallback(GLFWwindow* pWindow, double xPos, double yPos) override
    {
        App::baseCursorPosCallback(pWindow, xPos, yPos);
    }

    void appMouseButtonCallback(GLFWwindow* pWindow, int button, int action, int mods) override
    {
        App::baseMouseButtonCallback(pWindow, button, action, mods, mpCamera);
    }

private:
    void addGraphResources()
    {
        VkExtent2D extent = mpSwapchain->getExtent();
        VkExtent2D shadowExtent = {SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION};

        // The cloud attachments are written as color attachments and read back as storage images
        const VkImageUsageFlags attachmentAndStorage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

        mpGraph->addResource("cloudBackPosition", VK_FORMAT_R16G16B16A16_SFLOAT, extent, attachmentAndStorage);
        mpGraph->addResource("cloudBackDepth", mDepthFormat, extent, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

        // The shadow pass only renders to get its depth, but the fragment shader still writes a color
        mpGraph->addResource("shadowColor", VK_FORMAT_R8G8B8A8_UNORM, shadowExtent,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        mpGraph->addResource("shadowDepth", mDepthFormat, shadowExtent,
                             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

        mpGraph->addResource("cloudDepth", mDepthFormat, extent, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        mpGraph->addResource("cloudColor", VK_FORMAT_R16G16B16A16_SFLOAT, extent, attachmentAndStorage);

        // The final image leaves the graph as a blit into the swapchain image
        mpGraph->addResource("output", mpSwapchain->getImageFormat(), extent,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    }

    void createPasses()
    {
        mpCloudBackPosPass = mpDevice->createPass({mpGraph->getResource("cloudBackPosition")},
                                                  mpGraph->getResource("cloudBackDepth"));
        mpShadowPass = mpDevice->createPass({mpGraph->getResource("shadowColor")}, mpGraph->getResource("shadowDepth"));
        mpCloudDepthPass = mpDevice->createPass({}, mpGraph->getResource("cloudDepth"));
        mpCloudColorPass =
            mpDevice->createPass({mpGraph->getResource("cloudColor")}, mpGraph->getResource("cloudDepth"));
        mpResolvePass = mpDevice->createPass({mpGraph->getResource("output")}, nullptr);
    }

    void updatePasses()
    {
        mpCloudBackPosPass->update({mpGraph->getResource("cloudBackPosition")},
                                   mpGraph->getResource("cloudBackDepth"));
        mpShadowPass->update({mpGraph->getResource("shadowColor")}, mpGraph->getResource("shadowDepth"));
        mpCloudDepthPass->update({}, mpGraph->getResource("cloudDepth"));
        mpCloudColorPass->update({mpGraph->getResource("cloudColor")}, mpGraph->getResource("cloudDepth"));
        mpResolvePass->update({mpGraph->getResource("output")}, nullptr);
    }

    // Attach the graph's images and the camera matrices to the shaders
    void setShaderResources()
    {
        // The shadow map is sampled by the cloud shader, so the graph's depth image needs a sampler on top of it. The
        // graph creates its images again on every compile, so the texture follows.
        mpShadowMapTexture = mpDevice->createTextureFromImage(mpGraph->getResource("shadowDepth"));

        mPipelines[PIPELINE_SKY]->getShader()->setResource("camera", mpCamera->getUniformBuffer());
        mPipelines[PIPELINE_RESOLVE]->getShader()->setResource("inCloudRenderTarget",
                                                               mpGraph->getResource("cloudColor"));

        if (mpNeuralMesh) {
            auto pShader = mpNeuralMesh->getPipeline()->getShader();
            pShader->setResource("inBackPosition", mpGraph->getResource("cloudBackPosition"));
            pShader->setResource("inShadowMap", mpShadowMapTexture);
            pShader->setResource("light", mpShadowCamera->getUniformBuffer());
        }
    }

    // The cloud is drawn with a different shader in each pass, but Scene::createDescriptors() only prepares the
    // shaders of the pipelines its nodes carry. The mesh lives in a single node that is swapped between the pipelines
    // while rendering, so an empty node is registered for every other cloud pipeline to have the scene set up those
    // shaders too. The empty nodes are never drawn.
    void registerCloudPipelines()
    {
        for (uint32_t pipelineIndex : {PIPELINE_SHADOW, PIPELINE_CLOUD_BACK_POS, PIPELINE_CLOUD_FRONT_DEPTH}) {
            Node& node = mpScene->getNode(mpScene->addNode());
            node.setPipeline(mPipelines[pipelineIndex]);
            node.setVisible(false);
        }
    }

    void removeNeuralMesh()
    {
        if (!mpNeuralMesh) {
            return;
        }

        // The node stays in the scene, it just stops being rendered
        mpScene->getNode(mpNeuralMesh->getNodeIndex()).setVisible(false);
        mpNeuralMesh = {};
    }

    std::vector<ptr<Pipeline>> allPipelines() const
    {
        std::vector<ptr<Pipeline>> pipelines(mPipelines);
        if (mpNeuralMesh) {
            pipelines.push_back(mpNeuralMesh->getPipeline());
        }
        return pipelines;
    }

    // Push the frame's constants to whichever stages the shader declares them in
    void pushConstants(VkCommandBuffer cmd, ptr<Pipeline> pPipeline) const
    {
        VkShaderStageFlags stageFlags = 0;
        for (const auto& range : pPipeline->getShader()->getPushConstantRanges()) {
            stageFlags |= range.stageFlags;
        }

        if (stageFlags) {
            vkCmdPushConstants(cmd, pPipeline->getLayout(), stageFlags, 0, sizeof(PushConstant), &mPushConstant);
        }
    }

    // Bind the descriptor set that a named resource of the shader lives in
    static void bindResourceSet(VkCommandBuffer cmd, ptr<Shader> pShader, const std::string& name)
    {
        auto info = pShader->getResourceInfo(name);
        if (info) {
            pShader->bindResources(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info->set);
        }
    }

    // Draw the cloud mesh with one of the cloud pipelines. Only the node that holds the mesh is visible, so this is
    // what the whole scene renders as.
    void renderCloud(VkCommandBuffer cmd, ptr<Pipeline> pPipeline, ptr<Camera> pCamera)
    {
        mpScene->getNode(mpNeuralMesh->getNodeIndex()).setPipeline(pPipeline);

        // Binding a pipeline whose layout differs from the previously bound one leaves the push constants undefined,
        // so the pipeline is bound before they are pushed. The scene binds the same pipeline again when it draws.
        pPipeline->bind(cmd);
        pushConstants(cmd, pPipeline);

        mpScene->render(cmd, pCamera);
    }

    void cloudBackPositionPass(VkCommandBuffer cmd)
    {
        // The back-most surface is the one that survives a greater-than depth test, so the depth is cleared to zero
        VkClearDepthStencilValue depthClearValue = {.depth = 0.0f, .stencil = 0};
        mpCloudBackPosPass->begin(cmd, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), depthClearValue);

        if (mpNeuralMesh) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, mQueryPool,
                                 TIMESTAMP_COUNT * mpDevice->getFrameInFlightIndex() + TIMESTAMP_CLOUD_BEGIN);
            renderCloud(cmd, mPipelines[PIPELINE_CLOUD_BACK_POS], mpCamera);
        }

        mpCloudBackPosPass->end(cmd);
    }

    void shadowPass(VkCommandBuffer cmd)
    {
        mpShadowPass->begin(cmd, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        if (mpNeuralMesh) {
            renderCloud(cmd, mPipelines[PIPELINE_SHADOW], mpShadowCamera);
        }

        mpShadowPass->end(cmd);
    }

    void cloudPass(VkCommandBuffer cmd)
    {
        // Depth pre-pass, which leaves the front-most cloud surface in the depth buffer
        mpCloudDepthPass->begin(cmd);
        if (mpNeuralMesh) {
            renderCloud(cmd, mPipelines[PIPELINE_CLOUD_FRONT_DEPTH], mpCamera);
        }
        mpCloudDepthPass->end(cmd);

        // The graph synchronizes between its passes, not within one, so the color pass reading back the depth the
        // pre-pass just wrote needs a barrier of its own
        cloudDepthBarrier(cmd);

        // Overwrite the color, but load the depth from the pre-pass
        mpCloudColorPass->begin(cmd, glm::vec4(0.0f), {.depth = 1.0f, .stencil = 0}, VK_ATTACHMENT_LOAD_OP_CLEAR,
                                VK_ATTACHMENT_LOAD_OP_LOAD);

        if (mpNeuralMesh) {
            auto pPipeline = mpNeuralMesh->getPipeline();
            auto pShader = pPipeline->getShader();

            mpScene->getNode(mpNeuralMesh->getNodeIndex()).setPipeline(pPipeline);
            pPipeline->bind(cmd);
            pushConstants(cmd, pPipeline);

            // The sets the scene does not own: the network, the back positions, the shadow map and the light matrices
            mpNeuralMesh->bindMLP(cmd);
            bindResourceSet(cmd, pShader, "inBackPosition");
            bindResourceSet(cmd, pShader, "inShadowMap");
            bindResourceSet(cmd, pShader, "light");

            mpScene->render(cmd, mpCamera);

            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, mQueryPool,
                                 TIMESTAMP_COUNT * mpDevice->getFrameInFlightIndex() + TIMESTAMP_CLOUD_END);
        }

        mpCloudColorPass->end(cmd);
    }

    void cloudDepthBarrier(VkCommandBuffer cmd) const
    {
        VkImageSubresourceRange subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        if (mDepthFormat == VK_FORMAT_D32_SFLOAT_S8_UINT || mDepthFormat == VK_FORMAT_D24_UNORM_S8_UINT) {
            subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }

        Helpers::imageBarrier(cmd, mpGraph->getResource("cloudDepth")->getImage(),
                              VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                              VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                              VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                              VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, &subresourceRange);
    }

    void resolvePass(VkCommandBuffer cmd)
    {
        mpResolvePass->begin(cmd, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));

        // Render sky
        auto pSkyPipeline = mPipelines[PIPELINE_SKY];
        pSkyPipeline->bind(cmd);
        pushConstants(cmd, pSkyPipeline);
        pSkyPipeline->getShader()->bindResources(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
        vkCmdDraw(cmd, 3, 1, 0, 0);

        // Composite the clouds on top of it
        if (mpNeuralMesh) {
            auto pResolvePipeline = mPipelines[PIPELINE_RESOLVE];
            pResolvePipeline->bind(cmd);
            pushConstants(cmd, pResolvePipeline);
            pResolvePipeline->getShader()->bindResources(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }

        // Draw GUI on top of the final image
        App::renderGUI(cmd);

        mpResolvePass->end(cmd);
    }

    void createQueryPoolTimestamp()
    {
        VkQueryPoolCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = mpDevice->getFramesInFlightCount() * TIMESTAMP_COUNT,
        };
        Check::Vk(vkCreateQueryPool(mpDevice->getDevice(), &ci, nullptr, &mQueryPool));

        mTimestampsWritten.assign(mpDevice->getFramesInFlightCount(), false);
    }

    void destroyQueryPoolTimestamp()
    {
        vkDeviceWaitIdle(mpDevice->getDevice());
        vkDestroyQueryPool(mpDevice->getDevice(), mQueryPool, nullptr);
    }

    void cameraTrack(ptr<Camera> pCamera, float animation)
    {
        auto track = [](float r, float y, float a) -> glm::vec3 {
            return glm::vec3(r * cosf(2.0f * std::numbers::pi_v<float> * a), y,
                             r * sinf(2.0f * std::numbers::pi_v<float> * a));
        };

        pCamera->setPosition(track(mCameraTrackRadius, mCameraTrackHeight, animation));
        pCamera->setTarget(mCameraTrackTarget);
    }

    ptr<Device> mpDevice;
    ptr<Swapchain> mpSwapchain;
    VkFormat mDepthFormat = VK_FORMAT_UNDEFINED;

    ptr<RenderGraph> mpGraph;

    ptr<Pass> mpCloudBackPosPass;
    ptr<Pass> mpShadowPass;
    ptr<Pass> mpCloudDepthPass;
    ptr<Pass> mpCloudColorPass;
    ptr<Pass> mpResolvePass;
    std::vector<ptr<Pipeline>> mPipelines;

    ptr<Texture> mpShadowMapTexture;

    ptr<Camera> mpCamera;
    ptr<Camera> mpShadowCamera;
    ptr<Scene> mpScene;

    ptr<NeuralMesh> mpNeuralMesh;
    bool mCloudBackfaceCulling = false;

    VkQueryPool mQueryPool;
    uint64_t mTimestamps[TIMESTAMP_COUNT] = {};
    std::vector<bool> mTimestampsWritten;

    PushConstant mPushConstant = {};

    int mRenderMode = 0;
    float mExposure = 1.0f;
    float mTriplaneMipmap = 0.0f;
    float mSunElevation = std::numbers::pi_v<float> / 4.0f;
    float mSunRotation = 0.0f;

    float mCameraTrackHeight = 10.0f;
    float mCameraTrackRadius = 400.0f;
    glm::vec3 mCameraTrackTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    float mCameraTrackAnimation = 2.0f;
    bool mCameraTrackAnimateRadius = false;
    int32_t mCameraTrackSpeed = 100;
};

int main(int argc, char* argv[])
{
    // Without an argument the framework picks the first discrete device
    uint32_t physicalDeviceIndex = std::numeric_limits<uint32_t>::max();
    if (argc > 1) {
        physicalDeviceIndex = std::atoi(argv[1]);
    }

    Clouds app = Clouds(physicalDeviceIndex);
    app.run();
    return 0;
}
