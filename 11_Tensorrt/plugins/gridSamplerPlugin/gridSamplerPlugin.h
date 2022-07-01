/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GRID_SAMPLER_PLUGIN_H
#define GRID_SAMPLER_PLUGIN_H
#include "NvInfer.h"
#include <iostream>
#include <cstring>
#include <assert.h>
#include <vector>
#include "NvInferPlugin.h"
#include "plugin.h"
#include "gridSampler.h"

// using namespace nvinfer1::plugin;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2DynamicExt and BaseCreator classes.
// For requirements for overriden functions, check TensorRT API docs.
namespace nvinfer1
{
namespace plugin
{

using torch::detail::GridSamplerInterpolation;
using torch::detail::GridSamplerPadding;
using torch::detail::GridSamplerDataType;

class GridSampler : public IPluginV2DynamicExt
{
public:
    GridSampler(const std::string name, GridSamplerInterpolation interpolationMode, GridSamplerPadding paddingMode
        , bool alignCorners);
    GridSampler(const std::string name, int inputChannel, int inputHeight,
        int inputWidth, int gridHeight, int gridWidth, GridSamplerInterpolation interpolationMode,
        GridSamplerPadding paddingMode, bool alignCorners, DataType type);
    GridSampler(const std::string name, const void* serial_buf, size_t serial_size);
    // It doesn't make sense to make GridSampler without arguments, so we delete default constructor.
    // GridSampler() = delete;
    GridSampler(); // = delete;
    ~GridSampler() override;

    // IPluginV2DynamicExt Methods
    IPluginV2DynamicExt* clone() const noexcept override;

    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
        const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
        const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;


    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;

    // void attachToContext(
    //     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

    // void detachFromContext() override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* szNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    const std::string mLayerName;
    size_t mBatch;
    size_t mInputWidth, mInputHeight, mInputChannel, mGridHeight, mGridWidth;
    GridSamplerInterpolation mInterpolationMode;
    GridSamplerPadding mPaddingMode;
    bool mAlignCorners;
    DataType mType;


// protected:
    // For deprecated methods, To prevent compiler warnings.
    // using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    // using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    // using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    // using nvinfer1::IPluginV2DynamicExt::supportsFormat;
    // using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    // using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    // using nvinfer1::IPluginV2DynamicExt::enqueue;
private:
    using nvinfer1::IPluginV2Ext::configurePlugin;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2::enqueue;
    const char* mPluginNamespace;
    std::string mNamespace;
};


////////////////////// creator ////////////////////////////////
class GridSamplerPluginCreator : public nvinfer1::IPluginCreator
{
public:
    GridSamplerPluginCreator();

    // ~GridSamplerPluginCreator() override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin

} // namespace nvinfer1

#endif // GRID_SAMPLER_PLUGIN_H
