/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#ifndef GATHERELEMENT_H
#define GATHERELEMENT_H
#include "NvInfer.h"
#include <iostream>
#include <cstring>
#include <assert.h>
#include <vector>
#include "plugin.h"

namespace nvinfer1
{
namespace plugin
{

class GatherElements: public IPluginV2DynamicExt {
public:
    GatherElements(int axis);
    
    GatherElements(const void *buffer, size_t length);

    size_t getSerializationSize() const noexcept override;

    void serialize(void *buffer) const noexcept override;

    IPluginV2DynamicExt * clone() const noexcept override;

    int getNbOutputs() const noexcept override;

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;

    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;

    void setPluginNamespace(const char* szNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

    //! The combination of kLINEAR + kFLOAT is supported.
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;

private:

    int32_t calculateNumSlices(const Dims &indexTensorDims) const noexcept; 

    static constexpr  int indexTensorIdx  = 1;
    static constexpr  int dataTensorIdx   = 0;

    const char* mPluginNamespace;
    std::string mNamespace;

    using nvinfer1::IPluginV2Ext::configurePlugin;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2::enqueue;

    int axis_{1};
};

class GatherElementsPluginCreator : public nvinfer1::IPluginCreator {
public:
    GatherElementsPluginCreator();

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin

} // namespace nvinfer1

#endif