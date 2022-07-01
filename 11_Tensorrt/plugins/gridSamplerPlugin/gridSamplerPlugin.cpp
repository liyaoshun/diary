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
#include "gridSamplerPlugin.h"
#include "half.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include "NvInfer.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::plugin::GridSampler;
using nvinfer1::plugin::GridSamplerPluginCreator;

// plugin specific constants
namespace
{
static const char* GRID_SAMPLER_PLUGIN_VERSION{"1"};
static const char* GRID_SAMPLER_PLUGIN_NAME{"GridSampler"}; // creator will concat plugintype and namespace
static const char* GRID_SAMPLER_PLUGIN_NAMESPACE{""};
} // namespace

// Static class fields initialization
PluginFieldCollection GridSamplerPluginCreator::mFC{};
std::vector<PluginField> GridSamplerPluginCreator::mPluginAttributes;

// REGISTER_TENSORRT_PLUGIN(GridSamplerPluginCreator);

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

GridSampler::GridSampler(const std::string name, const void* serial_buf, size_t serial_size)
    : mLayerName(name)
{
    const char* d = reinterpret_cast<const char*>(serial_buf);
    const char* a = d;
    mInputChannel = readFromBuffer<size_t>(d);    
    mInputHeight = readFromBuffer<size_t>(d);    
    mInputWidth = readFromBuffer<size_t>(d);
    mGridHeight = readFromBuffer<size_t>(d);
    mGridWidth = readFromBuffer<size_t>(d);
    mInterpolationMode = readFromBuffer<GridSamplerInterpolation>(d);
    mPaddingMode = readFromBuffer<GridSamplerPadding>(d);
    mAlignCorners = readFromBuffer<bool>(d);
    mType = readFromBuffer<DataType>(d);
    assert(d == a + sizeof(size_t) * 5 + sizeof(GridSamplerInterpolation) + sizeof(GridSamplerPadding) + sizeof(bool) + sizeof(DataType));
}

GridSampler::GridSampler(const std::string name, 
GridSamplerInterpolation interpolationMode, 
GridSamplerPadding paddingMode,
bool alignCorners)
    : mLayerName(name)
    , mInterpolationMode(interpolationMode)
    , mPaddingMode(paddingMode)
    , mAlignCorners(alignCorners)
{
}

// for clone
GridSampler::GridSampler(const std::string name, int inputChannel, int inputHeight,
    int inputWidth, int gridHeight, int gridWidth, GridSamplerInterpolation interpolationMode,
    GridSamplerPadding paddingMode, bool alignCorners, DataType type)
    : mLayerName(name)
    , mInputChannel(inputChannel)
    , mInputHeight(inputHeight)
    , mInputWidth(inputWidth)
    , mGridHeight(gridHeight)
    , mGridWidth(gridWidth)
    , mInterpolationMode(interpolationMode)
    , mPaddingMode(paddingMode)
    , mAlignCorners(alignCorners)
    , mType(type)
{
}

GridSampler::~GridSampler() {}

const char* GridSampler::getPluginType() const noexcept
{
    return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSampler::getPluginVersion() const noexcept
{
    return GRID_SAMPLER_PLUGIN_VERSION;
}

int GridSampler::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs GridSampler::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // Validate input arguments
    assert(inputs[0].nbDims == 4);
    assert(inputs[1].nbDims == 4);
    
    // return N, C, H_g, W_g
    DimsExprs output(inputs[0]);
    output.d[2] = inputs[1].d[1];
    output.d[3] = inputs[1].d[2];
    return output;
}

int GridSampler::initialize() noexcept
{
    return 0;
}

size_t GridSampler::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int GridSampler::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int status = -1;

    GridSamplerDataType dataType = (mType == DataType::kFLOAT ? GridSamplerDataType::GFLOAT : GridSamplerDataType::GHALF);

    status = grid_sampler_2d_cuda(mBatch, inputs[0], inputs[1], outputs[0],
        mInputChannel, mInputHeight, mInputWidth, mGridHeight, mGridWidth,
        mInputChannel*mInputHeight*mInputWidth, mInputHeight*mInputWidth, mInputWidth, 1,
        mGridHeight*mGridWidth*2, mGridWidth*2, 2, 1,
        mInputChannel*mGridHeight*mGridWidth, mGridHeight*mGridWidth, mGridWidth, 1,
        mInterpolationMode, mPaddingMode, mAlignCorners, dataType, stream);

    return status;
}

size_t GridSampler::getSerializationSize() const noexcept
{
    return sizeof(size_t) * 5 + sizeof(GridSamplerInterpolation) + sizeof(GridSamplerPadding) + sizeof(bool) + sizeof(DataType);
}

void GridSampler::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    char* a = d;
    writeToBuffer<size_t>(d, mInputChannel);    
    writeToBuffer<size_t>(d, mInputHeight);    
    writeToBuffer<size_t>(d, mInputWidth);
    writeToBuffer<size_t>(d, mGridHeight);
    writeToBuffer<size_t>(d, mGridWidth);
    writeToBuffer<GridSamplerInterpolation>(d, mInterpolationMode);
    writeToBuffer<GridSamplerPadding>(d, mPaddingMode);
    writeToBuffer<bool>(d, mAlignCorners);
    writeToBuffer<DataType>(d, mType);
    assert(d == a + getSerializationSize());
}

bool GridSampler::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    assert(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
    
    bool condition = inOut[pos].format == TensorFormat::kLINEAR;

    condition &= inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF;
    condition &= inOut[pos].type == inOut[0].type;
    return condition;
}

void GridSampler::terminate() noexcept {} 

void GridSampler::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* GridSampler::clone() const noexcept
{
    GridSampler *plugin
        = new GridSampler(mLayerName, mInputChannel, mInputHeight, mInputWidth, 
        mGridHeight, mGridWidth, mInterpolationMode, mPaddingMode, mAlignCorners, mType);
    // plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void GridSampler::setPluginNamespace(const char* szNamespace) noexcept
{
    mNamespace = szNamespace;
}

const char* GridSampler::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// Return the DataType of the plugin output at the requested index. 
DataType GridSampler::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // one outputs
    assert(index == 0);
    assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
    return inputTypes[0];
}


void GridSampler::configurePlugin(            // configurePlugin 会在 enqueue之前运行，可以根据tensor的维度进行变量初始化
    const DynamicPluginTensorDesc* inputs, int32_t nbInputs, const DynamicPluginTensorDesc* outputs, int32_t nbOutputs) noexcept
{
    assert(nbInputs == 2);
    assert(nbOutputs == 1);

    // we only support 2d grid sampler now.
    assert(inputs[0].desc.dims.nbDims == 4);
    assert(inputs[1].desc.dims.nbDims == 4);

    mBatch = inputs[0].desc.dims.d[0];
    mInputChannel = inputs[0].desc.dims.d[1];
    mInputHeight = inputs[0].desc.dims.d[2];
    mInputWidth = inputs[0].desc.dims.d[3];
    mGridHeight = inputs[1].desc.dims.d[1];
    mGridWidth = inputs[1].desc.dims.d[2];
    mType = inputs[0].desc.type;

    assert(mBatch == inputs[1].desc.dims.d[0]);
    assert(inputs[1].desc.dims.d[3] == 2); // only supports coor = 2
}

// // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
// void GridSampler::attachToContext(
//     cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
// {
// }

// Detach the plugin object from its execution context.
// void GridSampler::detachFromContext() {}



//////////////////// creator ///////////////////////////////

GridSamplerPluginCreator::GridSamplerPluginCreator()
{
    // setPluginNamespace(GRID_SAMPLER_PLUGIN_NAMESPACE);
    mPluginAttributes.emplace_back(PluginField("interpolationmode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("paddingmode", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("aligncorners", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GridSamplerPluginCreator::getPluginName() const noexcept
{
     return GRID_SAMPLER_PLUGIN_NAME;
}

const char* GridSamplerPluginCreator::getPluginVersion() const noexcept
{ 
    return GRID_SAMPLER_PLUGIN_VERSION;
}

void GridSamplerPluginCreator::setPluginNamespace(const char* szNamespace) noexcept
{
    mNamespace = szNamespace;
}

const char* GridSamplerPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


const PluginFieldCollection* GridSamplerPluginCreator::getFieldNames() noexcept
{
    std::cout << __FUNCTION__ << std::endl;
    return &mFC;
}

IPluginV2* GridSamplerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept 
{
    std::cout << __FUNCTION__ << std::endl;
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int interpolationMode = 0, paddingMode = 0, alignCorners = 0;

    for (int i = 0; i < nbFields; ++i)
    {
        assert(fields[i].type == PluginFieldType::kINT32);

        if (!strcmp(fields[i].name, "interpolationmode"))           // 创建插件的时候进行参数检查
        {
            interpolationMode = *(reinterpret_cast<const int*>(fields[i].data));
        }

        if (!strcmp(fields[i].name, "paddingmode"))
        {
            paddingMode = *(reinterpret_cast<const int*>(fields[i].data));
        }

        if (!strcmp(fields[i].name, "aligncorners"))
        {
            alignCorners = *(reinterpret_cast<const int*>(fields[i].data));
        }
    }

    auto plugin = new GridSampler(name, static_cast<GridSamplerInterpolation>(interpolationMode)
        , static_cast<GridSamplerPadding>(paddingMode), static_cast<bool>(alignCorners));       // 前面解析创建插件的参数，这里创建插件
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2* GridSamplerPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed,
    IPluginV2Ext* plugin = new GridSampler(name, serialData, serialLength);
    // plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
