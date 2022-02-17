#include "GatherElement.h"
#include "half.h"
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::plugin::GatherElements;
using nvinfer1::plugin::GatherElementsPluginCreator;

// namespace
// {
const char* GATHERND_PLUGIN_VERSION{"1"};
const char* GATHERND_PLUGIN_NAME{"GatherElements"};
// } // namespace

PluginFieldCollection GatherElementsPluginCreator::mFC{};
std::vector<PluginField> GatherElementsPluginCreator::mPluginAttributes;

GatherElements::GatherElements(int axis): axis_(axis)
{
}

GatherElements::GatherElements(const void *buffer, size_t length) 
{
    const char* d = static_cast<const char*>(buffer);
    axis_ = read<int>(d);
    // memcpy(&m, buffer, sizeof(m));
}

size_t GatherElements::getSerializationSize() const noexcept
{
    return sizeof(axis_);
    // return 0;
}

void GatherElements::serialize(void *buffer) const noexcept 
{
    // char* d = static_cast<char*>(buffer);
    // // const char* const a = d;
    // write(d, axis_);
    memcpy(buffer, &axis_, sizeof(axis_));
    return;
}

IPluginV2DynamicExt * GatherElements::clone() const noexcept
{
    GatherElements* plugin = new GatherElements(&axis_, sizeof(axis_));
    // plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

int GatherElements::getNbOutputs() const noexcept
{
    // Plugin layer has 1 output
    return 1;
}

DimsExprs GatherElements::getOutputDimensions(int32_t outputIndex,const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder)  noexcept
{
    //output should have same dimensions as index tensor
    DimsExprs ret = inputs[indexTensorIdx];
    return ret;
}


size_t GatherElements::getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{    
    int32_t nSlices = calculateNumSlices(inputs[indexTensorIdx].dims);
    //transformCoeffs + transformed indices
    return outputs[0].dims.MAX_DIMS * sizeof(int32_t) + nSlices * sizeof(int32_t) + outputs[0].dims.nbDims*sizeof(int32_t) * 3;

}


int GatherElements::initialize() noexcept
{
    return 0;
}

void GatherElements::terminate() noexcept
{
}

void GatherElements::destroy() noexcept
{
    delete this;
}

void GatherElements::setPluginNamespace(const char* szNamespace) noexcept
{
    mNamespace = szNamespace;
}

const char* GatherElements::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

const char* GatherElements::getPluginType() const noexcept
{
    return GATHERND_PLUGIN_NAME;
}

const char* GatherElements::getPluginVersion() const noexcept
{
    return GATHERND_PLUGIN_VERSION;
}

void GatherElements::configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    assert(in  && nbInputs  == 2); // the plugin has two input tensors
    assert(out && nbOutputs == 1);
    assert(in[0].desc.type == out[0].desc.type);

    assert( in[0].desc.format == TensorFormat::kLINEAR); //data
    assert( in[1].desc.format == TensorFormat::kLINEAR); //indices
    assert(out[0].desc.format == TensorFormat::kLINEAR);
}

bool GatherElements::supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
        assert(nbInputs == 2 && nbOutputs == 1 && pos < nbInputs + nbOutputs);

        bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        if (pos == 1) // the datatype of the first input is kINT32
            condition &= inOut[pos].type == DataType::kINT32;
        else
            condition &= inOut[pos].type == DataType::kFLOAT;

        return condition;
}

DataType GatherElements::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[dataTensorIdx];
}


int32_t GatherElements::calculateNumSlices(const Dims &indexTensorDims) const noexcept
{
    int32_t nSlices = 1;
    for (int i = 0; i < indexTensorDims.nbDims; i++)
    {
        nSlices *= indexTensorDims.d[i];
    }
    return nSlices;
}


GatherElementsPluginCreator::GatherElementsPluginCreator()
{
    // TODO: batch_dims is optional in onnx graph
    // mPluginAttributes.emplace_back(PluginField("batch_dims", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}


const char* GatherElementsPluginCreator::getPluginName() const noexcept
{
     return GATHERND_PLUGIN_NAME;
}

const char* GatherElementsPluginCreator::getPluginVersion() const noexcept
{
    return GATHERND_PLUGIN_VERSION;
}

void GatherElementsPluginCreator::setPluginNamespace(const char* szNamespace) noexcept
{
    mNamespace = szNamespace;
}

const char* GatherElementsPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

const PluginFieldCollection* GatherElementsPluginCreator::getFieldNames() noexcept
{
    std::cout << __FUNCTION__ << std::endl;
    return &mFC;
}

IPluginV2* GatherElementsPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    std::cout << __FUNCTION__ << std::endl;
    int batch_dims = 0;
    for (int i = 0; i < fc->nbFields; i++) {
        // std::cout << "createPlugin    =======================   fc-name "<<fc->fields[i].name << std::endl;
        if (!strcmp(fc->fields[i].name, "axis")) {
            batch_dims = *(int *)fc->fields[i].data;
        }
    }

    auto* plugin = new GatherElements{batch_dims};
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2* GatherElementsPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept 
{
    IPluginV2Ext* plugin = new GatherElements(serialData, serialLength);
    return plugin;
}