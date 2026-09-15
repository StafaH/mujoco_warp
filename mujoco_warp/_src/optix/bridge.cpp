// Copyright 2026 The Newton Developers
// Licensed under the Apache License, Version 2.0.
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
static_assert(sizeof(OptixInstance) == 80, "OptixInstance packing changed");
static_assert(offsetof(OptixInstance, traversableHandle) == 64, "OptixInstance handle offset changed");
thread_local std::string error;
void check(OptixResult r) {
  if (r != OPTIX_SUCCESS) throw std::runtime_error(std::string(optixGetErrorName(r)) + ": " + optixGetErrorString(r));
}
void cuda_check(CUresult r) {
  if (r != CUDA_SUCCESS) {
    const char* text = nullptr;
    cuGetErrorString(r, &text);
    throw std::runtime_error(text ? text : "CUDA error");
  }
}
struct Buffer {
  CUdeviceptr ptr = 0;
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() {
    if (ptr) cuMemFree(ptr);
  }
  void alloc(size_t size) { cuda_check(cuMemAlloc(&ptr, std::max(size, size_t(1)))); }
};
struct Accel {
  OptixBuildInput input = {};
  OptixAccelBuildOptions options = {};
  Buffer storage, scratch;
  CUdeviceptr vertices = 0;
  unsigned flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  OptixTraversableHandle handle = 0;
  size_t output_size = 0, scratch_size = 0;
};
struct State {
  OptixDeviceContext context = nullptr;
  OptixModule module = nullptr;
  OptixPipeline pipeline = nullptr;
  std::vector<OptixProgramGroup> groups;
  std::vector<std::unique_ptr<Accel>> meshes;
  std::unique_ptr<Accel> scene;
  Buffer records, handles;
  OptixShaderBindingTable sbt = {};
  ~State() {
    if (pipeline) optixPipelineDestroy(pipeline);
    for (auto pg : groups) optixProgramGroupDestroy(pg);
    if (module) optixModuleDestroy(module);
    scene.reset();
    meshes.clear();
    if (context) optixDeviceContextDestroy(context);
  }
};
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
}  // namespace
extern "C" const char* mjwo_error() { return error.c_str(); }
extern "C" void* mjwo_create(CUcontext cuda_context) {
  try {
    check(optixInit());
    auto state = std::make_unique<State>();
    OptixDeviceContextOptions options = {};
    options.logCallbackLevel = 2;
    options.logCallbackFunction = [](unsigned, const char* tag, const char* message, void*) {
      std::fprintf(stderr, "OptiX [%s]: %s\n", tag, message);
    };
    check(optixDeviceContextCreate(cuda_context, &options, &state->context));
    return state.release();
  } catch (const std::exception& e) {
    error = e.what();
    return nullptr;
  }
}
extern "C" int mjwo_compile(State* s, const char* ptx, size_t length) {
  try {
    OptixModuleCompileOptions module_options = {};
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = 2;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.pipelineLaunchParamsVariableName = "mjwo_params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    char log[16384];
    size_t log_size = sizeof(log);
    auto result = optixModuleCreate(s->context, &module_options, &pipeline_options, ptx, length, log, &log_size, &s->module);
    if (result != OPTIX_SUCCESS) throw std::runtime_error(std::string(log, std::min(log_size, sizeof(log))));
    OptixProgramGroupOptions pg_options = {};
    for (int i = 0; i < 3; ++i) {
      OptixProgramGroupDesc desc = {};
      if (i == 0) {
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = s->module;
        desc.raygen.entryFunctionName = "__raygen__render";
      }
      if (i == 1) {
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      }
      if (i == 2) {
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleAH = s->module;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__triangle";
      }
      OptixProgramGroup pg;
      log_size = sizeof(log);
      check(optixProgramGroupCreate(s->context, &desc, 1, &pg_options, log, &log_size, &pg));
      s->groups.push_back(pg);
    }
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    result = optixPipelineCreate(s->context, &pipeline_options, &link_options, s->groups.data(), s->groups.size(), log,
                                 &log_size, &s->pipeline);
    if (result != OPTIX_SUCCESS) throw std::runtime_error(std::string(log, std::min(log_size, sizeof(log))));
    OptixStackSizes stack = {};
    for (auto pg : s->groups) check(optixUtilAccumulateStackSizes(pg, &stack, s->pipeline));
    unsigned direct_traversal, direct_state, continuation;
    check(optixUtilComputeStackSizes(&stack, 1, 0, 0, &direct_traversal, &direct_state, &continuation));
    check(optixPipelineSetStackSize(s->pipeline, direct_traversal, direct_state, continuation, 2));
    Record records[3] = {};
    for (int i = 0; i < 3; ++i) check(optixSbtRecordPackHeader(s->groups[i], &records[i]));
    s->records.alloc(sizeof(records));
    cuda_check(cuMemcpyHtoD(s->records.ptr, records, sizeof(records)));
    s->sbt.raygenRecord = s->records.ptr;
    s->sbt.missRecordBase = s->records.ptr + sizeof(Record);
    s->sbt.missRecordStrideInBytes = sizeof(Record);
    s->sbt.missRecordCount = 1;
    s->sbt.hitgroupRecordBase = s->records.ptr + 2 * sizeof(Record);
    s->sbt.hitgroupRecordStrideInBytes = sizeof(Record);
    s->sbt.hitgroupRecordCount = 1;
    return 0;
  } catch (const std::exception& e) {
    error = e.what();
    return -1;
  }
}
extern "C" int mjwo_launch(State* s, CUstream stream, CUdeviceptr params, size_t size, unsigned rays, unsigned worlds) {
  try {
    check(optixLaunch(s->pipeline, stream, params, size, &s->sbt, rays, worlds, 1));
    return 0;
  } catch (const std::exception& e) {
    error = e.what();
    return -1;
  }
}

extern "C" unsigned long long mjwo_mesh(State* s, CUstream stream, CUdeviceptr points, unsigned npoints, CUdeviceptr indices,
                                        unsigned ntriangles) {
  try {
    auto a = std::make_unique<Accel>();
    a->vertices = points;
    a->flags = OPTIX_GEOMETRY_FLAG_NONE;
    a->input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    auto& triangles = a->input.triangleArray;
    triangles.vertexBuffers = &a->vertices;
    triangles.numVertices = npoints;
    triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangles.vertexStrideInBytes = 12;
    triangles.indexBuffer = indices;
    triangles.numIndexTriplets = ntriangles;
    triangles.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangles.indexStrideInBytes = 12;
    triangles.flags = &a->flags;
    triangles.numSbtRecords = 1;
    a->options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    a->options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes;
    check(optixAccelComputeMemoryUsage(s->context, &a->options, &a->input, 1, &sizes));
    a->storage.alloc(sizes.outputSizeInBytes);
    a->scratch.alloc(sizes.tempSizeInBytes);
    check(optixAccelBuild(s->context, stream, &a->options, &a->input, 1, a->scratch.ptr, sizes.tempSizeInBytes, a->storage.ptr,
                          sizes.outputSizeInBytes, &a->handle, nullptr, 0));
    auto handle = a->handle;
    s->meshes.push_back(std::move(a));
    return handle;
  } catch (const std::exception& e) {
    error = e.what();
    return 0;
  }
}
extern "C" unsigned long long mjwo_handles(State* s) { return s->handles.ptr; }
extern "C" int mjwo_instances(State* s, CUstream stream, CUdeviceptr instances, unsigned count) {
  try {
    auto a = std::make_unique<Accel>();
    a->input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    a->input.instanceArray.instances = instances;
    a->input.instanceArray.numInstances = count;
    // Rebuild rather than UPDATE: callers can change geom_dataid on device,
    // replacing child traversable handles, which UPDATE does not permit.
    a->options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    a->options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes;
    check(optixAccelComputeMemoryUsage(s->context, &a->options, &a->input, 1, &sizes));
    a->storage.alloc(sizes.outputSizeInBytes);
    a->scratch.alloc(sizes.tempSizeInBytes);
    a->output_size = sizes.outputSizeInBytes;
    a->scratch_size = sizes.tempSizeInBytes;
    check(optixAccelBuild(s->context, stream, &a->options, &a->input, 1, a->scratch.ptr, a->scratch_size,
                          a->storage.ptr, a->output_size, &a->handle, nullptr, 0));
    s->handles.alloc(sizeof(a->handle));
    cuda_check(cuMemcpyHtoDAsync(s->handles.ptr, &a->handle, sizeof(a->handle), stream));
    cuda_check(cuStreamSynchronize(stream));
    s->scene = std::move(a);
    return 0;
  } catch (const std::exception& e) {
    error = e.what();
    return -1;
  }
}
extern "C" int mjwo_build_scene(State* s, CUstream stream) {
  try {
    auto& a = s->scene;
    check(optixAccelBuild(s->context, stream, &a->options, &a->input, 1, a->scratch.ptr, a->scratch_size,
                          a->storage.ptr, a->output_size, &a->handle, nullptr, 0));
    // BUILD may return a different handle even when reusing the output buffer.
    cuda_check(cuMemcpyHtoDAsync(s->handles.ptr, &a->handle, sizeof(a->handle), stream));
    return 0;
  } catch (const std::exception& e) {
    error = e.what();
    return -1;
  }
}
extern "C" void mjwo_destroy(State* s) { delete s; }
