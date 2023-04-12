// Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#ifndef PULSAR_NATIVE_COMMANDS_ROUTING_H_
#define PULSAR_NATIVE_COMMANDS_ROUTING_H_

#include "../global.h"

// Commands available everywhere.
#define MALLOC_HOST(VAR, TYPE, SIZE) \
  VAR = static_cast<TYPE*>(malloc(sizeof(TYPE) * (SIZE)))
#define FREE_HOST(PTR) free(PTR)

/* Include command definitions depending on CPU or GPU use. */

#ifdef __CUDACC__
// TODO: find out which compiler we're using here and use the suppression.
// #pragma push
// #pragma diag_suppress = 68
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
// #pragma pop
#include "../cuda/commands.h"
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <TH/TH.h>
#pragma clang diagnostic pop
#include "../host/commands.h"
#endif

#endif
