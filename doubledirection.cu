/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <omp.h>

// Max registered calls
#define DDIO_MAX_FC 0xFF
// Max registered kernel calls
#define DDIO_MAX_KC 0xFF
// Max function unit stack size
#define MAX_ddio_fu 	512

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

typedef struct {

} ddio_event;

typedef enum {
	DDIO_PIPELINE_CALL,
	DDIO_PRE_SYNC_CALL,
	DDIO_POST_SYNC_CALL,
	DDIO_ASYNC_CALL,
	DDIO_PRE_SYNC_KERNEL_CALL,
	DDIO_POST_SYNC_KERNEL_CALL,
	DDIO_ASYNC_KERNEL_CALL
} DDIO_CALL_SYNC;

typedef struct {
	union {
	} registers;
} ddio_variable;

typedef enum {
	NATIVE_HOST_CALL,
	NATIVE_DEVICE_CALL,
	DDIO_supported_call_types
} DDIO_call_type;

typedef unsigned short int ddio_function;
typedef void **ddio_function_params;

typedef struct {
	dim3 grid;
	dim3 block;
} t_useThreads;

typedef struct {
	DDIO_call_type callType;
	DDIO_CALL_SYNC sync;
	t_useThreads useThreads;
	ddio_function ddio_call;
	void **params;
} ddio_call;

typedef void *(*DDIO_function_call)(ddio_function_params params);

typedef enum {
	DDIO_SUCCESS,
	DDIO_UNKNOWN_ERROR
} DDIO_ERROR_CODES;

typedef union {
	bool ready;
	DDIO_call_type type;
} ThreadUnit;

// Function unit
typedef struct {
	union {
		bool ready;
		unsigned short int fp;
		ThreadUnit *threadUnit;
		unsigned int numThreads;
		unsigned int availableThreads;
		bool updated;
	} registers;
	ddio_call stack[MAX_ddio_fu];
} ddio_FU;

ddio_FU ddio_function_unit;

__shared__ DDIO_function_call ddio_device_calls[DDIO_MAX_FC];
DDIO_function_call ddio_host_calls[DDIO_MAX_FC];
DDIO_function_call ddio_host_kernels[DDIO_MAX_KC];
__shared__ DDIO_function_call ddio_device_kernels[DDIO_MAX_KC];

//#define DDIO_call(f,)

// NOP function for only host specific functions like I/O
extern __device__ void DDIO_NOP(ddio_function_params params);
extern __host__ void DDIO_KERNEL_CALL(ddio_function_params params);

extern __host__ __device__ void ddio_function_call(ddio_call call);
extern __host__ __device__ void ddio_kernel_call(ddio_call call);
extern __host__ __device__ void ddio_sync_fu();
extern __host__ __device__ void ddio_fu_process_call();

// Pass params macro
#define _VA(threadIdx,...) { \
		if(threadIdx !== NULL) { \
		   void *_args[] = { __VA_ARGS__ }; \
   		   return (void **)_args; \
		} else { \
		   void *_args[] = { __VA_ARGS__, &threadIdx }; \
		   return (void **)_args; \
		} \
}

__device__ void DDIO_NOP(void **params) {
}

__host__ void DDIO_DEVICE_KERNEL_CALL(ddio_function_params params) {
	ddio_call call = (ddio_call) *params[0];
	(ddio_device_kernels[call.ddio_call]) <<<call.useThreads.grid,call.useThreads.block>>> ((void **)call.params);
}

__host__ void DDIO_push_tohost_call(ddio_function_params parmas) {
	ddio_call call = (ddio_call) *params[0];
	call.callType = DDIO_HOST_CALL;

}


__host__ __device__ void ddio_kernel_call(ddio_call call) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 350
	ddio_device_kernels<<<call.useThreads.grid,call.useThreads.block>>>[call.ddio_call](call.params);
#else
	DDIO_push_tohost_call();
#endif
#else
	dim3 threadIdx;

if(call.useThreads.block.z == 1 && call.useThreads.block.y == 1 && call.useThreads.block.x > 1) {
#pragma omp parallel for
	for (threadIdx.x = 0;threadIdx.x < call.useThreads.block.x;threadIdx.x++) {
		// TODO: pass throught threadIdx...
		(ddio_host_kernels[call.ddio_call])(call.params);
	}
} else if(call.useThreads.block.z == 1 && call.useThreads.block.y > 1 && call.useThreads.block.x > 1) {
#pragma omp parallel for
	for (threadIdx.y = 0;threadIdx.y < call.useThreads.block.y;threadIdx.y++) {
		#pragma omp parallel for
			for (threadIdx.x = 0;threadIdx.x < call.useThreads.block.x;threadIdx.x++) {
				// TODO: pass throught threadIdx...
				(ddio_host_kernels[call.ddio_call])(call.params);
		}
	}
} else if(call.useThreads.block.z > 1 && call.useThreads.block.y > 1 && call.useThreads.block.x > 1) {
#pragma omp parallel for
	for (threadIdx.z = 0;threadIdx.z < call.useThreads.block.z;threadIdx.z++) {
	#pragma omp parallel for
		for (threadIdx.y = 0;threadIdx.y < call.useThreads.block.y;threadIdx.y++) {
		#pragma omp parallel for
			for (threadIdx.x = 0;threadIdx.x < call.useThreads.block.x;threadIdx.x++) {
				// TODO: pass throught threadIdx...
				(ddio_host_kernels[call.ddio_call])(call.params);
			}
		}
	}
}
#endif
}
__host__ __device__ void ddio_function_call(ddio_call call) {
#ifdef __CUDA_ARCH__
	switch (call.sync) {
	case DDIO_PIPELINE_CALL:
		(ddio_device_calls[call.ddio_call])(call.params);
		break;
	case DDIO_PRE_SYNC_CALL:
		__syncthreads();
		(ddio_device_calls[call.ddio_call])(call.params);
		break;
	case DDIO_POST_SYNC_CALL:
		(ddio_device_calls[call.ddio_call])(call.params);
		__syncthreads();
		break;
	case DDIO_ASYNC_CALL:
		(ddio_device_calls[call.ddio_call])(call.params);
		break;
	}
#else
	switch (call.sync) {
		case DDIO_PIPELINE_CALL:
		(ddio_host_calls[call.ddio_call])(call.params);
		break;
		case DDIO_PRE_SYNC_CALL:
#pragma omp barier
#pragma omp single {
		(ddio_host_calls[call.ddio_call])(call.params);
}
		break;
		case DDIO_POST_SYNC_CALL:
#pragma omp single {
		(ddio_host_calls[call.ddio_call])(call.params);
}
#pragma omp barier
		break;
		case DDIO_ASYNC_CALL:
		(ddio_host_calls[call.ddio_call])(call.params);
		break;
	}
#endif
}

__host__ __device__ void ddio_sync_fu() {
#ifdef __CUDA_ARCH__
	// if(ddio_function_unit.registers.updated) {
	__syncthreads();
#pragma omp barrier
	// }
#else // HOST CALL FUNCTION
	//if (ddio_function_unit.registers.updated) {
	cudaThreadSynchronize();
#pragma omp barrier
	//}
#endif
}

__host__ __device__ void ddio_fu_process_call() {

}

__global__ void bitreverse(void *data) {
	__syncthreads();
}

int main(void) {
	return 0;
}
