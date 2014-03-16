NVCC=nvcc
NVCCFLAGS += -Xcompiler -fopenmp -lgomp
LIBS += -lgomp
all:
	$(NVCC) $(NVCCFLAGS) doubledirection.cu -arch sm_30 -o doubledirection
