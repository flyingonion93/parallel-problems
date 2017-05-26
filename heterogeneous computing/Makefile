
NVCC=nvcc
ARCH=-arch sm_20
CUDAFLAGS=$(ARCH) -lineinfo
CUBLAS=-lcublas
BLAS=-Xcompiler -fopenmp -m64 -I$(MKLROOT)/include -I/opt/intel/include/intel64 -Xlinker -L/opt/intel/lib/intel64 -liomp5 -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm

all: MatrixMatrixMultiplication MatrixMatrixMultiplication_2GPU

MatrixMatrixMultiplication: MatrixMatrixMultiplication.cu 
	$(NVCC) $(CUDAFLAGS) $(BLAS) $(CUBLAS) MatrixMatrixMultiplication.cu -o MatrixMatrixMultiplication 
MatrixMatrixMultiplication_2GPU: MatrixMatrixMultiplication_2GPU.cu 
	$(NVCC) $(CUDAFLAGS) $(BLAS) $(CUBLAS) MatrixMatrixMultiplication_2GPU.cu -o MatrixMatrixMultiplication_2GPU 
clean:
	rm -f *.o MatrixMatrixMultiplication.linkinfo MatrixMatrixMultiplication MatrixMatrixMultiplication_2GPU.linkinfo MatrixMatrixMultiplication_2GPU


