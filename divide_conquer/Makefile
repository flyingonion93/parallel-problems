GCC=gcc
ICC=icc
MPI=mpicc
CFLAGS=-Wall
GCCFLAGS=-fopenmp -lm
ICCFLAGS=-qopenmp

all: test closest_pair_omp 

test: main.c vector.c closest_pair.c
	$(GCC) $(CFLAGS) $(GCCFLAGS) -g main.c vector.c closest_pair.c -DSIMPLE -o testvec_gcc
	$(ICC) $(CFLAGS) $(ICCFLAGS) -g main.c vector.c closest_pair.c -DSIMPLE -o testvec_icc

closest_pair_omp: main.c vector.c closest_pair.c
	$(GCC) $(CFLAGS) $(GCCFLAGS) -g main.c vector.c closest_pair.c -o divide_conquer_gcc
	$(ICC) $(CFLAGS) $(ICCFLAGS) -g main.c vector.c closest_pair.c -o divide_conquer_icc
	
docs: Doxyfile
	doxygen Doxyfile
