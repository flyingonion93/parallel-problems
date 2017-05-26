
/*************************************
 * Matrix-Matrix product with CUBLAS *
 *************************************/

#include <stdio.h>
#include <mkl_blas.h>
#include <omp.h>
#include "cublas_v2.h" /* Write here the name of the CUBLAS header file */

#define CUDA_SAFE_CALL( call ) { cuAssert((call), __FILE__, __LINE__); }
inline void cuAssert( cudaError_t err, const char *file, int line, bool abort=true)
{
		if( cudaSuccess != err )
		{
				fprintf(stderr, "CUDA: error ocurred in %s %s %d\n", cudaGetErrorString(err), file, line );
				if( abort )
						exit( err );

		}
}

#define CUBLAS_SAFE_CALL( call ) { cublasAssert((call), __FILE__, __LINE__); }
inline void cublasAssert( cublasStatus_t err, const char *file, int line, bool abort=true)
{
		if( CUBLAS_STATUS_SUCCESS != err )
		{
				fprintf(stderr, "CUBLAS: error ocurred in %s %s %d\n", err, file, line );
				if( abort )
						exit( err );

		}
}

/* Matrices stored by columns: BLAS style */
#define	A(i,j)			A[ (i) + ((j)*(n)) ]
#define	B(i,j)			B[ (i) + ((j)*(n)) ]
#define	C(i,j)			C[ (i) + ((j)*(n)) ]
#define	h_C(i,j)		h_C[ (i) + ((j)*(n)) ]
#define	h_C1(i,j)		h_C1[ (i) + ((j)*(n)) ]
#define	h_C2(i,j)		h_C2[ (i) + ((j)*(n)) ]
#define	het_C(i,j)	het_C[ (i) + ((j)*(n)) ]
#define	d_A(i,j) 		d_A[ (j) + ((i)*(n)) ]

int main( int argc, char *argv[] ) 
{
		int n, m, nm, m2, deviceCount, middle;
		float weigth;
		unsigned int i, j;

		if( argc < 3 ) 
		{
				printf( "Usage: %s n weight\n", argv[0] );
				exit( -EXIT_FAILURE );
		}

		sscanf( argv[1],"%d",&n );
		sscanf( argv[2],"%f",&weigth );

		m = n * weigth;
		nm = n - m;

		// General matrices
		double *A = (double *) malloc( n * n * sizeof(double) );
		double *B = (double *) malloc( n * n * sizeof(double) );

		// Result matrices
		double *C = (double *) malloc( n * n * sizeof(double) ); 			// CPU execution
		double *h_C = (double *) malloc( n * n * sizeof(double) );	 	// GPU execution
		double *het_C = (double *) malloc( n * n * sizeof(double) );	// Heterogeneous execution

		// GPU matrices
		double *d_A, *d_B, *d_C;

		// Heterogenous matrices
		double *d_A1, *d_A2, *d_B1, *d_B2, *d_C1, *d_C2;
		double *h_C1 = (double *) malloc( n * m * sizeof(double) );
		double *h_C2 = (double *) malloc( n * m * sizeof(double) );

		printf( "%s: Generating two random matrices of size %dx%d...\n", argv[0], n, n );

		for( i = 0; i < n; i++ )
		{
				for( j = 0; j < n; j++ )
						A( i, j ) = 2.0 * ( (double) rand() / RAND_MAX ) - 1.0;

		}

		for( i = 0; i < n; i++ )
		{
				for( j = 0; j < n; j++ )
						B( i, j ) = 2.0 * ( (double) rand() / RAND_MAX ) - 1.0;

		}

		/* STARTUP CUBLAS context */
		cublasHandle_t handle;
		CUBLAS_SAFE_CALL( cublasCreate( &handle ) );

		cudaEvent_t start, stop;
		CUDA_SAFE_CALL( cudaEventCreate( &start ) );
		CUDA_SAFE_CALL( cudaEventCreate( &stop ) );
		CUDA_SAFE_CALL( cudaGetDeviceCount( &deviceCount ) );

		const char trans = 'N';
		const double ONE = 1.0;
		const double ZERO = 0.0;

		// MKL execution (CPU)
		printf( "%s: C = A * B in CPU...\n", argv[0] );
		CUDA_SAFE_CALL( cudaEventRecord(start, NULL) );
		dgemm( &trans, &trans, &n, &n, &n, &ONE, A, &n, B, &n, &ZERO, C, &n );
		CUDA_SAFE_CALL( cudaEventRecord(stop, NULL) );
		CUDA_SAFE_CALL( cudaEventSynchronize( stop ) );
		float msecCPU = 0.0f;
		CUDA_SAFE_CALL( cudaEventElapsedTime( &msecCPU, start, stop ) );

		// CuBLAS execution (GPU)
		printf( "%s: C = A * B in GPU...\n", argv[0] );
		CUDA_SAFE_CALL( cudaMalloc( (void **) &d_A, n * n * sizeof(double) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void **) &d_B, n * n * sizeof(double) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void **) &d_C, n * n * sizeof(double) ) );
		CUBLAS_SAFE_CALL( cublasSetMatrix( n, n, sizeof(double), A, n, d_A, n ) );
		CUBLAS_SAFE_CALL( cublasSetMatrix( n, n, sizeof(double), B, n, d_B, n ) );

		CUDA_SAFE_CALL( cudaEventRecord(start, NULL) );
		CUBLAS_SAFE_CALL( cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &ONE, d_A, n, d_B, n, &ZERO, d_C, n ) );
		CUDA_SAFE_CALL( cudaEventRecord( stop, NULL ) );
		CUDA_SAFE_CALL( cudaEventSynchronize( stop ) );
		CUBLAS_SAFE_CALL( cublasGetMatrix( n, n, sizeof(double), d_C, n, h_C, n ) );
		float msecGPU = 0.0f;
		CUDA_SAFE_CALL( cudaEventElapsedTime( &msecGPU, start, stop ) );
		CUDA_SAFE_CALL( cudaFree( d_B ) );
		CUDA_SAFE_CALL( cudaFree( d_C ) );

		// Heterogeneous execution (CPU + GPU)
		printf( "%s: C = A * B in CPU + GPU...\n",argv[0] );
		m2 =  m/2;
		middle = ( nm + ( n - 1 ) ) / 2;
		CUDA_SAFE_CALL( cudaEventRecord( start, NULL ) );		
		#pragma omp parallel sections
		{
				#pragma omp section
				{
						CUDA_SAFE_CALL( cudaSetDevice( 0 ) );
						CUDA_SAFE_CALL( cudaMalloc( (void **) &d_A1, n * n * sizeof(double) ) );
						CUDA_SAFE_CALL( cudaMalloc( (void **) &d_B1, n * m2 * sizeof(double) ) );
						CUDA_SAFE_CALL( cudaMalloc( (void **) &d_C1, n * m2 * sizeof(double) ) );
						CUBLAS_SAFE_CALL( cublasSetMatrix( n, n, sizeof(double), A, n, d_A1, n ) );
						CUBLAS_SAFE_CALL( cublasSetMatrix( n, m2, sizeof(double), &B(0, nm), n, d_B1, n ) );
						CUBLAS_SAFE_CALL( cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m2, n, &ONE, d_A1, n, d_B1, n, &ZERO, d_C1, n ) );
				}
				#pragma omp section
				{
						CUDA_SAFE_CALL( cudaMalloc( (void **) &d_A2, n * n * sizeof(double) ) );
						CUDA_SAFE_CALL( cudaMalloc( (void **) &d_B2, n * m2 * sizeof(double) ) );
						CUDA_SAFE_CALL( cudaMalloc( (void **) &d_C2, n * m2 * sizeof(double) ) );
						CUBLAS_SAFE_CALL( cublasSetMatrix( n, n, sizeof(double), A, n, d_A2, n ) );
						CUBLAS_SAFE_CALL( cublasSetMatrix( n, m2, sizeof(double), &B(0, middle + 1), n, d_B2, n ) );
						CUBLAS_SAFE_CALL( cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m2, n, &ONE, d_A2, n, d_B2, n, &ZERO, d_C2, n ) );
				}
				#pragma omp section
				{
						dgemm( &trans, &trans, &n, &nm, &n, &ONE, A, &n, B, &n, &ZERO, het_C, &n );
				}
		}
		CUDA_SAFE_CALL( cudaEventRecord( stop, NULL ) );
		CUDA_SAFE_CALL( cudaEventSynchronize( stop ) );

		CUDA_SAFE_CALL( cudaSetDevice( 0 ) );
		CUBLAS_SAFE_CALL( cublasGetMatrix( n, m2, sizeof(double), d_C1, n, h_C1, n ) );
		memcpy( &het_C(0, nm), h_C1, n * m2 * sizeof(double) );

		CUDA_SAFE_CALL( cudaSetDevice( 1 ) );
		CUBLAS_SAFE_CALL( cublasGetMatrix( n, m2, sizeof(double), d_C2, n, h_C2, n ) );
		memcpy( &het_C(0, middle + 1), h_C2, n * m2 * sizeof(double) );


		float msecCPUGPU = 0.0f;
		CUDA_SAFE_CALL( cudaEventElapsedTime( &msecCPUGPU, start, stop ) );

		int one = 1;
		int maxid = idamax( &n, C, &one );
		double max = C[maxid];
		double error = ZERO;

		for( j = 1; j < n; j++ )
		{
				for( i = 1; i < n; i++ )
				{
						double a = fabs( C( i, j ) - h_C( i, j ) ) / max;
						error = a > error ? a : error;
				}
		}

		printf( "Error CPU/GPU = %.3e\n",error );

		one = 1;
		maxid = idamax( &n, C, &one );
		max = C[maxid];
		error = ZERO;

		for( j = 1; j < n; j++ )
		{
				for( i = 1; i < n; i++ )
				{
						double a = fabs( C( i, j ) - het_C( i, j ) ) / max;
						error = a > error ? a : error;
				}
		}

		printf( "Error CPU/CPU + GPU = %.3e\n",error );

		double flops = 2.0 * (double) n * (double) n * (double) n;

		float gigaFlopsCPU = ( flops * 1.0e-9f ) / ( msecCPU / 1000.0f );
		float gigaFlopsGPU = ( flops * 1.0e-9f ) / ( msecGPU / 1000.0f );
		float gigaFlopsCPUGPU = ( flops * 1.0e-9f ) / ( msecCPUGPU / 1000.0f );

		printf( "CPU time = %.2f msec.\n", msecCPU );
		printf( "GPU time = %.2f msec.\n", msecGPU );
		printf( "CPU + GPU time = %.2f msec.\n", msecCPUGPU );
		printf( "GFlops CPU = %.2f \n", gigaFlopsCPU );
		printf( "GFlops GPU = %.2f \n", gigaFlopsGPU );
		printf( "GFlops CPU + GPU = %.2f \n", gigaFlopsCPUGPU );

		// CPU matrices
		free( A );
		free( B );
		free( C );
		//free( h_C );
		free( het_C );
		free( h_C2 );

		//GPU matrices
		cudaFree( d_A );
		cudaFree( d_A1 );
		cudaFree( d_A2 );
		cudaFree( d_B );
		cudaFree( d_C );
		cublasDestroy( handle );
}

