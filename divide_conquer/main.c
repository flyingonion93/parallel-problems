
#include "closest_pair.h"

void PrintVector( vector *v, int n )
{
	printf( "V%d: (%f, %f, %f)\n", n, v->x, v->y, v->z );
}

void PrintDistance( float distance )
{
	printf( "Distance (2D): %f\n", distance );
}

int main( int argc, char *argv[] )
{
	double init_time;
	double total_time;
#ifdef SIMPLE
	int n = 10;
	
	vector v1;
	vector v2;
	vector v3;
	vector v4;
	vector v5;
	vector v6;
	vector v7;
	vector v8;
	vector v9;
	vector v10;
	InitializeVector2D( 0.0f, 0.0f, &v1 );
	InitializeVector2D( 1.0f, 2.0f, &v2 );
	InitializeVector2D( 7.0f, 8.0f, &v3 );
	InitializeVector2D( 5.0f, 3.0f, &v4 );
	InitializeVector2D( 6.0f, 7.0f, &v5 );
	InitializeVector2D( 1.0f, 8.0f, &v6 );
	InitializeVector2D( 4.0f, 8.0f, &v7 );
	InitializeVector2D( 3.0f, 6.0f, &v8 );
	InitializeVector2D( 2.0f, 4.0f, &v9 );
	InitializeVector2D( 4.0f, 1.0f, &v10 );
	vector *vector_list = malloc( sizeof( vector ) * n );
	vector *closest_pair = malloc( sizeof( vector ) * 2 );

	vector_list[0] = v1;
	vector_list[1] = v2;
	vector_list[2] = v3;
	vector_list[3] = v4;
	vector_list[4] = v5;
	vector_list[5] = v6;
	vector_list[6] = v7;
	vector_list[7] = v8;
	vector_list[8] = v9;
	vector_list[9] = v10;

	int i;
	for( i = 0; i < n; i++ )
	{
		PrintVector( &vector_list[i], i );
	}

	printf( "Executing sequential code (brute force)\n" );
	printf( "Thread count: %d\n", omp_get_num_threads() );
	init_time = omp_get_wtime();
	BruteForceSolveSeq( vector_list, n, closest_pair );
	total_time = omp_get_wtime() - init_time;
	printf( "Total time (seq): %f\n", total_time );

	printf( "Closest pair (seq): \n" );
	for( i = 0; i < 2; i++ )
	{
		PrintVector( &closest_pair[i], i );
	}
	float distance_2d = Distance2D( &closest_pair[0], &closest_pair[1] );
	PrintDistance( distance_2d );

	printf( "\nExecuting parallel code (brute force)\n" );
	init_time = omp_get_wtime();
	BruteForceSolveParOMP( vector_list, n, closest_pair );
	total_time = omp_get_wtime() - init_time;
	printf( "Total time (par): %f\n", total_time );

	printf( "Closest pair (par): \n" );
	for( i = 0; i < 2; i++ )
	{
		PrintVector( &closest_pair[i], i );
	}
	distance_2d = Distance2D( &closest_pair[0], &closest_pair[1] );
	PrintDistance( distance_2d );

	printf( "\nExecuting sequential code (divide & conquer)\n" );
	printf( "Thread count: %d\n", omp_get_num_threads() );
	init_time = omp_get_wtime();
	DivideConquerSolve( vector_list, n, closest_pair, SEQUENTIAL );
	total_time = omp_get_wtime() - init_time;
	printf( "Total time (seq): %f\n", total_time );

	printf( "Closest pair (seq): \n" );
	for( i = 0; i < 2; i++ )
	{
		PrintVector( &closest_pair[i], i );
	}
	distance_2d = Distance2D( &closest_pair[0], &closest_pair[1] );
	PrintDistance( distance_2d );

	printf( "\nExecuting parallel code (divide & conquer)\n" );
	printf( "Thread count: %d\n", omp_get_num_threads() );
	init_time = omp_get_wtime();
	DivideConquerSolve( vector_list, n, closest_pair, OPENMP );
	total_time = omp_get_wtime() - init_time;
	printf( "Total time (par): %f\n", total_time );

	printf( "Closest pair (par): \n" );
	for( i = 0; i < 2; i++ )
	{
		PrintVector( &closest_pair[i], i );
	}
	distance_2d = Distance2D( &closest_pair[0], &closest_pair[1] );
	PrintDistance( distance_2d );
#else
	if( argc != 2 )
	{
		printf( "Usage: ./divide_conquer n\nE.g. ./divide_conquer 100\n" );
		return -1;
	}	

	int n = atoi( argv[1] );

	int i;
	vector *vector_list = malloc( sizeof( vector ) * n );
	vector *closest_pair = malloc( sizeof( vector ) * 2 );

	GenerateRandom2DVector( n, vector_list );
	
	printf( "Executing sequential code (brute force)\n" );
	init_time = omp_get_wtime();
	BruteForceSolveSeq( vector_list, n, closest_pair );
	total_time = omp_get_wtime() - init_time;
	printf( "Total time (seq): %f\n", total_time );

	printf( "Closest pair (seq): \n" );
	for( i = 0; i < 2; i++ )
	{
		PrintVector( &closest_pair[i], i );
	}
	float distance_2d = Distance2D( &closest_pair[0], &closest_pair[1] );
	PrintDistance( distance_2d );

	printf( "\nExecuting parallel code (brute force)\n" );
	init_time = omp_get_wtime();
	BruteForceSolveParOMP( vector_list, n, closest_pair );
	total_time = omp_get_wtime() - init_time;
	printf( "Total time (par): %f\n", total_time );
	printf( "Closest pair (par): \n" );
	for( i = 0; i < 2; i++ )
	{
		PrintVector( &closest_pair[i], i );
	}
	distance_2d = Distance2D( &closest_pair[0], &closest_pair[1] );
	PrintDistance( distance_2d );		
	
	printf( "\nExecuting sequential code (divide & conquer)\n" );
	printf( "Thread count: %d\n", omp_get_num_threads() );
	init_time = omp_get_wtime();
	DivideConquerSolve( vector_list, n, closest_pair, SEQUENTIAL );
	total_time = omp_get_wtime() - init_time;
	printf( "Total time (seq): %f\n", total_time );

	printf( "Closest pair (seq): \n" );
	for( i = 0; i < 2; i++ )
	{
		PrintVector( &closest_pair[i], i );
	}
	distance_2d = Distance2D( &closest_pair[0], &closest_pair[1] );
	PrintDistance( distance_2d );

	printf( "\nExecuting parallel code (divide & conquer)\n" );
	printf( "Thread count: %d\n", omp_get_num_threads() );
	init_time = omp_get_wtime();
	DivideConquerSolve( vector_list, n, closest_pair, OPENMP );
	total_time = omp_get_wtime() - init_time;
	printf( "Total time (par): %f\n", total_time );

	printf( "Closest pair (par): \n" );
	for( i = 0; i < 2; i++ )
	{
		PrintVector( &closest_pair[i], i );
	}
	distance_2d = Distance2D( &closest_pair[0], &closest_pair[1] );
	PrintDistance( distance_2d );

#endif
	free( vector_list );
	free( closest_pair );
	return 0;
}

