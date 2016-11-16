#include "closest_pair.h"

/*!
 *	\brief Finds the closest pair of vectors (points) in a 2D space.
 *
 *	\par
 *	This brute force algorithm \f$O(n^2)\f$ compares each point with the rest
 *	to find one that makes a pair with minimum distance.
 *	(Sequential implementation)
 *	\param[in]		*vector_list			List of stored vectors
 *	\param[in]		n						Number of points to check
 *	\param[out]		*closest_pair_vector	List containing the two closest vectors
 */
void BruteForceSolveSeq( const vector *vector_list, const int n, vector *closest_pair_vector )
{
	float minimum_distance = FLT_MAX;
	int modified = 0;
	
	int i;
	int j;
	
	vector v1;
	vector v2;
	for( i = 0; i < n - 1; i++ )
	{
		v1 = vector_list[i];
		for( j = i + 1; j < n; j++ )
		{	
			v2 = vector_list[j];;
			float new_distance = Distance2D( &v1, &v2 );
			if( new_distance < minimum_distance )
			{
				minimum_distance = new_distance;
				closest_pair_vector[0] = v1;
				closest_pair_vector[1] = v2;
				modified = 1;
			}
		}
	}
	
	//In case anything is modified (< 2 points to check) vectors with infinite distance are created
	if( modified == 0 )
	{
		InitializeVector2D( FLT_MAX, FLT_MAX, &closest_pair_vector[0] );
		InitializeVector2D( 0, 0, &closest_pair_vector[1] );
	}
 }

/*!
 *	\brief Finds the closest pair of vectors (points) in a 2D space.
 *
 *	\par
 *	This brute force algorithm \f$O(n^2)\f$ compares each point with the rest
 *	to find one that makes a pair with minimum distance
 *	(Parallel implementation based on OpenMP)
 *	\param[in]		*vector_list			List of stored vectors
 *	\param[in]		n						Number of points to check
 *	\param[out]		*closest_pair_vector	List containing the two closest vectors
 */
void BruteForceSolveParOMP( const vector *vector_list, const int n, vector *closest_pair_vector )
{
	float minimum_distance = FLT_MAX;
	int i;
	int j;
	vector v1;
	vector v2;
	#pragma omp parallel for private(j, v1, v2)
	for( i = 0; i < n - 1; i++ )
	{
		if( 0 == i )
			printf( "Thread count: %d\n", omp_get_num_threads() );

		v1 = vector_list[i];
		for( j = i + 1; j < n; j++ )
		{
			v2 = vector_list[j];
			float new_distance = Distance2D( &v1, &v2 );
			if( new_distance < minimum_distance )
			{
				#pragma omp critical (CriticalZone1)
				if( new_distance < minimum_distance )
				{
					minimum_distance = new_distance;
					closest_pair_vector[0] = v1;
					closest_pair_vector[1] = v2;
				}	
			}
		}
	}
}

/*!
 *	\brief Obtains the closest points near the middle line with cost \f$O(n)\f$
 *	\param[in]		*vector_list			List of stored vectors
 *	\param[in]		size					Numbers of points to check
 *	\param[in]		distance				Maximum distance reference
 *	\param[out]		*closest_pair			List containing the two closest vectors
 */
void ObtainStripSeparation( const vector *vector_list, const int size, const float distance, vector *closest_pair )
{
	float minimum_distance = distance;
	int modified = 0;
	
	int i;
	int j;
	for( i = 0; i < size; i++ )
	{
		for( j = i + 1; j < size && ( vector_list[j].y - vector_list[i].y ) < minimum_distance; ++j )
		{
			float new_distance = Distance2D( &vector_list[i], &vector_list[j] );
			if( new_distance < minimum_distance )
			{
				minimum_distance = new_distance;
				closest_pair[0] = vector_list[i];
				closest_pair[1] = vector_list[j];
				modified = 1;
			}
		}
	}
	//In case anything is modifed, vectors with infinite distance are created
	if( modified == 0 )
	{
		InitializeVector2D( FLT_MAX, FLT_MAX, &closest_pair[0] );
		InitializeVector2D( 0, 0, &closest_pair[1] );
	}
}

/*!
 *	\brief Finds the closest pair of vectors (points) in a 2D space
 *
 *	\par
 *	This function call sorts the vector according the x and y coordinates
 *	and then calls the appropriatre solving function
 *	\param[in]		*vector_list			List of stored vectors
 *	\param[in]		n						Numbers of points to check
 *	\param[out]		*closest_pair			List containing the two closest vectors
 *	\param[in]		mode					Determines the type of execution
 */
void DivideConquerSolve( const vector *vector_list, const int n, vector *closest_pair, const int mode )
{
	vector *sorted_vector_x = (vector *)malloc( sizeof(vector) * n );
	vector *sorted_vector_y = (vector *)malloc( sizeof(vector) * n );
	vector *closest_pair_copy = (vector *)malloc( sizeof(vector) * 2 );

	// Copying the vector to have direct access to it, not only its reference, so recursion can work
	memcpy( sorted_vector_x, vector_list, sizeof(vector) * n );
	memcpy( sorted_vector_y, vector_list, sizeof(vector) * n );	
	
	qsort( sorted_vector_x, n, sizeof(vector), Compare2DByX );
	qsort( sorted_vector_y, n, sizeof(vector), Compare2DByY );
	
	switch( mode )
	{
		case SEQUENTIAL:
			DivideConquerSolveSeq( sorted_vector_x, sorted_vector_y, 0, n - 1, closest_pair_copy );
			break;
		case OPENMP:
			#pragma omp parallel
			#pragma omp single nowait
			{
				DivideConquerSolveParOMP( sorted_vector_x, sorted_vector_y, 0, n - 1, closest_pair_copy );
			}
			break;
	}
	memcpy( closest_pair, closest_pair_copy, sizeof( vector ) * 2 );
}

/*!
 *	\brief Finds the closest pair of vectors (points) in a 2D space
 *
 *	\par
 *	This divide and conquer algorithm \f$O(n\log{}n)\f$ splits the problem into
 *	two parts, left and right side of the space grid, recursively to search by
 *	brute force only a small subset of the problem. After the minimum distance
 *	of that subset has been found it compares its value with the one belonging
 *	to the other side of the grid and selects the minimum.
 *	(Sequential implementation)
 */
void DivideConquerSolveSeq( const vector *sorted_vector_x, const vector *sorted_vector_y, const int start, const int end, vector *closest_pair_vector )
{	
	int n = end - start + 1;
	if( n <= 3 )
	{	
		BruteForceSolveSeq( sorted_vector_x, ( end - start ) + 1, closest_pair_vector );
		return;
	}

	int mid_value = n / 2;
	vector mid_point = sorted_vector_x[mid_value];
	//Divide in y sorted arrays around the vertical line (midpoint)
	//to separate them according to their side of the line
	vector *vector_y_left = (vector *)malloc( sizeof( vector ) * ( mid_value + 1 ) );
	vector *vector_y_right = (vector *)malloc( sizeof( vector ) * ( n - mid_value  - 1) );
	vector *closest_pair_left = (vector *)malloc( sizeof( vector ) *  2 );
	vector *closest_pair_right = (vector *)malloc( sizeof( vector ) * 2 );
		
	int i;
	int left_index = 0;
	int right_index = 0;
	for( i = 0; i < n; i++ )
	{
		if( sorted_vector_y[i].x < mid_point.x )
			vector_y_left[left_index++] = sorted_vector_y[i];

		else
			vector_y_right[right_index++] = sorted_vector_y[i];
	}

	//Search for the minimum distance vector
	DivideConquerSolveSeq( sorted_vector_x, vector_y_left, 0, mid_value - 1, closest_pair_left );
	DivideConquerSolveSeq( sorted_vector_x + mid_value, vector_y_right, start + mid_value, end, closest_pair_right );

	//We check wether the left or the right side has minimum distance
	float distance_left = Distance2D( &closest_pair_left[0], &closest_pair_left[1] );	
	float distance_right = Distance2D( &closest_pair_right[0], &closest_pair_right[1] );
	float minimum_distance = MinimumDistance( distance_left, distance_right );
	
	if( minimum_distance == distance_left )
		memcpy( closest_pair_vector, closest_pair_left, sizeof( vector ) * 2 );

	else
		memcpy( closest_pair_vector, closest_pair_right, sizeof( vector ) * 2 );
	
	//Build a vector to check the nearest points to the middle
	vector *midpoint_close_vectors = (vector *)malloc( sizeof( vector ) * ( end - start ) );
	vector *midpoint_closest_vectors = (vector *)malloc( sizeof( vector ) * 2 );

	int j = 0;
	for( i = 0; i < n; i++ )
	{
		if( abs( sorted_vector_y[i].x - mid_point.x ) < minimum_distance )
		{
			midpoint_close_vectors[j] = sorted_vector_y[i];
			j++;
		}
	}
	
	//Check the minimum distance of the strip and compare with the one obtained while dividing
	ObtainStripSeparation( midpoint_close_vectors, j, minimum_distance, midpoint_closest_vectors );
	float midpoint_minimum_distance = Distance2D( &midpoint_closest_vectors	[0], &midpoint_closest_vectors[1] );
	float system_minimum_value = MinimumDistance( minimum_distance, midpoint_minimum_distance );

	//In case the new distance is less than the one that we already have, we write that on memory
	if( system_minimum_value == midpoint_minimum_distance )
		memcpy( closest_pair_vector, midpoint_closest_vectors, sizeof( vector ) * 2 );
	
}

/*!
 *	\brief Finds the closest pair of vectors (points) in a 2D space
 *
 *	\par
 *	This divide and conquer algorith \f$O(n\log{}n)\f$ bla bla bla
 *	(Parallel implementation based on OpenMP
 */
void DivideConquerSolveParOMP( const vector *sorted_vector_x, const vector *sorted_vector_y, const int start, const int end, vector *closest_pair_vector )
{
	int n = end - start + 1;
	if( n <= 3 )
	{	
		BruteForceSolveSeq( sorted_vector_x, ( end - start ) + 1, closest_pair_vector );
		return;
	}

	int mid_value = n / 2;
	vector mid_point = sorted_vector_x[mid_value];
	//Divide in y sorted arrays around the vertical line (midpoint)
	//to separate them according to their side of the line
	vector *vector_y_left = (vector *)malloc( sizeof( vector ) * ( mid_value + 1 ) );
	vector *vector_y_right = (vector *)malloc( sizeof( vector ) * ( n - mid_value  - 1) );
	vector *closest_pair_left = (vector *)malloc( sizeof( vector ) *  2 );
	vector *closest_pair_right = (vector *)malloc( sizeof( vector ) * 2 );
		
	int i;
	int left_index = 0;
	int right_index = 0;
	for( i = 0; i < n; i++ )
	{
		if( sorted_vector_y[i].x < mid_point.x )
			vector_y_left[left_index++] = sorted_vector_y[i];

		else
			vector_y_right[right_index++] = sorted_vector_y[i];
	}

	//Search for the minimum distance vector
	#pragma omp task
	DivideConquerSolveSeq( sorted_vector_x, vector_y_left, 0, mid_value - 1, closest_pair_left );
	#pragma omp task
	DivideConquerSolveSeq( sorted_vector_x + mid_value, vector_y_right, start + mid_value, end, closest_pair_right );
	
	#pragma omp taskwait
	//We check wether the left or the right side has minimum distance
	float distance_left = Distance2D( &closest_pair_left[0], &closest_pair_left[1] );	
	float distance_right = Distance2D( &closest_pair_right[0], &closest_pair_right[1] );
	float minimum_distance = MinimumDistance( distance_left, distance_right );
	
	if( minimum_distance == distance_left )
		memcpy( closest_pair_vector, closest_pair_left, sizeof( vector ) * 2 );

	else
		memcpy( closest_pair_vector, closest_pair_right, sizeof( vector ) * 2 );
	
	//Build a vector to check the nearest points to the middle
	vector *midpoint_close_vectors = (vector *)malloc( sizeof( vector ) * ( end - start ) );
	vector *midpoint_closest_vectors = (vector *)malloc( sizeof( vector ) * 2 );

	int j = 0;
	for( i = 0; i < n; i++ )
	{
		if( abs( sorted_vector_y[i].x - mid_point.x ) < minimum_distance )
		{
			midpoint_close_vectors[j] = sorted_vector_y[i];
			j++;
		}
	}
	
	//Check the minimum distance of the strip and compare with the one obtained while dividing
	ObtainStripSeparation( midpoint_close_vectors, j, minimum_distance, midpoint_closest_vectors );
	float midpoint_minimum_distance = Distance2D( &midpoint_closest_vectors	[0], &midpoint_closest_vectors[1] );
	float system_minimum_value = MinimumDistance( minimum_distance, midpoint_minimum_distance );

	//In case the new distance is less than the one that we already have, we write that on memory
	if( system_minimum_value == midpoint_minimum_distance )
		memcpy( closest_pair_vector, midpoint_closest_vectors, sizeof( vector ) * 2 );
	
}

