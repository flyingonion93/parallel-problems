#pragma once
/*!
 * 	\brief Module to obtain the closest pair in 3D space with multiple algorithms.
 *	
 *	\author	Carlos Gómez Morillas
 * 	\date	28-10-16
 *	
 *	Last modification: 29-10-16
 */

#ifndef CLOSEST_PAIR_h
	#define CLOSEST_PAIR_h

#define SEQUENTIAL	1
#define OPENMP		2
#define MPI			3
#define HYBRID		4

#include <omp.h>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include "vector.h"

void BruteForceSolveSeq( const vector *vector_list, const int n, vector *closest_pair );
void BruteForceSolveParOMP( const vector *vector_list, const int n, vector *closest_pair );
void BruteForceSolveParMPI( const vector *vector_list, const int n, vector *closest_pair );
void DivideConquerSolve( const vector *vector_list, const int n, vector *closest_pair, const int mode );
void DivideConquerSolveSeq( const vector *sorted_vector_x, const vector *sorted_vector_y, const int start, const int end, vector *closest_pair_vector );
void DivideConquerSolveParOMP( const vector *sorted_vector_x, const vector *sorted_vector_y, const int start, const int end, vector *closest_pair_vector );
void DivideConquerSolveParMPI( const vector *vector_list, const int n, vector *closest_pair );
void DivideConquerSolveHybrid( const vector *vector_list, const int n, vector *closest_pair );
void ObtainStripSeparation( const vector *vector_list, const int size, const float distance, vector *closest_pair );
#endif
