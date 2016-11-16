#pragma once
/*!
 * 	\brief Module to compute simple vector operations.
 *	
 *	\author	Carlos Gómez Morillas
 * 	\date	28-10-16
 *	
 *	Last modification: 29-10-16
 */
#ifndef VECTOR_h
	#define VECTOR_h

#include <math.h>
#include <stdlib.h>
#include <time.h>

/*!
 * Struct to define a vector in the space
 * The vector can also be interpreted as a point in 3D space
 */
typedef struct vector_st
{
	float x;
	float y;
	float z;
} vector;

void InitializeVector1D( const float x, vector *v );
void InitializeVector2D( const float x, const float y, vector *v );
void InitializeVector3D( const float x, const float y, const float z, vector *v );
void GenerateRandom2DVector( const int n, vector *vector_list );
void GenerateRandom3DVector( const int n, vector *vector_list );
float Distance2D( const vector *v1, const vector *v2 );
float Distance3D( const vector *v1, const vector *v2 );
void MidPoint2D( const vector *v1, const vector *v2, vector *mid_point );
void MidPoint3D( const vector *v1, const vector *v2, vector *mid_point );
int Compare1D( const void *a, const void *b );
int Compare2DByX( const void *a, const void *b );
int Compare2DByY( const void *a, const void *b );
int Compare3D( const void *a, const void *b );
float MinimumDistance( const float distance1, const float distance2 );
#endif
