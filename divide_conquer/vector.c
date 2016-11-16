#include "vector.h"

/*!
 *	\brief Initalizes a vector with 3D coordinates based on the value of x
 *	\param[in]		x				Value of the x component
 *	\param[out]		*v_res			Vector that is going to be initalized
 */
void InitializeVector1D( const float x, vector *v_res )
{
	v_res->x = x;
	v_res->y = 0.0;
	v_res->z = 0.0;
}

/*!
 *	\brief Initializes a vector with 3D coordinates based on the values of x and y
 *	\param[in]		x				Value of the x component
 *	\param[in] 		y				Value of the y component
 *	\param[out]		*v_res			Vector that is going to be initalized
 */
void InitializeVector2D( const float x, const float y, vector *v_res )
{
	InitializeVector1D( x, v_res );
	v_res->y = y;
}

/*!
 *	\brief Initializes a vector with 3D coordinates based on all of their values
 *	\param[in]		x				Value of the x component
 *	\param[in]		y				Value of the y component
 *	\param[in]		z				Value of the z component
 *	\param[out]		*v_res			Vector that is going to be initialized
 */
void InitializeVector3D( const float x, const float y, const float z, vector *v_res )
{
	InitializeVector2D( x, y, v_res );
	v_res->z = z;
}

/*!
 *	\brief Generates a vector of a determined size filled with random values
 *	\param[in]		n				Number of elements to generate
 *	\param[out]		*vector_list	List of vectors to store
 */
void GenerateRandom2DVector( const int n, vector *vector_list )
{
	srand( time( NULL ) );
	vector v;
	int double_n = n*n;
	int i;
	for( i = 0; i < n; i++ )
	{
		v.x = rand() % double_n;
		v.y = rand() % double_n;
		vector_list[i] = v;
	}
}

/*!
 *	\brief Computes the distance between two vectors in a 2D space
 *	The distance between \f$(p_1,p_2)\f$ and \f$(q_1,q_2)\f$ in a 2D space is
 *	\f$\sqrt{(q_1-p_1)^2+(q_2-p_2)^2}\f$
 *
 *	\param[in]		*v1				Vector p of the distance operation
 *	\param[in]		*v2				Vector q of the distance operation
 *	\return			Distance between p and q vectors
 */
float Distance2D( const vector *v1, const vector *v2 )
{
	float x_variation = v2->x - v1->x;
	float y_variation = v2->y - v1->y;
	x_variation *= x_variation;
	y_variation *= y_variation;
	return sqrt( x_variation + y_variation );
}

/*!
 *	\brief Computes the distance between two vectors in a 3D space
 *	The distance between \f$(p_1,p_2,p_3)\f$ and \f$(q_1,q_2,q_3)\f$ in a 3D space is
 *	\f$\sqrt{(q_1-p_1)^2+(q_2-p_2)^2+(q_3-p_3)^2}\f$
 *
 *	\param[in]		*v1				Vector p of the distance operation
 *	\param[in]		*v2				Vector q of the distance operation
 *	\return			Distance between p and q vectors
 */
float Distance3D( const vector *v1, const vector *v2 )
{
	float x_variation = v2->x - v1->x;
	float y_variation = v2->y - v1->y;
	float z_variation = v2->z - v1->z;
	x_variation *= x_variation;
	y_variation *= y_variation;
	z_variation *= z_variation;
	return sqrt( x_variation + y_variation + z_variation );	
}

/*!
 *	\brief Obtains a point in 2D space corresponding to the middle of two vector
 *	\param[in]		*v1				Vector p of the mid point operation
 *	\param[in]		*v2				Vector q of the mid point operation
 *	\param[out]		*mid_point		Position in space corresponding to the mid point of two vectors
 */
void MidPoint2D( const vector *v1, const vector *v2, vector *mid_point )
{
	mid_point->x = ( v1->x + v2->x ) / 2;
	mid_point->y = ( v1->x + v2->y ) / 2;
}

/*!
 *	\brief Obtains a point in 3D space corresponding to the middle of two vectors
 *	\param[in]		*v1				Vector p of the mid point operation
 *	\param[in]		*v2				Vector q of the mid point operation
 *	\param[out]		*mid_point		Position in space corresponding to the mid point of two vectors
 */
void MidPoint3D( const vector *v1, const vector *v2, vector *mid_point )
{
	MidPoint2D( v1, v2, mid_point );
	mid_point->z = ( v1->z + v2->z ) / 2;
}

/*!
 *	\brief Compares the value of the x component of two vectors
 *	\param[in]		*a				First vector to compare
 *	\param[in]		*b				Second vector to comapre
 *	\return The difference between components
 */
int Compare1D( const void *a, const void *b )
{
	vector *v1 = (vector *)a;
	vector *v2 = (vector *)b;
	int x1 = (int)trunc( v1->x );
	int x2 = (int)trunc( v2->x );
	return x1 - x2;
}

/*!
 *	\brief Compares the value of the y component of two vectors.
 *	\param[in]		*a				First vector to compare
 *	\param[in]		*b				Second vector to comapre
 *	\return	The difference between components.
 */
int Compare2DByX( const void *a, const void *b )
{
	vector *v1 = (vector *)a;
	vector *v2 = (vector *)b;
	int x1 = (int)trunc( v1->x );
	int x2 = (int)trunc( v2->x );
	int y1 = (int)trunc( v1->y );
	int y2 = (int)trunc( v2->y );
	int xdiff = x1 - x2;
	int ydiff = y1 - y2;
	if( x1 < x2 )
	{
		if( xdiff < ydiff )
			return xdiff;

	}else if( x1 > x2 )
	{
		if( xdiff > ydiff )
			return xdiff;

	}
	return xdiff + ydiff;
}

/*!
 *	\brief Compares the value of the y component of two vectors.
 *	\param[in]		*a				First vector to compare
 *	\param[in]		*b				Second vector to comapre
 *	\return	The difference between components.
 */
int Compare2DByY( const void *a, const void *b )
{
	vector *v1 = (vector *)a;
	vector *v2 = (vector *)b;
	int x1 = (int)trunc( v1->x );
	int x2 = (int)trunc( v2->x );
	int y1 = (int)trunc( v1->y );
	int y2 = (int)trunc( v2->y );
	int xdiff = x1 - x2;
	int ydiff = y1 - y2;
	if( y1 < y2 )
	{
		if( ydiff < xdiff )
			return ydiff;

	}else if( y1 > y2 )
	{
		if( ydiff > xdiff )
			return ydiff;

	}
	return xdiff + ydiff;
}

/*!
 *	\brief Compares the value of the z component of two vectors
 *	\param[in]		*v1				First vector to compare
 *	\param[in]		*v2				Second vector to compare
 *	\return The difference between components.
 */
int Compare3D( const void *a, const void *b )
{
	vector *v1 = (vector *)a;
	vector *v2 = (vector *)b;
	int x1 = (int)trunc( v1->x );
	int x2 = (int)trunc( v2->x );
	int y1 = (int)trunc( v1->y );
	int y2 = (int)trunc( v2->y );
	int z1 = (int)trunc( v1->z );
	int z2 = (int)trunc( v2->z );
	return (x1 - x2) + (y1 - y2) + (z1 - z2);
}

/*!
 *	\brief Compares which is the minimum value between two given distances
 *	\param[in]		distance1		First distance to check		
 *	\param[in]		distance2		Second distance to check
 *	\return The minimum distance value.
 */
float MinimumDistance( const float distance1, const float distance2 )
{
	return ( distance1 < distance2 ) ? distance1 : distance2;
}
