#ifndef VONMISESKAPPAFUNCTOR_H
#define VONMISESKAPPAFUNCTOR_H

/*!
* \file vonMisesKappaFunctor.hpp
* \author Christopher P Bridge
* \brief Contains declaration of the canopy::vonMisesKappaFunctor struct, used for numerically
* solving for the kappa parameter of the von Mises distribution
*/

#include <boost/math/special_functions/bessel.hpp>

namespace canopy
{

// Specific functor for the problem at hand (finding kappa)
/*! \brief A functor object to work with Eigen's non-linear solver to numerically
* solve for the kappa parameter of a von Mises distribution.
*/
struct vonMisesKappaFunctor
{
	float R; //!< R parameter of the problem (magnitude of the resultant vector)

	/*! \brief Constructor
	* \param R The R parameter of the problem (magnitude of the resultant vector)
	*/
	vonMisesKappaFunctor(float Rin) {R = Rin;}

	/*! \brief Calculates the value of the function to be solved
	* \param x Array containing the (single) parameter of the function
	* \param fvec Array in which the value of the function at x is returned by
	* reference
	* \return 0 for no error
	*/
	int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
	{
		using boost::math::cyl_bessel_i;
		// The function to solve for Kappa
		fvec(0,0) = cyl_bessel_i(1,x[0]) - R*cyl_bessel_i(0,x[0]);
		return 0;
	}

	/*! \brief Calculates the value of the derivative of the function to be solved
	* \param x Array containing the (single) parameter of the function
	* \param fjac Array in which the value of the derivative of the function at
	* x is returned by reference
	* \return 0 for no error
	*/
	int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
	{
		using boost::math::cyl_bessel_i;
		// Derivative of the above function using simple Bessel function identities
		fjac(0,0) = 0.5*(cyl_bessel_i(0,x[0]) + cyl_bessel_i(2,x[0])) - R*cyl_bessel_i(1,x[0]);
		return 0;
	}
};

}// end of namespace

#endif
