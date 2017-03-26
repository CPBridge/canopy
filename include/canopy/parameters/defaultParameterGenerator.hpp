#ifndef DEFAULTPARAMETERGENERATOR_HPP
#define DEFAULTPARAMETERGENERATOR_HPP

/*!
* \file defaultParameterGenerator.hpp
* \author Christopher P Bridge
* \brief Contains declaration of the canopy::defaultParameterGenerator class
*/

#include <random>
#include <array>
#include <algorithm>

namespace canopy
{

/*! \brief A simple parameter generator functor for training forest models
*
* Generates random combinations of parameters where all
* parameters are drawn independently from a uniform distribution
* between zero and some user-specified upper limit.
* \tparam TNumParams The number of parameters. This must match the
* corresponding template parameter of the forest model.
*/
template <unsigned TNumParams>
class defaultParameterGenerator
{
	public:
		/*! \brief Constructor where one limit is applied to all parameters
		* \param limit The upper limit for all variables. All parameter
		* values are generated from a uniform distribution over the integers
		* between 0 and limit inclusive.
		*/
		defaultParameterGenerator(const int limit)
		{
			// Setup RNG
			std::random_device rd{};
			rand_engine.seed(rd());

			// Set all limits to be the same value
			std::fill(param_limits.begin(),param_limits.end(),limit);
		}

		/*! \brief Constructor where different limits are used for each parameter.
		* \param limits Array of upper limits. The upper limit for all variables.
		* All parameter values for parameter with index p are generated from a
		* uniform distribution over the integers between 0 and limits[p] inclusive.
		*/
		defaultParameterGenerator(const std::array<int,TNumParams>& limits)
		{
			// Setup RNG
			std::random_device rd{};
			rand_engine.seed(rd());

			// Store limit values
			std::copy(limits.cbegin(),limits.cend(),param_limits.begin());
		}

		/*! \brief Function to generate random parameter combinations
		*
		* This is called automatically by the randomForestBase::train() method.
		*/
		void operator() (std::array<int,TNumParams>& params)
		{
			// Generate a valid combination of parameters and store in params
			for(unsigned p = 0; p < TNumParams; ++p)
			{
				params[p] = uni_int_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,param_limits[p]});
			}
		}

	private:
		std::array<int,TNumParams> param_limits; //!< The limits of valid values for each parameter
		std::default_random_engine rand_engine; //!< Random engine for random number generation
		std::uniform_int_distribution<int> uni_int_dist; //!< Random integer generator
};

} // end of namespace

#endif // DEFAULTPARAMETERGENERATOR_HPP
