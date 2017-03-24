#ifndef VONMISESDISTRIBUTION_HPP
#define VONMISESDISTRIBUTION_HPP

/*!
* \file vonMisesDistribution.hpp
* \author Christopher P Bridge
* \brief Contains the vonMisesDistribution class, which is the node and
* output distribution for the circularRegressor.
*/

#include <cmath>
#include <unsupported/Eigen/NonLinearOptimization>
#include <boost/math/special_functions/bessel.hpp>
#include <stdexcept>
#include <iostream>
#include <canopy/circularRegressor/vonMisesKappaFunctor.hpp>

namespace canopy
{

/*! \brief A distribution that defines the probabilities over a circular-valued
* label.
*
* The von Mises distribution has the characteristics of both a node distribution
* and an output distribution, and is used as the node and output distribution
* for the circularRegressor
*/
class vonMisesDistribution
{
	public:
		// Methods
		// -------

		/*! \brief Basic constructor
		*/
		vonMisesDistribution()
		: mu(0.0), kappa(0.0), S(0.0), C(0.0), pdf_normaliser(1.0)
		{
			// Nothing to do here
		}

		/*! \brief Initialise the distribution before fitting
		*/
		void initialise()
		{
			mu = 0.0;
			kappa = 0.0;
			S = 0.0;
			C = 0.0;
			pdf_normaliser = 1.0;
		}

		/*! \brief Fit the distribution to a set of labels.
		*
		* Fits the von Mises distribution to the set of labels between first_label and
		* last label.
		*
		* \tparam TLabelIterator The type of the iterator used to access the labels of
		* the training data. Must be a forward iterator that dereferences to numerical
		* floating point type.
		* \tparam TIdIterator The type of the iterator used to access the IDs of the
		* data points. The ID is unused but required for compatibility with randomForestBase.
		* \param first_label Iterator to the first label
		* \param last_label Iterator to the last label
		* \param - The third parameter is unused but required for compatibility with
		* randomForestBase
		*/
		template <class TLabelIterator, class TIdIterator>
		void fit(TLabelIterator first_label, const TLabelIterator last_label, TIdIterator /*unused*/)
		{
			double R2, Re;
			S = 0.0, C = 0.0;

			const int n_data = std::distance(first_label,last_label);

			// Loop through the data points, accumulating statistics
			for( ; first_label != last_label; ++first_label)
			{
				S += std::sin(*first_label);
				C += std::cos(*first_label);
			}

			R2 = S*S + C*C;

			// Mean direction
			mu = std::atan2(S,C);

			// Unbiased R statistic
			//Re = std::sqrt( ( double(n_data)/(double(n_data) + 1.0) ) * (R2 - 1.0/double(n_data)) );
			Re = std::sqrt(R2) / n_data;

			// Find the kappa parameter
			if(Re > 0.98)
			{
				// There appears to be no solution for kappa in this case (look into this further!)
				// Saturate at roughly the value for when Re = 0.98
				kappa = 25.0;
			}
			else
			{
				// Set up and solve the non-linear equation for kappa
				vonMisesKappaFunctor vmftrinstance(Re);
				Eigen::VectorXd kappa_vec(1);
				kappa_vec << 25.0;
				Eigen::HybridNonLinearSolver<vonMisesKappaFunctor> solver(vmftrinstance);
				//info =
				solver.hybrj1(kappa_vec);
				kappa = kappa_vec(0,0);
				pdf_normaliser = 1.0/(2.0*M_PI*boost::math::cyl_bessel_i(0,kappa));
			}
		}


		/*! \brief Returns the probability of a particular label
		*
		* This is the version used by the randomForestBase methods.
		* \tparam TId The type of the IDs of the data points. The ID is unused but
		* required for compatibility with randomForestBase.
		* \param x The angular label (in radians) of for which the probability is sought
		* \param - The second parameter is unused and but required for compatibility with
		* randomForestBase
		*/
		template<class TId>
		float pdf(const float x, const TId /*id*/) const
		{
			return pdf_normaliser*std::exp(kappa*std::cos(x - mu));
		}

		/*! \brief Returns the probability of a particular label
		*
		* This overloaded version does not require the ID and is intended
		* for use by user code.
		* \tparam TId The type of the IDs of the data points. The ID is unused but
		* required for compatibility with randomForestBase.
		* \param x The angular label (in radians) of for which the probability is sought
		*/
		float pdf(const float x) const
		{
			return pdf(x,0);
		}

		/*! \brief Prints the defining parameters of the distribution to an
		* output filestream
		*
		* \param stream The stream to which the parameters (mu and kappa) are printed
		*/
		void printOut(std::ofstream& stream) const
		{
			stream << mu << " " << kappa;
		}

		/*! \brief Reads the defining parameters of the distribution from a
		* filestream
		*
		* \param stream The stream from which the parameters (mu, kappa) are to be read
		*/
		void readIn(std::ifstream& stream)
		{
			stream >> mu;
			stream >> kappa;

			S = std::sin(mu);
			C = std::cos(mu);
			pdf_normaliser = 1.0/(2.0*M_PI*boost::math::cyl_bessel_i(0,kappa));
		}

		/*! \brief Return the (differential) entropy of the distribution */
		float entropy() const
		{
			using boost::math::cyl_bessel_i;
			return std::log(2.0*M_PI*cyl_bessel_i(0,kappa)) - kappa*cyl_bessel_i(1,kappa)/cyl_bessel_i(0,kappa);
		}

		/*! \brief Combine this distribution with a second by summing the probability
		* values, without normalisation.
		*
		* This method is used by the randomForestBase methods to aggregate the
		* distributions in several leaf nodes into one output distribution. In
		* this case, the sensor fusion approach of Stienne 2011 is used.
		*
		* \tparam TId The type of the IDs of the data points. The ID is unused but
		* required for compatibility with randomForestBase.
		* \param dist The distribution that this distribution should be combined with.
		* \param - The second parameter is unused and but required for compatibility with
		* randomForestBase
		*/
		template <class TId>
		void combineWith(const vonMisesDistribution& dist, const TId /*id*/)
		{
			// Add the weighted mu value to the sine and cosine sums
			S += (dist.kappa)*(dist.S);
			C += (dist.kappa)*(dist.C);
		}

		// Normalises a distribution to find the resulting mu and kappa
		// Uses the sensor fusion approach of Stienne 2011
		/*! \brief Normalise the distribution to ensure it is valid
		*
		* This may be used after several \c combineWith() operations to ensure
		* that the resulting distribution represents a valid probability distribution
		*/
		void normalise()
		{
			mu = std::atan2(S,C);
			kappa = std::hypot(S,C);

			// Don't want this to raise an exception if kappa is very large (can happen
			// if there are many trees)
			try
			{
				const double kappa_bessel = boost::math::cyl_bessel_i(0,kappa);
				pdf_normaliser = 1.0/(2.0*M_PI*kappa_bessel);
			}
			catch (std::overflow_error)
			{
				kappa = 500.0;
				pdf_normaliser = 6.35397e-217;
			}
		}

		/*! \brief Reset method
		*
		* Use this when using the class as an output distribution to create a
		* new blank distribution before combining with new node distributions
		*/
		void reset()
		{
			initialise();
		}

		/*! \brief Get the mu parameter
		* \return Mu, the circular mean parameter of the distribution
		*/
		float getMu() const
		{
			return mu;
		}

		/*! \brief Get the kappa parameter
		* \return Kappa, the concentration parameter of the distribution
		*/
		float getKappa() const
		{
			return kappa;
		}

		/*! \brief Allows the distribution to be written to a file via the
		* streaming operator '<<'
		*/
		friend std::ofstream& operator<< (std::ofstream& stream, const vonMisesDistribution& dist) { dist.printOut(stream); return stream;}

		/*! \brief Allows the distribution to be written to read from a file
		* via the streaming operator '>>'
		*/
		friend std::ifstream& operator>> (std::ifstream& stream, vonMisesDistribution& dist) { dist.readIn(stream); return stream;}

	protected:
		// Data
		float mu; //!< The distribution's circular mean parameter
		float kappa; //!< The distribution's concentration parameter
		double S; //!< Sum of sines, used during fitting and combining distributions
		double C; //!< Sum of cosines, used during fitting and combining distributions
		float pdf_normaliser; //!< Pre-calculated normalisation constant of the pdf equation

};

} // end of namespace

#endif
// VONMISESDISTRIBUTION_HPP
