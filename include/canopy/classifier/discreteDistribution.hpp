#ifndef DISCRETEDISTRIBUTION_HPP
#define DISCRETEDISTRIBUTION_HPP

/*!
* \file discreteDistribution.hpp
* \author Christopher P Bridge
* \brief Contains the canopy::discreteDistribution class, which is the node and
* output distribution for the classifier.
*/

#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>

namespace canopy
{

/*! \brief A distribution that defines the probabilities over a number of discrete
* (integer-valued) class labels.
*
* The discreteDistribution has the characteristics of both a node distribution
* and an output distribution, and is used as the node and output distribution
* for the classifier
*/
class discreteDistribution
{
	public:
		// Methods
		//--------

		/*! \brief Default constructor
		*
		* Initialises with 0 classes
		*/
		discreteDistribution() : n_classes(0) {}

		/*! \brief Constructor
		*
		* Initialises with a given number of classes
		* \param num_classes The number of discrete classes
		*/
		discreteDistribution(const int num_classes)
		{
			initialise(num_classes);
		}

		/*! \brief Initialise with a certain number of classes and reset
		* probabilities to zero
		* \param num_classes The number of discrete classes
		*/
		void initialise(const int num_classes)
		{
			n_classes = num_classes;
			prob.resize(n_classes);
			std::fill(prob.begin(),prob.end(),0.0);
		}

		/*! \brief Reset function - return probabilities to zero
		*
		* Use this when using the class as an output distribution to create a
		* new blank distribution before combining with new node distributions
		*/
		void reset()
		{
			std::fill(prob.begin(),prob.end(),0.0);
		}

		/*! \brief Returns the probability of a particular label
		*
		* This overloaded version does not require the ID and is intended
		* for use by user code.
		* \param x The label of for which the probability is sought
		*/
		float pdf(const int x) const
		{
			return prob[x];
		}

		/*! \brief Normalise the distribution to ensure it is valid
		*
		* This may be used after several \c combineWith() operations to ensure
		* that the resulting distribution represents a valid probability distribution
		*/
		void normalise()
		{
			float sum = 0.0;
			for(int c = 0; c < n_classes; ++c)
				sum += prob[c];

			for(int c = 0; c < n_classes; ++c)
				prob[c] /= sum;
		}

		/*! \brief Prints the defining parameters of the distribution to an
		* output filestream
		*
		* \param stream The stream to which the parameters (the probability
		* values for each class) are printed
		*/
		void printOut(std::ofstream& stream) const
		{
			for(int c = 0; c < n_classes - 1; c++)
				stream << prob[c] << " ";
			stream << prob[n_classes - 1];
		}

		/*! \brief Reads the defining parameters of the distribution from a
		* filestream
		*
		* \param stream The stream from which the parameters (probability values
		* for each class) are to be read
		*/
		void readIn(std::ifstream& stream)
		{
			for(int c = 0; c < n_classes; c++)
				stream >> prob[c];
		}

		/*! \brief Smooth the distribution using the softmax function
		*
		* This alters the probability distribution by replacing the probability
		* of class \f$ i \f$ according to
		* \f[ p_i \leftarrow \frac{ e^{\frac{p_i}{T}}}{\sum_{j=1}^N {e^\frac{p_j}{T}} } \f]
		*
		* where \f$ N \f$ is the number of classes and \f$ T \f$ is a temperature
		* parameter.
		* This has the effect of regularising the distribution, reducing the certainty.
		*
		* \param T The temperature parameter. The higher the temperature, the
		* more the certainty is reduced. T must be a strictly positive number,
		* otherwise this function will have no effect.
		*
		*/
		void raiseDistributionTemperature(const double T)
		{
			if(T > 0.0)
			{
				for(int c = 0; c < n_classes; ++c)
				prob[c] = std::exp(prob[c]/T);
				normalise();
			}
		}

		// Template methods, defined below
		// --------------------------------

		// Function to fit the parameters of the distribution, given a set of labels
		template <class TLabelIterator, class TIdIterator>
		void fit(TLabelIterator first_label, TLabelIterator last_label, TIdIterator /*unused*/);

		// Get the pdf of a given id and label
		template <class TId>
		float pdf(const int x, const TId /*id*/) const;

		// Aggregates the influence of the dist into the model, but does not necessarily normalise
		template <class TId>
		void combineWith(const discreteDistribution& dist, const TId /*id*/);

		/*! \brief Allows the distribution to be written to a file via the
		* streaming operator '<<'
		*/
		friend std::ofstream& operator<< (std::ofstream& stream, const discreteDistribution& dist) { dist.printOut(stream); return stream;}

		/*! \brief Allows the distribution to be written to read from a file
		* via the streaming operator '>>'
		*/
		friend std::ifstream& operator>> (std::ifstream& stream, discreteDistribution& dist) { dist.readIn(stream); return stream;}

	protected:
		// Data
		int n_classes; //!< The number of discrete classes
		std::vector<float> prob; //!< Vector containing the probabilities of each class

};



/*! \brief Fit the distribution to a set of labels.
*
* Fits the discrete distribution to the set of labels between first_label and
* last label. Expects the labels to take value between 0 and N-1 inclusive,
* where N is the number of classes that the distribution has been initialised with.
* There are no checks to ensure this.
*
* \tparam TLabelIterator The type of the iterator used to access the labels of
* the training data. Must be a forward iterator that dereferences to an integral
* type.
* \tparam TIdIterator The type of the iterator used to access the IDs of the
* data points. The ID is unused but required for compatibility with randomForestBase.
* \param first_label Iterator to the first label
* \param last_label Iterator to the last label
* \param - The third parameter is unused but required for compatibility with
* randomForestBase
*/
template <class TLabelIterator, class TIdIterator>
void discreteDistribution::fit(TLabelIterator first_label, const TLabelIterator last_label, TIdIterator /*unused*/)
{
	const int n_data = std::distance(first_label, last_label);

	if(n_data == 0)
	{
		std::fill(prob.begin(),prob.end(),1.0/float(n_classes));
	}
	else
	{
		std::fill(prob.begin(),prob.end(),0.0);

		for( ; first_label != last_label; ++first_label)
			prob[*first_label] += 1.0;

		std::for_each(prob.begin(),prob.end(), [=] (float& p) { p /= float(n_data); });
	}
}

/*! \brief Returns the probability of a particular label
*
* This is the version used by the randomForestBase methods.
* \tparam TId The type of the IDs of the data points. The ID is unused but
* required for compatibility with randomForestBase.
* \param x The label of for which the probability is sought
* \param - The second parameter is unused and but required for compatibility with
* randomForestBase
*/
template<class TId>
float discreteDistribution::pdf(const int x, const TId /*id*/) const
{
	return prob[x];
}

/*! \brief Combine this distribution with a second by summing the probability
* values, without normalisation.
*
* This method is used by the randomForestBase methods to aggregate the
* distributions in several leaf nodes into one output distribution.
*
* \tparam TId The type of the IDs of the data points. The ID is unused but
* required for compatibility with randomForestBase.
* \param dist The distribution that this distribution should be combined with.
* \param - The second parameter is unused and but required for compatibility with
* randomForestBase
*/
template <class TId>
void discreteDistribution::combineWith(const discreteDistribution& dist, const TId /*id*/)
{
	for(int c = 0; c < n_classes; c++)
		prob[c] += dist.prob[c];
}

}// end of namespace

#endif
// DISCRETEDISTRIBUTION_HPP
