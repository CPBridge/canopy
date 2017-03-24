#ifndef CIRCULARREGRESSOR_HPP
#define CIRCULARREGRESSOR_HPP

/*!
* \file circularRegressor.hpp
* \author Christopher P Bridge
* \brief Contains declaration of the circularRegressor class
*/

#include <canopy/circularRegressor/vonMisesDistribution.hpp>
#include <canopy/randomForestBase/randomForestBase.hpp>

namespace canopy
{

/*!
* \brief Implements a random forest classifier model to predict a circular-valued
* output label.
*
* This class uses the vonMisesDistribution as both the output distribution and
* the node distribution, and float as the type of the label to predict.
*
* \tparam TNumParams The number of parameters used by the features callback functor.
*/
template <unsigned TNumParams>
class circularRegressor : public randomForestBase<circularRegressor<TNumParams>,float,vonMisesDistribution,vonMisesDistribution,TNumParams>
{
	public:

		// Methods
		// -------
		circularRegressor();
		circularRegressor(const int num_trees, const int num_levels, const float info_gain_tresh = C_DEFAULT_MIN_INFO_GAIN);

	protected:
		/*! \brief Forward the definition of the type declared in the randomForestBase class */
		typedef typename randomForestBase<circularRegressor<TNumParams>,float,vonMisesDistribution,vonMisesDistribution,TNumParams>::scoreInternalIndexStruct scoreInternalIndexStruct;

		// Methods
		// -------

		void initialiseNodeDist(const int t, const int n);
		template <class TLabelIterator>
		float singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int /*tree*/, const int /*node*/) const;
		template <class TLabelIterator, class TIdIterator>
		void trainingPrecalculations(const TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator /*unused*/);
		void cleanupPrecalculations();
		template <class TLabelIterator>
		void bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int /*tree*/, const int /*node*/, const float initial_impurity,float& info_gain, float& thresh) const;
		float minInfoGain(const int /*tree*/, const int /*node*/) const;
		void printHeaderDescription(std::ofstream& /*stream*/) const;
		void printHeaderData(std::ofstream& /*stream*/) const;
		void readHeader(std::ifstream& /*stream*/);

		// Data
		// ----
		std::vector<double> sin_precalc; //!< Used during training to store pre-calculated sines of the training labels
		std::vector<double> cos_precalc; //!< Used during training to store pre-calculated cosines of the training labels
		float min_info_gain; //!< If during training, the best information gain at a node goes below this threshold, a lead node is declared

		// Constants
		// ---------
		static constexpr int C_NUM_SPLIT_TRIALS = 100; //!< This is the number of possible splits tested for each feature during training.
		static constexpr float C_DEFAULT_MIN_INFO_GAIN = 0.1; //!< Default value for the information gain threshold.
};

} // end of namespace

#include <canopy/circularRegressor/circularRegressor.tpp>
#endif
// CIRCULARREGRESSOR_HPP
