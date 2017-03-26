#ifndef CLASSIFER_HPP
#define CLASSIFER_HPP

/*!
* \file classifier.hpp
* \author Christopher P Bridge
* \brief Contains the declaration of the canopy::classifier class
*/

#include <canopy/randomForestBase/randomForestBase.hpp>
#include <canopy/classifier/discreteDistribution.hpp>

namespace canopy
{

/*!
* \brief Implements a random forest classifier model to predict a discrete
* output label.
*
* This class uses the discreteDistribution as both the output distribution and
* the node distribution, and int as the type of label to predict.
*
* \tparam TNumParams The number of parameters used by the features callback functor.
*/
template <unsigned TNumParams>
class classifier : public randomForestBase<classifier<TNumParams>,int,discreteDistribution,discreteDistribution,TNumParams>
{
	public:
		// Methods
		classifier(const int num_classes, const int num_trees, const int num_levels, const double info_gain_tresh = C_DEFAULT_MIN_INFO_GAIN); // constructor
		classifier();
		int getNumberClasses() const;
		void setClassNames(const std::vector<std::string>& new_class_names);
		void getClassNames(std::vector<std::string>& end_class_names) const;
		void raiseNodeTemperature(const double T);

	protected:
		/*! \brief Forward the definition of the type declared in the randomForestBase class */
		typedef typename randomForestBase<classifier<TNumParams>,int,discreteDistribution,discreteDistribution,TNumParams>::scoreInternalIndexStruct scoreInternalIndexStruct;

		// Methods
		void initialiseNodeDist(const int t, const int n);
		template <class TLabelIterator>
		void bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int /*tree*/, const int /*node*/, const float initial_impurity,float& info_gain, float& thresh) const;
		void printHeaderDescription(std::ofstream &stream) const;
		void printHeaderData(std::ofstream &stream) const;
		void readHeader(std::ifstream &stream);
		float minInfoGain(const int /*tree*/, const int /*node*/) const;
		template <class TLabelIterator>
		float singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int /*tree*/, const int /*node*/) const;
		template <class TLabelIterator, class TIdIterator>
		void trainingPrecalculations(const TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator/*unused*/);
		void cleanupPrecalculations();

		// Data
		int n_classes; //!< The number of classes in the discrete label space
		std::vector<std::string> class_names; //!< The names of the classes
		std::vector<double> xlogx_precalc; //!< Used for storing temporary precalculations of x*log(x) values during training
		double min_info_gain; //!< If during training, the best information gain at a node goes below this threshold, a lead node is declared

		// Constants
		static constexpr double C_DEFAULT_MIN_INFO_GAIN = 0.05; //!< Default value for the information gain threshold.
};

} // end of namespace

#include <canopy/classifier/classifier.tpp>
#endif
// CLASSIFER_HPP
