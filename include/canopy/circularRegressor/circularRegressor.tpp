/*!
* \file circularRegressor.tpp
* \author Christopher P Bridge
* \brief Contains implementations of the methods of the canopy::circularRegressor class
*/

#include <algorithm>
#include <limits>

namespace canopy
{

/*! \brief Default constructor
*
* Note that an object initialised in this way should not be trained, but may
* be used to read in a pre-trained model using \c readFromFile()
*/
template <unsigned TNumParams>
circularRegressor<TNumParams>::circularRegressor()
: randomForestBase<circularRegressor<TNumParams>,float,vonMisesDistribution,vonMisesDistribution,TNumParams>(), min_info_gain(C_DEFAULT_MIN_INFO_GAIN)
{

}

/*! \brief Full constructor
*
* Creates a full forest with a specified number of trees and levels, ready to be
* trained.
* \param num_trees The number of decision trees in the forest
* \param num_levels The maximum depth of any node in the trees
* \param info_gain_tresh The information gain threshold to use when training
* the model. Nodes where the best split is found to result in an information
* gain value less than this threshold are made into leaf nodes. Default:
* C_DEFAULT_MIN_INFO_GAIN
*/
template <unsigned TNumParams>
circularRegressor<TNumParams>::circularRegressor(const int num_trees, const int num_levels, const float info_gain_tresh)
: randomForestBase<circularRegressor<TNumParams>,float,vonMisesDistribution,vonMisesDistribution,TNumParams>(num_trees, num_levels), min_info_gain(info_gain_tresh)
{

}

/*! \brief Initialise a vonMisesDistribution as a node distribution for training
*
* This method is called automatically by the base class.
*
* \param t Index of the tree in which the distribution is to be initialised
* \param n Index of the node to be initialised within its tree
*/
template <unsigned TNumParams>
void circularRegressor<TNumParams>::initialiseNodeDist(const int t, const int n)
{
	this->forest[t].nodes[n].post[0].initialise();
}

/*! \brief Preliminary calculations to perform berfore training begins.
*
* In this case this pre-calculates an array values of the sines and cosines of
* the training labels to avoid calculating these many many times
*
* This method is called automatically by the base class.
*
* \tparam TLabelIterator Type of the iterator used to access the training labels
* Must be a random access iterator than dereferences to an floating point data type.
* \tparam TIdIterator Type of the iterator used to access the IDs of the training
* data. The IDs are unused by required for compatibility with randomForestBase .
*
* \param first_label Iterator to the first label in the training set
* \param last_label Iterator to the last label in the training set
* \param - The third parameter is unused but required for compatibility with
* randomForestBase
*/
template <unsigned TNumParams>
template <class TLabelIterator, class TIdIterator>
void circularRegressor<TNumParams>::trainingPrecalculations(const TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator /*unused*/)
{
	// Find the highest ID and create an array to hold sines and cosines up to this,
	// even if they are not used
	const int num_ids = std::distance(first_label,last_label);
	sin_precalc.resize(num_ids);
	cos_precalc.resize(num_ids);

	for(int d = 0; d < num_ids; ++d)
	{
		sin_precalc[d] = std::sin(first_label[d]);
		cos_precalc[d] = std::cos(first_label[d]);
	}
}

/*! \brief Clean-up of data to perform after training ends.
*
* In this case this clears the pre-calculated arrays created by
* \c trainingPrecalculations()
*
* This method is called automatically by the base class.
*/
template <unsigned TNumParams>
void circularRegressor<TNumParams>::cleanupPrecalculations()
{
	sin_precalc.clear();
	cos_precalc.clear();
}

// Function to find the best split of a set of sorted angle and score pairs - uses an approximate measure of spread instead of true entropy
/*! \brief Find the best way to split training data using the scores of a certain
* feature.
*
* This method takes a set of training data points and their scores resulting from
* some feature, and calculates the best score threshold that may be
* used to split the data into two partitions. The best split is the one that
* results in the greatest information gain in the child nodes, which in this case
* is based on squared circular distances from the circular mean.
*
* This method is called automatically by the base class.
*
* \tparam TLabelIterator Type of the iterator used to access the discrete labels.
* Must be a random access iterator that dereferences to a floating point data type.
* \param data_structs A vector in which each element is a structure containing
* the internal id (.id) and score (.score) for the current feature of the
* training data points. The vector is assumed to be sorted according to the score
* field in ascending order.
* \param first_label Iterator to the labels for which the entropy is to be
* calculated. The labels should be located at the offsets from this iterator given
* by the IDs of elements of the data_structs vector. I.e.
* \code
* first_label[data_structs[0].id]
* first_label[data_structs[1].id]
* \endcode
* etc.
* \param - The third parameter is unused but required for compatibility with
* randomForestBase
* \param - The fourth parameter is unused but required for compatibility with
* randomForestBase
* \param initial_impurity The initial impurity of the node before the split.
* This must be calculated with \c singleNodeImpurity() and passed in
* \param info_gain The information gain associated with the best split (i.e.
* the maximum achievable information gain with this feature) is returned by
* reference in this parameter
* \param thresh The threshold value of the feature score corresponding to tbe
* best split is returned by reference in this parameter
*/
template <unsigned TNumParams>
template <class TLabelIterator>
void circularRegressor<TNumParams>::bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int /*tree*/, const int /*node*/, const float initial_impurity,float& info_gain, float& thresh) const
{
	const double minval = data_structs.front().score;
	const double maxval = data_structs.back().score;
	const double hspace = (maxval-minval)/C_NUM_SPLIT_TRIALS;

	// Precalculate the cumulative sin and cosine of the labels for speed
	std::vector<double> cumcos(data_structs.size());
	std::vector<double> cumsin(data_structs.size());

	cumsin[0] = sin_precalc[data_structs[0].id];
	cumcos[0] = cos_precalc[data_structs[0].id];
	for(int d = 1; d < int(data_structs.size()); ++d)
	{
		cumsin[d] = cumsin[d-1] + sin_precalc[data_structs[d].id];
		cumcos[d] = cumcos[d-1] + cos_precalc[data_structs[d].id];
	}

	// Prepare for loop
	auto split_it = data_structs.cbegin();
	double best_impurity = std::numeric_limits<double>::max();
	double plateau_start_thresh;
	bool plateau_flag = false;

	// Loop through threshold values
	for(int h = 1; h < C_NUM_SPLIT_TRIALS; ++h)
	{
		// Find the score threshold value
		const double split_thresh = minval + h*hspace;

		// Check that this new threshold actually splits the data in
		// a different way to the previous threshold
		if( split_it->score >= split_thresh )
		{
			// Move the threshold to half way between this point and
			// the start of the plateau
			if(plateau_flag)
				thresh = (split_thresh + plateau_start_thresh)/2.0;

			// No need to calculate the purity again - it's the same!
			continue;
		}

		plateau_flag = false;

		// Find the point in the sorted vector
		// After this loop, split_t should point to the first data point that lies above the threshold
		while( split_it->score < split_thresh )
			++split_it;

		// Find numbers in the left and right sides
		const int Nl = std::distance(data_structs.cbegin(),split_it);

		// Find the mean of the left side and then ssd from it
		const double left_mean = std::atan2(cumsin[Nl-1],cumcos[Nl-1]);
		double ssd_left = 0.0;
		for(auto left_it = data_structs.cbegin() ; left_it != split_it; ++left_it)
			ssd_left += std::pow(0.5*(1.0 - std::cos(first_label[left_it->id]-left_mean)),2);

		// Find the mean of the right side and then ssd from it
		const double right_mean = std::atan2(cumsin[data_structs.size()-1] - cumsin[Nl-1], cumcos[data_structs.size()-1] - cumcos[Nl-1]);
		double ssd_right = 0.0;
		for(auto right_it = split_it ; right_it != data_structs.cend(); ++right_it)
			ssd_right += std::pow(0.5*(1.0 - std::cos(first_label[right_it->id]-right_mean)),2);

		// See whether this is the best split so far
		if(ssd_left + ssd_right < best_impurity)
		{
			best_impurity = ssd_left + ssd_right;
			thresh = split_thresh;

			plateau_flag = true;
			plateau_start_thresh = split_thresh;
		}

	}

	// return values
	info_gain = initial_impurity - best_impurity;
}

/*! \brief Calculate the impurity of the label set in a single node.
*
* This method takes the labels (angular labels) of a set of training
* data points and calculates the impurity of that set. In this case, this
* is based on the squared circular distance from the circular mean of the set.
*
* This method is called automatically by the base class.
* \tparam TLabelIterator Type of the iterator used to access the discrete labels.
* Must be a random access iterator that dereferences to an floating point data type.
* \param first_label Iterator to the labels for which the entropy is to be
* calculated. The labels should be located at the offsets from this iterator given
* by the elements of the nodebag vector. I.e.
* \code
* first_label[nodebag[0]]
* first_label[nodebag[1]]
* \endcode etc.
* \param nodebag Vector containing the internal training indices of the
* data points. These are the indices through which the labels may be accessed in
* first_label
* \param - The third parameter is unused but required for compatibility with
* randomForestBase
* \param - The fourth parameter is unused but required for compatibility with
* randomForestBase
*/
template <unsigned TNumParams>
template <class TLabelIterator>
float circularRegressor<TNumParams>::singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int /*tree*/, const int /*node*/) const
{
	// First find the mean
	double S = 0.0, C = 0.0;
	for(int id : nodebag)
	{
		S += sin_precalc[id];
		C += cos_precalc[id];
	}
	const double mean = std::atan2(S,C);

	// Use this to find sum of distances from the mean
	float ssd = 0.0;
	std::for_each(nodebag.cbegin(),nodebag.cend(),[&] (int d) {ssd += std::pow(0.5*(1.0 - std::cos(first_label[d]-mean)),2);} );

	return ssd;
}

/*! \brief Prints a string that allows a human to interpret the header information
* to a stream.
*
* This header is blank in the case of the circularRegressor .
* This method is called automatically by the base class.
*
* \param stream The stream to which the header description is printed.
*/
template <unsigned TNumParams>
void circularRegressor<TNumParams>::printHeaderDescription(std::ofstream& /*stream*/) const
{

}

/*! \brief Print the header information specific to the circularRegressor model
* to a stream.
*
* This header is blank in the case of the circularRegressor .
*
* This method is called automatically by the base class.
*
* \param stream The stream to which the header is printed.
*/
template <unsigned TNumParams>
void circularRegressor<TNumParams>::printHeaderData(std::ofstream& /*stream*/) const
{

}

/*! \brief Read the header information specific to the circularRegressor model
* from a stream.
*
* This header is blank in the case of the circularRegressor .
*
* This method is called automatically by the base class.
*
* \param stream The stream from which the header information is read.
*/
template <unsigned TNumParams>
void circularRegressor<TNumParams>::readHeader(std::ifstream& /*stream*/)
{

}

/*! \brief Get the information gain threshold for a given node
*
* In this case, this is a fixed value for all nodes. This method is called
* automatically by the base class.
* \param - The first parameter is unused but required for compatibility with
* \c randomForestBase
* \param - The second parameter is unused but required for compatibility with
* \c randomForestBase
* \return The threshold value for information gain. If a split results in a
* information gain less than this value, the node should be made a leaf instead.
*/
template <unsigned TNumParams>
float circularRegressor<TNumParams>::minInfoGain(const int /*tree*/, const int /*node*/) const
{
	return min_info_gain;
}

} // end of namespace
