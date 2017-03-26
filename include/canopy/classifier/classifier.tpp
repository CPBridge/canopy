/*!
* \file classifier.tpp
* \author Christopher P Bridge
* \brief Contains implementations of the methods of the canopy::classifier class
*/

#include <limits>
#include <sstream>

namespace canopy
{

/*! \brief Full constructor
*
* Creates a full forest with a specified number of trees and levels, ready to be
* trained.
* \param num_classes Number of discrete classes in the label space. The labels
* are assumed to run from 0 to num_classes-1 inclusive.
* \param num_trees The number of decision trees in the forest
* \param num_levels The maximum depth of any node in the trees
* \param info_gain_tresh The information gain threshold to use when training
* the model. Nodes where the best split is found to result in an information
* gain value less than this threshold are made into leaf nodes. Default:
* C_DEFAULT_MIN_INFO_GAIN
*/
template <unsigned TNumParams>
classifier<TNumParams>::classifier(const int num_classes, const int num_trees, const int num_levels, const double info_gain_tresh)
: randomForestBase<classifier<TNumParams>,int,discreteDistribution,discreteDistribution,TNumParams>(num_trees, num_levels), n_classes(num_classes), min_info_gain(info_gain_tresh)
{
}

/*! \brief Default constructor
*
* Note that an object initialised in this way should not be trained, but may
* be used to read in a pre-trained model using \c readFromFile()
*/
template <unsigned TNumParams>
classifier<TNumParams>::classifier()
: randomForestBase<classifier<TNumParams>,int,discreteDistribution,discreteDistribution,TNumParams>(), n_classes(0), min_info_gain(C_DEFAULT_MIN_INFO_GAIN)
{
}

/*! \brief Set the class name strings
*
* These will be stored within the model (including when written to file) and may
* be retrieved at a later date, however they do not affect the operation of the
* model in any way and are entirely optional.
*
* \param new_class_names Vector with each element containing the name of one class
*/
template <unsigned TNumParams>
void classifier<TNumParams>::setClassNames(const std::vector<std::string>& new_class_names)
{
	class_names = new_class_names;
}

/*! \brief Get the class name strings
*
* Retrieve a previously stored set of class names.

* \param class_names The class names are returned by reference in this vector.
* If none have been set, an empty vector is returned.
*/
template <unsigned TNumParams>
void classifier<TNumParams>::getClassNames(std::vector<std::string>& class_names) const
{
	class_names = this->class_names;
}

/*! \brief Initialise a discreteDistribution as a node distribution for training
*
* This method is called automatically by the base class.
*
* \param t Index of the tree in which the distribution is to be initialised
* \param n Index of the node to be initialised within its tree
*/
template <unsigned TNumParams>
void classifier<TNumParams>::initialiseNodeDist(const int t, const int n)
{
	this->forest[t].nodes[n].post[0].initialise(n_classes);
}

/*! \brief Preliminary calculations to perform berfore training begins.
*
* In this case this pre-calculates an array of values of x*log(x) to speed up entropy
* calculations.
*
* This method is called automatically by the base class.
*
* \tparam TLabelIterator Type of the iterator used to access the training labels
* Must be a random access iterator than dereferences to an integral data type.
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
void classifier<TNumParams>::trainingPrecalculations(const TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator /*unused*/)
{
	const int num_id = std::distance(first_label,last_label);
	xlogx_precalc = this->preCalculateXlogX(num_id);
}

/*! \brief Clean-up of data to perform after training ends.
*
* In this case this clears the pre-calculated array created by
* \c trainingPrecalculations()
*
* This method is called automatically by the base class.
*/
template <unsigned TNumParams>
void classifier<TNumParams>::cleanupPrecalculations()
{
	xlogx_precalc.clear();
}

/*! \brief Find the best way to split training data using the scores of a certain
* feature.
*
* This method takes a set of training data points and their scores resulting from
* some feature, and calculates the best score threshold that may be
* used to split the data into two partitions. The best split is the one that
* results in the greatest information gain in the child nodes, which in this case
* is based on the discrete entropy.
*
* This method is called automatically by the base class.
*
* \tparam TLabelIterator Type of the iterator used to access the discrete labels.
* Must be a random access iterator that dereferences to an integral data type.
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
void classifier<TNumParams>::bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int /*unused*/, const int /*unused*/, const float initial_impurity, float& info_gain, float& thresh) const
{
	// Number of data points (makes code more readbable)
	const int N = data_structs.size();

	// Call the base class routine for fast calculation of the best split
	double best_children_impurity;
	this->fastDiscreteEntropySplit(data_structs, n_classes, first_label, xlogx_precalc, best_children_impurity, thresh);

	// Values to return
	info_gain = initial_impurity - best_children_impurity/N;
}

// Calculates the impurity (entropy) of a single node
/*! \brief Calculate the impurity of the label set in a single node.
*
* This method takes the labels (discrete class labels) of a set of training
* data points and calculates the impurity of that set. In this case, this
* is based on the discrete entropy of the set.
*
* This method is called automatically by the base class.
* \tparam TLabelIterator Type of the iterator used to access the discrete labels.
* Must be a random access iterator that dereferences to an integral data type.
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
float classifier<TNumParams>::singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int /*tree*/, const int /*node*/) const
{
	return this->fastDiscreteEntropy(nodebag,n_classes,first_label,xlogx_precalc);
}

/*! \brief Prints a string that allows a human to interpret the header information
* to a stream.
*
* This method is called automatically by the base class.
*
* \param stream The stream to which the header description is printed.
*/
template <unsigned TNumParams>
void classifier<TNumParams>::printHeaderDescription(std::ofstream &stream) const
{
	stream << "n_classes [Class names]";
}

/*! \brief Print the header information specific to the classifier model
* to a stream.
*
* This prints out the number of classes and the class names to the stream.
*
* This method is called automatically by the base class.
*
* \param stream The stream to which the header is printed.
*/
template <unsigned TNumParams>
void classifier<TNumParams>::printHeaderData(std::ofstream &stream) const
{
	stream << n_classes;
	for(auto str : class_names)
		stream << " " << str;
}

/*! \brief Read the header information specific to the classifier model
* from a stream.
*
* This reads in the number of classes and the class names from the stream.
*
* This method is called automatically by the base class.
*
* \param stream The stream from which the header information is read.
*/
template <unsigned TNumParams>
void classifier<TNumParams>::readHeader(std::ifstream &stream)
{
	using namespace std;
	string line;
	getline(stream,line);
	stringstream ss(line);

	ss >> n_classes;
	string temp;
	while(ss >> temp)
		class_names.emplace_back(temp);
	while(int(class_names.size()) < n_classes)
		class_names.emplace_back(string("Class ") + to_string(class_names.size()));
}

/*! \brief Get the number of classes in the discrete label space of the model
*
* \return The number of classes
*/
template <unsigned TNumParams>
int classifier<TNumParams>::getNumberClasses() const
{
	return n_classes;
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
float classifier<TNumParams>::minInfoGain(const int, const int) const
{
	return min_info_gain;
}

/*! \brief Smooth the distributions in all of the leaf nodes using the softmax
* function
*
* This alters the probability distributions by replacing the probability
* of class \f$ i \f$ according to
* \f[ p_i \leftarrow \frac{ e^{\frac{p_i}{T}}}{\sum_{j=1}^N {e^\frac{p_j}{T}} } \f]
*
* where \f$ N \f$ is the number of classes and \f$ T \f$ is a temperature
* parameter.
* This has the effect of regularising the distributions, reducing the certainty.
*
* \param T The temperature parameter. The higher the temperature, the
* more the certainty is reduced. T must be a strictly positive number,
* otherwise this function will have no effect.
*/
template <unsigned TNumParams>
void classifier<TNumParams>::raiseNodeTemperature(const double T)
{
	for(int t = 0; t < this->n_trees; ++t)
	{
		for(int n = 0; n < this->n_nodes; ++n)
		{
			if(this->forest[t].nodes[n].is_leaf)
			{
				this->forest[t].nodes[n].post[0].raiseDistributionTemperature(T);
			}
		}
	}
}

} // end of namespace
