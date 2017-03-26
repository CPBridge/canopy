/*!
* \file randomForestBase.tpp
* \author Christopher P Bridge
* \brief Contains implementations of the methods of the canopy::randomForestBase class.
*/

// Include template class definition
#include <cmath>		 /* pow */
#include <utility>		/* pair */
#include <algorithm>	  /* min_element, max_element, shuffle */
#include <numeric>		/* iota */
#include <limits>		 /* numeric limits */
#include <assert.h>	   /* assert macro */
#include <boost/iterator/permutation_iterator.hpp>

namespace canopy
{

// Constructor
/*! \brief Full constructor
*
* Creates a full forest with a specified number of trees and levels, ready to be
* trained.
* \param num_trees The number of decision trees in the forest
* \param num_levels The maximum depth of any node in the trees
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::randomForestBase(const int num_trees, const int num_levels)
	: n_trees(num_trees), n_levels(num_levels)
{
	// Basic initialisation
	initialise();

	// Allocate memory
	allocateForestMemory();

}

// Basic constructor
/*! \brief Default constructor
*
* Note that an object initialised in this way should not be trained, but may
* be used to read in a pre-trained model using \c readFromFile()
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::randomForestBase()
{
	initialise();
}

/*! \brief Contains code common to both constructors.
*
* Removes any existing data and seeds the random engine.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::initialise()
{
	valid = false;

	// Seed the random number generator
	std::random_device rd{};
	rand_engine.seed(rd());

	feature_header = "";
	feature_string = "";

	// Remove any existing node data
	forest.clear();
}

/*! \brief Allocates memory for the forest with the required number of trees and
* levels/depth.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::allocateForestMemory()
{
	// Allocate memory for the forest
	forest.resize(n_trees);

	// Find number of nodes
	n_nodes = std::pow(2,n_levels+1) - 1;

	// Allocate memory for each tree
	for(int t = 0; t < n_trees; ++t)
	{
		(forest[t]).nodes.resize(n_nodes);
	}

}


/*! \brief Read a pre-trained model in from a file.
*
* Read in the parameters for a forest from a pre-trained model stored in a .tr
* file. After this function, the object will be ready to use for testing with
* the pre-trained model.
*
* \param filename The full name and path of the .tr file to read.
* \param trees_used The number of trees to read in from the file. If this is
* unspecified or set to a negative value, all the trees in the .tr file will be
* used. If the number specified is greater than the number trianed in .tr file,
* the function will fail and return zero.
* \param max_depth_used The maximum tree depth to read from the the file. If this
* is unspecified or set to a negative value, all the levels in the .tr file will
* be used. If the number specified is greater than the number trianed in .tr file,
* the function will fail and return zero. The .tr must have been trained with the
* fit_split_nodes option set to true for this option to be successful.
* \return True if the model was successfully read from the file, false otherwise.
* If false, the model should not be used.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
bool randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::readFromFile(const std::string filename, const int trees_used, const int max_depth_used)
{
	// Declarations
	std::string dummy_string;
	int last_split_node = -1, last_leaf_node = -1;

	// Open the text file and check it opened ok
	std::ifstream infile(filename.c_str());
	if (!infile.is_open()) return false;

	// Read in the feature defition strings
	getline(infile,dummy_string); // header line (ignore)
	getline(infile,feature_string); // the actual first feature line

	// The next line is blank, then it's a comment line
	getline(infile,dummy_string);
	getline(infile,dummy_string);

	// Read in the number of levels and trees
	infile >> n_trees;
	if(infile.fail()) return false;
	if(trees_used > n_trees)
		return false;
	else if(trees_used > 0)
		n_trees = trees_used;

	infile >> n_levels;
	if(infile.fail()) return false;
	if(max_depth_used > n_levels) return false;

	infile >> fit_split_nodes;
	if(infile.fail()) return false;

	const int n_nodes_in_file = std::pow(2,n_levels+1) - 1;
	if(max_depth_used > -1)
	{
		if(!fit_split_nodes)
			return false;
		last_split_node = std::pow(2,max_depth_used) - 2;
		last_leaf_node = std::pow(2,max_depth_used+1) - 2;
		n_levels = max_depth_used;
	}

	// Read in implementation-specfic parameters
	getline(infile,dummy_string);
	getline(infile,dummy_string);
	static_cast<derivedProxy*>(this)->readHeader(infile);
	if(infile.fail()) return false;

	// Allocate memory to hold this tree structure
	allocateForestMemory();

	// Loop through the trees
	for(int t = 0; t < n_trees; ++t)
	{
		// Flag that is true if the parent node is a leaf or orphan node
		std::vector<bool> orphan_flag(n_nodes_in_file,false);

		// Loop through nodes
		for(int n = 0; n < n_nodes_in_file; ++n)
		{
			// Skip if this is an orphan, and set children to orphans
			if(orphan_flag[n])
			{
				if(2*n + 2 < n_nodes_in_file)
				{
					orphan_flag[2*n+1] = true;
					orphan_flag[2*n+2] = true;
				}
				continue;
			}

			// Check that this node isn't beyond the max depth to read in
			if((max_depth_used > -1) && (n > last_leaf_node))
			{
				// If this was a leaf in the original tree, the children will be
				// orphans
				bool isleaf;
				infile >> isleaf;
				if(isleaf && (2*n + 2 < n_nodes_in_file) )
				{
					orphan_flag[2*n+1] = true;
					orphan_flag[2*n+2] = true;
				}
				// Skip the rest of the information on this line, then move on
				getline(infile,dummy_string);
				continue;
			}

			infile >> forest[t].nodes[n].is_leaf;
			if(infile.fail()) return false;

			// This node is a leaf if it is marked as one, or if it lies at
			// the maximum level to read in the trees
			if(forest[t].nodes[n].is_leaf || ( (max_depth_used > -1) && (n > last_split_node) ) )
			{
				if(forest[t].nodes[n].is_leaf)
				{
					if(2*n + 2 < n_nodes_in_file)
					{
						orphan_flag[2*n+1] = true;
						orphan_flag[2*n+2] = true;
					}
				}
				else
				{
					// Skip over the paramters and the threshold as they are irrelevant
					// if this is to be a leaf node
					float tempfloat;
					int tempint;
					for(unsigned p = 0; p < TNumParams; ++p)
						infile >> tempint;
					infile >> tempfloat;
					if(infile.fail()) return false;
					forest[t].nodes[n].is_leaf = true;
				}

				// Prepare and read in the posterior distribution
				forest[t].nodes[n].post.resize(1);
				static_cast<derivedProxy*>(this)->initialiseNodeDist(t,n);
				infile >> forest[t].nodes[n].post[0];
				if(infile.fail()) return false;

			}
			else
			{
				for(unsigned p = 0; p < TNumParams; ++p)
				{
					infile >> forest[t].nodes[n].params[p];
					if(infile.fail()) return false;
				}
				infile >> forest[t].nodes[n].thresh;
				if(infile.fail()) return false;
				forest[t].nodes[n].post.clear();
				if(fit_split_nodes)
					getline(infile,dummy_string); // skip unneeded posterior
			}
		} // node loop
	}

	infile.close();

	valid = true;
	return true;

}


// Function to write the forest structure to a file
/*! \brief Write a trained model to a .tr file to be stored and re-used.
*
* Ensure that \c setFeatureDefinitionString() is called before this function,
* otherwise a blank feature definition string will be stored.
*
* \param filename The full name and path of the file into which the model should
* be written.
* \return True if the model was successfully written to the specified file, false
* otherwise.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
bool randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::writeToFile(const std::string filename) const
{
	using namespace std;

	// Open the text file and check it opened ok
	ofstream outfile(filename.c_str());
	if (!outfile.is_open()) return false;

	// Output the feature definition lines
	outfile << "# " << feature_header << endl << feature_string << endl << endl;

	// Write the number of levels and trees
	outfile << "# Trees Levels Split_Dists\n";
	outfile << n_trees << " " << n_levels << " " << fit_split_nodes << '\n';

	// Write the implementation-specific information
	outfile << "# ";
	static_cast<const derivedProxy*>(this)->printHeaderDescription(outfile);
	outfile << '\n';
	static_cast<const derivedProxy*>(this)->printHeaderData(outfile);
	outfile << "\n\n";

	// Loop through the trees
	for(int t = 0; t < n_trees; ++t)
	{
		std::vector<bool> orphan_flag(n_nodes,false);

		// Loop through nodes in this level
		for(int n = 0; n < n_nodes; ++n)
		{
			// Skip if this is an orphan, and set children to orphans
			if(orphan_flag[n])
			{
				if(2*n + 2 < n_nodes)
				{
					orphan_flag[2*n+1] = true;
					orphan_flag[2*n+2] = true;
				}
				continue;
			}

			outfile << forest[t].nodes[n].is_leaf << " " ;

			if(forest[t].nodes[n].is_leaf)
			{
				outfile << forest[t].nodes[n].post[0] << '\n';
				if(2*n + 2 < n_nodes)
				{
					orphan_flag[2*n+1] = true;
					orphan_flag[2*n+2] = true;
				}
			}
			else
			{
				for(unsigned p = 0 ; p < TNumParams; ++p)
					outfile << forest[t].nodes[n].params[p] << " ";
				outfile << forest[t].nodes[n].thresh;
				if(fit_split_nodes)
				{
					outfile << " " ;
					outfile << forest[t].nodes[n].post[0] << '\n';
				}
				else
					outfile << '\n';
			}
		} // node loop

		outfile << '\n';
	} // tree loop

	outfile.close();

	return true;

}

/*! \brief Predict the output distribution for a number of IDs
*
* This function uses the forest model to predict the full output distribution for
* each of a number of data points, where each data point is identified by an ID
* variable.
*
* These ID variables are passed in as a pair of iterators pointing to the first
* and last IDs to be processed. The output distribution for each of these IDs
* is placed in a second container accessed by iterators.
*
* In this version of the function, the features needed by a single node are
* requested from the feature functor for all the IDs with a single function call.
* This involves some overhead, but may permit efficiencies resulting from
* calculating multiple features at once.
*
* Uses OpenMP to query the multiple tree models in parallel.
*
* \tparam TIdIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TId type expected by the feature functor.
* \tparam TOutputIterator Type of the iterator to the output distributions. Must be
* a forward output iterator that dereferences to TOutputDist.
* \tparam TFeatureFunctor The type of the feature functor object. Must meet the
* specifications for a \ref groupwise "groupwise feature functor" object, meaning
* it must define operator() with a certain form.
* \param first_id Iterator to the first ID whose output is to be predicted.
* \param last_id Iterator to the last ID whose output is to be predicted.
* \param out_it Iterator to the output distribution corresponding to the first ID.
* The container of output distributions must already exist, and contain enough
* elements for all of the IDs between first_id and last_id. At the end of this
* function, the output distributions in this container relate to the
* corresponding elements of the id container.
* \param feature_functor The feature functor object to be used as a callback to
* calculate the features. Must be safe to call from multiple threads
* simultaneously.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TIdIterator, class TOutputIterator, class TFeatureFunctor>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::predictDistGroupwise(TIdIterator first_id, const TIdIterator last_id, TOutputIterator out_it, TFeatureFunctor&& feature_functor) const
{
	// Arrays to use to hold leaf nodes
	const int num_id = std::distance(first_id,last_id);
	std::vector<std::vector<const TNodeDist*>> leaves(n_trees,std::vector<const TNodeDist*>(num_id));

	// Loop through all the trees in the forest and find the leaf distributions
	// that each id reaches
	#pragma omp parallel for
	for(int t = 0; t < n_trees; ++t)
		findLeavesGroupwise(first_id,last_id,t,leaves[t], std::forward<TFeatureFunctor>(feature_functor));

	// For each datapoint, go through the trees and combine the leaf distributions
	// Ideally would try to parallelise this...
	int d = 0;
	while(first_id != last_id)
	{
		// Reset any previous calculations
		out_it->reset();

		// Combine results
		for(int t = 0; t < n_trees; ++t)
			out_it->combineWith(*leaves[t][d],*first_id);

		// Normalise
		out_it->normalise();
		++out_it;
		++first_id;
		++d;
	}
}


/*! \brief Predict the output distribution for a number of IDs
*
* This function uses the forest model to predict the full output distribution for
* each of a number of data points, where each data poitn is identified by an ID
* variable.
*
* These ID variables are passed in as a pair of iterators pointing to the first
* and last IDs to be processed. The output distribution for each of these IDs
* is placed in a second container accessed by iterators.
*
* In this version of the function, the features needed by a single node are
* requested from the feature functor one-by-one.
*
* Uses OpenMP to query the multiple tree models in parallel.
*
* \tparam TIdIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TId type expected by the feature functor.
* \tparam TOutputIterator Type of the iterator to the output distributions. Must be
* a forward output iterator that dereferences to TOutputDist.
* \tparam TFeatureFunctor The type of the feature functor object. Must meet the
* specifications for a \ref single "single feature functor" object, meaning it
* must define operator() with a certain form.
* \param first_id Iterator to the first ID whose output is to be predicted.
* \param last_id Iterator to the last ID whose output is to be predicted.
* \param out_it Iterator to the output distribution corresponding to the first ID.
* The container of output distributions must already exist, and contain enough
* elements for all of the IDs between first_id and last_id. At the end of this
* function, the output distributions in this container relate to the
* corresponding elements of the id container.
* \param feature_functor The feature functor object to be used as a callback to
* calculate the features. Must be safe to call from multiple threads
* simultaneously.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TIdIterator, class TOutputIterator, class TFeatureFunctor>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::predictDistSingle(TIdIterator first_id, const TIdIterator last_id, TOutputIterator out_it, TFeatureFunctor&& feature_functor) const
{
	const int num_id = std::distance(first_id,last_id);

	// Loop over data
	#pragma omp parallel for
	for(int d = 0; d < num_id; ++d)
	{
		// Reset any previous calculations
		out_it[d].reset();

		// Vector to store the resulting leaf node from each tree
		std::vector<const TNodeDist*> leaves(n_trees);

		// Loop over trees in the forest
		//#pragma omp parallel for
		for(int t = 0; t < n_trees; ++t)
		{
			leaves[t] = findLeafSingle(first_id[d],t,std::forward<TFeatureFunctor>(feature_functor));
		}

		for(int t = 0; t < n_trees; ++t)
			out_it[d].combineWith(*leaves[t],first_id[d]);

		// Normalise
		out_it[d].normalise();

	}

}

/*! \brief Evaluate the probability of a certain value of the label for a set of
* data points.
*
* This function uses the forest model to evaluate the probability of a given
* value of the label (output) variable for a number of data points, where each
* point data is identified by an ID variable.
*
* These ID variables are passed in as a pair of iterators pointing to the first
* and last IDs to be processed. The value of the label for which the probability
* should be evaluated is passed in as a second iterator. The probability of the
* label for each of these IDs is placed in a third container accessed by iterators.
*
* In this version of the function, the features needed by a single node are
* requested from the feature functor for all the IDs with a single function call.
* This involves some overhead, but may permit efficiencies resulting from
* calculating multiple features at once.
*
* Uses OpenMP to query the multiple tree models in parallel.
*
* \tparam TIdIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TId type expected by the feature functor.
* \tparam TLabelIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TLabel type of the forest (or to something
* trivially convertible to that type).
* \tparam TOutputIterator Type of the iterator to the output. Must be
* a forward output iterator that dereferences to a type that supports assignment
* to float.
* \tparam TFeatureFunctor The type of the feature functor object. Must meet the
* specifications for a \ref groupwise "groupwise feature functor" object, meaning
* it must define operator() with a certain form.
* \param first_id Iterator to the ID of the first data point for which the
* probability of the label is to be evaluated.
* \param last_id Iterator to the ID of the last data point for which the
* probability of the label is to be evaluated.
* \param label_it Iterator to the label variable whose probability is to be
* evaluated.
* \param out_it Iterator to the output probability value for the first ID.
* The container of output values must already exist, and contain enough
* elements for all of the IDs between first_id and last_id. At the end of this
* function, the output values in this container relate to the
* corresponding elements of the id container.
* \param single_label If true, the value of the label whose probability is
* evaluated is the same for all the data points. This means that the label_it
* iterator is never advanced. If false, the value of the label is not necessarily
* the same for all data points, and the label_it iterator is advanced for each
* data point to give the value of the label to use.
* \param feature_functor The feature functor object to be used as a callback to
* calculate the features. Must be safe to call from multiple threads
* simultaneously.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TIdIterator, class TLabelIterator, class TOutputIterator, class TFeatureFunctor>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::probabilityGroupwise(TIdIterator first_id, const TIdIterator last_id, TLabelIterator label_it, TOutputIterator out_it, const bool single_label, TFeatureFunctor&& feature_functor) const
{
	const auto simple_pdf_functor = [] (const TNodeDist* node_ptr, const TLabel& label, const auto id)
	{
		return node_ptr->pdf(label,id);
	};
	const auto output_assignment_functor = [] (const auto /*unused*/, const float score) {return score;};
	probabilityGroupwiseBase(first_id,last_id,label_it,out_it,single_label,output_assignment_functor,std::forward<TFeatureFunctor>(feature_functor),simple_pdf_functor);
}

/*! \brief A generalised version of the \c probabilityGroupwise() function that
* enables the creation of more general functions.
*
* A generalised version of the \c probabilityGroupwise() function. There are
* two generalisations:
* -# The pdf value may be calculated from the node distribution in some way other
* than the calling the pdf() method. This enables, for example, accessing one
* distribution from a node distribution that contains multiple distributions over
* different variables. This behaviour is controlled by the pdf_functor object.
* -# The output probability value may be used for something other than simple
* assignment to a variable. This may be used, for example, to use the output
* value to update some other variable (via multiplication or addtition etc)
* in a single step without having to store results in a temporary array.
* This behaviour is controlled by the binary_function functor object.
*
* Unless otherwise specified, the behaviour is the same as the
* \c probabilityGroupwise() function.
*
* \tparam TIdIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TId type expected by the feature functor.
* \tparam TLabelIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TLabel type of the forest (or to something
* trivially convertible to that type).
* \tparam TOutputIterator Type of the iterator to the output. Must be
* a forward output iterator that dereferences to a type that supports assignment
* to float.
* \tparam TBinaryFunction The type of the binary_function argument. Must be a
* function object that has an operator() of the form float operator()(TOutput, float)
* where TOutput is the type that TOutputIterator dereferences to.
* \tparam TFeatureFunctor The type of the feature functor object. Must meet the
* specifications for a \ref groupwise "groupwise feature functor" object, meaning
* it must define operator() with a certain form.
* \tparam TPDFFunctor The type of the pdf_functor argument. Must be a function
* object that has an operator() of the form float operator()(TNodeDist*, TLabel, TId).
* \param first_id Iterator to the ID of the first data point for which the
* probability of the label is to be evaluated.
* \param last_id Iterator to the ID of the last data point for which the
* probability of the label is to be evaluated.
* \param label_it Iterator to the label variable whose probability is to be
* evaluated.
* \param out_it Iterator to the output probability value for the first ID.
* The container of output values must already exist, and contain enough
* elements for all of the IDs between first_id and last_id. At the end of this
* function, the output values in this container relate to the
* corresponding elements of the id container.
* \param single_label If true, the value of the label whose probability is
* evaluated is the same for all the data points. This means that the label_it
* iterator is never advanced. If false, the value of the label is not necessarily
* the same for all data points, and the label_it iterator is advanced for each
* data point to give the value of the label to use.
* \param binary_function A function object that takes the current value of the
* output variable (first argument) and the forest's predicted probability value
* (second) argument and returns the value that is then assigned to the output
* variable.
* \param feature_functor The feature functor object to be used as a callback to
* calculate the features. Must be safe to call from multiple threads
* simultaneously.
* \param pdf_functor A function object that takes a pointer to the leaf
* distribution reached by the forest (first argument), a lable value (second
* argument), and an ID (third argument) and returns the value used as the pdf
* for the that leaf distribution.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TIdIterator, class TLabelIterator, class TOutputIterator, class TBinaryFunction, class TFeatureFunctor, class TPDFFunctor>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::probabilityGroupwiseBase(TIdIterator first_id, const TIdIterator last_id, TLabelIterator label_it, TOutputIterator out_it, const bool single_label, TBinaryFunction&& binary_function, TFeatureFunctor&& feature_functor, TPDFFunctor&& pdf_functor) const
{
	// Arrays to use to hold leaf nodes
	const int num_id = std::distance(first_id,last_id);
	std::vector<std::vector<const TNodeDist*>> leaves(n_trees,std::vector<const TNodeDist*>(num_id));

	// Loop through all the trees in the forest and accumulate scores
	#pragma omp parallel for
	for(int t = 0; t < n_trees; ++t)
		findLeavesGroupwise(first_id,last_id,t,leaves[t],std::forward<TFeatureFunctor>(feature_functor));

	int d = 0;
	for( ; first_id != last_id; ++first_id)
	{
		float result = 0.0;

		// Combine results
		for(int t = 0; t < n_trees; ++t)
			result += std::forward<TPDFFunctor>(pdf_functor)(leaves[t][d],*label_it,*first_id);

		// Normalise by the number of trees and assign to output
		*out_it = std::forward<TBinaryFunction>(binary_function)(*out_it, result/n_trees);
		++out_it;

		if(!single_label)
			++label_it;

		++d;
	}
}

/*! \brief Evaluate the probability of a certain value of the label for a set of
* data points.
*
* This function uses the forest model to evaluate the probability of a given
* value of the label (output) variable for a number of data points, where each
* point data is identified by an ID variable.
*
* These ID variables are passed in as a pair of iterators pointing to the first
* and last IDs to be processed. The value of the label for which the probability
* should be evaluated is passed in as a second iterator. The probability of the
* label for each of these IDs is placed in a third container accessed by iterators.
*
* In this version of the function, the features needed by a single node are
* requested from the feature functor one-by-one.
*
* Uses OpenMP to query the multiple tree models in parallel.
*
* \tparam TIdIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TId type expected by the feature functor.
* \tparam TLabelIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TLabel type of the forest (or to something
* trivially convertible to that type).
* \tparam TOutputIterator Type of the iterator to the output. Must be
* a forward output iterator that dereferences to a type that supports assignment
* to float.
* \tparam TFeatureFunctor The type of the feature functor object. Must meet the
* specifications for a \ref single "single feature functor" object, meaning
* it must define operator() with a certain form.
* \param first_id Iterator to the ID of the first data point for which the
* probability of the label is to be evaluated.
* \param last_id Iterator to the ID of the last data point for which the
* probability of the label is to be evaluated.
* \param label_it Iterator to the label variable whose probability is to be
* evaluated.
* \param out_it Iterator to the output probability value for the first ID.
* The container of output values must already exist, and contain enough
* elements for all of the IDs between first_id and last_id. At the end of this
* function, the output values in this container relate to the
* corresponding elements of the id container.
* \param single_label If true, the value of the label whose probability is
* evaluated is the same for all the data points. This means that the label_it
* iterator is never advanced. If false, the value of the label is not necessarily
* the same for all data points, and the label_it iterator is advanced for each
* data point to give the value of the label to use.
* \param feature_functor The feature functor object to be used as a callback to
* calculate the features. Must be safe to call from multiple threads
* simultaneously.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TIdIterator, class TLabelIterator, class TOutputIterator, class TFeatureFunctor>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::probabilitySingle(TIdIterator first_id, const TIdIterator last_id, TLabelIterator label_it, TOutputIterator out_it, const bool single_label, TFeatureFunctor&& feature_functor) const
{
	const auto simple_pdf_functor = [] (const TNodeDist* const node_ptr, const TLabel& label, const auto id)
	{
		return node_ptr->pdf(label,id);
	};
	const auto output_assignment_functor = [] (const auto /*unused*/, const float score) {return score;};
	probabilitySingleBase(first_id,last_id,label_it,out_it,single_label,output_assignment_functor,std::forward<TFeatureFunctor>(feature_functor),simple_pdf_functor);
}

/*! \brief A generalised version of the \c probabilitySingle() function that
* enables the creation of more general functions.
*
* A generalised version of the \c probabilitySingle() function. There are
* two generalisations:
* -# The pdf value may be calculated from the node distribution in some way other
* than the calling the pdf() method. This enables, for example, accessing one
* distribution from a node distribution that contains multiple distributions over
* different variables. This behaviour is controlled by the pdf_functor object.
* -# The output probability value may be used for something other than simple
* assignment to a variable. This may be used, for example, to use the output
* value to update some other variable (via multiplication or addtition etc)
* in a single step without having to store results in a temporary array.
* This behaviour is controlled by the binary_function functor object.
*
* Unless otherwise specified, the behaviour is the same as the
* \c probabilitySingle() function.
*
* \tparam TIdIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TId type expected by the feature functor.
* \tparam TLabelIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TLabel type of the forest (or to something
* trivially convertible to that type).
* \tparam TOutputIterator Type of the iterator to the output. Must be
* a forward output iterator that dereferences to a type that supports assignment
* to float.
* \tparam TBinaryFunction The type of the binary_function argument. Must be a
* function object that has an operator() of the form float operator()(TOutput, float)
* where TOutput is the type that TOutputIterator dereferences to.
* \tparam TFeatureFunctor The type of the feature functor object. Must meet the
* specifications for a \ref single "single feature functor", meaning it must
* define operator() with a certain form.
* \tparam TPDFFunctor The type of the pdf_functor argument. Must be a function
* object that has an operator() of the form float operator()(TNodeDist*, TLabel, TId).
* \param first_id Iterator to the ID of the first data point for which the
* probability of the label is to be evaluated.
* \param last_id Iterator to the ID of the last data point for which the
* probability of the label is to be evaluated.
* \param label_it Iterator to the label variable whose probability is to be
* evaluated.
* \param out_it Iterator to the output probability value for the first ID.
* The container of output values must already exist, and contain enough
* elements for all of the IDs between first_id and last_id. At the end of this
* function, the output values in this container relate to the
* corresponding elements of the id container.
* \param single_label If true, the value of the label whose probability is
* evaluated is the same for all the data points. This means that the label_it
* iterator is never advanced. If false, the value of the label is not necessarily
* the same for all data points, and the label_it iterator is advanced for each
* data point to give the value of the label to use.
* \param binary_function A function object that takes the current value of the
* output variable (first argument) and the forest's predicted probability value
* (second) argument and returns the value that is then assigned to the output
* variable.
* \param feature_functor The feature functor object to be used as a callback to
* calculate the features. Must be safe to call from multiple threads
* simultaneously.
* \param pdf_functor A function object that takes a pointer to the leaf
* distribution reached by the forest (first argument), a lable value (second
* argument), and an ID (third argument) and returns the value used as the pdf
* for the that leaf distribution.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TIdIterator, class TLabelIterator, class TOutputIterator, class TBinaryFunction, class TFeatureFunctor, class TPDFFunctor>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::probabilitySingleBase(TIdIterator first_id, const TIdIterator last_id, TLabelIterator label_it, TOutputIterator out_it, const bool single_label, TBinaryFunction&& binary_function, TFeatureFunctor&& feature_functor, TPDFFunctor&& pdf_functor) const
{
	// Could not do this and use iterators, making it more general
	// But then would be much harder to parallelise
	const int num_id = std::distance(first_id,last_id);

	// Loop over the data ids
	# pragma omp parallel for
	for(int d = 0; d < num_id; ++d)
	{
		float result = 0.0;

		// Loop over the trees in the forest and accumulate pdf results
		for(int t = 0; t < n_trees; ++t)
		{
			const TNodeDist* const leaf_ptr = findLeafSingle(*first_id,t,std::forward<TFeatureFunctor>(feature_functor));
			result += std::forward<TPDFFunctor>(pdf_functor)(leaf_ptr,*label_it,*first_id);
		}

		// Normalise by the number of trees and assign to output
		*out_it = std::forward<TBinaryFunction>(binary_function)(*out_it, result/n_trees);

		++first_id;
		++out_it;
		if(!single_label)
			++label_it;
	}
}

/*! \brief Function to query a single tree model with a set of data points and
* store a pointer to the leaf distribution that each reaches.
*
* This is a basic operation that is used by higher-level processes. Using this
* method, the features needed by a single node are requested from the feature
* functor for all the IDs with a single function call. This involves some
* overhead, but may permit efficiencies resulting from calculating multiple
* features at once.
*
* \tparam TIdIterator Type of the iterator to the IDs. Must be a random access
* iterator and dereference to the TId type expected by the feature functor.
* \tparam TFeatureFunctor The type of the feature functor object. Must meet the
* specifications for a \ref groupwise "groupwise feature functor" object, meaning
* it must define operator() with a certain form.
* \param first_id Iterator to the ID of the first data point for which the
* leaf distribution is to be found.
* \param last_id Iterator to the ID of the last data point for which the
* leaf distribution is to be found.
* \param treenum Index of the tree to use.
* \param leaves After the function, this array contains pointers to the leaf
* distribution reached by the corresponding elements in the ID list. Expects to
* be pre-allocated to the correct size.
* \param feature_functor The feature functor object to be used as a callback to
* calculate the features. Must be safe to call from multiple threads
* simultaneously.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TIdIterator, class TFeatureFunctor>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::findLeavesGroupwise(TIdIterator first_id, const TIdIterator last_id, const int treenum, std::vector<const TNodeDist*>& leaves, TFeatureFunctor&& feature_functor) const
{
	// Create an array of vectors to store the contents of each node,
	// and initially place all the inputs into the first
	const int num_id = std::distance(first_id,last_id);
	std::vector<std::vector<int>> nodebag_rel(n_nodes);
	nodebag_rel[0].resize(num_id);
	std::iota(nodebag_rel[0].begin(),nodebag_rel[0].end(),0);
	std::vector<float> scores;
	scores.reserve(num_id);

	// Loop through the nodes, sending datapoints left and right
	for(int n = 0; n < n_nodes; ++n)
	{
		// Number of datapoint in this node
		const int num_data_thisnode = nodebag_rel[n].size();

		// If no datapoints have reached this node, skip
		if(num_data_thisnode == 0)
			continue;

		// Makes the code a bit more readable
		const node& thisnode = forest[treenum].nodes[n];

		if(thisnode.is_leaf)
		{
			// Update the scores for datapoints in this node
			for(int d = 0; d < int(nodebag_rel[n].size()); ++d)
				leaves[nodebag_rel[n][d]] = thisnode.post.data();
		}
		else
			// Not a leaf - send the contents left or right
		{

			// Use functor function to find the scores
			// Use a permutation iterator to access the ids through their relative
			// place in the list
			scores.resize(num_data_thisnode);
			const auto start_it = boost::make_permutation_iterator(first_id,nodebag_rel[n].cbegin());
			const auto end_it = boost::make_permutation_iterator(first_id,nodebag_rel[n].cend());
			std::forward<TFeatureFunctor>(feature_functor)(start_it,end_it,thisnode.params,scores.begin());

			// Reserve space in the bags of the children nodes
			// (this should make performing multiple push_backs less
			// expensive)
			nodebag_rel[2*n+1].reserve(num_data_thisnode);
			nodebag_rel[2*n+2].reserve(num_data_thisnode);

			// Send the datapoints left or right
			for(int d = 0; d < int(num_data_thisnode); ++d)
			{
				const int nextnode = (scores[d] < thisnode.thresh ) ? 2*n+1 : 2*n+2;
				nodebag_rel[nextnode].emplace_back(nodebag_rel[n][d]);
			}

			// Clear up
			nodebag_rel[n].clear();
			nodebag_rel[n].shrink_to_fit();
		}
	}
}

/*! \brief Function to query a single tree model with a single data point and
* return a pointer to the leaf distribution that it reaches.
*
* This is a basic operation that is used by higher-level processes. Using this
* method.
*
* \tparam TId Type of the ID used to identify the data point.
* \tparam TFeatureFunctor The type of the feature functor object. Must meet the
* specifications for a \ref single "single feature functor" object, meaning it
* must define operator() with a certain form.
* \param first_id ID of the data point for which the leaf distribution is to be
* found.
* \param treenum Index of the tree to use.
* \param feature_functor The feature functor object to be used as a callback to
* calculate the features. Must be safe to call from multiple threads
* simultaneously.
* \return A pointers to the leaf distribution reached by the data point.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template<class TId, class TFeatureFunctor>
const TNodeDist* randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::findLeafSingle(const TId id, const int treenum, TFeatureFunctor&& feature_functor) const
{
	int n = 0;
	while(!forest[treenum].nodes[n].is_leaf)
	{
		const node& thisnode = forest[treenum].nodes[n];
		const float score = std::forward<TFeatureFunctor>(feature_functor)(id, thisnode.params);
		n = (score < thisnode.thresh ) ? 2*n+1 : 2*n+2;
	}
	return forest[treenum].nodes[n].post.data();
}

/*! \brief Train the random forest model on training data.
*
* This function trains the random forest model to produce a valid model that may
* used for predictions or stored for future use. It takes iterators pointing
* to the IDs of the training data and the corresponding label variables, and functors
* to generate parameters of the feature functor and evaluate the features.
*
* This function uses OpenMP to train the trees in parallel threads.
*
* \tparam TIdIterator Type of the iterator used to access the training IDs.
* Must be a random access iterator that dereferences to the ID type expected by
* feature_functor.
* \tparam TLabelIterator Type of the iterator used to access the label variables.
* Must be a random access iterator that dereferences to type TLabel.
* \tparam TFeatureFunctor Type of the feature_functor parameter. Must be a
* \ref groupwise "groupwise feature functor" object with an operator() of a
* specified form.
* \tparam TParameterFunctor Type of the feature_functor parameter. Must be a
* \ref params "parameter generator functor" object with an operator() of the form
* void operator()(std::array<int,TNumParams>&)
* \param first_id Iterator to the ID of the first element in the training list.
* \param last_id Iterator to the ID of the last element in the training list.
* \param first_label Iterator to the label of the first element in the training
* list. This iterator will be advanced to find the labels of the subsequent IDs.
* \param feature_functor The function object that should be used to evaluate the
* features when training the split nodes. Must be safe to call from multiple
* threads simultaneously.
* \param parameter_functor The function object that should be called to generate
* a random set of split nodes parameters for use in the feature_functor. Should
* take a std::array<int,TNumParams> by reference and populate the elements with
* a valid combination of randomly chosen parameters. Must be safe to call from
* multiple threads simultaneously.
* \param num_param_combos_to_test The number of parameter combinations to test
* when training each split node.
* \param bagging If true, a random subset of the training data are used to train
* each tree. If false, the full set of training data are used to train each tree.
* Default: true.
* \param bag_proportion Proportion of the training data in the bag used to train
* each tree if bagging is true. If bagging is false, this parameter is ignored.
* If the value is not in the range 0 to 1, the training procedure will fail
* immediately. Default: C_DEFAULT_BAGGING_PROPORTION .
* \param train_split_nodes If true, a node distribution is fitted at every node in
* the forest, regardless of the lead nodes. This is typically slightly more
* time consuming and results is a larger .tr, but allows the trained model to
* be tested using a smaller depth than it was trained at. If false, the node
* distributions are only fitted to the leaf nodes. Default: true.
* \param min_training_data The threshold number of training data points in a node below which a
* leaf node is declared during training. Default: C_DEFAULT_MIN_TRAINING_DATA .
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TIdIterator, class TLabelIterator, class TFeatureFunctor, class TParameterFunctor>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::train(const TIdIterator first_id,
																			   const TIdIterator last_id,
																			   const TLabelIterator first_label,
																			   TFeatureFunctor&& feature_functor,
																			   TParameterFunctor&& parameter_functor,
																			   const unsigned num_param_combos_to_test,
																			   const bool bagging,
																			   const float bag_proportion,
																			   const bool train_split_nodes,
																			   const unsigned min_training_data)
{
	this->fit_split_nodes = train_split_nodes;

	const int num_ids = std::distance(first_id,last_id);

	if(bagging && (bag_proportion <= 0.0 || bag_proportion > 1.0))
		return;

	// Calculate the size of each bag
	const int bagsize = bagging ?  num_ids*bag_proportion : num_ids;

	// Perform any precalculations necessary
	static_cast<derivedProxy*>(this)->trainingPrecalculations(first_label, first_label + num_ids, first_id);

	// Loop through the trees in parallel, training each tree
	#pragma omp parallel for
	for(int t = 0; t < n_trees; ++t)
	{
		// Array of vectors of ids in each node and their number
		std::vector<std::vector<int>> nodebag(n_nodes);

		// Choose a random sample of training data to use for the first
		// node, or set it to the whole training set
		nodebag[0].resize(num_ids);
		std::iota(nodebag[0].begin(),nodebag[0].end(),0);
		if(bagging)
		{
			std::shuffle(nodebag[0].begin(), nodebag[0].end(), rand_engine);
			nodebag[0].resize(bagsize);
		}

		// Memory to hold scores from feature computations
		std::vector<float> score, best_score;
		score.reserve(bagsize);
		best_score.reserve(bagsize);

		// Loop through nodes
		for(int n = 0; n < n_nodes; ++n)
		{
			// If this is a leaf node, set the parameters and move on
			if(
				(n > std::pow(2,n_levels-1) - 2)
				|| (nodebag[n].size() < min_training_data)
				|| (forest[t].nodes[n].is_leaf)
			  )
			{
				// Fit leaf distribution to this node
				fitLeaf(t,n,nodebag[n],first_id,first_label);

				// Clear the list for this node
				nodebag[n].clear();
				nodebag[n].shrink_to_fit();

				// Advance to the next node
				continue;
			}

			float best_info_gain = std::numeric_limits<float>::lowest();
			float best_thresh = 0.0;
			std::array<int,TNumParams> best_params;
			std::fill(best_params.begin(),best_params.end(),-1); // mostly to avoid annoying compiler warnings about using best_params uninitialised

			// Calculate the impurity of the node before splitting
			const float initial_impurity = static_cast<derivedProxy*>(this)->singleNodeImpurity(first_label,nodebag[n],t,n);

			// Counter for parameter combinations that fail to distinguish the data at all
			unsigned failed_counter = 0;

			// Loop through a number of randomly chosen parameter sets
			for(unsigned p = 0; p < num_param_combos_to_test ; ++p)
			{
				std::vector<scoreInternalIndexStruct> data_structs;
				std::array<int,TNumParams> test_params;

				// Resize array to hold scores
				score.resize(nodebag[n].size());

				// Generate random parameter values
				std::forward<TParameterFunctor>(parameter_functor)(test_params);

				// Find the value of each of the training data for this
				// feature and store in vectors, by class.
				std::forward<TFeatureFunctor>(feature_functor)( boost::make_permutation_iterator(first_id,nodebag[n].cbegin()),
																boost::make_permutation_iterator(first_id,nodebag[n].cend()),
																test_params,score.begin());

				// Put the labels and scores into a vector where they can be sorted together
				data_structs.reserve(nodebag[n].size());
				for(unsigned d = 0; d < nodebag[n].size(); ++d)
					data_structs.emplace_back(scoreInternalIndexStruct(score[d],nodebag[n][d]));

				// Sort this vector
				sort(data_structs.begin(),data_structs.end(), [](const scoreInternalIndexStruct& l, const scoreInternalIndexStruct& r) {return l.score < r.score;});

				// Skip this parameter set if there is little or no variation between the feature values
				if( (data_structs.back().score - data_structs.front().score) <= std::numeric_limits<float>::min()*nodebag[n].size())
				{
					failed_counter++;
					data_structs.clear();
					continue;
				}

				// Call the function to find the best splitting threshold and the corresponding purity measure
				float thresh, info_gain;
				static_cast<derivedProxy*>(this)->bestSplit(data_structs, first_label, t, n, initial_impurity, info_gain, thresh);

				// If this is the best parameter set so far, update the
				// fields of the tree
				if(info_gain > best_info_gain)
				{
					best_params = test_params;
					best_thresh = thresh;
					best_info_gain = info_gain;

					// Change the best_score pointer to this new score array
					best_score.resize(score.size());
					best_score.swap(score);
				}

				data_structs.clear();

			} // loop over parameter combinations

			// Check to see whether the best information gain was enough to justify a split
			if(best_info_gain > (static_cast<derivedProxy*>(this)->minInfoGain(t,n)) && failed_counter < num_param_combos_to_test)
			{
				// Go ahead and split the node
				forest[t].nodes[n].params = best_params;
				forest[t].nodes[n].is_leaf = false;
				forest[t].nodes[n].thresh = best_thresh;
				forest[t].nodes[n].post.clear();
				for(unsigned d = 0; d < nodebag[n].size() ; ++d)
				{
					if( best_score[d] < forest[t].nodes[n].thresh)
						nodebag[2*n+1].emplace_back(nodebag[n][d]);
					else
						nodebag[2*n+2].emplace_back(nodebag[n][d]);
				}

				// Check that neither child is empty
				assert( (nodebag[2*n+1].size() != 0) && (nodebag[2*n+2].size() != 0) );

				// Fit a node distribution to this split node if the flag is set
				if(fit_split_nodes)
				{
					forest[t].nodes[n].post.resize(1);
					static_cast<derivedProxy*>(this)->initialiseNodeDist(t,n);
					const auto start_it_label = boost::make_permutation_iterator(first_label,nodebag[n].cbegin());
					const auto end_it_label = boost::make_permutation_iterator(first_label,nodebag[n].cend());
					const auto start_it_id = boost::make_permutation_iterator(first_id,nodebag[n].cbegin());
					forest[t].nodes[n].post[0].fit(start_it_label,end_it_label,start_it_id);
				}

			}
			else
			{
				// This node will become a leaf
				fitLeaf(t,n,nodebag[n],first_id,first_label);
			}

			// Clear up
			nodebag[n].clear();
			nodebag[n].shrink_to_fit();

		} // node loop

	} // tree loop

	// Clean-up any pre-calculated data
	static_cast<derivedProxy*>(this)->cleanupPrecalculations();
	valid = true;

}

// Given a vector of data samples in one node, fit a node distribution to the data
/*! \brief Fit a node distribution to the data that have arrived in a node during
* training.
*
* This is a subroutine of the train() method and should not be used otherwise.
*
* \tparam TIdIterator Type of the iterator used to access the IDs of the data
* points in the node
* \tparam TLabelIterator Type of the iterator used to access the labels of the
* data points in the node
* \param t Index of the tree whose node should be fitted.
* \param n Index of the node that should be fitted within the tree.
* \param nodebag Vector containing the internal indices of the data points in the
* node.
* \param first_id Iterator to the first ID in the training set (not the first in
* node)
* \param first_label Iterator to the label of the first data point in the training
* set (not the first in node)
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TIdIterator, class TLabelIterator>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::fitLeaf(const int t, const int n, const std::vector<int>& nodebag, const TIdIterator first_id, const TLabelIterator first_label)
{
	// Basic parameters of the node
	std::fill(forest[t].nodes[n].params.begin(),forest[t].nodes[n].params.end(),-1);
	forest[t].nodes[n].thresh = 0.0;
	forest[t].nodes[n].is_leaf = true;

	// If the parent is a leaf, this node is an orphan -
	// no need for a posterior
	if((n == 0) || (!forest[t].nodes[(n-1)/2].is_leaf))  // integer division deliberate
	{
		forest[t].nodes[n].post.resize(1);
		static_cast<derivedProxy*>(this)->initialiseNodeDist(t,n);
		const auto start_it_label = boost::make_permutation_iterator(first_label,nodebag.cbegin());
		const auto end_it_label = boost::make_permutation_iterator(first_label,nodebag.cend());
		const auto start_it_id = boost::make_permutation_iterator(first_id,nodebag.cbegin());
		forest[t].nodes[n].post[0].fit(start_it_label,end_it_label,start_it_id);
	}

	// If we're not already in the bottom layer, set the
	// children to leaves too
	if(2*n + 2 < n_nodes)
	{
		forest[t].nodes[2*n+1].is_leaf = true;
		forest[t].nodes[2*n+2].is_leaf = true;
	}
}

// Returns true if the forest has correctly been read from a file or trained
/*! \brief Check whether a forest model is valid.
*
* \return True if the forest has either been successfully trained or successfully
* read from a file and is therefore ready to use. False otherwise. If false, the
* model should not be used.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
bool randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::isValid() const
{
	return valid;
}

/*! \brief Store arbitrary strings that define parameters of the feature
* extraction process.
*
* This string is stored alongside the model, enabling the storage of information
* necessary to recreate the same feature extraction process at test time. The
* construction (and later parsing) of this string is left entirely up to the user.
* \param header_str This string will be printed above the feature string and
* is intended to help human readility of the string by explaining the meaning
* of the terms in the feature string.
* \param feat_str Arbitrary string containing data that can later be used to
* recreate the feature extraction process.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::setFeatureDefinitionString(const std::string& header_str, const std::string& feat_str)
{
	feature_header = header_str;
	feature_string = feat_str;
}

/*! \brief Retrieve a stored feature string.
*
* This method is used to retrieve a feature string previously stored in the model.
* This string may be used to store paratmers of the feature extraction process
* used to train the model. The parsing of this string is leftentirely up to
* the user.
*
* \param feature_string The stored string is returned by reference in this
* variable. If no string has been stored, an empty string is returned.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
void randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::getFeatureDefinitionString(std::string& feat_str) const
{
	feat_str = feature_string;
}

/*! \brief Calculate an array of x*log(x) for integer x.
*
* This is a utility routine provided for subclasses to make use of if convenient.
* The quantity x*log(x) arises in many entropy-based calculations, including the
* \c fastDiscreteEntropy() and \c fastDiscreteEntropySplit() calculations, and
* needs to be calculated a very large number of times in such routines. This
* method pre-calculates an array of x*log(x) for integers x in the range 0 to N
* inclusive, such that it may be used by other routines to speed up calculations.
*
* \param n Upper limit of the range of values for x
* \return A vector 'result' of length N+1 where result[i] has the value i*log(i),
* and result[0] = 0.0
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
std::vector<double> randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::preCalculateXlogX(const int N)
{
	if(N < 1)
		return std::vector<double>();
	std::vector<double> result(N+1);

	result[0] = 0.0;
	for(int i = 1; i <= N; ++i)
		result[i] = i*std::log(double(i));

	return result;
}


/*! \brief Calculates the entropy of the discrete labels of a set of data points
* using an efficient method.
*
* This is utility method that is provided for use in subclasses if convenient.
*
* \tparam TLabelIterator Type of the iterator used to access the discrete labels.
* Must be a random access iterator that dereferences to an integral data type.
* \param internal_index Vector containing the internal training indices of the
* data points. These are the indices through which the labels may be accessed in
* first_label
* \param n_labels The number of discrete labels. The possible values of the label
* are assumed to be the integers in the range 0 to n_labels-1
* \param first_label Iterator to the labels for which the entropy is to be
* calculated. The labels should be located at the offsets from this iterator given
* by the elements of the internal_index vector. I.e. first_label[internal_index[0]],
* first_label[internal_index[1]] etc.
* \param xlogx_precalc A pre-calculated array of the value of x*log(x), as
* calculated by the \c preCalculateXlogX() routine. This must be long enough
* to include the value for x = internal_index.size() or greater.
* \return The entropy of the set of labels.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TLabelIterator>
double randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::fastDiscreteEntropy(const std::vector<int>& internal_index, const int n_labels, const TLabelIterator first_label, const std::vector<double>& xlogx_precalc)
{
	const int n_data = internal_index.size();

	// Counts of each label
	std::vector<int> counts(n_labels,0);
	for(int i : internal_index)
		++counts[first_label[i]];

	// We can use the same formaulation as the "best split" function to avoid calculating log all the time
	// Multiply the expression by N to give in terms of the form NlogN, then divide at the end
	double ret = xlogx_precalc[n_data]; // total term

	// Now subtract all the "partial" terms
	for(int b = 0; b < n_labels; ++b)
		ret -= xlogx_precalc[counts[b]];

	// Divide by N again to give result
	return ret/n_data;

}

// A utility function that is useful to many subclasses (including the most basic classifier)
// This function uses a highly efficient method to test every possible split of a dataset and
// find the best in terms of discrete entropy.
// The input first_label is expected to be presorted in terms of the score. Furthermore, the user must
// provide an array of precalculated values for x*log(x)
// The method is based on multiplying the expression by N to give all terms in NlogN, then
// dividing again at the end.

/*! \brief Find the split in a set of training data that results in the best
* information gain for discrete labels.
*
* This is utility method that is provided for use in subclasses if convenient.
*
* \tparam TLabelIterator Type of the iterator used to access the discrete labels.
* Must be a random access iterator that dereferences to an integral data type.
* \param data_structs A vector in which each element is a structure containing
* the internal id (.id) and score (.score) for the current feature of the
* training data points. The vector is assumed to be sorted according to the score
* field in ascending order.
* \param n_labels The number of discrete labels. The possible values of the label
* are assumed to be the integers in the range 0 to n_labels-1
* \param first_label Iterator to the labels for which the entropy is to be
* calculated. The labels should be located at the offsets from this iterator given
* by the IDs of elements of the data_structs vector. I.e. first_label[data_structs[0].id],
* first_label[data_structs[1].id] etc.
* \param xlogx_precalc A pre-calculated array of the value of x*log(x), as
* calculated by the \c preCalculateXlogX() routine. This must be long enough
* to include the value for x = data_structs.size() or greater.
* \param best_split_impurity Returns by reference the impurity of the best split
* found.
* \param thresh Returns by reference the threshold of the feature score that
* gives the best split.
* \return The position 'd' of the best split in the training data. The partition of
* the data resulting in the best split has the first d+1 elements in one partiiton
* and the remainder in the other partition.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
template <class TLabelIterator>
int randomForestBase<TDerived,TLabel,TNodeDist,TOutputDist,TNumParams>::fastDiscreteEntropySplit(const std::vector<scoreInternalIndexStruct>& data_structs, const int n_labels, const TLabelIterator first_label, const std::vector<double>& xlogx_precalc, double& best_split_impurity, float& thresh)
{
	// Number of data samples
	const int n_data = data_structs.size();

	// Create two arrays of bins for the left and right partitions
	std::vector<int> left_binned(n_labels,0);
	std::vector<int> right_binned(n_labels,0);

	// Initialise the left partition to contain just the first datapoint,
	// and all others in the right partition
	left_binned[first_label[data_structs[0].id]] = 1;
	for(int d = 1; d < n_data; ++d)
		right_binned[first_label[data_structs[d].id]] += 1;

	// Intermediate results
	double left_running_total_partial = 0.0;
	double right_running_total_partial = 0.0;

	// Calculate the initial value for the sum of the right partition's partial
	// variables
	for(int b = 0; b < n_labels; ++b)
		right_running_total_partial += xlogx_precalc[right_binned[b]];

	// Compute the entropy for this first partition
	int best_d;
	double best_children_impurity;
	if(data_structs[0].score == data_structs[1].score)
	{
		best_children_impurity = std::numeric_limits<float>::max();
		best_d = -1;
	}
	else
	{
		best_children_impurity = (-left_running_total_partial + xlogx_precalc[1]) + (-right_running_total_partial + xlogx_precalc[n_data-1]);
		best_d = 0;
	}

	// Loop through other partitions, move the data into the new bins and compute entropy
	// At iteration d, datapoints 0,...,d are in the "left" partition (below the threshold)
	// and datapoints d+1,...,N-1 are in the "right" partition (above the threshold)
	for(int d = 1; d < n_data-1; ++d)
	{
		// The bin of the datapoint being moved
		const int b = first_label[data_structs[d].id];

		// Move out of the right bin and into the left bin
		right_binned[b] -= 1;
		left_binned[b] += 1;

		// Change to the left and right partial running totals
		// The count in bin 'bin' has just increased by one in the left histogram (new partial - old partial)
		left_running_total_partial += xlogx_precalc[left_binned[b]] - xlogx_precalc[left_binned[b]-1];
		// The count in bin 'bin' has just decreased by one in the right histogram (new partial - old partial)
		right_running_total_partial += xlogx_precalc[right_binned[b]] - xlogx_precalc[right_binned[b]+1];

		// If the score of this datapoint is the same as that of the next datapoint, we cannot actually split here
		if(data_structs[d].score == data_structs[d+1].score)
			continue;

		// Calculate the resulting purity
		const double left_side_impurity = -left_running_total_partial + xlogx_precalc[d+1];
		const double right_side_impurity = (-right_running_total_partial + xlogx_precalc[n_data-d-1]);
		const double this_children_impurity =  left_side_impurity + right_side_impurity;

		if(this_children_impurity < best_children_impurity)
		{
			best_children_impurity = this_children_impurity;
			best_d = d;
		}
	}

	// Check that at least one datapoint had a different score from the rest
	assert((best_d >= 0) && (best_d < n_data-1));

	// Values to return
	best_split_impurity = best_children_impurity;
	thresh = 0.5*(data_structs[best_d].score + data_structs[best_d+1].score);

	return best_d;
}

} // end of namespace
