#ifndef RANDOMFORESTBASE_HPP
#define RANDOMFORESTBASE_HPP

/*!
* \file randomForestBase.hpp
* \author Christopher P Bridge
* \brief Contains the declaration of the canopy::randomForestBase class.
*/

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <random>

/*! \brief Namespace containing the canopy library for random forest models */
namespace canopy
{

/*! \brief Base class for random forests models from which all specific models
* are derived using CRTP.
*
* This class implements the basic training and testing routines, and some utility
* functions that may be used by derived classs. This class cannot not be used
* directly.
*
* \tparam TDerived The type of the derived random forests model (e.g. classifier,
* regressor). Having the derived class as a template parameter implements the
* curiously recurring template (CRTP) idiom, which allows for static polymorphism.
* \tparam TLabel The type of the label that the model is used to predict. This
* is the output type of the forest model, for example an integer for a classifier
* or a float for a 1D regressor.
* \tparam TNodeDist The type of the node distribution, which is the distribution
* stored at each leaf node. The node distribution must have certain
* characteristics.
* \tparam TOutputDist The type of the output distribution, which is the type of
* the distribution predicted by the forest model. This may be same as or
* different from TNodeDist. The output distribution must have certain
* charaecteristics.
* \tparam TNumParams The number of parameters used by the features callback.
*/
template <class TDerived, class TLabel, class TNodeDist, class TOutputDist, unsigned TNumParams>
class randomForestBase
{
	public:
		// Methods
		// --------
		randomForestBase();

		randomForestBase(const int num_trees, const int num_levels); // constructor

		bool readFromFile(const std::string filename, const int trees_used = -1, const int max_depth_used = -1);

		bool writeToFile(const std::string filename) const;

		bool isValid() const;

		void setFeatureDefinitionString(const std::string& header_str, const std::string& feat_str);

		void getFeatureDefinitionString(std::string &feat_str) const;

		template <class TIdIterator, class TLabelIterator, class TFeatureFunctor, class TParameterFunctor>
		void train(const TIdIterator first_id, const TIdIterator last_id, const TLabelIterator first_label, TFeatureFunctor&& feature_functor, TParameterFunctor&& parameter_functor, const unsigned num_param_combos_to_test, const bool bagging = true, const float bag_proportion = C_DEFAULT_BAGGING_PROPORTION, const bool fit_split_nodes = true, const unsigned min_training_data = C_DEFAULT_MIN_TRAINING_DATA);

		template<class TIdIterator, class TOutputIterator, class TFeatureFunctor>
		void predictDistGroupwise(TIdIterator first_id, const TIdIterator last_id, TOutputIterator out_it, TFeatureFunctor&& feature_functor) const;

		template<class TIdIterator, class TOutputIterator, class TFeatureFunctor>
		void predictDistSingle(TIdIterator first_id, const TIdIterator last_id, TOutputIterator out_it, TFeatureFunctor&& feature_functor) const;

		template <class TIdIterator, class TLabelIterator, class TOutputIterator, class TFeatureFunctor>
		void probabilityGroupwise(TIdIterator first_id, const TIdIterator last_id, TLabelIterator label_it, TOutputIterator out_it, const bool single_label, TFeatureFunctor&& feature_functor) const;

		template <class TIdIterator, class TLabelIterator, class TOutputIterator, class TFeatureFunctor>
		void probabilitySingle(TIdIterator first_id, const TIdIterator last_id, TLabelIterator label_it, TOutputIterator out_it, const bool single_label, TFeatureFunctor&& feature_functor) const;

		template <class TIdIterator, class TLabelIterator, class TOutputIterator, class TBinaryFunction, class TFeatureFunctor, class TPDFFunctor>
		void probabilityGroupwiseBase(TIdIterator first_id, const TIdIterator last_id, TLabelIterator label_it, TOutputIterator out_it, const bool single_label, TBinaryFunction&& binary_function, TFeatureFunctor&& feature_functor, TPDFFunctor&& pdf_functor) const;

		template <class TIdIterator, class TLabelIterator, class TOutputIterator, class TBinaryFunction, class TFeatureFunctor, class TPDFFunctor>
		void probabilitySingleBase(TIdIterator first_id, const TIdIterator last_id, TLabelIterator label_it, TOutputIterator out_it, const bool single_label, TBinaryFunction&& binary_function, TFeatureFunctor&& feature_functor, TPDFFunctor&& pdf_functor) const;

	private:
		// Methods
		// -------
		void allocateForestMemory();

		void initialise();

		template <class TIdIterator, class TLabelIterator>
		void fitLeaf(const int t, const int n, const std::vector<int>& nodebag, const TIdIterator first_id, const TLabelIterator first_label);

	protected:

		// Types
		// -----

		/*! \brief Node structure - represents one node in a tree
		*/
		struct node
		{
			std::array<int,TNumParams> params; //!< Parameters for the split function
			bool is_leaf; //!< Indicates whether the node is a leaf (1 -> leaf)
			float thresh; //!< The decision threshold for an internal node
			std::vector<TNodeDist> post; //!< The posterior distribution over labels for a leaf node, shuld only ever have 1 or 0 elements
			node(): is_leaf(false), thresh(0.0) {} //!< Basic constructor
		};

		/*! \brief Tree structure - represents a single tree
		*
		* Nodes are arranged within the tree starting from the root node and
		* moving across levels followed by down levels such that index 0 is the
		* root, indices 1 and 2 are in the second layer, indices 3,4,5, and 6
		* are in the third layer and so on.
		* This means that the children of node n are 2*n+1 and 2*n+2.
		*/
		struct tree
		{
			std::vector<node> nodes; //!< Vector of the nodes
		};

		/*! \brief Structure for holding information about a data sample and its feature score
		*
		* This is used internally during the traning process to sort IDs along with
		* feature scores and pass the combination around between methods
		*/
		struct scoreInternalIndexStruct
		{
			float score; //!< The score of this data point according to the feature extraction
			int id; //!< The internal traning index of this data point
			/*! Simple constructor */
			scoreInternalIndexStruct(const float score, const int id): score(score), id(id) {}
		};

		/*! \brief Proxy for the derived class
		*
		* The proxy trick allows the interface functions within randomForestBase
		* to access protected members of the TDerived class via a static_cast to
		* a derivedProxy pointer, thus allowing member functions of the derived
		* class to be marked as protected.
		*/
		class derivedProxy: public TDerived
		{
			friend randomForestBase;
		};

		// Methods
		// -------

		template<class TIdIterator, class TFeatureFunctor>
		void findLeavesGroupwise(TIdIterator first_id, const TIdIterator last_id, const int treenum, std::vector<const TNodeDist*>& leaves, TFeatureFunctor&& feature_functor) const;

		template<class TId, class TFeatureFunctor>
		const TNodeDist* findLeafSingle(const TId id, const int treenum, TFeatureFunctor&& feature_functor) const;

		template <class TLabelIterator>
		static double fastDiscreteEntropy(const std::vector<int>& internal_index, const int n_labels, const TLabelIterator first_label, const std::vector<double>& xlogx_precalc);

		template <class TLabelIterator>
		static int fastDiscreteEntropySplit(const std::vector<scoreInternalIndexStruct>& data_structs, const int n_labels, const TLabelIterator first_label, const std::vector<double>& xlogx_precalc, double& best_split_impurity, float& thresh);

		static std::vector<double> preCalculateXlogX(const int N);

		// Data
		// ----
		int n_trees; //!< The number of trees in the forest
		int n_levels; //!< The maximum number of levels in each tree
		int n_nodes; //!< The number of nodes in each tree
		bool valid; //!< Whether the forest model is currently valid and usable for predictions (true = valid)
		bool fit_split_nodes; //!< Whether a node distribution is fitted to all nodes (true) or just the leaf nodes (false)
		std::vector<tree> forest; //!< Vector of tree models
		std::string feature_header; //!< String describing the content of the feature string
		std::string feature_string; //!< Arbitrary string describing the feature extraction process
		std::default_random_engine rand_engine; //!< Random engine for generating random numbers during training, may also be used by derived classes
		std::uniform_int_distribution<int> uni_dist; //!< For generating random integers during traning, may also be used derived classes

		// Constants
		// ---------
		static constexpr int C_DEFAULT_MIN_TRAINING_DATA = 50; //!< Default value for the minimum number of traning data points in a node before a leaf is declared
		static constexpr float C_DEFAULT_BAGGING_PROPORTION = 0.5; //!< Default value for the proportion of the traning set used to train each tree

};

} // end of namespace

// Include template class definition
#include <canopy/randomForestBase/randomForestBase.tpp>

// End include guard
#endif
