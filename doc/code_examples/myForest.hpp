template <unsigned TNumParams>
class myForest : public randomForestBase<myForest<TNumParams>,myLabel,myNodeDist,myOutputDist,TNumParams>
{
	public:
		/* You'll probably want to define a custom constructor here, plus any
		other public methods */

	protected:

		/*! Forward the definition of the type declared in the randomForestBase class.
		This isn't strictly necessary, but the protoype of bestSplit() becomes somewhat
		messy if you don't do this... */
		typedef typename randomForestBase<myForest<TNumParams>,myLabel,myNodeDist,myOutputDist,TNumParams>::scoreInternalIndexStruct scoreInternalIndexStruct;


		void printHeaderDescription(std::ofstream &stream) const
		{
			/* Print a human-readable description of the contents of the header
			data. Anything printed here is ignord by the library */
		}

		void printHeaderData(std::ofstream &stream) const
		{
			/* Print a single line containing any parameters that must be stored
			in order to reconstruct the model (such as number of classes etc) */
		}

		void readHeader(std::ifstream &stream)
		{
			/* Read in the data printed using printHeaderData() in order to
			reconstruct a stored forest model from file */
		}

		void initialiseNodeDist(const int t, const int n)
		{
			/* Initialise a node distribution before fitting it during training.
			This can be used to perform any arbitrary action on the node distribution
			in this->forest[t].nodes[n].post[0] to prepare for fitting, such as
			initialising it with certain parameters and/or calling a custom
			constructor */
		}

		float minInfoGain(const int tree, const int node) const
		{
			/* Return the value of information gain threshold for this node
			during training.

			If the actual information gain from the best split is below this,
			the node will become a leaf node. This can be give different
			behaviour in different nodes in the forest if desired, or can simply
			return a constant. */
		}

		template <class TLabelIterator>
		template <class TLabelIterator, class TIdIterator>
		void trainingPrecalculations(const TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator first_id)
		{
			/* This is called once at the start of the training routine and may
			be used to prepare for training on the supplied dataset, for example
			by precalculating values to speed up subsequent processes */
		}

		void cleanupPrecalculations()
		{
			/* This is called once at the end of the training routine and may be
			used, for example, to clear up any data no longer needed */
		}

		float singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int tree, const int node) const
		{
			/* Calculate the impurity of the labels in a given node before
			splitting in a given tree and node. This is used to compare to the
			value after splitting (found with bestSplit) in order to determine
			information gain. The labels are accessed via first_label[nodebag[0]],
			first_label[nodebag[1]] etc */
		}

		template <class TLabelIterator>
		void bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int tree, const int node, const float initial_impurity,float& info_gain, float& thresh) const
		{
			/* This the key function in the training routine. It takes a list
			of labels in data_structs which contains an integer-valued internal
			index (.id) and float-valued feature score (.score) according to the
			feature functor with the chosen parameters. The elements of
			data_structs are sorted by increasing values of the score before being
			passed to this method.

			The labels of each of the training samples in the node can be accessed
			via first_label[data_structs[0].id], first_label[data_structs[1].id]
			etc

			This method should find the best way to split the training samples
			with a single score threshold. The 'best' split is calculated with
			regards to the labels of the training samples, and is left to the
			user to define.

			The method returns (by reference) the chosen threshold (thresh) and
			the resulting information gain between the initial impurity before
			splitting (which is passed in as initial_impurity in order to avoid
			redundant repeated calculations), and the impurity after splitting.
			*/
		}

		/* Other data etc */
};
