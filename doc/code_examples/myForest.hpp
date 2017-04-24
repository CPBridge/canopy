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

		}

		void printHeaderData(std::ofstream &stream) const
		{

		}

		void readHeader(std::ifstream &stream)
		{

		}
		
		// Methods
		void initialiseNodeDist(const int t, const int n)
		{
			/* Initialise a node distribution before fitting it during training.
			This can be used to perform any arbitrary action on the node distribution
			in this->forest[t].nodes[n].post[0] to prepare for fitting */
		}

		float minInfoGain(const int tree, const int node) const
		{

		}

		template <class TLabelIterator>
		float singleNodeImpurity(const TLabelIterator first_label, const std::vector<int>& nodebag, const int tree, const int node) const
		{

		}

		template <class TLabelIterator, class TIdIterator>
		void trainingPrecalculations(const TLabelIterator first_label, const TLabelIterator last_label, const TIdIterator first_id)
		{

		}

		void cleanupPrecalculations()
		{

		}

		template <class TLabelIterator>
		void bestSplit(const std::vector<scoreInternalIndexStruct> &data_structs, const TLabelIterator first_label, const int tree, const int node, const float initial_impurity,float& info_gain, float& thresh) const
		{

		}

		/* Other data etc */
};
