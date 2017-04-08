class myOutputDist
{
	public:

		template <class TId>
		void combineWith(const myNodeDist& dist, const TId id)
		{
			/* Update the distribution to reflect the effect of combining it
			with a node distribution 'dist' */
		}

		void normalise()
		{
			/* Normalise the distribution after combining with several node
			distributions to ensure a valid distribution */
		}

		void reset()
		{
			/* Clear the results of all previous combinations to give a
			distribution that can be used to start the process on fresh data */
		}

	protected:

		// Distribution parameters etc
};
