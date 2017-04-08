#include <fstream>

class myNodeDist
{
	public:

		template <class TLabelIterator, class TIdIterator>
		void fit(TLabelIterator first_label, TLabelIterator last_label, TIdIterator first_id);
		{
			/* Function used to fit the distribution to training data during
			forest training. The data are passed in using iterators pointing
			to the set of labels to fit to, and their IDs. In most cases, the
			IDs will be unused and only the labels will be relevant.

			Due to the way the function is called by the randomForestBase class,
			TLabelIterator will be a random access iterator type (supports []
			syntax) that dereferences to myLabelType.
			*/
		}


		void printOut(std::ofstream& stream) const
		{
			/* Output parameters to 'stream' that can later be used by
			readIn() to fully reconstruct the distribution.
			This will probably involve reocrding parameters like mean and
			variance, and possibly other information.

			This will be called by randomForestBase when storing the model
			to a file. */
		}

		void readIn(std::ifstream& stream)
		{
			/* Read in parameters from 'stream' and use store them.
			This must match the format written by printOut()

			This will be called by randomForestBase when storing the model
			to a file.*/
		}

		template <class TId>
		float pdf(const myLabelType x, const TId id) const
		{
			/* Return the probability of label x under the distribution
			Note that the id paramater will be unused in many cases, as the
			probability will not depend on the ID, only the label.

			This is used by randomForestBase to perform the probability evaluation
			task. */
		}

		// Use operator<< to print to the file stream
		friend std::ofstream& operator<< (std::ofstream& stream, const myNodeDist& dist)
		{
			dist.printOut(stream); return stream;
		}

		//Use operator>> to read from the file stream
		friend std::ifstream& operator>> (std::ifstream& stream, myNodeDist& dist)
		{
			dist.readIn(stream); return stream;
		}

	protected:
		
		// Distribution parameters etc

};
