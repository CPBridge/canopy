// Standard Library Headers
#include <iostream>
#include <array>
#include <random>
#include <algorithm>
#include <string>

// The canopy classifier header
#include <canopy/classifier/classifier.hpp>

/* This programme demonstrates how to use the basic functionality of the canopy
library. It trains a random forest classifier to proedict the discrete label
of test data given some features. The features are randomly generated from
a Gaussian distribution with different, randomly-chosen mean and variance
parameters for each of the discrete classes.
*/

int main()
{
	/* Parameters of the test */
	constexpr unsigned N_CLASSES = 3; // number of discrete class labels
	constexpr unsigned TRAINING_DATA_PER_PER_CLASS = 200;
	constexpr unsigned TOTAL_TRAINING_DATA = N_CLASSES * TRAINING_DATA_PER_PER_CLASS;
	constexpr unsigned N_DIMS = 2; // dimensionality of the feature space
	constexpr double MIN_MU = 0.0; // range of the randomly-generated mean parameters (min)
	constexpr double MAX_MU = 10.0; // range of the randomly-generated mean parameters (max)
	constexpr double MAX_SIGMA = 3.0; // maximum value for randomly-generated standard deviation parameters
	constexpr int N_TREES = 128; //number of trees in the random forest
	constexpr int N_LEVELS = 10; // maximum number of levels in each tree
	constexpr unsigned N_TESTS = 10; //number of test data
	const std::string FILENAME = "example_model.tr"; // file to save the model in

	/* Set up random number generation */
	std::default_random_engine rand_engine;
	std::random_device rd{};
	rand_engine.seed(rd());
	std::normal_distribution<double> norm_dist;
	std::uniform_int_distribution<int> uni_int_dist;
	std::uniform_real_distribution<double> uni_real_dist;

	/* Randomly generate sigma and mu parameters for each class assuming
	axis-aligned distributions for simplicity. These are arrays with classes
	down the first dimension and the feature space dimension down the second */
	std::array<std::array<double,N_DIMS>,N_CLASSES> mu;
	std::array<std::array<double,N_DIMS>,N_CLASSES> sigma;
	for(unsigned c = 0; c < N_CLASSES; ++c)
	{
		for(unsigned d = 0; d < N_DIMS; ++d)
		{
			mu[c][d] = uni_real_dist(rand_engine,std::uniform_real_distribution<double>::param_type{MIN_MU,MAX_MU});
			sigma[c][d] = uni_real_dist(rand_engine,std::uniform_real_distribution<double>::param_type{0.0,MAX_SIGMA});
		}
	}

	/* Generate training data using these distributions */
	std::array<std::array<double,N_DIMS>,TOTAL_TRAINING_DATA> training_data_features;
	std::array<int,TOTAL_TRAINING_DATA> training_data_labels;
	for(unsigned c = 0; c < N_CLASSES; ++c)
	{
		for(unsigned n = 0; n < TRAINING_DATA_PER_PER_CLASS; ++n)
		{
			const unsigned i = c*TRAINING_DATA_PER_PER_CLASS + n;
			training_data_labels[i] = c;

			for(unsigned d = 0; d < N_DIMS; ++d)
			{
				training_data_features[i][d] = norm_dist(rand_engine,std::normal_distribution<double>::param_type{mu[c][d],sigma[c][d]});
			}
		}
	}

	/* Create a classifer object and initialise with the number of classes, trees
	and levels. The TNumParams template parameter is one because there is a single
	parameter of the feature calculation process, which indexes the different features
	in the list */
	canopy::classifier<1> the_classifier(N_CLASSES,N_TREES,N_LEVELS);

	/* Create a groupwise feature functor object in order to train the model.
	A C++14 generic lambda is a convenient way to do this
	as it can capture the data array by reference and figure out all the types
	for us */
	auto train_feature_lambda = [&] (auto first_id, const auto last_id, const std::array<int,1>& params, std::vector<float>::iterator out_it)
	{
		/* Iterate over the IDs */
		while(first_id != last_id)
		{
			/* ID for this data point found by dereferencing iterator */
			const int id = *first_id;

			/* The first and only parameter represents the dimension of the
			feature space to use */
			const int d = params[0];

			/* Look up the pre-calculated feature value for this ID and dimension
			and place it in the output iterator
			(The training_data_features array is captured by reference) */
			*out_it++ = training_data_features[id][d];

			/* Advance the iterator */
			++first_id;
		}
	};

	/* Create a parameter generator functor for training, which simply selects a
	random dimension. We could equally well use a canopy::defaultParameterGenerator
	here */
	auto param_lambda = [&] (std::array<int,1>& params)
	{
		params[0] = uni_int_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,N_DIMS-1});
	};

	/* Finally we need a way of identifying each of the data points in the training
	set. This is done by the index of the data point in the list */
	std::array<int,TOTAL_TRAINING_DATA> train_ids;
	std::iota(train_ids.begin(),train_ids.end(),0);

	/* With all this in place we are ready to train the model */
	the_classifier.train( train_ids.cbegin(), train_ids.cend(), training_data_labels.cbegin(), train_feature_lambda, param_lambda, N_DIMS/2 + 1);

	/* We can now write the model a file for later use */
	the_classifier.writeToFile(FILENAME);

	/* Generate some unseen test data from the same distributions */
	std::array<std::array<double,N_DIMS>,N_TESTS> test_data_features;
	std::array<int,N_TESTS> test_data_labels;
	for(unsigned n = 0; n < N_TESTS; ++n)
	{
		/* Choose a random class label */
		const int c = uni_int_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,N_CLASSES-1});
		test_data_labels[n] = c;

		/* Generate some features using this class's distribution parameters */
		for(unsigned d = 0; d < N_DIMS; ++d)
		{
			test_data_features[n][d] = norm_dist(rand_engine,std::normal_distribution<double>::param_type{mu[c][d],sigma[c][d]});
		}
	}

	/* We need a way of identifying each of the data points in the test
	set. This is again done by the index of the data point in the list */
	std::array<int,N_TESTS> test_ids;
	std::iota(test_ids.begin(),test_ids.end(),0);

	/* We need a functor to calculate the features for the test set.
	This is the same as before, but accesses the testing array instead of the
	training array */
	auto test_feature_lambda = [&] (auto first_id, const auto last_id, const std::array<int,1>& params, std::vector<float>::iterator out_it)
	{
		while(first_id != last_id)
		{
			const int id = *first_id;
			const int d = params[0];
			*out_it++ = test_data_features[id][d];
			++first_id;
		}
	};

	/* There are two basic ways to use the forest to analyse test data. The first
	is to predict the full distribution over the output space given the features.
	For the classification task, this means finding a discrete distribution over the
	class labels. First we need to create some discrete distribution objects and
	initialise them to the right number of classes */
	std::array<canopy::discreteDistribution,N_TESTS> d_dists;
	for(canopy::discreteDistribution& dist : d_dists)
		dist.initialise(N_CLASSES);

	/* Use the forest model to perform the prediction */
	the_classifier.predictDistGroupwise(test_ids.cbegin(),test_ids.cend(),d_dists.begin(),test_feature_lambda);

	/* Output the results to console */
	for(unsigned n = 0; n < N_TESTS; ++n)
	{
		std::cout << "True Label " << test_data_labels[n] << ", Predicted Distribution";
		for(unsigned c = 0; c < N_CLASSES; ++c)
		 	std::cout << " " << d_dists[n].pdf(c);
		std::cout << std::endl;
	}

	/* The other option is to use the forest model to evaluate the probability
	of a certain value of the label. Suppose we wanted to find the probability
	under the model that each of the test data belong their ground truth class.
	This results in floating point value for each test case. */
	std::array<double,N_TESTS> probabilities;
	the_classifier.probabilityGroupwise(test_ids.cbegin(),test_ids.cend(), test_data_labels.cbegin(), probabilities.begin(), false, test_feature_lambda);

	/* Output the result */
	std::cout << std::endl << "Probabilities:" << std::endl;
	for(double p : probabilities)
		std::cout << p << std::endl;
}
