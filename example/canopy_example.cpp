// Standard Library Headers
#include <iostream>
#include <array>
#include <random>
#include <algorithm>
#include <string>

// The canopy classifier header
#include <canopy/classifier/classifier.hpp>

int main()
{
	/* Parameters of the test */
	constexpr unsigned N_CLASSES = 3;
	constexpr unsigned TRAINING_DATA_PER_PER_CLASS = 200;
	constexpr unsigned TOTAL_TRAINING_DATA = N_CLASSES * TRAINING_DATA_PER_PER_CLASS;
	constexpr unsigned N_DIMS = 2; // dimensionality of the feature space
	constexpr double MIN_MU = 0.0;
	constexpr double MAX_MU = 10.0;
	constexpr double MAX_SIGMA = 3.0;
	constexpr int N_TREES = 128;
	constexpr int N_LEVELS = 10;
	constexpr unsigned N_TESTS = 10;
	const std::string FILENAME = "example_model.tr";

	/* Set up random number generation */
	std::default_random_engine rand_engine;
	std::random_device rd{};
	rand_engine.seed(rd());
	std::normal_distribution<double> norm_dist;
	std::uniform_int_distribution<int> uni_int_dist;
	std::uniform_real_distribution<double> uni_real_dist;

	/* Randomly generate 2D sigma and mu parameters for each class assuming
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
	and levels */
	canopy::classifier<1> the_classifier(N_CLASSES,N_TREES,N_LEVELS);

	/* In order to train the model, we need a function object to return the
	features from the array. A C++14 generic lambda is a convenient way to do this
	as it can capture the data array by reference and figure out all the types
	for us */
	auto train_feature_lambda = [&] (auto first_id, const auto last_id, const std::array<int,1>& params, std::vector<float>::iterator out_it)
	{
		while(first_id != last_id)
		{
			const int id = *first_id;
			*out_it++ = training_data_features[id][params[0]];
			++first_id;
		}
	};

	/* We also need a way of generating valid parameter values for training */
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

	/* Generate some unseen test data */
	std::array<std::array<double,N_DIMS>,N_TESTS> test_data_features;
	std::array<int,N_TESTS> test_data_labels;
	for(unsigned n = 0; n < N_TESTS; ++n)
	{
		/* Choose a random label */
		int c = uni_int_dist(rand_engine,std::uniform_int_distribution<int>::param_type{0,N_CLASSES-1});
		test_data_labels[n] = c;

		/* Generate some features from with this class */
		for(unsigned d = 0; d < N_DIMS; ++d)
		{
			test_data_features[n][d] = norm_dist(rand_engine,std::normal_distribution<double>::param_type{mu[c][d],sigma[c][d]});
		}
	}

	/* We need a way of identifying each of the data points in the test
	set. This is done by the index of the data point in the list */
	std::array<int,N_TESTS> test_ids;
	std::iota(test_ids.begin(),test_ids.end(),0);

	/* We need a functor to calculate the features for the test set */
	auto test_feature_lambda = [&] (auto first_id, const auto last_id, const std::array<int,1>& params, std::vector<float>::iterator out_it)
	{
		while(first_id != last_id)
		{
			const int id = *first_id;
			*out_it++ = test_data_features[id][params[0]];
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
