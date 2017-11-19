# Canopy - The Header-Only Library For Random Forests

Canopy is a C++ header-only template library for random forests. Random forests
are a highly flexible and effective method for constructing machine learning
models for a number of tasks by aggregating a number of decision trees.

The focus of this library is on providing an implementation that:
- Makes use of modern template-based programming techniques to provide a highly
flexible framework allowing the user to produce models for different tasks, such
as classification and regression.
- Is highly efficient in order to be suitable for using in time-critical
applications such as video processing. This is achieved with highly efficient
code as well as by taking advantage of the parallelisable nature of random
forests using multi-threading with OpenMP.
- Allows the user to execute arbitrary code to calculate features as required,
allowing for highly flexible and efficient models for image processing and other
applications.

Canopy is unashamedly an advanced tool, intended for users with a reasonable
familiarity with C++ who are prepared to dig into the details of how random
forests work to create new, efficient algorithms tailored to their own specific
purpose. If you just want a quick tool to classify your personal collection of
[iris stamens](https://en.wikipedia.org/wiki/Iris_flower_data_set), it probably
isn't what you are looking for...

### Features

The library contains a base class, `randomForestBase`, from which a range of
models may be derived. There are also two predefined models that you can use
straight away:

- `classifier` - A random forest classifier
- `circularRegressor` - A random forest model for predicting circular-valued
(wrapped) variables

Others may be added in the future... if you develop one, feel free to contribute
it!

### Dependencies

Canopy requires a C++11 enabled compiler (preferably C++14) and depends upon the
following popular, open-source libraries:

- Boost
- OpenMP (if you want to take advantage of multi-threading)
- Eigen (only for the circularRegressor model)

### Documentation

The full documentation for the library is provided [here](https://cpbridge.github.io/canopy/index.html), and includes
installation instructions, explanations and examples.

### Author

Canopy was written by [Chris Bridge](http://chrisbridge.science) at the
University of Oxford's Institute of Biomedical Engineering.

### Related

An early version of canopy was used in the implementation of a model to analyse
medical ultrasound videos of the fetal heart. More details are available in these
documents:

- C.P. Bridge, “Computer-Aided Analysis of Fetal Cardiac Ultrasound Videos”, DPhil Thesis, University of Oxford, 2017. Available on [my website](https://chrisbridge.science/publications.html).
- C.P. Bridge, C. Ioannou, and J.A. Noble, “Automated Annotation and Quantitative Description of Ultrasound Videos of the Fetal Heart”, *Medical Image Analysis 36* (February 2017) pp. 147-161. Open access available [here](http://dx.doi.org/10.1016/j.media.2016.11.006).
- C.P. Bridge, Christos Ioannou, and J.A. Noble, “Localizing Cardiac Structures in Fetal Heart Ultrasound Video”, *Machine Learning in Medical Imaging Workshop, MICCAI, 2017*, pp. 246-255. Original article available [here](https://link.springer.com/chapter/10.1007/978-3-319-67389-9_29). Authors' manuscript available on [my website](https://chrisbridge.science/publications.html).

or on the author's website at <http://chrisbridge.science/research.html>.
