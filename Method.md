# Method and validation

The aim of this exercise is to predict incidence rates for new product surveys, based on known incidences of past surveys. 

The exercise specifically targets the addition of a survey about a new product category, so the model will have to make predictions based on unseen product categories.

It seemed important thus to find common properties between the product categories, such as categorising in broad groups or attributing features to each of the categories. 

Given the shortness of time, manually classifying the product list was not an option, so I used OpenAI's API to come up with a classification. This is stored in [broad_categories.json](broad_categories.json).

I first asked for a set of broad groups, then asked to classify the product list into these groups, while also attributing some subjective features to the categories:

- How broadly is the product expected to be used by the public
- How often an average person is expected to purchase it
- How often is she expected to use it
- How expensive it is
- ...

This is done in the notebook [1-LLMFeatures.ipynb](1-LLMFeatures.ipynb). 

The resulting product features are stored in [product_features.json](product_features.json).

## Feature extraction

The product features are merged with the training dataset, then a set of features are extracted from the questions and answers using regexes. Individual countries are gouped into continents. This is expected to help for countries with small sample sizes. 

The dataset with new features is created with [2-AddFeatures.ipynb](2-AddFeatures.ipynb) and stored in [train_augmented.csv](train_augmented.csv). Similarly, the prediction set is augmented with the new features: [test_augmented.csv](test_augmented.csv).

## Baseline

A baseline model, predicting all values with the mean of a training set (80% of the incidence dataset) established the following performance indicators on the logit target:
* Prediction MAE = 0.76
* 80% prediction interval coverage = 79%

## Prediction model

The model that was chosen is a linear mixed effects model, allowing for a hierarchical grouping of variances that was expected to help for groups with small sample sizes. It is also a model that is easier to interpret than more advanced machine learning models or neural networks. 

I first split the incidence dataset into a training and a validation set. The two sets do not share any product categories in order to evaluate the performance of the model when predicting incidence for new products.

The incidence value is logit-transformed, as its values are in the range 0-1 and the regression can predict outside these.


The model only uses a small set of features that were seen in [exploratory analysis](3-EDA.ipynb) to have more predictive value (higher correlation with target). These are:
- Broad category, extracted by the LLM
- Product category, from the original dataset
- Audience breadth, extracted by the LLM
- Cost tier, extracted by the LLM
- Number of choices in positive and negative answer


After training, the model is used to predict the validation set and the predictions compared to the actual incidence value. To do this, no random effect coefficient may be used for the product category level, which is set to and inheriting the coefficient from the parent (`broad_category`). 

Some performance indicators show that the model has some predictive power, and is estimating the prediction interval fairly well:
* Prediction MAE = 0.58
* 80% prediction interval coverage = 79%

The prediction interval is based on the variance of product means (random effect) + expected variance within a group (residual variance). The sources of variance merit some attention:
* Variance between products seen as random effects is important because we will be predicting for unseen target groups. The value of the mixed effects model is that separates variability associated with product categories from residual variability. Group-level effects allow information to be shared across observations, while the estimated variance of unseen product categories is included in the prediction intervals. The `broad_category` was chosen as a grouping feature but it is seen to capture much less variance than what is left fot the child `product_category` feature. A different gruping could be investigated.
* The residual variance is due to unexplained variability in the sample. It could be decreased by proposing different features independent of the product categorisation (e.g. relying on different information about the surveys, etc)
* There is another source of variance that I did not include in the estimation. The known incidence values are themselves uncertain, with a level of uncertainty that depends on the surveyed population sample. The value of the uncertainty can be estimated with a binomial distribution and produce values between 0.04 and 0.9 in logit incidence. This should be taken into account in the modelling but was not at present.

## Future work

Given more time, I intended to explore different machine learning techniques. I would expect that some of the relationships between features are highly non-linear and would like to take advantage of decision tree based methods to explore these, such as with CatBoost. I am also particularly interested in attempting bayesian approaches for these kind of hierarchical problems. 

Improvements in the feature sets could also provide big benefits. One idea would be to replace the product category grouping with product similarity, for example based on word-vector or document-vector representations of the product categories, possibly with dimensionality reduction with techniques such as PCA. The regression could then take advantage of product dimension similarity.


# Question 3

To answer the question about transposing one measured incidence from Australia to the United States, I calculated the distribution of incidence differences across all product categories that were present in both countries.

I used the mean difference to calculate the predicted incidence in the US by adding it to the measured incidence in Australia. 

To estimate the uncertainty on this value I used a validation split approach, by holding out each product, making the prediction on the remaining products as above and calculating the residual of this prediction. I then used the distribution of residuals to measure the 10th and 90th percentile, add to the incidence value in Australia and estimate upper an lower bounds of the prediction. 

One caveat is that unlike other product categories, Air Conditioners are affected by the climate of each country and the relationship between the population response in both countries may be different than for most other categories. 

More details and calculations can be found in [5-Country-Transfer.ipynb](5-Country-Transfer.ipynb)


