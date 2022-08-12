# Customer Churn Reduction
## Evaluating Multiple Machine Learning Classification Models for Unbalanced Data

### Contributors(s)
Victoria Hall

### Description
The purpose of this project is to utilize different machine learning techniques to create a tool that a company can use to predict whether a customer is likely to churn or not. 
By providing these predictions any number of business stakeholders like, a marketing or merchant team, can produce strategies to intervene and try to prevent churn. 
Multiple classification models were tested before the final results was chosen.



### Models Evaluated
1. Logistic Regression - Baseline linear classifier for binary data
2. Random Forest - Ensemble Bagging Classifier (Bootstrap Aggregation)
    - Makes predictions by combining classifiers on bootstrapped subsets 
3. K-Nearest Neighbors - Instance Based Classifier
    - Makes predictions based on the training set only by searching for the most similar instances
4. Support Vector Machine - Maximal Margin Classifier
    - Finds the hyperplane that splits the data into two groups, and chooses the the hyperplane that maximizes the margin (distance between hyperplane and close points)
5. XGBoost - Ensemble Boosting Classifier
    - Type of gradient boosting. Traings many models in a gradual additive manner and identifies shortcomings with a loss function.
6. Multilayer Perceptron Classifier (MLP Classifer)
    - Relies on an underlying neural network


### Software Requirement(s)
- Python 3.9

### Helpful Resources
1. https://towardsdatascience.com/quickly-test-multiple-models-a98477476f0 
