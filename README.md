# AutoML and Feature Stores Assignment

## Problem Description

In this assignment, you will explore two important concepts in modern machine learning:

1. **AutoML with TPOT**: Use the TPOT library to automatically find a high-performing machine learning pipeline for the Wisconsin Breast Cancer dataset.
2. **Feature Stores**: Create a simple feature store to manage and track features used in your models.

This assignment will give you practical experience with automated machine learning and feature management systems.

## Instructions

### Part 1: AutoML with TPOT

1. Open the `assignment.py` file.
2. Complete the `find_pipeline_for_cancer_dataset()` function:
   * Load the breast cancer dataset using `sklearn.datasets.load_breast_cancer()`.
   * Split the data into training and testing sets.
   * Create an instance of `TPOTClassifier`. Use small generations (e.g., 2) and population size (e.g., 10) for faster execution.
   * Fit the TPOT classifier to the training data.
   * Return the best pipeline found by TPOT.

### Part 2: Feature Store Implementation

3. Complete the `create_feature_store()` function:
   * Convert the input data to a pandas DataFrame.
   * Add the target column to the DataFrame.
   * Create feature statistics including mean, standard deviation, min, max, and missing values.
   * Return a dictionary containing the feature store information.

### Part 3: Performance Evaluation

4. Complete the `evaluate_pipeline_performance()` function:
   * Calculate accuracy using the pipeline's `score()` method.
   * Get predictions using the pipeline's `predict()` method.
   * Calculate precision, recall, and F1-score using sklearn metrics.
   * Return a dictionary with all performance metrics.

## Hints

* The dataset is available in `sklearn.datasets`.
* Use `train_test_split` from `sklearn.model_selection` to split your data.
* The best pipeline is stored in the `fitted_pipeline_` attribute of the TPOT object after you have run `.fit()`.
* Set `random_state=42` in `train_test_split` and `TPOTClassifier` for reproducible results.
* Use `pd.DataFrame()` to convert numpy arrays to DataFrames.
* Use pandas methods like `.mean()`, `.std()`, `.min()`, `.max()` for feature statistics.
* Import required metrics: `from sklearn.metrics import precision_score, recall_score, f1_score`

## Testing Your Solution

Run the tests to verify your implementation:
```bash
python test.py
```

## Further Exploration (Optional)

* TPOT has an `export()` method that saves the Python code for the best pipeline to a file. Try using it!
* Explore how you could extend the feature store to include feature versioning and lineage tracking.
* TPOT can also be used for regression problems with `TPOTRegressor`. How would you use it on a dataset like the California Housing dataset?
* Consider how you could integrate this with a real feature store like Feast or Tecton.

## Learning Objectives

By completing this assignment, you will:
- Understand how AutoML tools like TPOT work
- Learn to create and manage feature stores
- Gain experience with automated pipeline optimization
- Practice evaluating model performance with multiple metrics
- Understand the importance of feature management in ML workflows
