# Causal Direction Estimation from Data

## Overview

This repository contains a Python script for estimating causal relationships between two variables using machine learning models. The primary goal is to determine the direction of causality from observational data without relying on underlying physical theories or experimental interventions. This approach is particularly useful in scenarios where the causal relationship is non-obvious and cannot be established through traditional experimental methods.

### Theoretical Background

To illustrate, consider a dataset with altitude and temperature measurements. Without resorting to physical laws, our objective is to discern whether altitude affects temperature or vice versa. The strategy involves:

1. **Building Two Models:**
   - Model A: Altitude as the independent variable and temperature as the dependent variable.
   - Model B: Temperature as the independent variable and altitude as the dependent variable.

2. **Evaluating Prediction Errors:**
   - We analyze the error distributions for each model — the discrepancies between predicted and actual values.
   - If a model's error distribution appears more 'random' or 'noise-like,' it likely represents the correct causal direction. Conversely, systematic errors in the other model indicate an inaccurate causal assumption.

3. **Advanced Error Distribution Analysis:**
   - We implement a sophisticated technique to quantify error distribution 'regularity.' This involves assessing deviations from a hyper-ellipsoid shape, which would symbolize a normal distribution. Distinctly shaped distributions may indicate structural model issues, potentially caused by misjudged causality.

### Why The Basic Approach Makes Sense

This approach to determining causal direction from data leverages several core theoretical insights from statistics and machine learning:

1. **Causal Mechanisms in Predictive Modeling:**
   - Causality implies a specific directionality where one variable influences another. Regression models capture this directionality by using one variable to predict the other. If 'A' causes 'B', a model with 'A' as the independent variable should predict 'B' effectively, capturing the causal mechanism.

2. **Interpreting Model Errors:**
   - Errors in a well-specified model, where the causal direction aligns with reality, should be mostly random, representing unexplained noise. This randomness indicates that the model has successfully captured the systematic variations driven by the causal variable.
   - In contrast, errors in a model with the incorrect causal direction will exhibit non-random patterns, revealing the model's failure to capture the true generating process.

3. **Mahalanobis Distance in Error Analysis:**
   - Mahalanobis distance measures how deviations in error distributions compare to normal randomness, considering the data's covariance. This metric helps determine which model's errors align more closely with random noise, suggesting a correctly specified causal direction.

4. **Comparative Error Structure Analysis:**
   - By contrasting two models, each assuming a different causal direction, and evaluating their error distributions, we assess which model aligns more closely with the principle of causality leading to random residual noise.

5. **Assumptions and Practical Considerations:**
   - The method presumes linear or adequately captured relationships by the chosen models. Complex or nonlinear causal relationships might not be accurately depicted.
   - The presence of confounding variables can mislead the analysis by introducing spurious correlations.
   - While this method suggests a more likely causal direction, it doesn't conclusively prove causation. It should be viewed as part of a broader exploratory analysis, supplemented by theoretical knowledge and, if feasible, experimental validation.

### Intuitive Understanding of the Approach

The rationale behind this data-driven approach for determining causal direction is grounded in the fundamental nature of causality and predictive modeling:

1. **Causality and Predictability:**
   - A causal relationship implies that changes in one variable systematically result in changes in another. If this holds true, a model should accurately predict the effect variable using the cause variable, leaving only random noise in the residuals.

2. **Error Characteristics as Indicators:**
   - Randomness in a model's residuals suggests successful capture of the systematic component, indicative of the correct causal direction. Structured or patterned residuals hint at a mis-specified model, likely with an incorrect assumption about causality.

3. **Diagnostic Role of Mahalanobis Distance:**
   - This statistical measure helps evaluate how error distributions deviate from expected randomness. Lesser deviation implies a better causal model, capturing the true directionality.

4. **Efficacy of Comparative Analysis:**
   - Comparing error distributions from models with reversed variables offers insights into which direction is more causally plausible, based on the nature of the residuals.

5. **Data-Centric Insights:**
   - The approach leverages the intrinsic properties of the data, assuming the data reflects the underlying causal relationship. It uses the error structures to infer the more probable causal direction.

In essence, this methodology exploits the principle that correct causality modeling results in random-like errors, while incorrect modeling leaves structured errors. The comparison of these error structures from two different causal assumptions in the models helps infer the likely direction of causality. It is important to apply this method with a comprehension of its theoretical underpinnings and inherent limitations, ensuring a holistic approach to causal analysis.


### Unique Aspects of the Script

1. **Fully Automated Analysis:**
   - Distinct from similar tools, this script offers a fully automated process requiring minimal user intervention. Users need only to provide a dataset, and the script executes comprehensive causal analysis.

2. **Broad Applicability:**
   - Designed for real-world scenarios where causality is not evident and experimental validation is unfeasible, such as in economics, epidemiology, and social sciences.

3. **Comprehensive Model Testing:**
   - Employs a wide array of machine learning models to ensure robustness in causal estimation, accommodating various data characteristics.


## Repository Structure

- `causal_direction_estimation_from_data.py`: Main script implementing the causal direction estimation methodology.

## Script Functionality

The script performs the following functions:

1. **Model Fitting:** 
   - Automatically fits a range of machine learning models using scikit-learn, testing both possible causal directions between each pair of variables.

2. **Error Analysis:** 
   - Analyzes and quantifies the structure of prediction errors for each model, assessing the 'randomness' of the error distributions.

3. **Causal Direction Estimation:** 
   - Calculates a ratio representing the 'noise-like' quality of error distributions for each direction. A higher ratio in one direction suggests a stronger likelihood of that being the correct causal relationship.

4. **Visualization:** 
   - Generates heatmaps to visually represent the strength of causal relationships and associated p-values for permutation and bootstrap tests.

## Usage

1. **Data Preparation:** 
   - Prepare a dataset in CSV format with columns representing different variables for which causal relationships are to be tested.

2. **Running the Script:** 
   - Modify the file path in the script to point to your dataset.
   - Execute the script to perform causal analysis. The script will automatically handle data normalization, model fitting, error analysis, and heatmap generation.

3. **Interpreting Results:** 
   - Analyze the output, focusing on the Mahalanobis ratio and visual heatmaps to infer potential causal relationships.

## Requirements

- Python 3.x
- Libraries: pandas, numpy, scikit-learn, scipy, seaborn, matplotlib

## Limitations and Considerations

- This method is observational and does not provide definitive proof of causality.
- Works best with large datasets where sufficient variability exists in the data.
- The approach relies on the assumption that the causal relationship, if present, is captured by the models used.

## Detailed Function Breakdown of Causal Direction Estimation Script

### Function: `get_models()`
- **Purpose:** This function creates and returns a dictionary of various regression and machine learning models.
- **Theoretical Underpinning:** Each model encapsulates different assumptions and methods for capturing relationships in data. Linear models assume a linear relationship between variables, while others, like KNeighbors and SVR, can model more complex, non-linear interactions.
- **Intuitive Explanation:** Think of each model as a different lens to view the data. Linear models view relationships as straight lines, while others might see curves or more complex patterns. This variety helps in examining the data from multiple perspectives.
- **Models Included:**
  - `LinearRegression`: Standard linear regression without regularization.
  - `Ridge`: Linear regression with L2 regularization.
  - `BayesianRidge`: Bayesian approach to ridge regression.
  - `Lasso`: Linear regression with L1 regularization, using a reduced alpha (0.01) for softer penalization.
  - `ElasticNet`: Combines L1 and L2 regularization, also with reduced alpha.
  - `GradientBoostingRegressor`: A machine learning model that uses gradient boosting.
  - `KNeighborsRegressor`: Regression based on k-nearest neighbors.
  - `SVR`: Support Vector Regression.

### Function: `fit_models_and_analyze_errors(X, y, models, column_names)`
- **Purpose:** Fits each model to the data and analyzes the prediction errors.
- **Theoretical Underpinning:** The core concept is based on regression analysis, where the goal is to predict an outcome (dependent variable) from one or more predictors (independent variables). The quality of these predictions, and the errors they produce, can inform us about the relationships in the data.
- **Intuitive Explanation:** Imagine trying to predict tomorrow's temperature (y) based on today's altitude (X) using different methods. Some methods might get close to the actual temperature, with small errors, while others might be way off. Analyzing where and how these predictions go wrong (the errors) can tell us a lot about the relationship between altitude and temperature.
- **Process:**
  1. **Data Preparation:** Converts input series `X` and `y` to NumPy arrays and reshapes them if they are 1-dimensional.
  2. **Error Collection:** Initializes a dictionary to hold error information for each model.
  3. **Model Fitting:**
     - Iterates through each model, splitting the dataset using `TimeSeriesSplit` for cross-validation.
     - Fits the model on each train-test split, then predicts and collects errors (difference between actual and predicted values).
  4. **Error Analysis:**
     - Aggregates errors across all folds.
     - Checks for low variance in predictions and aggregated errors as an indicator of potential model underperformance.
     - Stores errors, mean cross-validated mean squared error (MSE), and R² scores in the `errors` dictionary.

### Function: `calculate_mahalanobis_distance(errors)`
- **Purpose:** Computes the Mahalanobis distance for the given error array, indicating how far the errors deviate from a normal distribution.
- **Theoretical Underpinning:** The Mahalanobis distance is a multivariate statistical measure that gauges the distance of a point from a distribution, considering the distribution's covariance. It's a way to measure the 'strangeness' or 'unusualness' of the errors in a multivariate context.
- **Intuitive Explanation:** If the errors from our predictions are just random noise, they should scatter randomly and typically around zero. Mahalanobis distance checks how 'normal' this scatter is. If the errors are not just random noise but have some pattern, this distance will be larger, signaling something more than mere chance at play.
- **Process:**
  1. **Pre-processing Checks:** Ensures the errors array is suitable for calculation (not empty, not singular).
  2. **Calculation:**
     - Computes the mean and covariance of the error array.
     - Calculates the Mahalanobis distance for each error point.
     - Averages these distances and converts them into cumulative distribution function (CDF) values using the chi-square distribution.

### Function: `calculate_mahalanobis_ratio(data1, data2, models, column_names)`
- **Purpose:** Calculates the ratio of the average Mahalanobis distances for two sets of errors (each assuming different causal directions).
- **Theoretical Underpinning:** This function is based on the concept that the Mahalanobis distance is a measure of the divergence of a point from a distribution. In causal analysis, this helps to quantify how 'structured' or 'random' the error distribution is in each model, providing insights into which model more accurately captures the causal relationship.
- **Intuitive Explanation:** Imagine throwing darts at a target. If your throws (errors) are all over the place (random), it suggests you're aiming correctly but with some natural variability. If your throws consistently miss in one direction, it implies something is systematically off. This function is like comparing the pattern of your throws in two scenarios to see which one is more random, indicating a better aim (or causal direction).
- **Process:**
  1. Fits models and analyzes errors for both directions (data1 → data2 and data2 → data1).
  2. Computes average Mahalanobis distances for each direction.
  3. Returns the ratio of these average distances.

### Function: `permutation_test_mahalanobis(data1, data2, models, column_names, n_iterations)`
- **Purpose:** Performs a permutation test to assess the significance of the calculated Mahalanobis ratio.
- **Theoretical Underpinning:** Permutation tests are a non-parametric statistical method used to test hypotheses. By shuffling the data and recalculating the Mahalanobis ratio, this function assesses whether the observed ratio is significant or just a result of random chance.
- **Intuitive Explanation:** Think of this like shuffling a deck of cards repeatedly and checking if a specific arrangement (the observed ratio) occurs often or is rare. If it's rare, it suggests that the original arrangement (the observed Mahalanobis ratio) wasn't just a fluke and might have meaningful implications.
- **Process:**
  1. Computes the observed Mahalanobis ratio.
  2. Randomly permutes the data pairs and recalculates the ratio for each permutation.
  3. Counts how often the permuted ratio exceeds the observed ratio.
  4. Calculates a p-value based on this count.

### Function: `bootstrap_test(data1, data2, n_iterations, size)`
- **Purpose:** Conducts a bootstrap test to assess the difference in means between two datasets.
- **Theoretical Underpinning:** Bootstrap testing is a resampling technique used to estimate statistics on a population by sampling a dataset with replacement. It allows for understanding the variability of the estimated difference in means without making strict assumptions about the distribution of the data.
- **Intuitive Explanation:** Imagine you have two buckets of apples, each with different average weights. You repeatedly take a handful from each bucket, measure the average weights, and put them back. By doing this many times, you get a sense of how often the difference in average weight you initially observed could happen just by chance. If it's rarely observed in your repeated samples, the initial observation is likely significant.
- **Process:**
  1. Calculates the observed difference in means.
  2. Performs resampling with replacement to create bootstrap samples.
  3. Calculates the mean difference for each bootstrap sample.
  4. Counts how often the bootstrap difference exceeds the observed difference.
  5. Calculates a p-value based on this count.

### Function: `analyze_causal_direction(data, column1, column2, models)`
- **Purpose:** Analyzes the causal direction between two variables using the previously defined functions.
- **Theoretical Underpinning:** 
  - This function embodies the concept of causality in statistical modeling, where the direction of the causal relationship is crucial. The models aim to capture the dependency of one variable on another, and the comparison of error structures from both directions provides insight into which model aligns better with the causal reality.
- **Intuitive Explanation:** 
  - Think of this function as a detective examining evidence from two scenarios. Each model represents a different story about who influences whom. By looking at which story leaves less 'unexplained' (random noise in errors), the function makes an educated guess about the actual causal relationship.
- **Process:**
  1. Standardizes the data for the two variables.
  2. Fits models and analyzes errors in both possible causal directions.
  3. Calculates the Mahalanobis ratio.
  4. Determines the likely causal direction based on the ratio.
  5. Returns the results, including the determined causal direction and error arrays.

### Function: `normalize_data(data)`
- **Purpose:** Normalizes each column in the dataset.
- **Theoretical Underpinning:** 
  - Normalization is a fundamental step in data preprocessing, especially important for machine learning models. It ensures that all variables contribute equally to the analysis and prevents variables with larger scales from dominating the model's behavior.
- **Intuitive Explanation:** 
  - Imagine trying to compare the importance of apples and oranges in a fruit salad. If apples are counted in dozens and oranges in singles, the count of apples will overshadow the oranges. Normalization is like ensuring each fruit is counted in a comparable way, say, by individual pieces.
- **Process:** Subtracts the mean and divides by the standard deviation for each column in the dataset.

### Function: `create_heatmap(matrix, labels, title)`
- **Purpose:** Creates and displays a heatmap based on
 the given matrix.
- **Theoretical Underpinning:** 
  - The heatmap visualizes complex data in an easily interpretable format. By representing numerical values with color gradients, it allows for quick identification of patterns and relationships, which is crucial in making sense of statistical outputs.
- **Intuitive Explanation:** 
  - Consider a heatmap as a colorful map showing the landscape of your data. Just like how a geographical map uses colors to show elevations, a heatmap uses colors to show the intensity or frequency of your data, making it easier to spot where the 'highs' and 'lows' are.
- **Process:**
  1. Initializes a mask to only show the upper triangle of the matrix.
  2. Creates a heatmap using Seaborn with the provided matrix, labels, and title.

#### Main Script: `process_data(file_path)`
- **Purpose:** Executes the entire causal analysis process for the given file path.
- **Process:**
  1. Loads and preprocesses the data.
  2. Performs causal analysis on all combinations of variables.
  3. Stores results in matrices for Mahalanobis ratios and p-values from permutation and bootstrap tests.
  4. Generates and displays heatmaps to visualize the results.
