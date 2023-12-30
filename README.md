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
   - Our approach extends beyond traditional methods by employing a sophisticated scoring system to analyze the regularity and characteristics of error distributions from various machine learning models. This system includes:
     - **Statistical Tests**: We apply the Shapiro-Wilk test to assess normality, along with skewness and kurtosis tests to understand the shape of the error distributions. These tests help determine if the errors behave like random noise or exhibit systematic patterns.
     - **Functional Data Analysis (FDA)**: FDA is used to transform the error distributions into a functional data form. We then analyze the variance and depth of these distributions. A significant difference in the variance or depth between two sets of errors suggests a potential causal direction.
     - **Distribution Fitting and AIC Scoring**: We fit different statistical distributions to the errors and use the Akaike Information Criterion (AIC) for model selection. This step aims to identify which error distribution aligns more closely with a theoretical distribution, indicating a better model fit and thus a more likely causal direction.
   - The scoring system assigns weights to the results of these tests and analyses. The total scores for each possible causal direction are compared, with a higher score indicating a more likely causal relationship. This comprehensive analysis helps in inferring the most plausible causal direction between two variables, based on the behavior of their prediction errors.

### Why The Basic Approach Makes Sense

This approach to determining causal direction from data leverages several core theoretical insights from statistics and machine learning:

1. **Causal Mechanisms in Predictive Modeling:**
   - The underlying assumption is that causality implies directionality, where one variable influences another. We use a variety of regression models, each capturing this relationship differently. If 'A' causes 'B', a model treating 'A' as the independent variable should effectively predict 'B', implying a causal influence.

2. **Multifaceted Error Analysis:**
   - The analysis goes beyond examining random errors in well-specified models. It involves:
     - **Statistical Tests on Errors**: We apply Shapiro-Wilk, skewness, kurtosis, and autocorrelation tests to the prediction errors. These tests help assess whether errors are random (indicative of a good model fit and potential causal direction) or have non-random patterns (suggesting incorrect causal assumptions).
     - **Functional Data Analysis (FDA)**: We transform error structures into functional data, allowing us to analyze their variance and mean differences. This step is crucial in understanding the depth and variability of errors in a more nuanced manner.

3. **Comparative Error Structure and Distribution Analysis:**
   - We contrast models assuming different causal directions by analyzing their error distributions and applying Functional Data Analysis (FDA).
   - Distribution fitting to errors, with the Akaike Information Criterion (AIC) for model selection, helps in understanding how the errors conform to theoretical distributions. Lower AIC values suggest a better fit and, by extension, a more plausible causal direction.

4. **Scoring System for Causal Inference:**
   - We employ a scoring system that aggregates results from statistical tests, FDA, and distribution fitting. This comprehensive scoring reflects the complex nature of causal relationships better than any single metric.

5. **Ratio-Based Decision Making:**
   - The method calculates a ratio of scores for each possible causal direction, using a predefined threshold to suggest the more likely direction. If the ratio does not meet the threshold, the analysis remains inconclusive.

### Intuitive Understanding of the Approach

This methodology for determining causal direction is deeply grounded in predictive modeling, comprehensive statistical testing, and functional data analysis (FDA). It adopts a multifaceted approach, utilizing various statistical methods to deduce the most probable causal relationship between variables in time series data.

1. **Causality and Predictive Modeling:**
   - The approach is based on the premise that a true causal link between two variables should enable accurate predictions. In such cases, using the causal variable to predict the effect variable results in residuals that predominantly exhibit random noise.

2. **Detailed Error Analysis as Indicators of Causality:**
   - Beyond seeking randomness in model residuals, this method conducts an extensive error analysis across multiple models. It investigates whether error characteristics, such as distribution and structure, significantly differ when hypothesized causal directions are reversed.

3. **Integration of Statistical Tests and Functional Data Analysis:**
   - Statistical tests including Shapiro-Wilk for normality, kurtosis, skewness, and autocorrelation are employed to scrutinize the error structure. FDA further deepens this analysis, exploring variance and depth in the error patterns across various models.

4. **Distribution Fitting and Akaike Information Criterion (AIC) Scoring:**
   - Different statistical distributions are fitted to the model errors, and the Akaike Information Criterion (AIC) is used to select the model that best approximates randomness. This step aids in identifying the most accurate causal model based on error distribution.

5. **Scoring System for Causal Direction Inference:**
   - A scoring system, based on the outcomes of statistical tests, FDA, and AIC scores, is applied. This system compares the scores for each potential causal direction, using a predefined threshold to deduce the more likely causal relationship, or declaring the analysis inconclusive when the evidence is insufficient.

6. **Comprehensive Ratio-Based Analysis for Causality:**
   - The method employs a ratio of scores for the two potential causal directions. If this ratio surpasses a certain threshold, it suggests a stronger likelihood of that direction being the causal one. This ratio-based approach ensures a balanced assessment of diverse statistical elements.

In essence, this methodology leverages the principle that accurate causal modeling leads to error structures that are close to random, while incorrect causal assumptions result in distinct error patterns. By comparing these structures under different causal hypotheses and integrating a broad spectrum of statistical techniques, the approach offers a nuanced and robust way to infer the likely direction of causality.

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
   - Generates heatmaps to visually represent the strength of causal relationships.

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


### Function: `transform_to_fdata(errors)`
- **Purpose:** Transforms an array of errors into functional data format suitable for Functional Data Analysis (FDA).
- **Theoretical Underpinning:** 
  - This function is central to FDA, where data is treated as a continuum rather than discrete points. The transformation facilitates advanced analytical techniques like variance and depth analysis.
- **Intuitive Explanation:** 
  - Imagine errors not just as isolated points but as part of a continuous curve. This function reshapes error data into a format that lets us analyze it as a continuous function, revealing patterns not visible in discrete analysis.
- **Process:** 
  - Converts the error array into a one-dimensional array.
  - Generates a grid of points and reshapes the data into a matrix.
  - Creates a functional data object, `FDataGrid`, from this matrix.

### Function: `add_jitter(data, jitter_amount=1e-6)`
- **Purpose:** Adds a small amount of random noise (jitter) to the data.
- **Theoretical Underpinning:** 
  - Jittering is a technique often used in data visualization and analysis to prevent overplotting and to handle cases of zero variance in data.
- **Intuitive Explanation:** 
  - If data points overlap exactly or are too uniform, it's hard to see their true distribution. Adding jitter is like slightly shaking a pile of overlapping papers to spread them out and see each one better.
- **Process:** 
  - Applies a small, random fluctuation to each data point, controlled by the `jitter_amount` parameter.

### Function: `synchronize_errors(errors_X1, errors_X2)`
- **Purpose:** Synchronizes two error arrays by ensuring they are of the same length and free of non-numeric values.
- **Theoretical Underpinning:** 
  - Error synchronization is crucial for comparative analysis, especially when performing statistical tests or functional data analysis on paired error data.
- **Intuitive Explanation:** 
  - It's like aligning two strings of beads so that each bead from one string has a corresponding bead in the other. This alignment is necessary to make meaningful comparisons between the two strings.
- **Process:** 
  - Converts error lists into numpy arrays.
  - Removes any non-numeric values (NaN or infinity) and synchronizes the lengths of both arrays.

### Function: `fda_analysis(errors_X1, errors_X2)`
- **Purpose:** Performs Functional Data Analysis on two sets of errors to compare their statistical properties.
- **Theoretical Underpinning:** 
  - FDA is used here to analyze and compare the variance, mean, and depth of error functions, providing a deeper understanding of error characteristics in a continuous domain.
- **Intuitive Explanation:** 
  - Imagine analyzing two different waves in the ocean. Instead of just looking at individual waves (errors), you're examining the overall shape, size, and movement patterns of these waves to understand their nature.
- **Process:** 
  - Synchronizes and cleans the error data.
  - Transforms error arrays into functional data.
  - Computes and compares various statistical measures like variance, mean, and functional depth between the two sets of errors.
  - Handles cases with insufficient data or zero variance by returning appropriate values or flags.

### Function: `fit_distributions(errors)`
- **Purpose:** Fits various statistical distributions to the error data from models and selects the best fit using the Akaike Information Criterion (AIC).
- **Theoretical Underpinning:** 
  - This function is based on the principle that understanding the distribution of errors can give insights into the model's performance and the underlying data structure. Different distributions provide different lenses to view and interpret these errors.
- **Intuitive Explanation:** 
  - Imagine the errors as a set of data points scattered in a certain way. `fit_distributions` tries on different 'glasses' (distributions) to see which one views these points most clearly and accurately, thereby understanding the error pattern better.
- **Process:**
  1. Checks for empty or problematic error data.
  2. Tries fitting different statistical distributions (normal, gamma, beta, exponential) to the errors.
  3. For distributions like Beta, normalizes the error data.
  4. Selects the distribution that best fits the errors based on the lowest AIC value, indicating the most efficient model in terms of information loss.

### Function: `prediction_error_analysis(errors_X1, errors_X2, threshold=3)`
- **Purpose:** Analyzes prediction errors from various models to determine the most likely causal direction between two variables.
- **Theoretical Underpinning:** 
  - Combines concepts from statistical testing, functional data analysis, and model comparison to infer causal relationships. It assumes that the correct causal direction will yield a particular pattern in the prediction errors.
- **Intuitive Explanation:** 
  - Think of this function as a judge weighing evidence from two sides (X1 causing X2 and vice versa). It examines the 'quality' of errors from models in both scenarios using various statistical measures and decides which causal direction seems more convincing.
- **Process:**
  1. Validates the input as error dictionaries from models.
  2. Assigns weights to different types of error analysis (statistical tests, FDA, AIC scores).
  3. Scores each potential causal direction based on how the errors conform to expectations (e.g., randomness, distribution shape).
  4. Compares these scores using a ratio to determine the more likely causal direction, or concludes inconclusiveness if the evidence is not strong enough.
  5. Returns the ratio and the inferred causal direction, or a statement of inconclusiveness.

### Function: `fit_models_and_analyze_errors(X, y, models, column_names)`
- **Purpose:** This function fits a variety of regression models to the data and performs an in-depth analysis of the prediction errors for each model.
- **Theoretical Underpinning:** The function is grounded in regression analysis principles, aiming to understand the relationship between an independent variable (X) and a dependent variable (y) through various predictive models. It emphasizes the importance of error analysis in understanding model performance and the underlying data relationships.
- **Intuitive Explanation:** Consider you're attempting to predict a dependent variable (y) based on an independent variable (X) using different models. This function systematically evaluates how each model performs, not just in terms of its predictive accuracy but also by examining the nature and variance of its errors.
- **Process:**
  1. **Variation Check:** Ensures sufficient variation exists in both `X` and `y` to allow for meaningful analysis.
  2. **Error Collection and Analysis Setup:** Prepares for collecting detailed error information for each model, including mean squared error (MSE) and R² scores.
  3. **Model Fitting and Error Tracking:**
     - Iterates through each provided model, splitting the dataset using `TimeSeriesSplit` for time series cross-validation.
     - In each split, the model is fitted to the training data, and predictions are made for the test set.
     - Prediction errors (difference between actual and predicted values) are collected, along with a flag for low variance in predictions, which may indicate model underperformance.
  4. **Comprehensive Error Analysis:**
     - Aggregates errors from all splits and checks for low variance, both in individual predictions and aggregated errors, which could signal issues with the model.
     - Compiles and stores detailed error information, including aggregated errors, cross-validated MSE, and R² scores, for each model in a dictionary.


### Function: `analyze_causal_direction(data, column1, column2, models, threshold=3)`
- **Purpose:** This function evaluates the causal direction between two variables by analyzing the errors of various predictive models in both potential causal directions.
- **Theoretical Underpinning:** 
  - At its core, this function applies the principles of statistical modeling to uncover causal relationships. It scrutinizes how well each model can predict one variable from another and compares the error structures from both directions to discern the more likely causal link.
- **Intuitive Explanation:** 
  - Imagine the function as a detective analyzing two narratives: one where `column1` influences `column2`, and another where `column2` influences `column1`. The function assesses each narrative by examining the 'leftover' errors (or clues) from various models and determines which narrative is more plausible based on the error characteristics and a scoring system.
- **Process:**
  1. Validates the specified columns and standardizes the data for the two variables.
  2. Fits multiple machine learning models to the data in both possible causal directions, analyzing the errors generated.
  3. Utilizes the `prediction_error_analysis` function to compare the errors and compute a score ratio, based on statistical tests, functional data analysis, and distribution fitting.
  4. Determines the most likely causal direction by comparing the score ratios to a predefined threshold.
  5. Returns the results, including the identified causal direction, score ratio, and error arrays for further analysis.


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
- **Purpose:** To execute the full causal analysis process for the specified file path.
- **Process:**
  1. **Data Loading and Preprocessing:** 
     - Loads data from the given file path.
     - Preprocesses the data, including handling infinite values and ensuring data consistency.
  2. **Normalization (Conditional):** 
     - Normalizes the data if the `use_normalize_input_data` flag is set. This step is crucial for ensuring that the data is on a comparable scale across different variables.
  3. **Causal Analysis on Variable Combinations:**
     - Iteratively performs causal analysis for every possible pair of variables in the dataset.
     - Utilizes a range of machine learning models to assess the causal relationship between each pair.
  4. **Ratio Matrix Construction:**
     - Constructs a matrix to store the ratio scores derived from the causal analysis, indicating the strength and direction of the inferred causal relationships.
     - Each entry in the matrix represents the causal direction strength from one variable to another.
  5. **Progress Updates:**
     - Provides ongoing updates on the progress of the analysis, indicating how many combinations of variables have been processed out of the total.

- **Output:** 
  - Returns a matrix of ratio scores representing the causal relationship strengths, along with a list of the column names (variables) analyzed.