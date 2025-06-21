# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a binary classifier that was trained using a Random Forest Classifier from Scikit-learn. It’s used to predict whether an individual's income is more than $50,000 a year. The model uses demographic and job-related features from the census data to make its predictions. The frameworks used in this model include Python, Pandas, NumPy, and the previously mentioned Scikit-learn.

## Intended Use

This model is intended to be used for demographic income patterns research and analysis.  However, this model should be not used for such decisions as this was designed for educational purposes and would need extensive testing and adjustments before deploying. 

## Training Data

The training data is from the UCI Adult Census Income Dataset (https://archive.ics.uci.edu/ml/datasets/adult), that was provided by Udacity. The data was split into a training set (80% of the data) and a test set (20%) of the data.

## Evaluation Data

The evaluation data from from the test split of the data. The 20% of the data that was set aside for testing was used for the evaluation. 

## Metrics
Precision - 0.7419
This was the proportion of positive identifications that were correct.

Recall - 0.6384
This measures the proportions of of actual positives that were identified correctly.

F1 Score - 0.6863
This represents harmonic mean of precision and recall. This provides a balanced measure of the model's accuracy. 

## Ethical Considerations

It should be noted that this dataset is older and may have underlying biases present. The data set could, and likely does, reflect social patterns and economical disparities across demographic groups. Due to this possibly existing in the data set, this model could produce results that are clouded by the performances in specific groups.

## Caveats and Recommendations
Ongoing evaluation and improvements should be made before using the model publicly. This was intended to be used as an education project and not released for public use and analysis. The data used is older and no longer relevant for analysis of the current year, new data would need to be collected for modern decision making. 