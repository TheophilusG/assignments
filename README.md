# Gender Prediction Model

## Overview
This project implements a machine learning model to predict customer gender based on their shopping session data. It uses a Random Forest Classifier with extensive feature engineering, data preprocessing, and visualization techniques.

## Prerequisites
Before running the project, ensure you have Python installed (Python 3.7+ recommended). The following Python libraries are required:

### Required Libraries
To install the required dependencies, run:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn plotly
```

Alternatively, you can install all dependencies using a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Main Dependencies
- `pandas` - For handling tabular data
- `numpy` - For numerical computations
- `scikit-learn` - For machine learning models and preprocessing
- `imbalanced-learn` - For handling imbalanced datasets (SMOTE)
- `matplotlib` & `seaborn` - For data visualization
- `plotly` - For interactive visualizations

## How to Run the Model

1. **Prepare Data**: Ensure the dataset files (`trainingData.csv` and `trainingLabels.csv`) are in the working directory.

2. **Run the Script**: Execute the Python script to train the model and generate results.
```bash
python gender_prediction.py
```

3. **Outputs Generated**:
   - A trained model
   - Performance metrics (Accuracy, Confusion Matrix, Classification Report)
   - Visualizations (saved in the `visualizations/` directory)
   - An interactive dashboard (`dashboard.html`)

## Using the Model for Prediction
To make predictions on new data, modify the script to load new input data and call the `predict()` method:
```python
predictor = GenderPredictor()
predictor.train("trainingData.csv", "trainingLabels.csv")
new_data = pd.read_csv("new_data.csv")
predictions = predictor.predict(new_data)
print(predictions)
```

## Notes
- Ensure the dataset follows the expected format.
- If adding new features, update the preprocessing pipeline accordingly.
- Hyperparameters can be adjusted in the `GridSearchCV` section for better model tuning.

For any issues, refer to the documentation or raise an issue in the repository.

---

This project is part of the Scentronix Technical Test.

