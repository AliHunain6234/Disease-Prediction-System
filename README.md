# Disease Prediction System

A machine-learning project that predicts whether a patient is likely to have **heart disease** from 13 medical measurements, and serves the prediction through a small **Gradio** web form. It was built as a task for a Code Alpha machine-learning internship.

> **Educational project only.** It is not a medical device and must not be used to diagnose or treat anyone.

## What it does

1. Loads `heart.csv` (1,025 rows, 13 inputs and a `target` column, no missing values).
2. Splits the data 80/20 with a fixed random seed, stratified by the target, and scales the inputs with `StandardScaler`.
3. Trains four models: Logistic Regression, Random Forest (200 trees, depth 10), SVM (RBF kernel) and XGBoost.
4. Picks the model with the highest test accuracy, plots its confusion matrix, and saves the model and the scaler to `models/` with `joblib`.
5. Loads them again, predicts for a sample patient with a confidence value, and launches a Gradio form where you type the measurements and get "Disease Detected" or "No Disease Detected" with the confidence.

## The inputs

| Column | Meaning (standard UCI heart-disease coding) |
|---|---|
| `age` | Age in years |
| `sex` | 0 = female, 1 = male |
| `cp` | Chest pain type (0 to 3) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar above 120 mg/dl (0 or 1) |
| `restecg` | Resting ECG result (0 to 2) |
| `thalach` | Maximum heart rate reached |
| `exang` | Exercise-induced angina (0 or 1) |
| `oldpeak` | ST depression during exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels coloured by fluoroscopy |
| `thal` | Thalassemia test result code |
| `target` | 1 = disease present, 0 = no disease (the value to predict) |

## Results, and an important caveat

Test accuracy on the 205 held-out rows, as printed by the notebook:

| Model | Test accuracy |
|---|---|
| Logistic Regression | 80.98% |
| SVM | 92.68% |
| Random Forest | 100.00% |
| XGBoost | 100.00% |

**Do not read the 100% as real performance.** Only **302 of the 1,025 rows are unique; 723 (70.5%) are exact duplicates.** When rows are split at random, almost every test row has an identical copy in the training set (about 98% on average across random 80/20 splits), so a model that simply memorises the training rows scores near-perfectly. The 80.98% of the simplest model is the more believable number. To measure real performance, remove the duplicates (`df.drop_duplicates()`) before splitting, or split by patient if patient IDs are available.

## Project layout

```text
Disease_Prediction_System.ipynb   data loading, training, evaluation, saving and the Gradio app
heart.csv                         the dataset
requirements.txt                  Python packages
```

Running the notebook creates a `models/` folder (`disease_model.pkl` and `scaler.pkl`); it is not stored in this repository.

## Run it

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook Disease_Prediction_System.ipynb
```

Run the cells from top to bottom. The last cell starts the Gradio app and prints a local address to open in the browser (in Colab or a hosted notebook Gradio also prints a temporary public link).

## Limitations

- The dataset is small and has many duplicates (see above), so the reported accuracies overstate real-world performance.
- The model predicts one binary label from 13 measurements; it says nothing about the type or the severity of a condition.
- The form takes raw numeric codes (for example chest pain type 0 to 3), so a person without medical training can easily enter a wrong code.
- No threshold tuning, cross-validation or calibration was done.
