# EduMetrics — College Placement & Salary Analytics (Python)

A beginner-friendly Streamlit project for exploring student placement patterns, estimating placement probability, and predicting salary with simple machine-learning models.

**Data note:** `data/student_placement_sample.csv` contains 600 synthetic demo records. They are not real college placement statistics.

## Run on Windows

1. Install Python 3.10 or newer.
2. Extract this ZIP and open the `EduMetrics_Python_Fixed` folder in VS Code.
3. In the terminal, run:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

4. Open the local URL printed in the terminal, usually `http://localhost:8501`.
5. Stop the app with `Ctrl+C` in the terminal.

## App sections

- **Overview:** student count, placement rate, average CGPA, average salary, and charts.
- **Explore Data:** filter records, inspect correlations, and download filtered data.
- **Predict:** enter a student profile to estimate placement probability and salary.

There is no Viva or Methodology tab in the UI. `VIVA_GUIDE.md` is a separate optional study document.

## Data Science concepts

- Descriptive statistics and visualization
- Correlation analysis
- Logistic regression for placement classification
- Linear regression for salary estimation among placed students
- Train/test split and evaluation with accuracy, MAE, and R²
- Median imputation and feature scaling

## CSV format for your own data

Required columns: `cgpa`, `internships`, `projects`, `communication_score`, `aptitude_score`, `technical_score`, `placed`.

For salary prediction, include `salary_lpa`; `branch` and `student_id` are optional. Placement outcomes should be `Yes` or `No` (also accepts common 1/0 and true/false labels).

## Important limitations

- The included dataset is synthetic and cannot support claims about real placement outcomes.
- Predictions are educational estimates, not guarantees of employment or salary.
- Correlation does not prove causation.
- Get permission and protect personal information when using real student data.
