## Fundamentals of Python & Applications in Data Science
### Group Project Proposal

**Project Title**: Airbnb Price Prediction and Explainable Machine Learning Analysis

**Module**: Fundamentals of Python & Applications in Data Science  
**Selected Option**: Option B – Machine Learning

### Group Members
- **Joel Suhner** (`suhnejoe@students.zhaw.ch`)
- **Maximillian Galm** (`galmmax1@students.zhaw.ch`)
- **Anastasiia Bühler** (`buehlana@students.zhaw.ch`)
- **Lobsang Dadutsang** (`dadutlob@students.zhaw.ch`)
- **Jason Winter** (`winteja2@students.zhaw.ch`)

### Project Description
The objective of this project is to develop and evaluate regression models to predict Airbnb listing prices based on structural and geographic features. The project will implement a reproducible, end-to-end machine learning pipeline including:

- **Data ingestion**
- **Exploratory data analysis (EDA)**
- **Data cleaning**
- **Feature engineering**
- **Data validation**
- **Model training and evaluation**

As baseline and comparison models, we plan to use **Linear Regression** and **Random Forest**, with a potential extension to **Gradient Boosting** methods. Model performance will be assessed using **MAE**, **RMSE**, **R²**, and **cross-validation**.

To ensure explainability, we will apply **SHAP** to interpret model predictions and analyze both global and local feature importance. We will also critically reflect on:

- **Model limitations**
- **Potential bias** (e.g., location-related effects)
- **Data quality issues**
- **Our use of AI tools throughout the project**

### Potential Data Sources
- **Airbnb Open Data (Kaggle)**
- **Airbnb Prices in European Cities (Kaggle)**
- **Additional Airbnb datasets from Kaggle (if needed)**

### Project Structure
- `data/raw/`: original dataset files
- `data/processed/`: processed outputs (e.g. combined dataset)
- `notebooks/`: exploratory notebooks
- `src/data_loader.py`: data ingestion, loading, and combining
- `src/preprocessing.py`: preprocessing functions
- `src/train.py`: model training entry point (placeholder)
- `src/evaluate.py`: model evaluation entry point (placeholder)

### Quick Start
1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Combine all city CSV files from `data/raw`:
   - `python data_loader.py --combine-all --raw-dir data/raw --output-path data/processed/airbnb_combined.csv`
3. Run first EDA in notebook:
   - Open `notebooks/01_eda.ipynb`
