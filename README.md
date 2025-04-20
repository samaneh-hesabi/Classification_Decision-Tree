<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Titanic Dataset Classification with Decision Tree</div>

# 1. Project Overview
This project implements a machine learning solution for predicting survival on the Titanic using a Decision Tree classifier. The project includes comprehensive data analysis, preprocessing, and model evaluation. The goal is to predict whether a passenger survived the Titanic disaster based on various features.

# 2. Dataset Description
The Titanic dataset contains information about 891 passengers who were on board the Titanic. The dataset includes both demographic and travel-related information.

## 2.1 Dataset Features
| Feature | Description | Type | Notes |
|---------|-------------|------|-------|
| PassengerId | Unique identifier for each passenger | Integer | Dropped during preprocessing |
| Survived | Survival status (0 = No, 1 = Yes) | Binary | Target variable |
| Pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | Integer | Socio-economic status |
| Name | Passenger name | String | Used to extract titles |
| Sex | Gender | Categorical | Converted to binary |
| Age | Age in years | Float | Missing values filled with median |
| SibSp | Number of siblings/spouses aboard | Integer | Used to create family features |
| Parch | Number of parents/children aboard | Integer | Used to create family features |
| Ticket | Ticket number | String | Dropped during preprocessing |
| Fare | Passenger fare | Float | Scaled during preprocessing |
| Cabin | Cabin number | String | Dropped due to many missing values |
| Embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) | Categorical | Converted to numerical |

## 2.2 Derived Features
| Feature | Description | Type | Creation Method |
|---------|-------------|------|----------------|
| FamilySize | Total family members aboard | Integer | SibSp + Parch + 1 |
| IsAlone | Whether passenger is traveling alone | Binary | 1 if FamilySize = 1, else 0 |
| Title | Passenger's title (Mr, Mrs, Miss, etc.) | Categorical | Extracted from Name |

# 3. Project Structure
- `src/`: Contains all Python source files
  - `titanic_analysis.py`: Main data preprocessing and analysis script
  - `titanic_eda.py`: Exploratory data analysis script
  - `ddd.py`: Decision tree implementation and evaluation
- `notebooks/`: Jupyter notebooks for interactive analysis
- `data/`: Directory for storing datasets
- `results/`: Contains output visualizations and model results
- `requirements.txt`: Project dependencies
- `environment.yml`: Conda environment configuration

# 4. Data Preprocessing Steps
1. Handle missing values:
   - Age: Filled with median
   - Embarked: Filled with mode
   - Cabin: Dropped due to high missing values
2. Feature engineering:
   - Created FamilySize and IsAlone features
   - Extracted Title from Name
3. Data transformation:
   - Converted categorical variables to numerical
   - Scaled numerical features
   - Dropped unnecessary columns

# 5. Installation
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

# 6. Usage
1. Run the data preprocessing and analysis:
   ```bash
   python src/titanic_analysis.py
   ```
2. For interactive analysis, open the Jupyter notebooks in the `notebooks/` directory

# 7. Results
The project generates several visualizations:
- `decision_tree.png`: Visualization of the trained decision tree
- `confusion_matrix.png`: Model performance metrics
- `titanic_eda_plots.png`: Exploratory data analysis visualizations

# 8. Dependencies
- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter

# 9. License
This project is licensed under the MIT License - see the LICENSE file for details.
