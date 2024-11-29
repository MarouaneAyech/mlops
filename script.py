# %%
## -- main lib
import pandas as pd
from joblib import dump
import os


## -- sklearn data processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
## -- sklearn modeling
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
## -- sklearn metrics
from sklearn.metrics import r2_score

# %%
def impute_outliers(data, low, high, replacement):
    return data.apply(lambda x : replacement if x<low or x>high else x).values.reshape(-1,1)

# %%
outlier_imputer_num = ColumnTransformer(
    transformers=[
            ('outlier_replacer_size_m2', 
            FunctionTransformer(impute_outliers, kw_args={'low': 10, 'high': 500, 'replacement': 140}),
            'size_m2'),
            ('outlier_replacer_num_bedrooms',
            FunctionTransformer(impute_outliers, kw_args={'low': 0, 'high': 10, 'replacement': 3}),
	        'num_bedrooms'),
            ('outlier_replacer_num_bathrooms', 
            FunctionTransformer(impute_outliers, kw_args={'low': 0, 'high': 5, 'replacement': 2}),
            'num_bathrooms')
    ],
    # Garde les colonnes restantes telles quelles
    remainder='passthrough'
)
outlier_imputer_num

# %%
# Définir une Fonction personnalisée pour extraire l'année à partir de la date
def extract_year(data):
    # Convertir le type des valeurs de la colonne data en type datetime
    data = pd.to_datetime(data)
    # Extraire l'année de la date
    annee = data.dt.year
    # Transformer la colonne annee (de type Series) en un array numpy 2D
    return annee.values.reshape(-1, 1) 

# %%
# Instancier le transformateur personnalisé
year_extractor = FunctionTransformer(extract_year)
year_extractor

# %%
def to_dataframe_with_columns(data, columns):
    # Retourner un dataframe
    return pd.DataFrame(data, columns=columns)

# %%
cols_num = ['size_m2', 'num_bedrooms', 'num_bathrooms', 'distance_school','public_transport_access', 'property_tax']
preprocessor_num = Pipeline(
    steps=[
        ('nan_imputer_num', SimpleImputer(strategy='mean')), # Imputation des NaN
        ('to_dataframe', FunctionTransformer(to_dataframe_with_columns, kw_args={'columns': cols_num})),
        ('outlier_imputer_num', outlier_imputer_num)
])


# %%
preprocessor = ColumnTransformer(
    transformers=[
            # Imputation pour les colonnes numériques
            ('num', preprocessor_num, cols_num), 
            # Encodage pour les colonnes catégorielles
            ('cat', OneHotEncoder(), ['city']),
             ('date', year_extractor, 'date_built')
    ]
)
preprocessor

# %%
pipeline = Pipeline(steps=[
    # Imputation des valeurs manquantes
    ('preprocessor', preprocessor),
    # Normalisation des données
    ('scaler', StandardScaler()),
    # reduction dim
    ('pca', PCA()),
    # Modèle de régression
    ('regressor', LinearRegression())
])
pipeline

# %%
df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'data.csv'))

X = df.drop('price',axis=1)
y = df['price']

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=45)

# %%
param_grid = {
    'pca__n_components': [2, 3 , 4, 5, 6],
}

# %%
grid_search = GridSearchCV( 
	estimator = pipeline, 	
	param_grid = param_grid, 
	scoring = 'r2', 
	cv = 5
)
grid_search

# %%
# Entraîner le pipeline
grid_search.fit(X_train, y_train)

# %%
best_model = grid_search.best_estimator_
best_model.score(X_test, y_test)

# %%
print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score R² :", grid_search.best_score_)

# %%
cv_results = pd.DataFrame(grid_search.cv_results_).sort_values(by='rank_test_score')
print(cv_results[['params', 'mean_test_score', 'std_test_score','rank_test_score']])

# %%
train_score = grid_search.score(X_train, y_train)
print("Score R² sur train :", train_score)
test_score = grid_search.score(X_test, y_test)
print("Score R² sur test :", test_score)

# %%
# save model
dump(best_model, os.path.join(os.getcwd(), 'models', "best_pipeline.pkl"))


# %%
# save metrics
with open('metrics.txt','w') as f :
    f.write(f"Score R2 sur train : {train_score:.2f}\n")
    f.write(f"Score R2 sur test : {test_score:.2f}\n")



