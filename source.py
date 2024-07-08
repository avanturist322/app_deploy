import streamlit as st
import pandas as pd

allowed_extensions = ['csv', 'xlsx']

def load_file(file: bytes, filetype: str) -> pd.DataFrame:
    """
    Загружает файл и возвращает DataFrame.
    
    Parameters:
    file (bytes): Загруженный файл.
    filetype (str): Расширение файла ('csv' или 'xlsx').
    
    Returns:
    pd.DataFrame: DataFrame с данными из файла.
    
    Raises:
    Exception: Если расширение файла не соответствует допустимым форматам.
    """
    if filetype in allowed_extensions:
        if filetype == 'csv':
            df = pd.read_csv(file)
        elif filetype == 'xlsx':
            df = pd.read_excel(file)
        return df
    else:
        raise Exception("Неподходящее расширение файла. Пожалуйста, загрузите файл в формате .csv или .xlsx.")

def load_file_to_st(uploaded_file: bytes, warning_name: str) -> pd.DataFrame:
    """
    Загружает файл в Streamlit и выводит данные DataFrame.
    
    Parameters:
    uploaded_file (file): Загруженный файл через streamlit.file_uploader.
    
    Returns:
    pd.DataFrame: DataFrame с данными из загруженного файла.
    """
    if uploaded_file is not None:
        if hasattr(uploaded_file, 'name'):
            file_type = uploaded_file.name.split('.')[-1]
            try:
                df = load_file(uploaded_file, file_type)
                # st.write(f"Размер файла: {df.shape} \n\n Колонки: {', '.join(df.columns.values.tolist())}")
                # st.write(df.head())
                
                return df
            
            except Exception as e:
                st.write(e)
        else:
            st.write("Неподходящее расширение файла. Пожалуйста, загрузите файл в формате .csv или .xlsx.")
    else:
        st.error(f"Пожалуйста, загрузите файл {warning_name}.")


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def get_preprocessed_data(X_train_orig, X_test_orig, ALL_GIS, mode='ss', do_ohe=True,
                          lith_name='Код Prime', mode_pred='tc_par', feature_names=None, tc_par_name='TC_par_ups'):
    """_summary_

    Args:
        X_train_orig (_type_): _description_
        X_test_orig (_type_): _description_
        ALL_GIS (_type_): _description_
        mode (str, optional): _description_. Defaults to 'ss'.

    Returns:
        _type_: _description_
    """
    # One-hot encoding of categorial featurs ('Код Prime')
    X_train_one_hot_encoded = pd.get_dummies(X_train_orig[lith_name], prefix=lith_name, dtype=float)
    X_train = pd.concat([X_train_orig.drop('Код Prime', axis=1), X_train_one_hot_encoded], axis=1)

    X_test_one_hot_encoded = pd.get_dummies(X_test_orig[lith_name], prefix=lith_name, dtype=float)
    X_test = pd.concat([X_test_orig.drop(lith_name, axis=1), X_test_one_hot_encoded], axis=1)

    all_gis_one_hot_encoded = pd.get_dummies(ALL_GIS[lith_name], prefix=lith_name, dtype=float)
    ALL_GIS_encoded = pd.concat([ALL_GIS.drop(lith_name, axis=1), all_gis_one_hot_encoded], axis=1)

    # Numerica features selection
    ALL_GIS_encoded = ALL_GIS_encoded.reindex(columns=X_train.columns, fill_value=0)
    # if 'TC_par_ups' in X_train_orig.columns.values:
    #     numeric_features = ['TC_par_ups', 'AK', 'BKS', 'DS_DN','GGP', 'GKS', 'NGKS', 'K', 'Th', 'U']
    # elif 'TC_par_pred' in X_train_orig.columns.values:
    #     numeric_features = ['TC_par_pred', 'AK', 'BKS', 'DS_DN','GGP', 'GKS', 'NGKS', 'K', 'Th', 'U']
    # else:
    #     numeric_features = ['AK', 'BKS', 'DS_DN','GGP', 'GKS', 'NGKS', 'K', 'Th', 'U']

    if mode_pred in ['tc_par', 'vhc']:
        numeric_features = feature_names
    elif mode_pred in ['anisotropy']:
        numeric_features = feature_names + [tc_par_name]

    ohe_cat_features = [item for item in X_train.columns.values.tolist() if item not in numeric_features]

    X_train_numeric = X_train[numeric_features]
    X_test_numeric = X_test[numeric_features]
    ALL_GIS_numeric = ALL_GIS_encoded[numeric_features]

    # Scaler creation
    if mode == 'ss':
        scaler = StandardScaler()
    elif mode == 'mms':
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)
    ALL_GIS_scaled = scaler.transform(ALL_GIS_numeric)

    # Create DataFrame for scaled numeric attributes
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train_numeric.columns)
    ALL_GIS_scaled_df = pd.DataFrame(ALL_GIS_scaled, columns=X_train_numeric.columns)

    X_train_categorical = X_train[ohe_cat_features]
    X_test_categorical = X_test[ohe_cat_features]
    ALL_GIS_categorical = ALL_GIS_encoded[ohe_cat_features]

    # Transferring cat dataframe indexes to numerical dataframe
    X_train_scaled_df.set_index(X_train_categorical.index, inplace=True)
    X_test_scaled_df.set_index(X_test_categorical.index, inplace=True)
    ALL_GIS_scaled_df.set_index(ALL_GIS_categorical.index, inplace=True)

    # Combining scaled numeric and categorical features
    if do_ohe:
        X_train_combined = pd.concat([X_train_scaled_df, X_train_categorical], axis=1)
        X_test_combined = pd.concat([X_test_scaled_df, X_test_categorical], axis=1)
        ALL_GIS_combined = pd.concat([ALL_GIS_scaled_df, ALL_GIS_categorical], axis=1)
    else:
        X_train_combined = pd.concat([X_train_scaled_df, X_train_orig[lith_name].astype(str)], axis=1)
        X_test_combined = pd.concat([X_test_scaled_df, X_test_orig[lith_name].astype(str)], axis=1)
        ALL_GIS_combined = pd.concat([ALL_GIS_scaled_df, ALL_GIS[lith_name].astype(str)], axis=1)

    return X_train_combined, X_test_combined, ALL_GIS_combined, scaler


import numpy as np

class Metrics:
    def __init__(self):
        pass
    
    def mse(self, y_true, y_pred):
        """
        Mean Squared Error
        """
        return np.mean((y_true - y_pred)**2)

    def rmse(self, y_true, y_pred):
        """
        Root Mean Squared Error
        """
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def mae(self, y_true, y_pred):
        """
        Mean Absolute Error
        """
        return np.mean(np.abs(y_true - y_pred))

    def mad(self, y_true, y_pred):
        """
        Mean Absolute Deviation
        """
        return np.median(np.abs(y_true - y_pred))

    def r_squared(self, y_true, y_pred):
        """
        R-squared
        """
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean)**2)
        ss_res = np.sum((y_true - y_pred)**2)
        return 1 - (ss_res / ss_tot)
    


metrics = Metrics()
def get_metrics(y_test, y_pred):
    mse = metrics.mse(y_test, y_pred)
    rmse = metrics.rmse(y_test, y_pred)
    mae = metrics.mae(y_test, y_pred)
    mad = metrics.mad(y_test, y_pred)
    r_squared = metrics.r_squared(y_test, y_pred)

    data_met = {
        'MSE': [mse], 'RMSE': [rmse],
        'MAE': [mae], 'MAD': [mad],
        'R^2': [r_squared]
    }
    df = pd.DataFrame(data_met)

    st.write(df)



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_predict
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor






def metaregressor(base_clfs, final_classifier, X_train, X_test, y_train, cv):
    """
    Meta classifier prediction using stacking. 
    Input:
    :param base_clfs: list,  base classifiers which will be stacked together.
    :param final_classifier: estimator, a classifier which will be used to combine the base estimators. 
    :param X_train: numpy array or pandas table, train set.
    :param X_test: numpy array or pandas table, target for train set.
    :param X_train: numpy array or pandas table, test set.
    :param cv: number of cross-validation folds.
    
    Output:
    :param y_pred: numpy array or pandas table, prediction of meta classifier using stacking on test set.
    :param final_classifier(optional): estimator, trained final_calssifier.
    
    More details https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
    
    """
    ### BEGIN Solution (do not delete this comment)

    # Initializing of empty arrays to store scores
    length_tr = X_train.shape[0]
    length_te = X_test.shape[0]
    width = len(base_clfs)
    train_score_mem = np.zeros([length_tr, width])
    test_score_mem = np.zeros([length_te, width])

    # Filling the empty arrays with scores
    i = 0
    for model in base_clfs:
      train_score_mem[:, i] = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs = -1)
      model.fit(X_train, y_train)
      test_score_mem[:, i] = model.predict(X_test)
      i += 1

    # Training the final classifier on scores from X_train
    final_classifier.fit(train_score_mem, y_train)
    y_pred = final_classifier.predict(test_score_mem)
    
    return y_pred, final_classifier