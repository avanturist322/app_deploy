import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy

import source


#################################################################################################################
def run():
    random_state = 322
    metrics = source.Metrics()

    st.set_page_config(
        page_title="Прогноз тепловых свойств горных пород по данным ГИС",
        page_icon="🖥",
    )

    st.title("Прогноз тепловых свойств горных пород по данным ГИС")

    st.write(r"Данное приложение предназначено для осуществления прогноза тепловых свойств горных пород \
             ($\lambda_{\parallel}, \lambda_{\bot}, C_p$) по данным геофизических исследований скважин (ГИС).")

    st.sidebar.success("В данном разделе осуществляется загрузка входных данных, \
                     их автоматическая предобработка и осуществление прогноза \
                     с использованием различных алгоритмов ML с управляемой \
                     настройкой основных гиперпараметров.")
    st.sidebar.info(r"**Важно:** для осуществления прогноза $\lambda_{\bot}$ или $K$ сначала выполните прогноз $\lambda_{\parallel}$ и сохраните результат, а потом загрузите его в соответствующем окне.")


    st.write("### 1. Загрузка данных")
    st.write("Загрузите входные данные в формате `.csv` или `.xlsx`.")
    st.info("Вам не нужно упорядочивать колонки в таблицах / \
            использовать названия как в примерах - при выполнении соответствующего \
            шага будет доступна возможность указать название \
            колонки с прогнозируемой величиной, глубиной или типом пород. \
            **Важно**: названия колонок с глубиной и типами пород \
            в таблицах в разделах 1.1. и 1.2. должны быть одинаковы.")
    st.write("#### 1.1. Данные ГИС")
    st.info("Загрузите таблицу, содержазую колонку с глубиной, колонки с \
            данными ГИС и (опционально) колонку с типами пород. Таблица \
            должна быть отсортирована по возрастанию глубины с приведенными \
            к одной глубине данными ГИС.")
    
    st.info(r"**Пример конфигурации таблицы** \
            Глубина | ГИС-1 | ГИС-2 | ... | ГИС-n| Тип пород (опционально)")
    
    uploaded_gis = st.file_uploader("TC_par", 
                                     type=['csv', 'xlsx'], 
                                     label_visibility='collapsed')
    ALL_GIS = source.load_file_to_st(uploaded_gis)

    st.write("#### 1.2. Результаты измерений на керне")
    st.info("Загрузите таблицу, содержащую колонку с глубиной, колонки с \
            предобработанными (скорректированными на пластовые условия \
            / upscaled) значениями прогнозируемых величин.")
    st.info(r"**Примеры минимальных конфигураций таблицы** \
            1. Для прогноза $\lambda_{\parallel}$ или $C_p$: \
            Глубина | $\lambda_{\parallel}$ / $C_p$| Тип пород (опционально) \
            2. Для прогноза $\lambda_{\bot}$ или $K$: \
            Глубина | $\lambda_{\parallel}$ | $K$| Тип пород (опционально)")

    uploaded_data = st.file_uploader("data", 
                                     type=['csv', 'xlsx'], 
                                     label_visibility='collapsed')
    data = source.load_file_to_st(uploaded_data)
    
    use_lith = st.checkbox('Использовать типы пород при прогнозе?', value=False)
    lith_name = st.text_input("Введите название колонки с типами пород, если она есть в данных:", "Код Prime")
    if use_lith:
        st.info("Будут использованы типы пород.")
        # lith_name = st.text_input("Введите название колонки с типами пород:", "Код Prime")
        st.write(f"Выбрана колонка {lith_name} с типами пород.")

        assert lith_name in ALL_GIS.columns.values, 'В данных нет такой колонки с типами пород'

        fig, ax = plt.subplots(1, 2, figsize=(12,5))
        
        ax[0].hist(data.sort_values(by=lith_name)[lith_name].astype(int).astype(str), bins=20)
        ax[1].hist(ALL_GIS.sort_values(by=lith_name)[lith_name].astype(int).astype(str), bins=20)
        
        ax[0].xaxis.set_tick_params(rotation=45)
        ax[1].xaxis.set_tick_params(rotation=45)

        st.pyplot(fig)

    else:
        st.info("Типы пород не будут использованы.")


    # * Удаление отсутствующих в обучающей выборке типов пород

    if use_lith:
        shape_before = ALL_GIS.shape
        ALL_GIS = ALL_GIS[ALL_GIS[lith_name].isin(data[lith_name])].reset_index(drop=True)
        shape_after = ALL_GIS.shape
        st.write(f"Размер таблицы с данными ГИС до исключения не представленных \
                 образцами керна типов пород:")
        st.write(shape_before)
        st.write(f"Размер после исключения:")
        st.write(shape_after)

        fig, ax = plt.subplots(1, 2, figsize=(12,5))
        
        ax[0].hist(data.sort_values(by=lith_name)[lith_name].astype(int).astype(str), bins=20)
        ax[1].hist(ALL_GIS.sort_values(by=lith_name)[lith_name].astype(int).astype(str), bins=20)
        
        ax[0].xaxis.set_tick_params(rotation=45)
        ax[1].xaxis.set_tick_params(rotation=45)

        st.pyplot(fig)



    # * Разбиение на обучающую и тестовую выборки
    depth_name = st.text_input("Введите название колонки с глубиной:", "DEPT")
    st.write(f"Выбрана колонка {depth_name} с глубинами.")

    from sklearn.model_selection import train_test_split
    import pandas as pd

    feature_names = ALL_GIS.columns.values.tolist()
    feature_names.remove(depth_name)

    if not use_lith:# and lith_name is not None:
        feature_names.remove(lith_name)
    
    st.write("Признаки:")
    st.write(feature_names)


    # ? На этом этапе есть названия колонок с признаками с или без литотипов
    all_gis = ALL_GIS.copy()
    if lith_name in ALL_GIS.columns.values: 
        all_gis = all_gis.drop(lith_name, axis=1)
    
    final_list = {}
    for col_name in all_gis.columns.values:
        if col_name == depth_name:
            continue
        else:
            final_list[col_name] = all_gis[[depth_name, col_name]]


    df_new4 = data.copy()
    min_depth, max_depth = ALL_GIS[depth_name].min(), ALL_GIS[depth_name].max()
    df_new4 = df_new4[(df_new4[depth_name] >= min_depth) \
                        & (df_new4[depth_name] <= max_depth)].reset_index(drop=True)
    
    for key in tqdm(final_list.keys()):
        selected_b = final_list[key][(final_list[key]['DEPT'] >= data['DEPT'].min()) & (final_list[key]['DEPT'] <= data['DEPT'].max())]
        new_y2 = np.interp(data['DEPT'], selected_b['DEPT'], selected_b[final_list[key].columns.values[1]])
        result = pd.DataFrame({final_list[key].columns.values[1]: new_y2})
        df_new4 = pd.concat([df_new4, result], axis=1)

    data_to_pred = df_new4.copy()
        
    st.success("Результат интерполяции данных ГИС по глубинам керновых данных")
    st.write(data_to_pred)

    # !  ===================================  ПРОГНОЗ ТЕПЛОВЫХ СВОЙСТВ ==========================================
    st.write("### 2. Прогноз тепловых свойств")
    st.write("Что предсказываем?")

    what_to_predict = st.selectbox("Предсказывать: ", 
                                    ("TC_par", 
                                    "TC_per",
                                    "VHC"))
    
    st.info(f"Конфигурация прогноза: {what_to_predict}")

    
    # !  ................................  ПРОГНОЗ TC_par | VHC ........................................

    # go = st.checkbox("Едем дальше?", True)
    # if go or not go:
    #     st.write(1)

    if what_to_predict == 'TC_par':
        tc_par_name = st.text_input(r"Введите название колонки с $\lambda_{\parallel}$:", "TC_par_ups")
        st.info(f"Прогнозируемая величина: {tc_par_name}")
    elif what_to_predict == 'TC_per':
        anisotropy_name = st.text_input(r"Введите название колонки с $K = \frac{\lambda_{\parallel}}{\lambda_{\bot}}$:", "Anisotropy_ups")
        tc_par_name = st.text_input(r"Введите название колонки с $\lambda_{\parallel}$:", "TC_par_ups")
        tc_per_name = st.text_input(r"Введите название колонки с $\lambda_{\bot}$:", "TC_per_ups")
        st.info(rf"Прогнозируемая величина: {anisotropy_name}, последующий перерасчет в TC_per")
    elif what_to_predict == 'VHC':
        vhc_name = st.text_input(r"Введите название колонки с $C_p$:", "VHC_ups")
        st.info(f"Прогнозируемая величина: {vhc_name}")
        

    
    if what_to_predict == 'TC_par':
        data_to_pred = data_to_pred[[depth_name] + feature_names + [tc_par_name]].dropna().reset_index(drop=True)
        if use_lith:
            unique_value_counts = data_to_pred[lith_name].value_counts()
            need_to_replicate = unique_value_counts[unique_value_counts<=2].index.values
            for val in need_to_replicate:
                data_to_pred = pd.concat((data_to_pred, data_to_pred[data_to_pred[lith_name] == val]), axis=0).reset_index(drop=True)
        y = data_to_pred[tc_par_name]
        mode_pred = 'tc_par'
        X = data_to_pred[feature_names]

    elif what_to_predict == 'TC_per':
        data_to_tc_par = data_to_pred[[depth_name] + feature_names + [tc_par_name]].dropna().reset_index(drop=True)
        if use_lith:
            unique_value_counts = data_to_tc_par[lith_name].value_counts()
            need_to_replicate = unique_value_counts[unique_value_counts<=2].index.values
            for val in need_to_replicate:
                data_to_tc_par = pd.concat((data_to_tc_par, data_to_tc_par[data_to_tc_par[lith_name] == val]), axis=0).reset_index(drop=True)
        y_to_tc_par = data_to_tc_par[tc_par_name]
        X_to_tc_par = data_to_tc_par[feature_names]
        Dept_to_tc_par = data_to_tc_par[depth_name]

        data_to_pred = data_to_pred[[depth_name] + feature_names + [anisotropy_name] + [tc_par_name] + [tc_per_name]].dropna().reset_index(drop=True)
        if use_lith:
            unique_value_counts = data_to_pred[lith_name].value_counts()
            need_to_replicate = unique_value_counts[unique_value_counts<=2].index.values
            for val in need_to_replicate:
                data_to_pred = pd.concat((data_to_pred, data_to_pred[data_to_pred[lith_name] == val]), axis=0).reset_index(drop=True)
        y = data_to_pred[anisotropy_name]
        mode_pred = 'anisotropy'
        X = data_to_pred[feature_names + [tc_par_name]]

    elif what_to_predict == 'VHC':
        data_to_pred = data_to_pred[[depth_name] + feature_names + [vhc_name]].dropna().reset_index(drop=True)
        if use_lith:
            unique_value_counts = data_to_pred[lith_name].value_counts()
            need_to_replicate = unique_value_counts[unique_value_counts<=2].index.values
            for val in need_to_replicate:
                data_to_pred = pd.concat((data_to_pred, data_to_pred[data_to_pred[lith_name] == val]), axis=0).reset_index(drop=True)
        y = data_to_pred[vhc_name]
        mode_pred = 'vhc'
        X = data_to_pred[feature_names]


        
    Dept = data_to_pred[depth_name]

    def df_to_csv(df, filename, button_text, add_key=''):
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')


        csv = convert_df(df)

        st.download_button(f"{button_text}", # label
                            csv, # data
                            f"{filename}.csv", # filename
                            "text/csv",
                            key='download-csv'+add_key)
    

    if use_lith:
        X_train_orig, X_test_orig, \
            y_train_tc_par, y_test_tc_par, \
            Dept_train_tc_par, Dept_test_tc_par = train_test_split(X, y, Dept, test_size=0.3, 
                                                            random_state=random_state, 
                                                            shuffle=True, stratify=X[lith_name])

        feature_names2 = copy.deepcopy(feature_names)
        feature_names2.remove(lith_name)

        X_train_combined, X_test_combined, \
            ALL_GIS_combined, scaler = source.get_preprocessed_data(X_train_orig, X_test_orig, 
                                                                    ALL_GIS, mode='ss', do_ohe=True,
                                                                    lith_name=lith_name, mode_pred=mode_pred,
                                                                    feature_names=feature_names2, 
                                                                    tc_par_name=tc_par_name if what_to_predict != 'VHC' else '')
    else:
        X_train_orig, X_test_orig, \
        y_train_tc_par, y_test_tc_par, \
        Dept_train_tc_par, Dept_test_tc_par = train_test_split(X, y, Dept, test_size=0.3, 
                                                        random_state=random_state, 
                                                        shuffle=True)#, stratify=X[lith_name])

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train_orig)
        X_train_combined = scaler.transform(X_train_orig)
        X_test_combined = scaler.transform(X_test_orig)
        
        try:
            ALL_GIS_combined = scaler.transform(ALL_GIS.drop(columns=[depth_name, lith_name]))
        except:
            st.write(r"Для получения прогноза $\lambda_{\bot}$ на весь интервал, загрузите полученный ранее прогноз $\lambda_{\parallel}$ по всему интервалу:")
            uploaded_tc_par_pred = st.file_uploader("TC_par_pred", 
                                                    type=['csv', 'xlsx'], 
                                                    label_visibility='collapsed',
                                                    key='0')
            pred_all_tc_par = source.load_file_to_st(uploaded_tc_par_pred)

            
            ALL_GIS_combined = scaler.transform(pd.concat((ALL_GIS, pred_all_tc_par.iloc[:,-1]), axis=1).rename(columns={'TC_par_pred': tc_par_name}).drop(columns=[depth_name, lith_name]))



    # st.write(X_train_combined.shape, X_test_combined.shape, ALL_GIS.shape, ALL_GIS_combined.shape)

    # ? PREDICTOR
    model_mode = st.selectbox("Выберите тип модели:", 
                ("Linear Regression",
                    "Decision Tree",
                    "Gradient Boosting", 
                    "XGBoost",
                    "CatBoost",
                    "Stacking"))
    
    st.info(f"Выбранная модель: {model_mode}")

    def predictor(pred_name):
        def display_title_metrics(pred_name):
            if pred_name == 'TC_par':
                st.write(r"Метрики для $\lambda_{\parallel}$")
            elif pred_name == 'TC_per':
                st.write(r"Метрики для $\lambda_{\bot}$")
            elif pred_name == 'Anisotropy':
                st.write(r"Метрики для $K$")
            elif pred_name == 'VHC':
                st.write(r"Метрики для $C_p$")
            elif pred_name is None:
                pass

        if model_mode == "Linear Regression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train_combined, y_train_tc_par)
            y_pred_tc_par = model.predict(X_test_combined)
            pred_all_tc_par = model.predict(ALL_GIS_combined)
            if pred_name is not None:
                display_title_metrics(pred_name)
                source.get_metrics(y_test_tc_par, y_pred_tc_par)

        elif model_mode == "Decision Tree":
            from sklearn.tree import DecisionTreeRegressor

            params = {'max_depth': None, # None or int
                    'min_samples_split': 2,
                    'min_samples_leaf': 1}

            # Create sliders for non-None parameters
            for param, default in params.items():
                if param != 'max_depth':
                    if isinstance(default, int):
                        value = st.slider(f"{param}", 1, 20, default, key=f"dt_solo_{param}")
                    else:
                        value = st.selectbox(f"{param}", [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key=f"dt_solo_{param}")
                else:
                    # value = st.selectbox(f"{param}", [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                    value = st.slider(f"{param}", None, 20, default, key=f"dt_solo_{param}")
                    if value == 0:
                        value = None

                params[param] = value

            # Display selected parameters
            st.write("Selected Parameters:")
            st.write(params)
            params['random_state'] = 42

            model = DecisionTreeRegressor(**params)
            model.fit(X_train_combined, y_train_tc_par)
            y_pred_tc_par = model.predict(X_test_combined)
            pred_all_tc_par = model.predict(ALL_GIS_combined)
            if pred_name is not None:
                display_title_metrics(pred_name)
                source.get_metrics(y_test_tc_par, y_pred_tc_par)          

        elif model_mode == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingRegressor

            params = {'max_depth': 3, # None or int
                    'learning_rate': 0.01,
                    'n_estimators': 200}

            # Create sliders for non-None parameters
            for param, default in params.items():
                if param == 'max_depth':
                    value = st.slider(f"{param}", None, 20, default, step=1, key=f"gb_solo_{param}")
                    if value == 0:
                        value = None
                elif param == 'learning_rate':
                    value = st.slider(f"{param}", 0.01, 1.0, default, step=0.01, key=f"gb_solo_{param}")
                elif param == 'n_estimators':
                    value = st.slider(f"{param}", 0, 1000, default, step=10, key=f"gb_solo_{param}")
                    if value == 0:
                        value = 1

                params[param] = value

            # Display selected parameters
            st.write("Selected Parameters:")
            st.write(params)
            params['random_state'] = 42

            model = GradientBoostingRegressor(**params)
            model.fit(X_train_combined, y_train_tc_par)
            y_pred_tc_par = model.predict(X_test_combined)
            pred_all_tc_par = model.predict(ALL_GIS_combined)
            if pred_name is not None:
                display_title_metrics(pred_name)
                source.get_metrics(y_test_tc_par, y_pred_tc_par)

        elif model_mode == "XGBoost":
            import xgboost as xgb

            params = {'max_depth': 7, # None or int
                    'learning_rate': 0.5,
                    'n_estimators': 200}

            # Create sliders for non-None parameters
            for param, default in params.items():
                if param == 'max_depth':
                    value = st.slider(f"{param}", None, 20, default, step=1, key=f"xgb_solo_{param}")
                    if value == 0:
                        value = None
                elif param == 'learning_rate':
                    value = st.slider(f"{param}", 0.01, 1.0, default, step=0.01, key=f"xgb_solo_{param}")
                elif param == 'n_estimators':
                    value = st.slider(f"{param}", 0, 1000, default, step=10, key=f"xgb_solo_{param}")
                    if value == 0:
                        value = 1

                params[param] = value

            # Display selected parameters
            st.write("Selected Parameters:")
            st.write(params)
            params['random_state'] = 42

            model = xgb.XGBRegressor(**params)
            model.fit(X_train_combined, y_train_tc_par)
            y_pred_tc_par = model.predict(X_test_combined)
            pred_all_tc_par = model.predict(ALL_GIS_combined)
            if pred_name is not None:
                display_title_metrics(pred_name)
                source.get_metrics(y_test_tc_par, y_pred_tc_par) 

        elif model_mode == "CatBoost":
            from catboost import CatBoostRegressor

            params = {'depth': 7, # None or int
                    'learning_rate': 0.5,
                    'l2_leaf_reg': 5,
                    'iterations': 700}

            # Create sliders for non-None parameters
            for param, default in params.items():
                if param == 'depth':
                    value = st.slider(f"{param}", None, 20, default, step=1)
                    if value == 0:
                        value = None
                elif param == 'learning_rate':
                    value = st.slider(f"{param}", 0.01, 1.0, default, step=0.01)
                elif param == 'l2_leaf_reg':
                    value = st.slider(f"{param}", 0, 10, default, step=1)
                elif param == 'iterations':
                    value = st.slider(f"{param}", 0, 1000, default, step=10)
                    if value == 0:
                        value = 1

                params[param] = value

            # Display selected parameters
            st.write("Selected Parameters:")
            st.write(params)
            params['random_state'] = 42
            params['task_type'] = 'CPU'
            params['silent'] = True

            model = CatBoostRegressor(**params)
            model.fit(X_train_combined, y_train_tc_par)
            y_pred_tc_par = model.predict(X_test_combined)
            pred_all_tc_par = model.predict(ALL_GIS_combined)
            if pred_name is not None:
                display_title_metrics(pred_name)
                source.get_metrics(y_test_tc_par, y_pred_tc_par)

        elif model_mode == "Stacking":
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import GradientBoostingRegressor
            import xgboost as xgb
            from catboost import CatBoostRegressor

            params_gb = {'max_depth': 3, # None or int
                    'learning_rate': 0.01,
                    'n_estimators': 200}
            
            params_xgb = {'depth': 7, # None or int
                    'learning_rate': 0.5,
                    'n_estimators': 200}

            params_cb = {'depth': 7, # None or int
                    'learning_rate': 0.5,
                    'l2_leaf_reg': 5,
                    'iterations': 700}


            st.write("Gradient Boosting")
            for param, default in params_gb.items():
                if param == 'max_depth':
                    value = st.slider(f"{param}", None, 20, default, step=1, key=f"gb_{param}")
                    if value == 0:
                        value = None
                elif param == 'learning_rate':
                    value = st.slider(f"{param}", 0.01, 1.0, default, step=0.01, key=f"gb_{param}")
                elif param == 'n_estimators':
                    value = st.slider(f"{param}", 0, 1000, default, step=10, key=f"gb_{param}")
                    if value == 0:
                        value = 1
                params_gb[param] = value

            st.write("XGBoost")
            for param, default in params_xgb.items():
                if param == 'depth':
                    value = st.slider(f"{param}", None, 20, default, step=1, key=f"xgb_{param}")
                    if value == 0:
                        value = None
                elif param == 'learning_rate':
                    value = st.slider(f"{param}", 0.01, 1.0, default, step=0.01, key=f"xgb_{param}")
                elif param == 'n_estimators':
                    value = st.slider(f"{param}", 0, 1000, default, step=10, key=f"xgb_{param}")
                    if value == 0:
                        value = 1
                params_xgb[param] = value

            st.write("CatBoost")
            for param, default in params_cb.items():
                if param == 'depth':
                    value = st.slider(f"{param}", None, 20, default, step=1, key=f"cb_{param}")
                    if value == 0:
                        value = None
                elif param == 'learning_rate':
                    value = st.slider(f"{param}", 0.01, 1.0, default, step=0.01, key=f"cb_{param}")
                elif param == 'l2_leaf_reg':
                    value = st.slider(f"{param}", 0, 10, default, step=1, key=f"cb_{param}")
                elif param == 'iterations':
                    value = st.slider(f"{param}", 0, 1000, default, step=10, key=f"cb_{param}")
                    if value == 0:
                        value = 1
                params_cb[param] = value

            # Display selected parameters
            st.write("Selected Parameters:")
            st.write("Gradient Boosting")
            st.write(params_gb)
            st.write("XGBoost")
            st.write(params_xgb)
            st.write("CatBoost")
            st.write(params_cb)

            params_gb['random_state'] = 42
            params_xgb['random_state'] = 42
            params_cb['random_state'] = 42

            params_cb['task_type'] = 'CPU'
            params_cb['silent'] = True

            model_gb = GradientBoostingRegressor(**params_gb)
            model_xgb = xgb.XGBRegressor(**params_xgb)
            model_cb = CatBoostRegressor(**params_cb)
            models = [model_gb, model_xgb, model_cb]
            
            for model in models:
                model.fit(X_train_combined, y_train_tc_par)
            
            for model in  models:
                print('MSE of ', model.__class__.__name__, ' on test:', metrics.mse(y_test_tc_par, model.predict(X_test_combined)))
            
            meta_model = LinearRegression()
            meta_model.fit(X_train_combined, y_train_tc_par)
            print('MSE of metaregressor on original datasets features on test: ', metrics.mse(y_test_tc_par, meta_model.predict(X_test_combined)))

            n = 5
            y_pred_tc_par, model = source.metaregressor(models, meta_model, X_train_combined, X_test_combined, y_train_tc_par, n)
            pred_all_tc_par, model = source.metaregressor(models, meta_model, X_train_combined, ALL_GIS_combined, y_train_tc_par, n)
            if pred_name is not None:
                display_title_metrics(pred_name)
                source.get_metrics(y_test_tc_par, y_pred_tc_par)

        return y_pred_tc_par, pred_all_tc_par, model
    



    if what_to_predict in ['TC_par', 'VHC']:
        if what_to_predict == 'TC_par':
            pred_name = 'TC_par'
            y_pred_tc_par, pred_all_tc_par, model_tc_par = predictor(pred_name)
            
            if ' ' in model_mode: 
                model_name = "".join(model_mode.split())
            else: 
                model_name = model_mode
            df_to_csv(pd.DataFrame({depth_name: ALL_GIS[depth_name], 
                                    'TC_par_pred': pred_all_tc_par}), 
                                    filename=model_name +'_'+'TC_par',
                                    button_text='Скачать прогноз TC_par по всему интервалу')
        elif what_to_predict == 'VHC':
            pred_name = 'VHC'
            y_pred_vhc, pred_all_vhc, model_vhc = predictor(pred_name)

            if ' ' in model_mode: 
                model_name = "".join(model_mode.split())
            else: 
                model_name = model_mode
            df_to_csv(pd.DataFrame({depth_name: ALL_GIS[depth_name], 
                                    'VHC_pred': pred_all_vhc}), 
                                    filename=model_name +'_'+'VHC',
                                    button_text='Скачать прогноз VHC по всему интервалу')
    elif what_to_predict in ['TC_per']:
        pred_name = 'Anisotropy'
        y_pred_anisotropy, pred_all_anisotropy, model_anisotropy = predictor(pred_name)

        if ' ' in model_mode: 
            model_name = "".join(model_mode.split())
        else: 
            model_name = model_mode
        df_to_csv(pd.DataFrame({depth_name: ALL_GIS[depth_name], 
                                'K_pred': pred_all_anisotropy}), 
                                filename=model_name +'_'+'K',
                                button_text='Скачать прогноз K по всему интервалу')
    
        y_pred_tc_per = X_test_orig[tc_par_name].values / y_pred_anisotropy
        y_test_tc_per = data_to_pred.loc[X_test_orig.index, tc_per_name].values

        if use_lith:
            try:
                pred_all_tc_per = pred_all_tc_par / pred_all_anisotropy
            except:
                st.write(r"Для получения прогноза $\lambda_{\bot}$ на весь интервал, загрузите полученный ранее прогноз $\lambda_{\parallel}$ по всему интервалу:")
                uploaded_tc_par_pred = st.file_uploader("TC_par_pred", 
                                                        type=['csv', 'xlsx'], 
                                                        label_visibility='collapsed')
                pred_all_tc_par = source.load_file_to_st(uploaded_tc_par_pred)
            

            
        a = pred_all_tc_par.loc[:, pred_all_tc_par.columns != depth_name].iloc[:, 0].values
        b = pred_all_anisotropy
        pred_all_tc_per = a / b


        if ' ' in model_mode: 
            model_name = "".join(model_mode.split())
        else: 
            model_name = model_mode
        df_to_csv(pd.DataFrame({depth_name: ALL_GIS[depth_name], 
                                'TC_per_pred': pred_all_tc_per}), 
                                filename=model_name +'_'+'TC_per',
                                button_text='Скачать прогноз TC_per по всему интервалу',
                                add_key='1')

        st.write(r"Метрики для $\lambda_{\bot}$")
        source.get_metrics(y_test_tc_per, y_pred_tc_per)


    # fig, ax = plt.subplots(1, 2, figsize=(12,5))
    
    # ax[0].hist(pred_all_anisotropy, bins=20)
    # ax[1].hist(pred_all_tc_per, bins=20)

    # st.pyplot(fig)





# st.toast("Warming up...")
# st.error("Error message")
# st.warning("Warning message")
# st.info("Info message")
# st.success("Success message")


if __name__ == "__main__":
    run()