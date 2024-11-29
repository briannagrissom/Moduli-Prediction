import pandas as pd
# from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
# from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import xgboost as xgb

pd.options.mode.chained_assignment = None

"""The purpose of this file is to train the pipeline and evaluate its performance on 
testing data using K-fold cross validation. """

df2 = pd.read_excel('/Users/briannagrissom/moduli_prediction/excel_files/dataset_with_L21_DO3.xlsx')


def train_models(X, predictors, constant_indices):
    """Fit ExtraTreesRegressor and Ridge with training data."""
    params = {'tree_method': 'hist', 'objective': 'reg:squarederror'}
    arr_C11 = [3 * X[:, 4] - 2 * X[:, 9], X[:, 9]]  # C11 = 3*K -2*C12
    arr_C12 = [(3 * X[:, 4] - X[:, 8]) / 2, X[:, 8]]  # C12 = (3*K - C11)/2
    arr_C44 = (5 / 3) * X[:, 3] - (1 / 3) * X[:, 8] + (1 / 3) * X[:, 9]  # C44 = (3/5)*G - (1/3)*C11 + (1/3)*C12
    arr_K = (X[:, 8] + 2 * X[:, 9]) / 3  # K = (C11 + 2*C12)/3
    arr_dict = {'C11': [arr_C11, predictors[0], constant_indices[0]],
                'C12': [arr_C12, predictors[1], constant_indices[1]],
                'C44': [arr_C44, predictors[3], constant_indices[3]], 'K': [arr_K, predictors[2], constant_indices[2]]}
    models = dict()
    for key in arr_dict.keys():
        def custom_loss_more_data(y_pred, dtrain, arrs, constant):
            y_true = dtrain.get_label()
            if constant == 'K':
                grad = 2 * (y_pred - y_true) + 2 * (y_pred - arrs)  # first derivative w.r.t.yhat
                hess = np.full_like(y_pred, 4)  # second derivative w.r.t. yhat is 4
            if constant == 'C11':
                relu = np.maximum(0, arrs[1] - y_pred)  # bad if C12 > C11
                grad = 2 * (y_pred - y_true) + 2 * (y_pred - arrs[0]) + 0 * 0.1 * relu * np.sign(
                    arrs[1] - y_pred)  # first derivative w.r.t.yhat
                hess = np.full_like(y_pred, 4)
            if constant == 'C12':
                grad = 2 * (y_pred - y_true) + 2 * (y_pred - arrs[0])  # first derivative w.r.t.yhat
                hess = np.full_like(y_pred, 4)  # second derivative w.r.t. yhat is 4
            if constant == 'C44':
                grad = 2 * (y_pred - y_true) + 2 * (y_true - arrs)
                hess = np.full_like(y_pred, 4)

            return grad, hess

        arr = arr_dict[key][0]
        feature_vars = arr_dict[key][1]
        response_var = arr_dict[key][2]

        def custom_loss(y_pred, dtrain):
            return custom_loss_more_data(y_pred, dtrain, arrs=arr, constant=key)

        features = X[:, feature_vars]
        response = X[:, response_var]
        train_data = xgb.DMatrix(features, response)
        model = xgb.train(params=params, dtrain=train_data, num_boost_round=600, obj=custom_loss)
        models[f'{key}_model'] = model

    return models


def get_metrics(data, iterations=1, return_df=False, num_k_folds=10):
    """Returns the mean absolute percentage error (MAPE) and its standard deviation for the prediction of each constant
    over a specified number of iterations. Predictions are done using K-fold cross validation. If return_df is True,
    return a copy of the original DataFrame with predictions, errors, and percentage errors for each row averaged
    across all iterations."""

    C11_predictors = ['K_VRH', 'Δa', 'ΔTm', 'Δχ', 'δ', 'ΔHmix', 'ΔSmix', 'λ', 'Poisson_ROM',
                      'Lamé_ROM', 'C12_ROM_EAH>0', 'C44_ROM_EAH=0', 'C11_ROM_EAH=0']
    C12_predictors = ['K_VRH', 'ΔTm', 'Δχ', 'ΔHmix', 'Lamé_ROM', 'C44_ROM_EAH=0', 'C11_ROM_EAH=0']
    K_predictors = ['K_VRH', 'ΔHmix', 'Δχ', 'G_VRH', 'ΔTm']
    C44_predictors = ['K_VRH', 'G_VRH', 'E_VRH', 'Δa', 'ΔTm', 'Δχ', 'δ', 'ΔHmix', 'ΔSmix', 'λ',
                      'Poisson_ROM', 'Lamé_ROM', 'C12_ROM_EAH>0']

    C11_predictors_indices, C12_predictors_indices, K_predictors_indices, C44_predictors_indices = [], [], [], []
    predictor_dict = {}
    for idx, col in enumerate(data.columns):
        predictor_dict[col] = idx
    C11_index, C12_index, K_index, C44_index = predictor_dict['C11'], predictor_dict['C12'], \
        predictor_dict['Bulk modulus'], predictor_dict['C44']
    for col in predictor_dict.keys():
        if col in C11_predictors:
            index = predictor_dict[col]
            C11_predictors_indices.append(index)
        if col in C12_predictors:
            index = predictor_dict[col]
            C12_predictors_indices.append(index)
        if col in K_predictors:
            index = predictor_dict[col]
            K_predictors_indices.append(index)
        if col in C44_predictors:
            index = predictor_dict[col]
            C44_predictors_indices.append(index)
    c_indices = [C11_index, C12_index, K_index, C44_index]
    all_predictors = [C11_predictors_indices, C12_predictors_indices, K_predictors_indices, C44_predictors_indices]

    df = data.copy()
    averaged_mapes = {}
    std_of_mapes = {}
    if return_df:
        relevant_columns = []
    for n in range(iterations):
        err_dict = {}
        mape_dict = {}
        for i in range(1, num_k_folds + 1):
            err_dict[i] = []
            mape_dict[i] = []
        X = data.values
        Kfold = KFold(n_splits=num_k_folds, shuffle=True)
        fold_number = 1
        for train_index, test_index in Kfold.split(X):
            # Dividing data into training and test set
            X_train, X_test = X[train_index], X[test_index]
            mdls = train_models(X=X_train, predictors=all_predictors, constant_indices=c_indices)

            C11_model = mdls['C11_model']
            C11_X = X_test[:, C11_predictors_indices]
            C11_vals = X_test[:, 8]
            C11_test_data = xgb.DMatrix(C11_X, C11_vals)
            C11_predicted = C11_model.predict(C11_test_data)
            err1 = C11_vals - C11_predicted
            mape1 = mean_absolute_percentage_error(C11_vals, C11_predicted)

            C12_model = mdls['C12_model']
            C12_X = X_test[:, C12_predictors_indices]
            C12_vals = X_test[:, 9]
            C12_test_data = xgb.DMatrix(C12_X, C12_vals)
            C12_predicted = C12_model.predict(C12_test_data)
            err2 = C12_vals - C12_predicted
            mape2 = mean_absolute_percentage_error(C12_vals, C12_predicted)

            # K_X = X_test[:, K_predictors_indices]
            # K_predicted = K_predictor.predict(K_X)

            K_predicted = []  # has better MAPE when we solve for K
            for i in range(len(C12_predicted)):
                K_predicted.append((2 * C12_predicted[i] + C11_predicted[i]) / 3)

            err3 = X_test[:, 4] - K_predicted
            mape3 = mean_absolute_percentage_error(X_test[:, 4], K_predicted)

            C44_model = mdls['C44_model']
            C44_X = X_test[:, C44_predictors_indices]
            C44_vals = X_test[:, 10]
            C44_test_data = xgb.DMatrix(C44_X, C44_vals)
            C44_predicted = C44_model.predict(C44_test_data)
            err4 = C44_vals - C44_predicted
            mape4 = mean_absolute_percentage_error(C44_vals, C44_predicted)

            G_predicted = []
            for i in range(len(X_test)):
                G_predicted.append((3 / 5) * C44_predicted[i] + (1 / 5) * C11_predicted[i] - (1 / 5) * C12_predicted[i])
            err5 = X_test[:, 3] - G_predicted
            mape5 = mean_absolute_percentage_error(X_test[:, 3], G_predicted)

            E_predicted = []
            for i in range(len(X_test)):
                E_predicted.append((9 * K_predicted[i] * G_predicted[i]) / (3 * K_predicted[i] + G_predicted[i]))
            err6 = X_test[:, 6] - E_predicted
            mape6 = mean_absolute_percentage_error(X_test[:, 6], E_predicted)

            err_dict[fold_number] = [err1, err2, err3, err4, err5, err6]
            mape_dict[fold_number] = [mape1, mape2, mape3, mape4, mape5, mape6]
            fold_number = fold_number + 1
            count = 0
            if return_df:
                for idx in test_index:
                    df.loc[idx, f'C11 pred {n}'] = C11_predicted[count]
                    df.loc[idx, f'C11 err {n}'] = err1[count]
                    df.loc[idx, f'C11 pe {n}'] = abs((C11_predicted[count] - df.loc[idx, 'C11']) / df.loc[idx, 'C11'])

                    df.loc[idx, f'C12 pred {n}'] = C12_predicted[count]
                    df.loc[idx, f'C12 err {n}'] = err2[count]
                    df.loc[idx, f'C12 pe {n}'] = abs((C12_predicted[count] - df.loc[idx, 'C12']) / df.loc[idx, 'C12'])

                    df.loc[idx, f'K pred {n}'] = K_predicted[count]
                    df.loc[idx, f'K err {n}'] = err3[count]
                    df.loc[idx, f'K pe {n}'] = abs(
                        (K_predicted[count] - df.loc[idx, 'Bulk modulus']) / df.loc[idx, 'Bulk modulus'])

                    df.loc[idx, f'C44 pred {n}'] = C44_predicted[count]
                    df.loc[idx, f'C44 err {n}'] = err4[count]
                    df.loc[idx, f'C44 pe {n}'] = abs((C44_predicted[count] - df.loc[idx, 'C44']) / df.loc[idx, 'C44'])

                    df.loc[idx, f'G pred {n}'] = G_predicted[count]
                    df.loc[idx, f'G err {n}'] = err5[count]
                    df.loc[idx, f'G pe {n}'] = abs(
                        (G_predicted[count] - df.loc[idx, 'Shear Voigt']) / df.loc[idx, 'Shear Voigt'])

                    df.loc[idx, f'E pred {n}'] = E_predicted[count]
                    df.loc[idx, f'E err {n}'] = err6[count]
                    df.loc[idx, f'E pe {n}'] = abs(
                        (E_predicted[count] - df.loc[idx, 'Young modulus 2']) / df.loc[idx, 'Young modulus 2'])
                    count = count + 1

        averaged_mapes[n] = pd.DataFrame(mape_dict).T.mean().values
        std_of_mapes[n] = pd.DataFrame(mape_dict).T.std().values
        if return_df:
            relevant_columns = relevant_columns + [f'C11 pred {n}', f'C11 err {n}', f'C11 pe {n}',
                                                   f'C12 pred {n}', f'C12 err {n}', f'C12 pe {n}',
                                                   f'K pred {n}', f'K err {n}', f'K pe {n}',
                                                   f'C44 pred {n}', f'C44 err {n}', f'C44 pe {n}',
                                                   f'G pred {n}', f'G err {n}', f'G pe {n}',
                                                   f'E pred {n}', f'E err {n}', f'E pe {n}']

    overall_mapes = pd.DataFrame(pd.DataFrame(averaged_mapes).T.mean()).T
    overall_std = pd.DataFrame(pd.DataFrame(std_of_mapes).T.mean()).T

    overall_mapes.rename(columns={0: 'C11', 1: 'C12', 2: 'K', 3: 'C44', 4: 'G', 5: 'E'}, inplace=True)
    overall_std.rename(columns={0: 'C11', 1: 'C12', 2: 'K', 3: 'C44', 4: 'G', 5: 'E'}, inplace=True)

    if return_df:
        result_df = df[df.columns[:33]]
        constants = ['C11', 'C12', 'K', 'C44', 'G', 'E']
        for constant in constants:
            cols = [col for col in relevant_columns if constant in col]
            metrics = ['pred', 'pe', 'err']
            for metric in metrics:
                cols2 = [col for col in cols if metric in col]
                mean = np.mean(df[cols2], axis=1)
                result_df[f'{constant} {metric} (mean)'] = mean
        return [overall_mapes, overall_std, result_df]

    else:
        return [overall_mapes, overall_std]


mapes, std = get_metrics(data=df2, iterations=1, return_df=False)
print(mapes)
