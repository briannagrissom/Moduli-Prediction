import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler

pd.options.mode.chained_assignment = None

"""The purpose of this file is to train the pipeline and evaluate its performance on 
testing data using K-fold cross validation. """


# df2 = pd.read_excel('excel_files/blocking_data.xlsx')
# df2 = df2.loc[df2['Crystal structure'] != 'None', :]
df2 = pd.read_excel('excel_files/dataset_with_L21_DO3.xlsx')
# df2 = df2.loc[df2['Crystal structure'] != 'BCC', :]


def train_models(X, C11_model, C12_model, K_model, C44_model, predictors, constant_indices):
    """Fit ExtraTreesRegressor and Ridge with training data."""
    # Use ExtraTreesRegressor to predict C11. This model was found to have the least error over many iterations.
    pred_C11 = X[:, predictors[0]]
    C11_vals = X[:, constant_indices[0]].ravel()
    C11_model.fit(pred_C11, C11_vals)

    # Use ExtraTreesRegressor to predict C12. This model was found to have the least error over many iterations.
    pred_C12 = X[:, predictors[1]]
    C12_vals = X[:, constant_indices[1]].ravel()
    C12_model.fit(pred_C12, C12_vals)

    # Use LinearRegression to predict K. This model had nearly the lowest MAPE w/o overfitting
    pred_K = X[:, predictors[2]]
    K_vals = X[:, constant_indices[2]].ravel()
    K_model.fit(pred_K, K_vals)

    # Predict C44
    pred_C44 = X[:, predictors[3]]
    C44_vals = X[:, constant_indices[3]]
    C44_model.fit(pred_C44, C44_vals)

    return [C11_model, C12_model, K_model, C44_model]


def get_metrics(data, iterations=1, return_df=False, C11_model=ExtraTreesRegressor(),
                C12_model=ExtraTreesRegressor(), K_model=Ridge(),
                C44_model=ExtraTreesRegressor(), num_k_folds=10):
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
            C11_predictor, C12_predictor, K_predictor, C44_predictor = train_models(X=X_train, C11_model=C11_model,
                                                                                    C12_model=C12_model,
                                                                                    K_model=K_model,
                                                                                    C44_model=C44_model,
                                                                                    predictors=all_predictors,
                                                                                    constant_indices=c_indices)

            C11_X = X_test[:, C11_predictors_indices]

            C11_predicted = C11_predictor.predict(C11_X)
            err1 = X_test[:, 8] - C11_predicted
            mape1 = mean_absolute_percentage_error(X_test[:, 8], C11_predicted)

            C12_X = X_test[:, C12_predictors_indices]
            C12_predicted = C12_predictor.predict(C12_X)
            err2 = X_test[:, 9] - C12_predicted
            mape2 = mean_absolute_percentage_error(X_test[:, 9], C12_predicted)

            # K_X = X_test[:, K_predictors_indices]
            # K_predicted = K_predictor.predict(K_X)

            K_predicted = []  # has better MAPE when we solve for K
            for i in range(len(C12_predicted)):
                K_predicted.append((2 * C12_predicted[i] + C11_predicted[i]) / 3)

            err3 = X_test[:, 4] - K_predicted
            mape3 = mean_absolute_percentage_error(X_test[:, 4], K_predicted)

            C44_X = X_test[:, C44_predictors_indices]
            C44_predicted = C44_predictor.predict(C44_X)
            err4 = X_test[:, 10] - C44_predicted
            mape4 = mean_absolute_percentage_error(X_test[:, 10], C44_predicted)

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
