import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, \
    HistGradientBoostingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from py_files.evaluate_performance import get_metrics

data = pd.read_excel('excel_files/moduli_prediction_dataset.xlsx')
# data.drop(columns=['Unnamed: 0'], inplace=True)

gbrs = [GradientBoostingRegressor(random_state=1), GradientBoostingRegressor(random_state=1),
        GradientBoostingRegressor(random_state=1), GradientBoostingRegressor(random_state=1)]
dtrs = [DecisionTreeRegressor(random_state=20), DecisionTreeRegressor(random_state=20),
        DecisionTreeRegressor(random_state=20), DecisionTreeRegressor(random_state=20)]
lrs = [LinearRegression(), LinearRegression(), LinearRegression(), LinearRegression()]
brs = [BayesianRidge(), BayesianRidge(), BayesianRidge(), BayesianRidge()]
rfrs = [RandomForestRegressor(random_state=20), RandomForestRegressor(random_state=20),
        RandomForestRegressor(random_state=20), RandomForestRegressor(random_state=20)]
abrs = [AdaBoostRegressor(random_state=20), AdaBoostRegressor(random_state=20),
        AdaBoostRegressor(random_state=20), AdaBoostRegressor(random_state=20)]
bars = [BaggingRegressor(), BaggingRegressor(), BaggingRegressor(), BaggingRegressor()]
hgbrs = [HistGradientBoostingRegressor(), HistGradientBoostingRegressor(), HistGradientBoostingRegressor(),
         HistGradientBoostingRegressor()]
etrs = [ExtraTreesRegressor(), ExtraTreesRegressor(), ExtraTreesRegressor(), ExtraTreesRegressor()]
ens = [ElasticNet(), ElasticNet(), ElasticNet(), ElasticNet()]
krs = [KernelRidge(), KernelRidge(), KernelRidge(), KernelRidge()]
svrs = [SVR(), SVR(), SVR(), SVR()]
knrs = [KNeighborsRegressor(), KNeighborsRegressor(), KNeighborsRegressor(), KNeighborsRegressor()]

model_names = ['gbr', 'dtr', 'lr', 'br', 'rfr', 'abr', 'bar', 'hgbr', 'etr', 'en', 'kr', 'svr', 'knr']
models = [gbrs, dtrs, lrs, brs, rfrs, abrs, bars, hgbrs, etrs, ens, krs, svrs, knrs]
model_dict = {}
for i, model_name in enumerate(model_names):
    model_dict[model_name] = [models[i][0], models[i][1], models[i][2], models[i][3]]

results = {}
for index, model in enumerate(model_dict.keys()):
    models = model_dict[model]
    res = get_metrics(data=data, C11_model=models[0], C12_model=models[1], C44_model=models[2], K_model=models[3],
                      iterations=10)
    results[model] = res.values.flatten()

df = pd.DataFrame(results)
