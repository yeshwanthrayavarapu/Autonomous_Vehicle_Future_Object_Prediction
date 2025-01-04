import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures


def preprocessing(X, Y):
    print(X.shape, Y.shape)
    x = X.T
    y = Y.reshape(-1, 1)
    print(x.shape, y.shape)

    df = pd.DataFrame(x, columns=['Distance', 'Speed', 'Mask area', 'Acceleration_x', 'Acceleration_y',
                                  'Relative_Position'])
    df['Relative_Position'] = df['Relative_Position'].astype('category').cat.codes
    df['Time duration'] = Y.tolist()

    df_train = df.sample(frac=0.6, random_state=1)
    df_test = df.drop(index=df_train.index)

    Y_train = df_train[['Time duration']]
    X_train = df_train.drop(['Time duration'], axis=1)

    Y_test = df_test[['Time duration']]
    X_test = df_test.drop(['Time duration'], axis=1)

    # normalized_df=(df-df.mean())/df.std()

    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    Y_train = Y_train - Y_train.mean()
    Y_test = Y_test - Y_test.mean()

    return df, X_train, Y_train, X_test, Y_test


def linear_regression(X_train, Y_train, X_test, Y_test):
    linear_rg = LinearRegression().fit(X_train, Y_train)
    print("Linear regression score - training", linear_rg.score(X_train, Y_train))

    y_pred = linear_rg.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    # Print the results
    print("Linear regression")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-Squared: {r2:.2f}")


def PolynomialRegression(X_train, Y_train, X_test, Y_test, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_X_train = poly.fit_transform(X_train)
    poly_regr = LinearRegression()
    poly_regr.fit(poly_X_train, Y_train)
    print("Polynomial regression score - training", poly_regr.score(poly_X_train, Y_train))
    poly_X_test = poly.fit_transform(X_test)
    y_pred = poly_regr.predict(poly_X_test)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    # Print the results
    print("Linear regression")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-Squared: {r2:.2f}")


def RandomForestRegression(X_train, Y_train, X_test, Y_test, max_depth, n_estimators):
    rf_regr = RandomForestRegressor(max_depth=max_depth, random_state=0, n_estimators=n_estimators)
    Y_train = (Y_train.to_numpy()).T[0]
    rf_regr.fit(X_train, Y_train)
    print("Random forest regressor - training score", rf_regr.score(X_train, Y_train))
    Y_test = (Y_test.to_numpy()).T[0]

    y_pred = rf_regr.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    # Print the results
    print("Random forest regression")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-Squared: {r2:.2f}")


def DecisionTreeRegression(X_train, Y_train, X_test, Y_test, max_depth):
    dt_regr = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    Y_train = (Y_train.to_numpy()).T[0]
    dt_regr.fit(X_train, Y_train)
    print("Decision tree regressor - training", dt_regr.score(X_train, Y_train))
    Y_test = (Y_test.to_numpy()).T[0]

    y_pred = dt_regr.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    # Print the results
    print("Decision tree regression")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-Squared: {r2:.2f}")


def GradientBoostingRegression(X_train, Y_train, X_test, Y_test, alpha, estimators):
    gb_regr = GradientBoostingRegressor(random_state=0, learning_rate=alpha, n_estimators=estimators)
    Y_train = (Y_train.to_numpy()).T[0]
    gb_regr.fit(X_train, Y_train)
    print("Gradient boost regressor - training", gb_regr.score(X_train, Y_train))
    Y_test = (Y_test.to_numpy()).T[0]

    y_pred = gb_regr.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    # Print the results
    print("Gradient boosting regression")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-Squared: {r2:.2f}")


def BaggingRegression(X_train, Y_train, X_test, Y_test, n_estimators):
    bag_regr = BaggingRegressor(estimator=SVR(), n_estimators=n_estimators, random_state=0)
    Y_train = (Y_train.to_numpy()).T[0]
    bag_regr.fit(X_train, Y_train)
    print("Bagging regressor - training", bag_regr.score(X_train, Y_train))
    Y_test = (Y_test.to_numpy()).T[0]

    y_pred = bag_regr.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, y_pred)
    r2 = r2_score(Y_test, y_pred)

    # Print the results
    print("Bagging regression")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R-Squared: {r2:.2f}")


"""

def principalcomponentanalysis(X_train, Y_train, X_test, Y_test):
    pca = PCA()
    pca.fit(X_train)
    X_pca = pca.transform(X_train)
    scores = cross_val_score(LinearRegression(), X_pca, Y_train, cv=5)
    print("PCR R-squared scores: ", scores.mean())



# Partial Least Squares Regression (PLSR)

def PLSR(X_train, Y_train, X_test, Y_test):
    pls = PLSRegression(n_components=3)
    pls.fit(X_train, Y_train)
    scores = cross_val_score(pls, X_train, Y_train, cv=5)
    print("PLSR R-squared scores: ", scores.mean())

"""

