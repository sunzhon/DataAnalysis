
from sklearn.metrics import r2_score, mean_squared_error as mse
def get_scores(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse(y_true, y_pred))
    mae = np.mean(abs((y_true - y_pred)))
    r_rmse = rmse / (y_true.max() - y_true.min())

    return r2, rmse, mae, r_rmse
    
