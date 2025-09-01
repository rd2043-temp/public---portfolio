import pandas as pd; import numpy as np; import matplotlib.pyplot as plt; from sklearn.metrics import r2_score
import os 

def evaluate_strategies(df, predictions, date_column='Date', ticker_column='Ticker', returns_column='Return', 
                        trade_cost : float = 0 # 0.000_05
                        ):
    """
    Evaluate profitability of three strategies based on predicted Sharpe ratios.
    
    Parameters:
    - df: DataFrame with columns including 'target', 'Returns', and features
    - predictions: Dictionary with (date, ticker) keys and predicted Sharpe ratios as values
    - asset_universe: List of tickers
    - date_column: Name of date column in multi-index
    - ticker_column: Name of ticker column in multi-index
    - returns_column: Name of returns column for profitability
    
    Returns:
    - Plots and metrics for portfolio performance
    """
    # Ensure DataFrame has Date and Ticker in index
    df = df.reset_index()
    df = df.sort_values([ticker_column, date_column])
    df["Return"] = df.groupby(ticker_column)["Return_lag_1"].shift(-1)  # Next day return
    df['Predictions'] = df.apply(lambda row: predictions.get((row[date_column], row[ticker_column]), 0), axis=1)


    dates =  sorted(df[date_column].unique())
    strategy_returns = {
        'Long Top 3': [],
        'Short Bottom 3': [],
        'Long-Short': []
    }
    
    for date in dates:
        daily_data = df[df[date_column] == date].copy()
        
        daily_data = daily_data.sort_values('Predictions', ascending=False)
        top_3 = daily_data.head(3); bottom_3 = daily_data.tail(3)
        
        long_top_3_return = top_3[returns_column].mean() - trade_cost # Equal-weighted
        strategy_returns['Long Top 3'].append(long_top_3_return)
        
        short_bottom_3_return = -bottom_3[returns_column].mean() - trade_cost # Negative for short
        strategy_returns['Short Bottom 3'].append(short_bottom_3_return)
        
        long_short_return = (top_3[returns_column].mean()  - bottom_3[returns_column].mean()) / 2  - trade_cost # Equal-weighted
        strategy_returns['Long-Short'].append(long_short_return)

    returns_df = pd.DataFrame(strategy_returns, index=dates)
    
    # calc cumulative returns
    cumulative_returns = (1 + returns_df).cumprod()
    
    # calc annualized Sharpe ratio (assuming 252 trading days)
    number_of_days_between_dates = (dates[-1] - dates[0]).days if len(dates) > 1 else 365
    obs_per_year = (len(returns_df) / (number_of_days_between_dates / 365.25)) if number_of_days_between_dates > 0 else 252
    if obs_per_year <= 0:
        obs_per_year = 252
    annualized_sharpe = {}
    for strategy in strategy_returns:
        ret = returns_df[strategy].dropna()
        mean_ret = ret.mean() if not ret.empty else 0.0
        std_ret = ret.std() if not ret.empty else 0.0
        annualized_ret = mean_ret * obs_per_year #- risk_free_rate
        annualized_std = std_ret * (obs_per_year ** 0.5)
        sharpe = annualized_ret / annualized_std if annualized_std != 0 else np.nan
        annualized_sharpe[strategy] = sharpe
    
    # plot and eval cumulative returns
    plt.figure(figsize=(10, 6))
    for strategy in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[strategy], label=strategy)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns of Strategies')
    plt.legend(); plt.grid(True); plt.show()
    print("Annualized Sharpe Ratios:")
    for strategy, sharpe in annualized_sharpe.items():
        print(f"{strategy}: {sharpe:.4f}")
    
    return cumulative_returns, annualized_sharpe, returns_df

def train_predict_evaluate(df, model_func, 
                           model_name: str, 
                           train_intervals: list = [60, 365], 
                           lookback_days : int = 365 * 2, 
                           weights: list = [0.7, 0.3], 
                           base_path: str = "sub_models/weekly",
                           trade_cost : float = 0 # 0.000_05
                           ):
    """
    Train, predict, and evaluate using a specified model function for given intervals.
    
    Parameters:
    - df: Input DataFrame with MultiIndex ('Date', 'Ticker') and 'target' column
    - model_func: Model function with signature (df, train: bool, model_path: str)
    - model_name: Name of the model (e.g., 'xgboost', 'lightgbm', 'randomforest', 'neuralnetwork')
    - train_intervals: List of training intervals in days (e.g., [60, 365])
    - weights: Weights for combining predictions from different intervals
    - base_path: Base directory for saving models
    """
    # create directories for each interval
    for interval in train_intervals:
        os.makedirs(f"{base_path}/{interval}d", exist_ok=True)

    if not isinstance(df.index, pd.MultiIndex) or 'Date' not in df.index.names:
        raise ValueError("DataFrame index must be a MultiIndex with 'Date' and 'Ticker' levels")


    dates = sorted(set(df.index.get_level_values('Date')))
    last_train_dates = {interval: dates[0] - pd.Timedelta(days=interval) for interval in train_intervals}

    predictions = {}
    if 'Predictions' not in df.columns:
        df['Predictions'] = np.nan

    for date in dates:
        if date < dates[0] + pd.Timedelta(days=lookback_days):
            continue
        for interval in train_intervals:
            if date >= last_train_dates[interval] + pd.Timedelta(days=interval):
                df_for_train = df[(df.index.get_level_values('Date') < date) & (df.index.get_level_values('Date') > date - pd.Timedelta(days=lookback_days))].reset_index()
                df_for_train = df_for_train.fillna(0).replace([np.inf, -np.inf], 0)
                
                last_train_dates[interval] = date
                model_path = f"{base_path}/{interval}d/{model_name}_model_{last_train_dates[interval].date()}_{interval}.{'h5' if model_name.lower() == 'neuralnetwork' else 'json' if model_name.lower() == 'xgboost' else 'txt' if model_name.lower() == 'lightgbm' else 'joblib'}"
                model_func(df_for_train.drop(columns=['Date', 'Ticker']), train=True, model_path=model_path)
                
                print(f"model created :: type = {interval}d :: at date = {date} under path = {model_path}")

        # make predictions using all models
        df_for_pred = df[df.index.get_level_values('Date') == date].reset_index()
        df_for_pred = df_for_pred.fillna(0).replace([np.inf, -np.inf], 0)
        
        combined_pred = np.zeros(len(df_for_pred))
        for idx, interval in enumerate(train_intervals):
            model_path = f"{base_path}/{interval}d/{model_name}_model_{last_train_dates[interval].date()}_{interval}.{'h5' if model_name.lower() == 'neuralnetwork' else 'json' if model_name.lower() == 'xgboost' else 'txt' if model_name.lower() == 'lightgbm' else 'joblib'}"
            pred = model_func(df_for_pred.drop(columns=['Date', 'Ticker']), train=False, model_path=model_path)
            combined_pred += weights[idx] * pred

        i=0
        for ticker in df_for_pred['Ticker']:
            predictions[(date, ticker)] = combined_pred[i]

            idx_date = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
            idx = (idx_date, ticker)
            df.loc[idx, 'Predictions'] = combined_pred[i]
            i+=1

    # eval
    np.random.seed(42)
    df_reset = df.reset_index()
    actuals = df_reset['target'].fillna(0)
    pred_values = [predictions.get((row['Date'], row['Ticker']), 0) for _, row in df_reset.iterrows()]
    r2 = r2_score(actuals, pred_values)
    print(f"R^2 on test set: {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, pred_values, alpha=0.5)
    plt.xlabel('True Target')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Target')
    plt.grid(True); plt.show()

    cumulative_returns, sharpe_ratio, returns_df = evaluate_strategies(df, predictions, trade_cost=trade_cost)

    return predictions,df, cumulative_returns, sharpe_ratio, returns_df



def simulate_diff_trading_costs(df, predictions, 
                                trade_costs: list = [0, 0.000_01, 0.000_05, 0.000_1], date_column='Date', 
                                ticker_column='Ticker', returns_column='Return'):
    
    results = {}
    all_cum_returns = pd.DataFrame()
    
    for cost in trade_costs:
        cum_returns, sharpe, returns_df = evaluate_strategies(df, predictions, date_column, ticker_column, returns_column, cost)
        results[cost] = {
            'cumulative_returns': cum_returns,
            'sharpe_ratios': sharpe,
            'returns_df': returns_df
        }
        
        # Rename columns for combined plot
        renamed_cum = cum_returns.rename(columns={col: f"{col} (Cost {cost})" for col in cum_returns.columns})
        all_cum_returns = pd.concat([all_cum_returns, renamed_cum], axis=1)
    
    # Combined plot
    plt.figure(figsize=(12, 8))
    for col in all_cum_returns.columns:
        plt.plot(all_cum_returns.index, all_cum_returns[col], label=col)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns for Different Trading Costs')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Sharpe Ratios for Different Trading Costs:")
    for cost, data in results.items():
        print(f"\nTrading Cost: {cost}")
        for strategy, sharpe in data['sharpe_ratios'].items():
            print(f"{strategy}: {sharpe:.4f}")
    
    return results




# =================== Models =====================
#2.1 model construction
def XGBoost(df, train : bool = True, model_path : str = "xgboost_model.json"):
    if train : 
        # train the model
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error
        X_train, y_train = df.drop(columns=['target']), df['target']
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,              # Reduced to prevent overfitting
            'learning_rate': 0.03,            # Lower for gradual learning
            'max_depth': 4,                   # Shallower trees for noisy data
            'min_child_weight': 3,            # Increased to control overfitting
            'subsample': 0.7,                 # Lower to add randomness
            'colsample_bytree': 0.6,          # Use fewer features per tree
            'reg_alpha': 0.5,                 # Increased L1 regularization
            'reg_lambda': 2.0,                # Increased L2 regularization
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'             # Faster training
        }
        model = xgb.XGBRegressor(**params)
        
        
        
        model.fit(X_train, y_train)
        model.save_model(model_path)  
        
    else : 
        # load the model
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        # make predictions
        X = df.drop(columns=['target'])
        predictions = model.predict(X)
        return predictions 

def LightGBM(df, train: bool = True, model_path: str = "lightgbm_model.txt"):
    if train:
        # train the model
        import lightgbm as lgb
        from sklearn.metrics import mean_squared_error
        X_train, y_train = df.drop(columns=['target']), df['target']
        model = lgb.LGBMRegressor(objective='regression',
                                 n_estimators=500,
                                 learning_rate=0.05,
                                 max_depth=6,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 reg_alpha=0.1,
                                 reg_lambda=1.5,
                                 random_state=42,
                                 n_jobs=-1)
        
        model.fit(X_train, y_train, eval_metric='rmse')
        model.booster_.save_model(model_path)
        
    else:
        # load the model
        import lightgbm as lgb
        model = lgb.Booster(model_file=model_path)
        # make predictions
        X = df.drop(columns=['target'])
        predictions = model.predict(X)
        return predictions

def RandomForest(df, train: bool = True, model_path: str = "randomforest_model.joblib"):
    if train:
        # train the model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        import joblib
        X_train, y_train = df.drop(columns=['target']), df['target']
        model = RandomForestRegressor(n_estimators=500,
                                     max_depth=6,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     random_state=42,
                                     n_jobs=-1)
        
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        
    else:
        # load the model
        import joblib
        model = joblib.load(model_path)
        # make predictions
        X = df.drop(columns=['target'])
        predictions = model.predict(X)
        return predictions

def NeuralNetwork(df, train: bool = True, model_path: str = "neuralnetwork_model.h5"):
    if train:
        # train the model
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.optimizers import Adam
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        
        X_train, y_train = df.drop(columns=['target']), df['target']
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Build neural network
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.01),
                     loss='mse',
                     metrics=['mae'])
        
        model.fit(X_train_scaled, y_train,
                 epochs=100,
                 batch_size=32,
                 validation_split=0.2,
                 verbose=0)
        
        model.save(model_path)
        # Save the scaler for inference
        import joblib
        joblib.dump(scaler, model_path.replace('.h5', '_scaler.joblib'))
        
    else:
        # load the model
        from keras.models  import load_model
        import joblib
        model = load_model(model_path)
        scaler = joblib.load(model_path.replace('.h5', '_scaler.joblib'))
        
        # make predictions
        X = df.drop(columns=['target'])
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled).flatten()
        return predictions
