import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Function to prepare data, except for Unemployment data
def PrepareMacro(Macro_Data,Begin_Year,Begin_Month,Name_col,Name_Var):
    
    #Initilising the data
    month = Begin_Month
    dates = []  #list to store dates
    values = [] #list to store values
    shape = Macro_Data.shape
    n_columns = shape[1]
    col = 1

    #Loop to extract the data
    for i in range(0,n_columns+1):
        year = (Begin_Year + i) % 100
    
        
        while month <= 12:
            if col == n_columns:
                break
            else:
                year_string = str("{:02d}".format(year))
                month_string = str(month)


                col_name = Name_col + year_string + 'M' + month_string
                A = Macro_Data[col_name]
                B = pd.value_counts(A.isna().values)
            
                if A.count() == shape[0]: #Check for when no NaNs
                    values.append(A.iloc[-1])
                else:
                    values.append(A[B.iloc[1]-1])

                if year >= Begin_Year:
                    dates.append('19'+year_string+'-'+month_string)
                else:
                    dates.append('20'+year_string+'-'+month_string)

                month += 1
                col += 1
        month = 1

    #Saving everything in a dataframe
    d = {'Dates':dates, Name_Var:values}
    y = pd.DataFrame(data=d)
    y['Dates'] = pd.to_datetime(y['Dates'], format='%Y-%m').dt.to_period('M')
    
    return y


def read_merge_prepare_data(forecast_period, Macro_Data):
    # Read CSV files
    forecast_file_path = f"data/processed_data/{forecast_period}.csv"
    df = pd.read_csv(forecast_file_path)


    # Merge DataFrames on Dates and drop unnecessary columns
    Merged_Data = pd.merge(df, Macro_Data[['GDP_log_return', 'Cons_log_return', 'IPT_log_return', 'Unempl', 'Dates']],
                           right_on='Dates', left_on='rankdate').drop('Dates', axis=1)

    Merged_Data['Date'] = pd.to_datetime(Merged_Data['rankdate'], format='%Y-%m').dt.to_period('M')
    Merged_Data = Merged_Data[(Merged_Data['Date'].dt.year >= 1985) & (Merged_Data['Date'].dt.year <= 2019)].drop(['rankdate'], axis=1)

    # Preparing the datasets
    Merged_Data.sort_values(by='Date', ascending=True, inplace=True)

    # Columns to drop
    columns_to_drop = ['cfacshr', 'Unnamed: 0', 'ticker', 'cusip', 'cname', 'fpedats', 'statpers', 'anndats_act', 'fpi', 'actual', 'meanest']
    Merged_Data.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Missing values per column
    missing_values = Merged_Data.isna().sum()
    print(f"{forecast_period} Missing Values:")
    print(missing_values[missing_values > 0])

    # Drop rows with missing values 
    Merged_Data.dropna(axis=0, inplace=True)

    print(f' len extreme data droped {len(Merged_Data.loc[abs(Merged_Data.adj_actual) > 10])}')
    Merged_Data = Merged_Data.loc[abs(Merged_Data['adj_actual']) < 10]
    print(f'len data {len(Merged_Data)}')

    return Merged_Data


def train_test_random_forest_rolling(period, data_frame):
    # Filter data for training and testing based on date (train on 1988, test after; except A2, train on 2 years)
    data_frame = data_frame[(data_frame['Date']>= '1985-01') & (data_frame['Date']<= '2019-12' )]
    start_train = pd.to_datetime('1985-01', format='%Y-%m').to_period('M')
    y_hat_test = pd.Series()
    length_train = 11 # 12 months, hence add 11 to first month 
    n_loops = 408
    if period == 'A2':
        length_train = 23 # 24 months 
        n_loops = 396

    for i in range(0, n_loops): # till 12-2019 420 months; last for loop 420-12= 408, for A2 = 420-24=396
        train_start_date = (start_train.to_timestamp() + pd.DateOffset(months=i)).to_period('M')
        train_end_date = (start_train.to_timestamp() + pd.DateOffset(months=length_train+i)).to_period('M')
        train_data = data_frame[(data_frame['Date'] >= train_start_date) & (data_frame['Date'] <= train_end_date)]
        print(f'train btw {train_start_date} and {train_end_date}')

        test_date = (start_train.to_timestamp() + pd.DateOffset(months=length_train + 1 + i)).to_period('M')
        test_data = data_frame[data_frame['Date'] == test_date]
        print(f'test on {test_date}')

        if len(test_data)!=0:
            print(period)
            print(f'test data length {len(test_data)}')
            # Separate predictors and target variable
            y_train = train_data['adj_actual']
            X_train_full = train_data.loc[:, ~train_data.columns.isin(['adj_actual'])]

            X_test_full = test_data.loc[:, ~test_data.columns.isin(['adj_actual'])]

            X_train = X_train_full.drop(['Date', 'permno'], axis=1)
            X_test = X_test_full.drop(['Date', 'permno'], axis=1)
            print(f' min y train {y_train.min()}')
            print(f' max y train {y_train.max()}')
            # Instantiate a RandomForestRegressor 
            forest_model_rf = RandomForestRegressor(n_estimators=2000, max_depth=7, max_samples=0.01,  min_samples_leaf=5,  n_jobs=-1) #max_samples=sample_fractions[period]
            print("Training Random Forest")
            forest_model_rf.fit(X_train, y_train)
            print("Random Forest training completed")

            y_hat_test = pd.concat([y_hat_test, pd.Series(forest_model_rf.predict(X_test))])
            print(f' len prediciton {len(y_hat_test.values)}')

    # Dataframe with permno, date, predictor, real value and predicted value
    result_df = pd.DataFrame(data_frame[(data_frame['Date']>= '1986-01') & (data_frame['Date']<= '2019-12') ])
    if period == 'A2':
        result_df = pd.DataFrame(data_frame[(data_frame['Date']>= '1987-01') & (data_frame['Date']<= '2019-12') ])
    result_df['predicted_adj_actual'] = y_hat_test.values

    result_df['biased_expectation'] = (result_df.adj_meanest - result_df.predicted_adj_actual) / result_df.price

    return result_df