# import parameters
from sys import argv
script, csv_train, csv_test = argv

#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#import data
df_train = pd.read_csv(csv_train)
df_test = pd.read_csv(csv_test)



def preprocess(df):
  '''
  Add X and y columns to dataframe
  '''
  
  #convert timestamps
  df['event time:timestamp'] = pd.to_datetime(df['event time:timestamp'], format='%d-%m-%Y %H:%M:%S', exact=False)
  df['case REG_DATE'] = pd.to_datetime(df['case REG_DATE'], format='%Y-%m-%dT%H:%M:%S', exact=False)
  df['case REG_DATE'] = df['case REG_DATE'].apply(lambda x: x.replace(tzinfo=None))
  df['case REG_DATE'] = df['case REG_DATE'].apply(lambda x: x.replace(microsecond=0))
  df['time_after_event_start'] = (df['event time:timestamp'] - df['case REG_DATE'])/pd.to_timedelta('1Min')

  df = df.sort_values(by = ['event time:timestamp'])
    
  #add traceposition
  df['tracePosition'] = 0
  for trace in list(df['case concept:name'].unique()):
    df_trace = df[df['case concept:name'] == trace]
    for position in range(len(df_trace)):
      eventID = df_trace['eventID '].tolist()[position]
      df.at[df.index[df['eventID '] == eventID].tolist()[0], 'tracePosition'] = position

  #add weekdays
  df['dayOfWeek'] = df['event time:timestamp'].dt.dayofweek
  df['weekday'] = np.where(df['dayOfWeek'] <= 4, 1, 0)
  df['saturday'] = np.where(df['dayOfWeek'] == 5, 1, 0)
  df['sunday'] = np.where(df['dayOfWeek'] == 6, 1, 0)

  #create dummies
  df_dummies = pd.get_dummies(df['event concept:name'])
  df = df.join(df_dummies)
  df_dummies = pd.get_dummies(df['event lifecycle:transition'])
  df = df.join(df_dummies)

  #encode columns
  le = LabelEncoder()
  df['activity_name'] = df['event concept:name'] + df['event lifecycle:transition']
  df['event concept:name_trans'] = le.fit_transform(df['event concept:name']) + 1 
  df['event lifecycle:transition_trans'] = le.fit_transform(df['event lifecycle:transition']) + 1
  df['activity_transformed'] = le.fit_transform(df['activity_name']) + 1

  #caculate target variables
  target_time = []
  target_activity = []
  target_activity_life = []
  df = df.sort_values(['case concept:name', 'event time:timestamp'])

  for i in range(len(df)):
    if i == len(df)-1: 
      target_time.append(0)
      target_activity.append(0)
      target_activity_life.append(0)
      break
    if df.iloc[i]['case concept:name'] == df.iloc[i+1]['case concept:name']:
      target_time.append(df.iloc[i+1]['time_after_event_start'] - df.iloc[i]['time_after_event_start'])
      target_activity.append(df.iloc[i+1]['event concept:name'])
      target_activity_life.append(df.iloc[i+1]['activity_transformed'])
    else: 
      target_time.append(0)
      target_activity.append(0)
      target_activity_life.append(0)

  #add target variables
  df['true_activity'] = target_activity
  df['true_time'] = target_time
  df['true_activity_life'] = target_activity_life

  return df

df_train = preprocess(df_train)
df_test = preprocess(df_test)



def activityPredictionBaseline(df_train, df_test):
  '''
  return list of case id of the longest trace and its length
  '''
  train_cases = list(df_train['case concept:name'].unique())
  
  longestTraceLength = 0
  for trace in train_cases:
    traceLength = len(df_train[df_train['case concept:name'] == trace])
    if traceLength > longestTraceLength:
        longestTraceLength = traceLength


  activity_list = [[] for i in range(longestTraceLength)]
  time_list = [[df_train.loc[df_train.index[0], 'event time:timestamp'] - df_train.loc[df_train.index[0], 'event time:timestamp']] for i in range(longestTraceLength)]
  for trace in train_cases:
    df_cases = df_train[df_train['case concept:name'] == trace]
    for i in range(len(df_cases)):
      activity_list[i].append(df_cases.loc[df_cases.index[i], 'activity_transformed'])

  common = []
  for activity in activity_list:
    c = Counter(activity)
    common.append(c.most_common(1)[0][0])

  df_train['activity_pred_baseline'] = 99 # all values will be overwritten
  df_train['activity_pred_baseline'] = 99 # all values will be overwritten

  for i in range(len(df_train)):
    if df_train['tracePosition'][i] == len(common) - 1:
      df_train.at[i, 'activity_pred_baseline'] = 0
    else:
      df_train.at[i, 'activity_pred_baseline'] = common[df_train['tracePosition'][i] + 1]
  
  for i in range(len(df_test)):
    df_test.at[i, 'activity_pred_baseline'] = common[df_test['tracePosition'][i] + 1]

  return df_train, df_test

df_train, df_test = activityPredictionBaseline(df_train, df_test)

accuracy_train = accuracy_score(df_train['activity_pred_baseline'], df_train['true_activity_life'])
accuracy_test = accuracy_score(df_test['activity_pred_baseline'], df_test['true_activity_life'])
print(f"Baseline activity-lifecycle prediction accuracy train set: {round(accuracy_train, 3)}")
print(f"Baseline activity-lifecycle prediction accuracy test set: {round(accuracy_test, 3)}")




def timePredictionBaseline(df_train, df_test):
  '''
  predict time until next activity by calculating
  mean time until next activity for given trace position
  '''

  train_cases = list(df_train['case concept:name'].unique())

  longest_trace_length = 0
  for trace in train_cases:
    trace_length = len(df_train[df_train['case concept:name'] == trace])
    if trace_length > longest_trace_length:
        longest_trace_length = trace_length

  time_list = [[df_train.loc[df_train.index[0], 'event time:timestamp'] - df_train.loc[df_train.index[0], 'event time:timestamp']] for i in range(longest_trace_length)]
  for trace in train_cases:
    df_cases = df_train[df_train['case concept:name'] == trace]
    for i in range(1, len(df_cases)):
      tdelta = df_cases.loc[df_cases.index[i], 'event time:timestamp'] - df_cases.loc[df_cases.index[i-1], 'event time:timestamp']
      time_list[i - 1].append(tdelta)

  avg_time = []
  for i in range(len(time_list)):
    sum = df_train.loc[df_train.index[0], 'event time:timestamp'] - df_train.loc[df_train.index[0], 'event time:timestamp']
    for tdelta in time_list[i]:
      sum += tdelta
    avg_time.append(sum/len(time_list[i]))

  df_train['time_pred_baseline'] = 0
  df_test['time_pred_baseline'] = 0

  for i in range(len(df_train)):
    minutes = pd.Timedelta(avg_time[df_train['tracePosition'][i]]) / np.timedelta64(1, 'm')
    df_train.at[i, 'time_pred_baseline'] = minutes

  for i in range(len(df_test)):
    minutes = pd.Timedelta(avg_time[df_test['tracePosition'][i]]) / np.timedelta64(1, 'm')
    df_test.at[i, 'time_pred_baseline'] = minutes

  return df_train, df_test

df_train, df_test = timePredictionBaseline(df_train, df_test)

rmse_train = mse(df_train['true_time'], df_train['time_pred_baseline'], squared=False)
rmse_test = mse(df_test['true_time'], df_test['time_pred_baseline'], squared=False)
print(f"Baseline time prediction RMSE train set: {round(rmse_train, 1)}")
print(f"Baseline time prediction RMSE test set: {round(rmse_test, 1)}")



def activityPredictionRF(df_train, df_test):
  '''
  Predict the combination of next activity and next lifecycle transition
  '''

  predictors = ['case AMOUNT_REQ', 'event concept:name_trans', 'event lifecycle:transition_trans', 'time_after_event_start']

  X_train = df_train[predictors].to_numpy()
  X_test = df_test[predictors].to_numpy()
  y_train = df_train['true_activity_life']
  y_test = df_test['true_activity_life']

  rf = RandomForestClassifier(n_estimators=30, max_depth=15, min_samples_leaf=2, min_impurity_decrease=0.000001, random_state=40)
  rf.fit(X_train, y_train)

  df_train['activity_pred_rf'] = rf.predict(X_train)
  df_test['activity_pred_rf'] = rf.predict(X_test)

  return df_train, df_test

df_train, df_test = activityPredictionRF(df_train, df_test)

accuracy_train = accuracy_score(df_train['activity_pred_rf'], df_train['true_activity_life'])
accuracy_test = accuracy_score(df_test['activity_pred_rf'], df_test['true_activity_life'])
print(f"Random Forest activity-lifecycle prediction accuracy train set: {round(accuracy_train, 3)}")
print(f"Random Forest activity-lifecycle prediction accuracy test set: {round(accuracy_test, 3)}")




def timePredictionOLS(df_train, df_test):
  '''
  Predict time until next activity in minutes for OLS regression and 
  '''

  predictors = ['A_ACCEPTED', 'A_ACTIVATED', 'A_APPROVED', 'A_CANCELLED', 'A_DECLINED', 'A_FINALIZED', 'A_PARTLYSUBMITTED', 'A_PREACCEPTED', 'A_REGISTERED', 'A_SUBMITTED', 
                'O_ACCEPTED', 'O_CANCELLED', 'O_CREATED', 'O_DECLINED', 'O_SELECTED', 'O_SENT', 'O_SENT_BACK', 
                'W_Afhandelen leads', 'W_Beoordelen fraude', 'W_Completeren aanvraag', 'W_Nabellen incomplete dossiers', 'W_Nabellen offertes', 'W_Valideren aanvraag', 
                'COMPLETE', 'SCHEDULE', 'START', 'case AMOUNT_REQ', 'time_after_event_start', 'tracePosition', 'weekday', 'saturday', 'sunday']

  X_train = df_train[predictors].to_numpy()
  X_test = df_test[predictors].to_numpy()

  reg = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
  reg.fit(X_train, df_train['true_time'])

  df_train['time_pred_OLS'] = reg.predict(X_train)
  df_test['time_pred_OLS'] = reg.predict(X_test)

  #change negative values to 0
  df_train.loc[df_train['time_pred_OLS'] < 0, 'time_pred_OLS'] = 0
  df_test.loc[df_test['time_pred_OLS'] < 0, 'time_pred_OLS'] = 0

  return df_train, df_test

df_train, df_test = timePredictionOLS(df_train, df_test)

rmse_train = mse(df_train['true_time'], df_train['time_pred_OLS'], squared=False)
rmse_test = mse(df_test['true_time'], df_test['time_pred_OLS'], squared=False)
print(f"OLS time prediction RMSE train set: {round(rmse_train, 1)}")
print(f"OLS time prediction RMSE test set: {round(rmse_test, 1)}")




def activityPredictionRFGS(df_train, df_test):
  '''
  Predict the combination of next activity and next lifecycle transition
  '''

  predictors = ['case AMOUNT_REQ', 'event concept:name_trans', 'event lifecycle:transition_trans', 'time_after_event_start', 'dayOfWeek', 'tracePosition', 'activity_transformed']

  X_train = df_train[predictors].to_numpy()
  X_test = df_test[predictors].to_numpy()
  y_train = df_train['true_activity_life']
  y_test = df_test['true_activity_life']

  rf = RandomForestClassifier(max_depth=15, max_features=6, min_samples_leaf=2, min_impurity_decrease=0.000001, random_state=40)
  rf.fit(X_train, y_train)

  df_train['activity_pred_rfgs'] = rf.predict(X_train)
  df_test['activity_pred_rfgs'] = rf.predict(X_test)

  return df_train, df_test

df_train, df_test = activityPredictionRFGS(df_train, df_test)

accuracy_train = accuracy_score(df_train['activity_pred_rfgs'], df_train['true_activity_life'])
accuracy_test = accuracy_score(df_test['activity_pred_rfgs'], df_test['true_activity_life'])
print(f"Random Forest (Grid Searched) activity-lifecycle prediction accuracy train set: {round(accuracy_train, 3)}")
print(f"Random Forest (Grid Searched) activity-lifecycle prediction accuracy test set: {round(accuracy_test, 3)}")





def timePredictionXGB(df_train, df_test):
  '''
  Predict time until next activity in minutes for OLS regression and 
  '''

  predictors = ['case AMOUNT_REQ', 'event concept:name_trans', 'event lifecycle:transition_trans', 'time_after_event_start', 'tracePosition', 'dayOfWeek', 'activity_pred_rfgs']

  X_train = df_train[predictors].to_numpy()
  X_test = df_test[predictors].to_numpy()

  xgbr = XGBRegressor(n_estimators=30, max_depth=6, objective='reg:squarederror')
  xgbr.fit(X_train, df_train['true_time'])

  df_train['time_pred_XGB'] = xgbr.predict(X_train)
  df_test['time_pred_XGB'] = xgbr.predict(X_test)

  df_train.loc[df_train['time_pred_XGB'] < 0, 'time_pred_XGB'] = 0
  df_test.loc[df_test['time_pred_XGB'] < 0, 'time_pred_XGB'] = 0

  return df_train, df_test

df_train, df_test = timePredictionXGB(df_train, df_test)

rmse_train = mse(df_train['true_time'], df_train['time_pred_XGB'], squared=False)
rmse_test = mse(df_test['true_time'], df_test['time_pred_XGB'], squared=False)
print(f"XGBoost time prediction RMSE train set: {round(rmse_train, 1)}")
print(f"XGBoost time prediction RMSE test set: {round(rmse_test, 1)}")



#output datasets
df_train_output = pd.read_csv(csv_train)
df_train_output['activity_pred_baseline'] = df_train['activity_pred_baseline']
df_train_output['time_pred_baseline'] = df_train['time_pred_baseline']
df_train_output['activity_pred_RF'] = df_train['activity_pred_rf']
df_train_output['time_pred_OLS'] = df_train['time_pred_OLS']
df_train_output['activity_pred_RFGS'] = df_train['activity_pred_rfgs']
df_train_output['time_pred_XGB'] = df_train['time_pred_XGB']

df_test_output = pd.read_csv(csv_test)
df_test_output['activity_pred_baseline'] = df_test['activity_pred_baseline']
df_test_output['time_pred_baseline'] = df_test['time_pred_baseline']
df_test_output['activity_pred_RF'] = df_test['activity_pred_rf']
df_test_output['time_pred_OLS'] = df_test['time_pred_OLS']
df_test_output['activity_pred_RFGS'] = df_test['activity_pred_rfgs']
df_test_output['time_pred_XGB'] = df_test['time_pred_XGB']

df_train_output.to_csv('train_output.csv')
df_train_output.to_csv('test_output.csv')