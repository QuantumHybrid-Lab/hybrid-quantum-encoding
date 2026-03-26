import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

cat_features = ['Gender','family_history_with_overweight',
                'FAVC','SMOKE','SCC','CAEC','CALC','MTRANS']

cont_features = ['Age','Height','Weight','FCVC',
                 'NCP','CH2O','FAF','TUE']

le = LabelEncoder()
for col in cat_features:
    df[col] = le.fit_transform(df[col])

scaler_cat = MinMaxScaler(feature_range=(0, np.pi))
X_angle = scaler_cat.fit_transform(df[cat_features])

scaler_cont = MinMaxScaler()
X_cont = scaler_cont.fit_transform(df[cont_features])
X_amplitude = normalize(X_cont, norm='l2')

le_target = LabelEncoder()
y = le_target.fit_transform(df['NObeyesdad'])

X_cat_train, X_cat_test, X_amp_train, X_amp_test, y_train, y_test = \
    train_test_split(X_angle, X_amplitude, y, test_size=0.2, random_state=42)

print("Kategorik train shape:", X_cat_train.shape)
print("Sürekli train shape:", X_amp_train.shape)
print("Sınıf sayısı:", len(np.unique(y)))
print("İlk satır açısal:", X_angle[0])
print("İlk satır genlik:", X_amplitude[0])