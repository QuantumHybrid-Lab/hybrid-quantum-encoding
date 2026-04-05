import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. VERİYİ YÜKLE
file_name = 'ObesityDataSet_raw_and_data_sinthetic.csv'
df = pd.read_csv(file_name)

# 2. HATA ÖNLEYİCİ SAYISALLAŞTIRMA
# Tüm sütunları kontrol et ve 'object' (metin) olanları sayıya çevir
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'string':
        # Önce tüm veriyi stringe çeviriyoruz, sonra sayısal etikete (0,1,2..) dönüştürüyoruz
        df[col] = le.fit_transform(df[col].astype(str))

# 3. NORMALİZASYON (Kuantum ve SVM için şart)
scaler = MinMaxScaler()
# Hedef değişken hariç her şeyi 0-1 arasına çekiyoruz
target = 'NObeyesdad'
features = df.drop(target, axis=1)
labels = df[target]

features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# 4. VERİYİ EĞİTİM VE TEST OLARAK AYIR
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# 5. KLASİK BASELINE: SVM MODELİ
# Hızlı sonuç için linear kernel kullanıyoruz
model_svm = SVC(kernel='linear')
model_svm.fit(X_train, y_train)
y_pred = model_svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("-" * 35)
print("DENEY BAŞARIYLA TAMAMLANDI!")
print(f"KLASİK MODEL (SVM) DOĞRULUK ORANI: %{accuracy*100:.2f}")
print("-" * 35)
print("\n 'Temizlenmiş Veri' hazır!")# Mevcut kodunun en altına ekle:
features_scaled['Target'] = labels.values
features_scaled.to_csv('temiz_obezite_verisi.csv', index=False)
print(" dosya oluşturuldu: temiz_obezite_verisi.csv")