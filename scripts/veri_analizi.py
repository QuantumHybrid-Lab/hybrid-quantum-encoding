import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. VERİYİ YÜKLE
df_temiz = pd.read_csv('data/obesity/temiz_obezite_verisi.csv')

X = df_temiz.iloc[:, :-1]
y = df_temiz.iloc[:, -1]

print(f"Veri yüklendi. Toplam sütun sayısı: {len(df_temiz.columns)}")

# 2. EN ÖNEMLİ 3 ÖZELLİĞİ SEÇ (Amplitude Encoding: 2^2=4 >= 3, 2 qubit)
selector = RandomForestClassifier(random_state=42)
selector.fit(X, y)

feature_imp = pd.Series(selector.feature_importances_, index=X.columns)
top_3_features = feature_imp.nlargest(3).index.tolist()

print(f"\n--- SEÇİLEN 3 ÖZELLİK: {top_3_features} ---")

# 3. L2 NORMALİZASYON (Amplitude Encoding için kareler toplamı = 1)
X_selected = X[top_3_features]
scaler = Normalizer(norm='l2')
X_scaled = scaler.fit_transform(X_selected)

# 4. BASELINE MODEL TESTİ (SVM)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

model = SVC(kernel='rbf', C=10)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("\n" + "="*35)
print(f"BASELINE MODEL SONUCU")
print(f"Test Doğruluk Oranı: %{accuracy*100:.2f}")
print("="*35)
