import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. VERİYİ YÜKLE
ddf_temiz = pd.read_csv('temiz_obezite_verisi.csv')

# Hata almamak için sütun ismine bakmadan bölüyoruz:
# iloc[:, :-1] -> Son sütun hariç tüm sütunlar (Özellikler)
# iloc[:, -1]  -> Sadece son sütun (Hedef Değişken)
X = ddf_temiz.iloc[:, :-1]
y = ddf_temiz.iloc[:, -1]

print(f"Veri yüklendi. Toplam sütun sayısı: {len(ddf_temiz.columns)}")

# 2. EN ÖNEMLİ 10 ÖZELLİĞİ SEÇ
selector = RandomForestClassifier(random_state=42)
selector.fit(X, y)

feature_imp = pd.Series(selector.feature_importances_, index=X.columns)
top_10_features = feature_imp.nlargest(10).index.tolist()

print(f"\n--- SEÇİLEN 10 ÖZELLİK ---")
for i, feature in enumerate(top_10_features, 1):
    print(f"{i}. {feature}")

# 3. L2 NORMALİZASYON (Amplitude Encoding Hazırlığı)
X_selected = X[top_10_features]
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