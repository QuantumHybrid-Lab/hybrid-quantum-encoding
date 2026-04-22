import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. VERİYİ YÜKLE
# Dosya artık doğrudan bu kodun yanında olduğu için sadece ismini yazıyoruz
try:
    df_kanser = pd.read_csv('wdbc.data', header=None)
    print("Kanser veri seti başarıyla yüklendi.")
except Exception as e:
    print(f"Hata oluştu: {e}")
    exit()

# WDBC Veri Yapısı: 0: ID, 1: Hedef (M/B), 2-31: Özellikler [cite: 1]
X = df_kanser.iloc[:, 2:] 
y = df_kanser.iloc[:, 1]  

# Etiketleri sayısal yap (M=1, B=0)
y = y.map({'M': 1, 'B': 0})

# 2. EN ÖNEMLİ 10 ÖZELLİĞİ SEÇ
selector = RandomForestClassifier(random_state=42)
selector.fit(X, y)

feature_imp = pd.Series(selector.feature_importances_, index=X.columns)
top_10_features = feature_imp.nlargest(10).index.tolist()

print(f"\n--- SEÇİLEN 10 ÖZELLİK (İndeksler): {top_10_features} ---")

# 3. L2 NORMALİZASYON (Kuantum Genlik Kodlama Hazırlığı)
X_selected = X[top_10_features]
scaler = Normalizer(norm='l2')
X_scaled = scaler.fit_transform(X_selected)

# 4. BASELINE MODEL TESTİ (SVM)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

model = SVC(kernel='rbf', C=10)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print("\n" + "="*35)
print(f"KANSER VERİSİ BASELINE SONUCU")
print(f"Test Doğruluk Oranı: %{accuracy*100:.2f}")
print("="*35)