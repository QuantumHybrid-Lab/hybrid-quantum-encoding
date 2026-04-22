import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 1. VERİYİ YÜKLE VE TEMİZLE
df_kalp = pd.read_csv('processed.cleveland.data', header=None, na_values='?')
df_kalp = df_kalp.dropna()
X = df_kalp.iloc[:, :-1]
y = (df_kalp.iloc[:, -1] > 0).astype(int)

# 2. ÖN İŞLEME
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)

# 3. YÖNTEM 2: XGBOOST İLE ÖZELLİK SEÇİMİ
# XGBoost, özellik önem derecelerini daha keskin belirler
xgb_selector = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
xgb_selector.fit(X_std, y)

# En önemli 10 özelliğin indeksini alıyoruz
top_10_idx = pd.Series(xgb_selector.feature_importances_).nlargest(10).index
X_selected = X_std[:, top_10_idx]

print(f"\n--- XGBOOST SEÇİLEN 10 ÖZELLİK İNDEKSİ: {top_10_idx.tolist()} ---")

# 4. KUANTUM GENLİK HAZIRLIĞI (L2)
l2_scaler = Normalizer(norm='l2')
X_final = l2_scaler.fit_transform(X_selected)

# 5. MODEL TESTİ (SVC)
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.20, random_state=42)

# Modeli en iyi performansı verecek şekilde sıkı tutuyoruz
model = SVC(kernel='rbf', C=100, gamma='auto')
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("\n" + "="*35)
print(f"YÖNTEM 2 (XGBoost Hibrit) SONUCU")
print(f"Yeni Doğruluk Oranı: %{accuracy*100:.2f}")
print("="*35)