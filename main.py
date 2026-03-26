import wfdb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Records
records = ['100', '101']

features = []
labels = []

window_size = 50
step_size = 10

for rec in records:
    record = wfdb.rdrecord(f'data/raw/hrv/{rec}')
    signal = record.p_signal[:, 0]

    for i in range(0, len(signal) - window_size, step_size):
        window = signal[i:i+window_size]

        diff_signal = np.diff(window)

        sdnn = np.std(diff_signal)
        rmssd = np.sqrt(np.mean(diff_signal**2))

        features.append([sdnn, rmssd])

        noise = np.random.normal(0, 0.002)
        rmssd_noisy = rmssd + noise

        if rmssd_noisy < 0.01:
            labels.append(0)
        elif rmssd_noisy < 0.03:
            labels.append(1)
        else:
            labels.append(2)

X = np.array(features)
y = np.array(labels)

print("Total samples:", len(X))


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X_scaled, y)


X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# Models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC()
knn = KNeighborsClassifier()

# Train
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)


rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)
knn_pred = knn.predict(X_test)


print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))

def get_frs_label(pred):
    if pred == 0:
        return "LOW"
    elif pred == 1:
        return "MEDIUM"
    else:
        return "HIGH"

# Show sample predictions
print("\nSample Fatigue Risk Predictions:")

for i in range(10):
    pred = svm_pred[i]   # using best model (SVM)
    print(f"Sample {i+1}: Risk Level → {get_frs_label(pred)}")