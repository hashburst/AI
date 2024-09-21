from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Carica il dataset
df = pd.read_csv('data.csv')

# Dividi in input e output
X = df.drop('target', axis=1)
y = df['target']

# Divisione in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inizializza il modello
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Addestra il modello
clf.fit(X_train, y_train)

# Predice il set di test
y_pred = clf.predict(X_test)

# Valuta il modello
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
