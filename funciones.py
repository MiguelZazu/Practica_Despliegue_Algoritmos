from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def cargar_datos():
    datos = load_wine()
    X = datos['data']
    y = datos['target']
    return X, y, datos['feature_names'], datos['target_names']

def preprocesar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def entrenar_modelo(X_train, y_train, modelo='lr', random_state=42):
    if modelo == 'lr':
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    elif modelo == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError("Modelo no soportado. Usa 'lr' o 'rf'.")
    clf.fit(X_train, y_train)
    return clf

def evaluar_modelo(modelo, X_test, y_test):
    pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average='macro')
    rec = recall_score(y_test, pred, average='macro')
    reporte = classification_report(y_test, pred, output_dict=True)
    return acc, prec, rec, reporte