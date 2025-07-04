import argparse
import mlflow
from funciones import cargar_datos, preprocesar_datos, entrenar_modelo, evaluar_modelo
from sklearn.model_selection import train_test_split

def main(modelo, random_state):
    mlflow.set_experiment("Clasificacion_Vino_Script")
    run_name=f'Clasificacion_Vino-{modelo}-{random_state}'
    with mlflow.start_run(run_name=run_name):
        X, y, features, target_names = cargar_datos()
        X_scaled, scaler = preprocesar_datos(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)

        clf = entrenar_modelo(X_train, y_train, modelo=modelo, random_state=random_state)
        acc, prec, rec, reporte = evaluar_modelo(clf, X_test, y_test)

        mlflow.log_param("modelo", modelo)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
        print("Reporte:", reporte)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", type=str, required=True, help="Modelo a usar: 'lr' o 'rf'.")
    parser.add_argument("--random", type=int, default= 42, required=True, help="Valor para random_state (entero).")
    args = parser.parse_args()
    main(args.modelo, args.random)