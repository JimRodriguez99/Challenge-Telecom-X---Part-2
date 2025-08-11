# Telecom Churn Prediction

Modelo de Machine Learning para predecir cancelación de servicios de telecomunicaciones.

## Resultados

- **AUC-ROC**: 0.8314
- **Precisión**: 76%, Recall: 74%
- **Dataset**: 7,267 clientes, tasa churn: 25.72%

## Modelo

Gradient Boosting optimizado con GridSearchCV:
- `learning_rate=0.1, max_depth=5, n_estimators=200`
- Preprocesamiento: StandardScaler + OneHotEncoder
- Balanceamiento: SMOTE
- Feature selection: SelectKBest (15 features)

## Variables Importantes

| Variable | Importancia |
|----------|-------------|
| Duración cliente | 20% |
| Contratos 2 años | 19% |
| Contratos 1 año | 15% |
| Pago electrónico | 11% |
| Cargos mensuales | 10% |

## Uso

```bash
pip install -r requirements.txt
python main.py
```

```python
import joblib
model = joblib.load('models/telecom_churn_model.pkl')
predictions = model.predict_proba(data)[:, 1]
```

## Estructura

```
├── notebooks/          # Análisis y entrenamiento
├── models/             # Modelo y preprocessors
├── src/                # Scripts de procesamiento
└── data/              # Datasets
```
