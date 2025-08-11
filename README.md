# Telecom X - Parte 2

## Propósito del Análisis

Modelo predictivo para identificar clientes con **alta probabilidad de cancelar servicios de telecomunicaciones (churn)** basado en variables demográficas, de servicio y facturación. El objetivo principal es predecir el comportamiento de abandono para implementar estrategias de retención proactivas.

## Estructura del Proyecto

```
telecom-x-part2/
├── notebooks/
│   └── telecom_churn_analysis.ipynb    # Cuaderno principal
├── data/
│   ├── raw/
│   │   └── telecom_data.csv           # Datos originales
│   └── processed/
│       └── telecom_processed.csv      # Datos preprocesados
├── models/
│   ├── telecom_churn_model.pkl        # Modelo entrenado
│   └── preprocessor.pkl               # Pipeline de preprocesamiento
├── visualizations/
│   ├── churn_distribution.png         # Distribución de churn
│   ├── correlation_matrix.png         # Matriz de correlación
│   └── feature_importance.png         # Importancia de variables
└── README.md
```

## Preparación de los Datos

### Clasificación de Variables

**Variables Numéricas:**
- `tenure`: Duración del cliente (meses)
- `MonthlyCharges`: Cargos mensuales
- `TotalCharges`: Cargos totales

**Variables Categóricas:**
- `gender`: Género del cliente
- `Contract`: Tipo de contrato (Month-to-month, One year, Two year)
- `PaymentMethod`: Método de pago
- `InternetService`: Tipo de servicio de internet

### Proceso de Preprocesamiento

1. **Limpieza de datos:**
   - Conversión de `TotalCharges` a numérico
   - Tratamiento de valores faltantes

2. **Codificación:**
   ```python
   # Variables numéricas: StandardScaler
   # Variables categóricas: OneHotEncoder
   preprocessor = ColumnTransformer([
       ('num', StandardScaler(), numerical_features),
       ('cat', OneHotEncoder(drop='first'), categorical_features)
   ])
   ```

3. **División de datos:**
   - **Entrenamiento**: 80% (5,814 registros)
   - **Prueba**: 20% (1,453 registros)
   - Estratificación por variable target para mantener proporción de churn

4. **Balanceamiento:**
   - Aplicación de SMOTE debido al desbalance de clases (25.72% churn)

### Justificaciones de Modelización

- **Gradient Boosting** seleccionado por mejor balance precisión-recall
- **Feature selection** (15 variables) para reducir overfitting
- **Validación cruzada** estratificada para evaluación robusta
- **GridSearchCV** para optimización de hiperparámetros

## Insights del EDA

### Gráficos Principales

1. **Distribución de Churn:**
   - 74.28% clientes activos vs 25.72% churn
   - Desbalance significativo requiere técnicas de balanceamiento

2. **Variables Críticas:**
   - Clientes con contratos month-to-month: 42.7% churn rate
   - Contratos de 2 años: solo 2.8% churn rate
   - Primeros 12 meses: período de mayor riesgo

3. **Correlaciones Relevantes:**
   - Correlación negativa entre tenure y churn (-0.35)
   - Clientes con fibra óptica muestran mayor propensión al abandono

## Resultados del Modelo

- **AUC-ROC**: 0.8314
- **Precisión**: 76%
- **Recall**: 74%
- **F1-Score**: 0.62

### Variables Más Importantes
| Variable | Importancia | Interpretación |
|----------|-------------|----------------|
| tenure | 20% | Clientes nuevos mayor riesgo |
| Contract_Two year | 19% | Contratos largos retienen más |
| Contract_One year | 15% | Retención moderada |
| PaymentMethod_Electronic check | 11% | Método menos estable |
| MonthlyCharges | 10% | Sensibilidad al precio |

## Instrucciones de Ejecución

### Instalación de Dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn joblib
```

### Ejecución del Cuaderno

1. **Carga de datos:**
   ```python
   # Datos originales
   df = pd.read_csv('data/raw/telecom_data.csv')
   
   # Datos preprocesados (opcional)
   df_processed = pd.read_csv('data/processed/telecom_processed.csv')
   ```

2. **Ejecución completa:**
   - Abrir `notebooks/telecom_churn_analysis.ipynb`
   - Ejecutar celdas secuencialmente
   - Los datos preprocesados se generan automáticamente

3. **Predicción con modelo entrenado:**
   ```python
   import joblib
   model = joblib.load('models/telecom_churn_model.pkl')
   predictions = model.predict_proba(new_data)[:, 1]
   ```

### Requisitos del Sistema
- Python 3.8+
- Jupyter Notebook o Google Colab
- RAM mínima: 4GB
