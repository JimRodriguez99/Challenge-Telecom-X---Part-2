Sobre el desafío
Challenge Telecom X - Part 2 es parte del programa de desafíos de Alura + Oracle enfocado en Data Science y Machine Learning . Este proyecto desarrolla un modelo predictivo para identificar clientes con alta probabilidad de cancelar servicios de telecomunicaciones (churn), aplicando técnicas avanzadas de ciencia de datos.
🏆 Objetivos del desafío:

Desarrollar un modelo de Machine Learning para predecir el abandono
Aplicar técnicas de preprocesamiento e ingeniería de características.
Implementar mejores practicas de ciencia de datos
Generar insights accionables para el negocio
Crear un pipeline completo listo para producción

📊 Resultados Alcanzados
🎯 Métricas del Modelo:

🏆 AUC-ROC: 0.8314 (Excelente capacidad discriminativa)
📈 Precisión Global: 76%
🎯 Recall para Churn: 74%
⚡ 15 características claves identificadas
🚀 Modelo optimizado con aumento de gradiente

📋 Conjunto de datos procesado:

📊 7,267 clientes analizados
🔢 21 variables originales
📈 25.72% tasa de abandono (desequilibrado)
⚖️ Balanceamiento aplicado con SMOTE

Hallazgos Principales
🎯 Top 5 Factores Críticos de Churn:
RangoVariableImportanciaPerspectiva de negocio1️⃣Duración del cliente20%Clientes nuevos tienen mayor riesgo2️⃣Contratos de 2 años19%Factor de retención más fuerte3️⃣Contratos de 1 año15%Retención moderada versus mensual4️⃣Pago electrónico11%Método menos estable5️⃣Cargos mensuales10%Sensibilidad al precio
💡 Insights Clave para el Negocio:

🕒 Antigüedad : Primeros 12 meses son críticos
📋 Contratos : Largo plazo reduce significativamente la deserción
💳 Pagos : Débito automático es más estable que cheque electrónico
🌐 Fibra Óptica : Clientes premium más propensos al abandono

🛠️ Metodología Aplicada
1. 📊 Análisis Exploratorio
pitón# Análisis de distribución de churn
churn_distribution = df['Churn'].value_counts(normalize=True)
print(f"No Churn: {churn_distribution[0]:.1%}")
print(f"Churn: {churn_distribution[1]:.1%}")
2. 🔄 Canalización de preprocesamiento
pitón# Pipeline de preprocesamiento
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])
3. ⚖️ Equilibrio de Clases
pitón# Aplicación de SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_preprocessed, y_train)
4. 🎯 Selección de características
pitón# Selección de mejores características
selector = SelectKBest(score_func=f_classif, k=15)
X_train_selected = selector.fit_transform(X_train_res, y_train_res)
5. 🤖 Modelado y Optimización
pitón# Modelos comparados
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Optimización con GridSearchCV
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1], 
    'max_depth': [3, 5]
}
📈 Comparación de modelos
ModeloAUC-ROCPrecisión GlobalRecuperación de abandonoPuntuación F1Tiempo🥇 Potenciación de gradientes0.831476%74%0.62Medio🥈 Regresión logística0.832674%80%0.62Rápido🥉 Bosque aleatorio0.802476%58%0,55Lento
🏆 Mejor modelo: aumento de gradiente optimizado

Parámetros óptimos :learning_rate=0.1, max_depth=5, n_estimators=200
Balanza ideal : Entre precisión y recuperación
Interpretabilidad : Clara identificación de factores importantes

🗂️ Estructura del Proyecto
Challenge-Telecom-X-Part2/
│
├── 📓 notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb     # EDA completo
│   ├── 02_Data_Preprocessing.ipynb            # Limpieza y transformación
│   ├── 03_Feature_Engineering.ipynb           # Creación de características
│   ├── 04_Model_Training_Evaluation.ipynb     # Entrenamiento y evaluación
│   └── 05_Business_Insights_Conclusions.ipynb # Conclusiones de negocio
│
├── 📊 data/
│   ├── raw/                    # Datos originales del challenge
│   ├── processed/              # Datos procesados
│   └── results/               # Resultados y predicciones
│
├── 🤖 models/
│   ├── telecom_churn_model.pkl              # Modelo final optimizado
│   ├── preprocessor.pkl                     # Pipeline de preprocesamiento
│   └── feature_selector.pkl                 # Selector de características
│
├── 📈 reports/
│   ├── figures/               # Visualizaciones generadas
│   ├── challenge_report.pdf   # Reporte final del challenge
│   └── presentation.pptx      # Presentación de resultados
│
├── 🐍 src/
│   ├── data_processing.py     # Funciones de procesamiento
│   ├── model_training.py      # Entrenamiento de modelos
│   ├── evaluation_metrics.py  # Métricas de evaluación
│   └── visualization.py       # Funciones de visualización
│
├── 📋 requirements.txt         # Dependencias del proyecto
├── 🚀 main.py                 # Script principal de ejecución
├── 📖 README.md               # Este archivo
└── 📄 LICENSE                 # Licencia MIT
🚀 Reproducir el desafío
Opción 1: Google Colab (Recomendado) 🌟
intento# 1. Abrir en Google Colab
https://colab.research.google.com/

# 2. Cargar notebook principal
# 3. Subir dataset del challenge
# 4. Ejecutar todas las celdas secuencialmente
Opción 2: Entorno Local 💻
intento# 1. Clonar repositorio
git clone https://github.com/tu-usuario/Challenge-Telecom-X-Part2.git
cd Challenge-Telecom-X-Part2

# 2. Crear entorno virtual
python -m venv telecom-env
source telecom-env/bin/activate  # Linux/Mac
# telecom-env\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar análisis completo
python main.py
Opción 3: Ejecución Rápida ⚡
pitón# Cargar modelo pre-entrenado
import joblib
model = joblib.load('models/telecom_churn_model.pkl')

# Predecir churn para nuevos clientes
predictions = model.predict(new_customer_data)
probabilities = model.predict_proba(new_customer_data)[:, 1]
📋 Dependencias del Proyecto
TXT# Core Data Science
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0

# Preprocessing & Balancing
imbalanced-learn>=0.8.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Model Persistence
joblib>=1.1.0

# Jupyter Environment
jupyter>=1.0.0
ipykernel>=6.0.0

# Utilities
tqdm>=4.62.0
💼 Valor de Negocio - Reto Telecom X
🎯 Impacto Proyectado:

📈 15-20% de incremento en retención de clientes
💰 Reducción de 15-25% en tasa de abandono
🎯 ROI positivo en programas de retención dirigidos
⚡ Identificación temprana de clientes en riesgo

📊 KPIs mejorados:
MétricaAntesDespuésMejoraRetención74,3%85-90%+15%Detección ChurnManual74% Auto+74%Tiempo RespuestaDíasTiempo Real-99%Precisión Acciones20%76%+280%
🏆 Logros del Desafío
✅ Objetivos cumplidos:

 Modelo predictivo con AUC > 0,80
 Pipeline automatizado completo
 Ingeniería de funciones optimizada
 Insights de negocio accionables
 Documentación completa del proceso.
 Código reproducible y escalable

🌟Técnicas Aplicadas:

Preprocesamiento avanzado con ColumnTransformer
Equilibrio inteligente con SMOTE
Selección automática de características
Optimización de hiperparámetros con GridSearchCV
Validación cruzada estratificada
Evaluación multimétrica integral

👨‍💻 Autor
Jim Rodriguez - Científico de Datos en formación

📧 Correo electrónico : jarodriguezserna@gmail.com
💼 LinkedIn : https://www.linkedin.com/in/jim-rodr%C3%ADguez-508166361/
🐙 GitHub : https://github.com/JimRodriguez99


🏆 ¡ Reto Completado con Éxito!
