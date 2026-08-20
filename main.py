import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px

df_churn = pd.read_csv('churn_clientes.csv')

X = df_churn.drop('churn', axis=1)
y = df_churn['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Acurácia no Treino: {model.score(X_train, y_train):.2%}")
print(f"Acurácia no Teste:  {accuracy_score(y_test, y_pred):.2%}\n")

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))


df_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importancia': model.feature_importances_
}).sort_values(by='Importancia', ascending=True)

fig = px.bar(
    df_imp, 
    x='Importancia', 
    y='Feature', 
    orientation='h',
    title='<b>Importância das Variáveis - Random Forest (Churn)</b>',
    labels={'Importancia': 'Peso no Modelo', 'Feature': 'Variável'},
    color='Importancia',
    color_continuous_scale='Teal'
)

fig.update_layout(width=700, height=400)
fig.show()