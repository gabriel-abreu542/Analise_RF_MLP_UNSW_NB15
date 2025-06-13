from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import numpy as np
import pandas as pd
import time
import sys

# Funcao para realizar o Label Encoding em colunas categoricas
def label_encode(treino, teste):
    categorical_columns = treino.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(list(treino[col].astype(str).values) + list(teste[col].astype(str).values))
        treino[col] = le.transform(list(treino[col].astype(str).values))
        teste[col] = le.transform(list(teste[col].astype(str).values))
    return treino, teste

def normalizacao(treino, teste, scaler):    
    # Separar os dados em X e y
    X_treino = treino.drop('attack_cat', axis=1)
    y_treino = treino['attack_cat'] 
    
    X_teste = teste.drop('attack_cat', axis=1)
    y_teste = teste['attack_cat'] 
    
    # Aplicar o MinMaxScaler nas variaveis de entrada
    X_treino_normalizado = scaler.fit_transform(X_treino)
    X_teste_normalizado = scaler.transform(X_teste)
    
    # Retornar os dados normalizados, mantendo as attack_cats originais
    treino_normalizado = pd.DataFrame(X_treino_normalizado, columns=X_treino.columns)
    teste_normalizado = pd.DataFrame(X_teste_normalizado, columns=X_teste.columns)
    
    # Adicionar as colunas de attack_cats de volta aos datasets normalizados
    treino_normalizado['attack_cat'] = y_treino
    teste_normalizado['attack_cat'] = y_teste
    
    return treino_normalizado, teste_normalizado

def avaliar_dataset_RF(treino, teste, n_arvores):
    X_treino = treino.drop('attack_cat', axis=1)
    y_treino = treino['attack_cat']
    X_teste = teste.drop('attack_cat', axis=1)
    y_teste = teste['attack_cat']

    print("Classes no treino:", np.unique(y_treino))
    print("Classes no teste :", np.unique(y_teste))

    modelo = RandomForestClassifier(random_state=42, n_estimators=n_arvores)

    # Treinar o modelo
    inicio = time.time()
    modelo.fit(X_treino, y_treino)
    fim = time.time()

    tempo = fim - inicio

    # Fazer previsoes
    y_pred = modelo.predict(X_teste)

    acc = metrics.accuracy_score(y_teste, y_pred)
    pre = metrics.precision_score(y_teste, y_pred, average='weighted', zero_division=np.nan)
    rec = metrics.recall_score(y_teste, y_pred, average='weighted', zero_division=np.nan)
    f1 = metrics.f1_score(y_teste, y_pred, average='weighted')

    print("\nRANDOM FOREST:")
    print(f"({n_arvores} Árvores)")
    print(f"Resultados:")

    print(f"Accuracy: {acc}")
    print(f"Precision: {pre}")
    print(f"Recall: {rec}")
    print(f"F1-score: {f1}")

    auc_roc = roc_auc_score(y_teste, modelo.predict_proba(X_teste), average='weighted', multi_class='ovr')
    print("AUC-ROC: ", auc_roc)

    print(f"Tempo de treino: {tempo} segundos")
    
    
def avaliar_dataset_MLP(treino, teste, camadas):
    X_treino = treino.drop('attack_cat', axis=1)
    y_treino = treino['attack_cat']
    X_teste = teste.drop('attack_cat', axis=1)
    y_teste = teste['attack_cat']

    modelo = MLPClassifier(hidden_layer_sizes=(camadas))

    inicio = time.time()
    # Treinar o modelo
    modelo.fit(X_treino, y_treino)
    fim = time.time()

    tempo = fim - inicio

    # Fazer previsoes
    y_pred = modelo.predict(X_teste)    

    acc = metrics.accuracy_score(y_teste, y_pred)
    pre = metrics.precision_score(y_teste, y_pred, average='weighted', zero_division=np.nan)
    rec = metrics.recall_score(y_teste, y_pred, average='weighted', zero_division=np.nan)
    f1 = metrics.f1_score(y_teste, y_pred, average='weighted')

    print("\nMULTI-LAYER PERCEPTRON:")
    print(f"(Camadas: {camadas})")
    print(f"Resultados :")

    print(f"Accuracy: {acc}")
    print(f"Precision: {pre}")
    print(f"Recall: {rec}")
    print(f"F1-score: {f1}")

    auc_roc = roc_auc_score(y_teste, modelo.predict_proba(X_teste), average='weighted', multi_class='ovr')
    print("AUC-ROC: ", auc_roc)
    print(f"Tempo de treino: {tempo} segundos")
    
# Carregar os dados
treino = pd.read_csv(r"UNSW_NB15\UNSW_NB15_training-set.csv")
teste = pd.read_csv(r"UNSW_NB15\UNSW_NB15_testing-set.csv")

# stdout_orig = sys.stdout

# with open('resultados/resultadosUNSW_NB15_Attack_cat.txt', 'w') as f:
#     sys.stdout = f

dist_treino = treino.groupby('attack_cat').size()
dist_teste = teste.groupby('attack_cat').size()
# print("Distribuição dos registros: \n")
# print(f"Treino: {dist_treino[0]} registros normais, {dist_treino[1]} registros de ataque")
# print(f"Teste: {dist_teste[0]} registros normais, {dist_teste[1]} registros de ataque")
# print(f"Treino: {dist_treino}")
# print(f"Teste: {dist_teste}")

# Remover as colunas 'label' e 'id'
drop_columns = ['label', 'id']
treino_orig = treino.drop(drop_columns, axis=1, inplace=False)
teste_orig = teste.drop(drop_columns, axis=1, inplace=False)

colunas_vazias_treino = treino.columns[treino.isnull().all()].tolist()
colunas_vazias_teste = teste.columns[teste.isnull().all()].tolist()

print("\nColunas VAZIAS:")
print("Treino:")
print(colunas_vazias_treino)
print("Teste:")
print(colunas_vazias_teste)

if(len(colunas_vazias_treino) > 0):
    treino = treino.drop(colunas_vazias_treino, axis=1, inplace=False)
if(len(colunas_vazias_teste) > 0):
    teste = teste.drop(colunas_vazias_teste, axis=1, inplace=False)


# Aplicar Label Encoding
treino_orig, teste_orig = label_encode(treino_orig, teste_orig)

print("\nRESULTADOS ANTES DA NORMALIZAÇÃO:")

# avaliar_dataset_MLP(treino_orig, teste_orig, (10,10,20))
# avaliar_dataset_MLP(treino_orig, teste_orig, (20,40,20))
# avaliar_dataset_MLP(treino_orig, teste_orig, (40,60,40))
# avaliar_dataset_MLP(treino_orig, teste_orig, (60,40,60))
# avaliar_dataset_MLP(treino_orig, teste_orig, (60,80,60))
# avaliar_dataset_MLP(treino_orig, teste_orig, (80,100,80))
# avaliar_dataset_MLP(treino_orig, teste_orig, (100,120,100))
# avaliar_dataset_MLP(treino_orig, teste_orig, (16,8))
# avaliar_dataset_MLP(treino_orig, teste_orig, (32,16))
# avaliar_dataset_MLP(treino_orig, teste_orig, (64,32))
# avaliar_dataset_MLP(treino_orig, teste_orig, (100, 80, 60, 40, 20))
# avaliar_dataset_MLP(treino_orig, teste_orig, (128, 128, 64, 64))


# avaliar_dataset_RF(treino_orig, teste_orig, 10)
# avaliar_dataset_RF(treino_orig, teste_orig, 25)
# avaliar_dataset_RF(treino_orig, teste_orig, 50)
# avaliar_dataset_RF(treino_orig, teste_orig, 75)
# avaliar_dataset_RF(treino_orig, teste_orig, 100)
# avaliar_dataset_RF(treino_orig, teste_orig, 125)
# avaliar_dataset_RF(treino_orig, teste_orig, 150)

print("============================================================================================================")
print("RESULTADOS APÓS A NORMALIZAÇÃO MINMAX")
# Aplicar normalização MinMaxScaler
treino_normalizado, teste_normalizado = normalizacao(treino_orig, teste_orig, MinMaxScaler())

# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (16,8))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (32,16))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (64,32))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (10,20,10))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (20,40,20))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (40,60,40))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (60,40,60))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (60,80,60))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (80,100,80))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (100,120,100))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (100, 80, 60, 40, 20))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (128, 128, 64, 64))

# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 10)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 25)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 50)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 75)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 100)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 125)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 150)

print("============================================================================================================")
print("RESULTADOS APÓS A NORMALIZAÇÃO STANDARD (Z-SCORE)")
# Aplicar normalização StandardScaler
treino_normalizado, teste_normalizado = normalizacao(treino_orig, teste_orig, StandardScaler())

# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (10,20,10))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (20,40,20))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (40,60,40))  
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (60,40,60))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (60,80,60))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (80,100,80))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (100,120,100))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (16,8))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (32,16))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (64,32))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (100, 80, 60, 40, 20))
# avaliar_dataset_MLP(treino_normalizado, teste_normalizado, (128, 128, 64, 64))

# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 10)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 25)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 50)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 75)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 100)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 125)
# avaliar_dataset_RF(treino_normalizado, teste_normalizado, 150)

# sys.stdout = stdout_orig

# print("Execução finalizada. Resultados salvos em 'resultados/resultadosUNSW_NB15_Attack_cat.txt'")