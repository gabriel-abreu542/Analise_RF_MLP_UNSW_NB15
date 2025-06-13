from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import metrics
import pandas as pd
import time

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

def contribuicao_por_feature(pca, features): 
    componentes = pd.DataFrame(
        pca.components_,
        columns=features,
        index=[f'PC{i+1}' for i in range(len(pca.components_))]
    )

    variancia_explicada = pca.explained_variance_ratio_

    contribuicao = (componentes.abs().T * variancia_explicada).sum(axis=1)
    contribuicao_ordenada = contribuicao.sort_values(ascending=False)
    
    return contribuicao_ordenada

def aplicar_PCA(treino, variancia_alvo=0.99):
    X_treino = treino.drop('attack_cat', axis=1)
    y_treino = treino['attack_cat']

    pca = PCA(n_components=variancia_alvo)
    X_treino_pca = pca.fit_transform(X_treino)

    colunas = [f'PC{i+1}' for i in range(X_treino_pca.shape[1])]
    treino_pca = pd.DataFrame(X_treino_pca, columns=colunas)

    treino_pca['attack_cat'] = y_treino.values

    return treino_pca, pca

def analisar_variancia_PCA(pca):
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var:.6f}")
    print(f"\nVariância total explicada pelos {len(pca.explained_variance_ratio_)} componentes: {sum(pca.explained_variance_ratio_):.6f}")

def analisar_componentes_PCA(pca, feature_names):
    componentes_df = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=[f'PC{i+1}' for i in range(len(pca.components_))]
    )
    for i in range(len(componentes_df)):
        print(f'\nComponente PC{i+1} mais influenciado por:')
        print(componentes_df.loc[f'PC{i+1}'].abs().sort_values(ascending=False))

# Carregar os dados
treino = pd.read_csv(r"UNSW_NB15\UNSW_NB15_training-set.csv")
teste = pd.read_csv(r"UNSW_NB15\UNSW_NB15_testing-set.csv") 

drop_columns = ['label', 'id']

treino_orig = treino.drop(drop_columns, axis=1, inplace=False)
teste_orig = teste.drop(drop_columns, axis=1, inplace=False)

# Aplicar Label Encoding
treino_orig, teste_orig = label_encode(treino_orig, teste_orig)

treino_z_score, teste_z_score = normalizacao(treino_orig, teste_orig, StandardScaler())

# PCA após normalização Z-Score
print("===================================================================")
print("Resultados do PCA após a normalização Z-Score:")
treino_pca_z, pca_z_score = aplicar_PCA(treino_z_score)

colunas = treino_z_score.drop('attack_cat', axis=1).columns
analisar_variancia_PCA(pca_z_score)
print("\nContribuição total de cada atributo para os componentes principais retidos:")
contribuicao = contribuicao_por_feature(pca_z_score, colunas)
print(contribuicao)

treino_min_max, teste_min_max = normalizacao(treino_orig, teste_orig, MinMaxScaler())

# PCA após normalização MinMax
print("===================================================================")
print("Resultados do PCA após a normalização MinMax:")
treino_pca_min_max, pca_min_max = aplicar_PCA(treino_min_max)

analisar_variancia_PCA(pca_min_max)
print("\nContribuição total de cada atributo para os componentes principais retidos:")
contribuicao = contribuicao_por_feature(pca_min_max, colunas)
print(contribuicao)

