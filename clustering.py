import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, data_file):
        self.data_file = data_file

    def load_data(self):
        # Carga el conjunto de datos
        self.df = pd.read_csv(self.data_file)

    def preprocess_data(self):
        # Filtra las características relevantes
        caracteristicas_relevantes = ['title', 'genres', 'vote_average', 'vote_count']

        # Crea un nuevo DataFrame con las características relevantes
        self.df_relevante = self.df[caracteristicas_relevantes]

        # Preprocesamiento de datos: Normaliza las características y realiza imputación
        scaler = StandardScaler()
        imputer = SimpleImputer(strategy='mean')

        # Selecciona las características numéricas a normalizar e imputar
        columnas_numericas = ['vote_average', 'vote_count']

        # Normaliza y luego imputa los valores faltantes
        self.df_relevante[columnas_numericas] = scaler.fit_transform(imputer.fit_transform(self.df_relevante[columnas_numericas]))

    def apply_dbscan(self, eps=0.5, min_samples=5):
        # Aplica DBSCAN a tus datos
        X = self.df_relevante[['vote_average', 'vote_count']]
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X)

        # Agrega los resultados de los clusters al DataFrame original
        self.df_relevante['cluster'] = clusters

    def save_results(self, output_file):
        # Guarda el DataFrame con las asignaciones de cluster en un archivo CSV
        self.df_relevante.to_csv(output_file, index=False)

# if __name__ == "__main__":
#     # Ejemplo de uso
#     data_processor = DataProcessor('movies_metadata.csv')
#     data_processor.load_data()
#     data_processor.preprocess_data()
#     data_processor.apply_dbscan()
#     data_processor.save_results('resultados_clusters.csv')
