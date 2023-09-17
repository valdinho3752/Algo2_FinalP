import glob
from langchain.document_loaders import CSVLoader
from clustering import DataProcessor


class LoadData:

  def __init__(self):
    self.csvLoad = DataProcessor('movies_metadata.csv')
    self.csvLoad.load_data()
    self.csvLoad.preprocess_data()
    self.csvLoad.apply_dbscan()
    self.csvLoad.save_results('resultados_clusters.csv')
    self.docs = self.load_data('resultados_clusters.csv')
          
  def load_data(self, file):
    loader = CSVLoader(file_path=file, encoding="utf-8", csv_args={'delimiter': ','})
    docs = loader.load()
    return docs

loader = LoadData()
print(loader.docs)
# print(type(loader.docs))
# print(type(loader.docs[0]))