import rps_databases
import pandas as pd

#Conexão com postgres
db = rps_databases.cloud_connect()

#Consulta tabela de produtos
products = pd.read_sql('SELECT * FROM portfolio.products', db.engine)

print(products)