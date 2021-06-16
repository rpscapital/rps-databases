import rps_databases
import pandas as pd

#Conexão com mysql
db = rps_databases.local_connect()

#Consulta tabela de produtos antiga
products = pd.read_sql('SELECT * FROM base_lista_ativos.base_lista_ativos', db.engine)

print(products)