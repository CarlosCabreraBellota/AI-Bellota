#%%
import tabula
from pycparser.ply.yacc import token
from tabula import read_pdf
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from thefuzz import fuzz
from thefuzz import process
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')


# 1. Leer las referencias de Excel
references = pd.read_excel('Libro1.xlsx')
references_df = references['CODART']
references_df = references_df.astype(str)
# Normalizar y limpiar references_df cargado desde Excel
references_df = references_df.str.strip()  # Eliminar espacios al principio y al final
references_df = references_df.str.replace(" ", "").str.replace("-", "").str.upper()  # Eliminar espacios internos, guiones, y convertir a mayúsculas
references_df = references_df.dropna().drop_duplicates()  # Eliminar valores nulos y duplicados
references = references['CODART'].astype(str).tolist()

# 2. Leer y combinar las tablas del PDF
pdf = "ordine del 01-12-2023 Spluga Srl.pdf" #Pedido de compras CIA.ESPAÑOLA AISLAMIENTOS S.A. Nº 644506 #P-2302490
tablas = read_pdf(pdf, pages='all', multiple_tables=True)
tablas_combined = pd.concat(tablas, ignore_index=True)

# Convertimos todas las celdas a una lista única, eliminando NaN y espacios innecesarios
pdf_references = tablas_combined.stack().astype(str).unique()  # Usar stack y convertir a str
pdf_references = [ref.strip().replace("–", "").lower() for ref in pdf_references if ref != 'nan']

# Tokenizar cada referencia alfanumérica
tokenized_references = [word_tokenize(ref) for ref in pdf_references]

tokenized_references_df = pd.DataFrame(tokenized_references)
print(tokenized_references_df)
tokenized_references_df = pd.DataFrame(tokenized_references_df.values.flatten(), columns=['reference'])

tokenized_references_df["reference"] = tokenized_references_df["reference"].replace({'–':''}, regex=True)
tokenized_references_df = tokenized_references_df.dropna()
tokenized_references_df = tokenized_references_df.drop_duplicates()
print(tokenized_references_df)

# Aplanar la lista de listas en una sola lista
flattened_references = [token for sublist in tokenized_references for token in sublist]

# Mostrar las primeras tokenizaciones como ejemplo
for i, tokens in enumerate(tokenized_references[:40]):  # Muestra las primeras 5 tokenizaciones
    print(f"Referencia {i+1}: {tokens}")

# Reunir los tokens nuevamente en una cadena (esto es necesario porque las referencias pueden estar divididas)
reconstructed_references = [''.join(tokens) for tokens in tokenized_references]

# 5. Normalizar las referencias del Excel (eliminar espacios y convertir a minúsculas)
normalized_excel_references = [ref.replace("-", "").lower() for ref in references]

# Convertir normalized_excel_references a un DataFrame
df_normalized_excel_references = pd.DataFrame(normalized_excel_references, columns=['reference'])
df_normalized_excel_references['reference'] = df_normalized_excel_references['reference'].str.strip()
# Convertir flattened_references a un DataFrame
df_flattened_references = pd.DataFrame(flattened_references, columns=['reference'])
df_flattened_references['reference'] = df_flattened_references['reference'].str.replace("–", "", regex=True)

# Encontrar elementos en común entre ambos DataFrames
coincidencias = pd.merge(df_flattened_references, df_normalized_excel_references, on='reference')

# Mostrar las coincidencias
print(f"Coincidencias encontradas: {coincidencias}")
#coincidencias = coincidencias.drop_duplicates(subset='reference', keep='first')
df_sin_duplicados = coincidencias.drop_duplicates(subset='reference', keep='first')
# Mostrar el DataFrame sin duplicados
print("DataFrame sin duplicados:")
print(df_sin_duplicados)