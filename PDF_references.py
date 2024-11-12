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

#%%

# 1. Leer las referencias de Excel
references = pd.read_excel('Libro1.xlsx')
references_df = references['CODART']
references_df = references_df.astype(str)
# Normalizar y limpiar references_df cargado desde Excel
references_df = references_df.str.strip()  # Eliminar espacios al principio y al final
references_df = references_df.str.replace(" ", "").replace("-", "").str.upper()  # Eliminar espacios internos, guiones, y convertir a mayúsculas
references_df = references_df.dropna().drop_duplicates()  # Eliminar valores nulos y duplicados
references = references['CODART'].astype(str).tolist()



# 2. Leer y combinar las tablas del PDF
pdf = "P-2302490.pdf"
tablas = read_pdf(pdf, pages='all', multiple_tables=True)
tablas_combined = pd.concat(tablas, ignore_index=True)

# Convertimos todas las celdas a una lista única, eliminando NaN y espacios innecesarios
pdf_references = tablas_combined.stack().astype(str).unique()  # Usar stack y convertir a str
pdf_references = [ref.strip() for ref in pdf_references if ref != 'nan']
pdf_references = [ref.replace("-", "").lower() for ref in pdf_references]
print(pdf_references)

# Tokenizar cada referencia alfanumérica
tokenized_references = [word_tokenize(ref) for ref in pdf_references]

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

# Encontrar elementos en común entre ambos DataFrames
coincidencias = pd.merge(df_flattened_references, df_normalized_excel_references, on='reference')

# Mostrar las coincidencias
print(f"Coincidencias encontradas: {coincidencias}")
#coincidencias = coincidencias.drop_duplicates(subset='reference', keep='first')
df_sin_duplicados = coincidencias.drop_duplicates(subset='reference', keep='first')
# Mostrar el DataFrame sin duplicados
print("DataFrame sin duplicados:")
print(df_sin_duplicados)


# 6. Comparar las referencias del PDF con las del Excel usando fuzzywuzzy
# Usamos fuzz.partial_ratio para obtener una medida de similitud entre las cadenas
threshold = 100  # Umbral de similitud para considerar que las referencias son iguales
matched_references = []

for pdf_ref in flattened_references:
    for excel_ref in normalized_excel_references:
        similarity = fuzz.partial_ratio(pdf_ref, excel_ref)  # Similaridad con partial_ratio
        if similarity >= threshold:  # Si la similitud es mayor que el umbral
            matched_references.append({
                'pdf_reference': pdf_ref,
                'excel_reference': excel_ref,
                'similarity': similarity
            })

# 7. Crear un DataFrame con las referencias similares
matched_df = pd.DataFrame(matched_references)



#Obtener las referencias únicas en 'excel_reference'
references_def = matched_df['excel_reference'].drop_duplicates()

# Convertir a mayúsculas
references_def = [ref.upper() for ref in references_def]

# Crear un DataFrame con las referencias únicas en mayúsculas
references_def = pd.DataFrame(references_def, columns=['excel_reference'])

# Eliminar duplicados si hubiera alguno y reiniciar el índice sin especificar el nivel
references_def = references_def.drop_duplicates().reset_index(drop=True)

# Asegúrate de que ambas series estén en mayúsculas para una comparación coherente
references_def['excel_reference'] = references_def['excel_reference'].str.upper()

# Filtrar el DataFrame eliminando cadenas con longitud menor a 4 en la columna 'excel_reference'
references_def = references_def[references_def['excel_reference'].str.len() >= 4]

# Reiniciar el índice del DataFrame después de filtrar
references_def = references_def.reset_index(drop=True)
references_df = references_df.str.upper()

# Filtrar coincidencias: selecciona solo las referencias en references_def que también están en references_df
coincidencias_df = references_def[references_def['excel_reference'].isin(references_df)]

# Mostrar el DataFrame con las coincidencias
print(coincidencias_df)

#extraer las cantidades y meter en el dataframe







