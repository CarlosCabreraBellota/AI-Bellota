#%%
# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
from tabula.io import read_pdf
from nltk.tokenize import word_tokenize
import os
import re
from gmft.pdf_bindings import PyPDFium2Document
from gmft.auto import TableDetector, AutoTableFormatter, AutoFormatConfig
# Descargar el tokenizador de nltk si es necesario
import nltk
import tabula
nltk.download('punkt')
# Configure table detector and formatter
detector = TableDetector()
config = AutoFormatConfig()
config.semantic_spanning_cells = True
config.enable_multi_header = True
formatter = AutoTableFormatter(config)

#%% ESTRATEGIA 1: albert soler ferreteria y ESTRATEGIAS BIGMAT
pdf_path = "2024-12-09_4000000001_24025439.pdf" #2024-12-09_4000000001_24025439 #j3495473output_1
def extract_tables_to_dataframes(pdf_path):
    """
    Extract tables from a PDF using gmft and return them as pandas DataFrames.
    """
    # Open the PDF document
    doc = PyPDFium2Document(pdf_path)
    dataframes = []

    try:
        for page in doc:
            # Detect tables on the page
            tables = detector.extract(page)
            for table in tables:
                try:
                    # Format and convert each table to a DataFrame
                    formatted_table = formatter.extract(table)
                    df = formatted_table.df()
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error formatting a table: {e}")
    finally:
        doc.close()

    return dataframes
tables = extract_tables_to_dataframes(pdf_path = pdf_path)

filtered_tables=[]
for i in range(len(tables) - 1):  
    if len(tables[i].columns) > 5 and len(tables[i]) > 3:
        print(f"Table {i} with more than 4 columns:")
        print(tables[i])
        filtered_tables.append(tables[i])
        
filtered_tables = pd.concat(filtered_tables)
filtered_tables = filtered_tables.reset_index()
filtered_tables = filtered_tables.drop('index', axis = 1)
filtered_tables = filtered_tables.reset_index(drop='True')

#filtered_tables = filtered_tables.reset_index
print(filtered_tables)
filtered_tables.to_excel('pruebita.xlsx')
#base_columns = filtered_tables[0].columns
#filtered_same_structure = [df for df in filtered_tables if list(df.columns) == list(base_columns)]

# Palabras clave relacionadas con "cantidad" en varios idiomas y abreviaturas
keywords = ["quantity", "qty", "cantidad", "cant", "cant.","CANTIDAD UD.","quantité", "qté", "unidades","und",
            "quantidade", "qtd", "quantità", "qtà"]     

matched_columns = [col for col in filtered_tables.columns if any(keyword.lower() in col.lower() for keyword in keywords)]

cantidad_ref = filtered_tables[matched_columns].dropna().replace("un", "", regex=True)
#cantidad_ref = cantidad_ref.dropna().replace("un", "", regex=True)

excel_path = "Libro1.xlsx"  # Ruta del archivo Excel
df_referencias = pd.read_excel(excel_path)
df_referencias = df_referencias['CODART']

#normalización de datos de la tabla

columna_objetivo = None
for col in filtered_tables.columns:
    if "REF.PROVEEDOR" in col.upper() or "REFERENCIA" in col.upper() or "REF.PROV" in col.upper() or "REF.PROV." in col.upper() or "VUESTRA REF." in col.upper() or "REFERENCIA" in col.upper():
        columna_objetivo = col
        break

# Guardar en otra variable si existe
if columna_objetivo:
    referencias = filtered_tables[columna_objetivo]
    print(f"Se encontró la columna: {columna_objetivo}")
else:
    print("No se encontró ninguna columna.")
    def buscar_patron_en_dataframe(df):
        """
        Busca un patrón específico en todas las celdas de un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame con los datos.
            patron (str): Patrón regex para buscar.

        Returns:
            pd.DataFrame: DataFrame con las coincidencias encontradas.
        """
        patron = r'Ref\.Proveedor:\s*([\w\s-]+)'
        # Función para procesar cada celda
        def buscar_patron_celda(celda):
            if isinstance(celda, str):  # Verificar que la celda sea una cadena
                match = re.search(patron, celda)
                if match:
                    return match.group(0)  # Devolver el texto que coincide con el patrón
            return None  # Si no hay coincidencia, devolver None

            # Aplicar la búsqueda del patrón a todas las celdas
        return df.map(buscar_patron_celda).dropna(axis=1)


#%%
def buscar_patron_en_dataframe(df):
    """
    Busca un patrón específico en todas las celdas de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        patron (str): Patrón regex para buscar.

    Returns:
        pd.DataFrame: DataFrame con las coincidencias encontradas.
    """
    patron = r'Ref\.Proveedor:\s*([\w\s-]+)'
    # Función para procesar cada celda
    def buscar_patron_celda(celda):
        if isinstance(celda, str):  # Verificar que la celda sea una cadena
            match = re.search(patron, celda)
            if match:
                return match.group(0)  # Devolver el texto que coincide con el patrón
        return None  # Si no hay coincidencia, devolver None

    # Aplicar la búsqueda del patrón a todas las celdas
    return df.map(buscar_patron_celda).dropna(axis=1)

referencias = buscar_patron_en_dataframe(filtered_tables)


#referencias = filtered_tables.apply(lambda col: col.apply(extraer_referencias))        

#%%


referencias=[]    
def extraer_referencia(texto):
     if isinstance(texto, str):  # Verificar que el valor sea una cadena
         match = re.search(r'Ref\.Proveedor:\s*([\w\s-]+)', texto)  #Ref\.Proveedor:\s*([\w\s]+) Ref\.Proveedor:\s*(\S+)
         if match:
             return match.group(1)  # Devolver solo lo que está después
         if not match and columna_objetivo in filtered_tables.columns:
             filtered_tables[columna_objetivo]
             return filtered_tables[columna_objetivo]
         elif len(filtered_tables[columna_objetivo]) > 0:
            return None  # Devolver None si no hay coincidencia
    


referencias == filtered_tables.map(extraer_referencia).dropna(axis=1)

referencias = extraer_referencia(filtered_tables, columna_objetivo)
filtered_tables['REFERENCIAS'] = referencias

#%%
pedido = pd.concat([referencias,cantidad_ref], axis=1)
# Mostrar el DataFrame resultante


excel_filename = os.path.splitext(os.path.basename(pdf_path))[0] + "_pedido.xlsx"
pedido.to_excel(excel_filename, index=False)

#%% estrategia 2: BIGMAT

pdf_path = "a4034881output.pdf"

dfs = tabula.read_pdf(pdf_path, pages = "all", stream=True, multiple_tables=True)
# read_pdf returns list of DataFrames
print(len(dfs))
dfs

def extract_tables_to_dataframes(pdf_path):
    """
    Extract tables from a PDF using gmft and return them as pandas DataFrames.
    """
    # Open the PDF document
    doc = PyPDFium2Document(pdf_path)
    dataframes = []

    try:
        for page in doc:
            # Detect tables on the page
            tables = detector.extract(page)
            for table in tables:
                try:
                    # Format and convert each table to a DataFrame
                    formatted_table = formatter.extract(table)
                    df = formatted_table.df()
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error formatting a table: {e}")
    finally:
        doc.close()

    return dataframes

tables = extract_tables_to_dataframes(pdf_path = pdf_path)


#si solo tenemos una table
filtered_tables=[]
if len(tables) == 1 and len(tables[0].columns) > 5 and len(tables[0]) > 3:
    tables = pd.DataFrame.from_dict(tables[0])
    filtered_tables = tables

else:
    for i in range(len(tables) - 1):  
        if len(tables[i].columns) > 5 and len(tables[i]) > 3:
            print(f"Table {i} with more than 4 columns:")
            print(tables[i])
            filtered_tables.append(tables[i])
     

#%%

# for i in range(len(tables) - 1):
#     if len(tables)  <= 1:
#        filtered_tables = tables

#     else:
#         filtered_tables = pd.concat(filtered_tables)
#         filtered_tables = filtered_tables.reset_index()
#         filtered_tables = filtered_tables.drop('index', axis = 1)
#         filtered_tables = filtered_tables.reset_index(drop='True')
        
# Palabras clave relacionadas con "cantidad" en varios idiomas y abreviaturas
keywords = ["quantity", "qty", "cantidad", "cant", "cant.","CANTIDAD UD.","quantité", "qté", "unidades","und",
            "quantidade", "qtd", "quantità", "qtà"]     

matched_columns = [col for col in filtered_tables.columns if any(keyword.lower() in col.lower() for keyword in keywords)]
cantidad_ref = filtered_tables[matched_columns]

print(filtered_tables)
columna_objetivo = None
for col in filtered_tables.columns:
    if "REF.PROVEEDOR" in col.upper() or "REFERENCIA" in col.upper() or "REF.PROV" in col.upper() or "VUESTRA REF." in col.upper() or "REFERENCIA" in col.upper():
        columna_objetivo = col
        break
# %%
