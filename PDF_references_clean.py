#%%
# Importar bibliotecas necesarias
import pandas as pd
from tabula.io import read_pdf
from nltk.tokenize import word_tokenize
import os
from gmft.pdf_bindings import PyPDFium2Document
from gmft.auto import TableDetector, AutoTableFormatter, AutoFormatConfig

# Descargar el tokenizador de nltk si es necesario
import nltk
nltk.download('punkt')

#%%
# Configure table detector and formatter
detector = TableDetector()
config = AutoFormatConfig()
config.semantic_spanning_cells = True
config.enable_multi_header = True
formatter = AutoTableFormatter(config)

# 1. Leer las referencias desde el archivo Excel
references_df = pd.read_excel('Libro1.xlsx')['CODART']

# Normalizar y limpiar el DataFrame
references_df = (references_df
                 .astype(str)  # Convertir a cadena
                 .str.strip()  # Eliminar espacios al principio y al final
                 .str.replace(" ", "")  # Eliminar espacios internos
                 .str.replace("-", "")  # Eliminar guiones
                 .str.upper())  # Convertir a mayúsculas

# Eliminar valores nulos y duplicados
references_df = references_df.dropna().drop_duplicates()
# Convertir a lista
references = references_df.tolist()

# 2. Leer y combinar las tablas del PDF
pdf = "ordine del 01-12-2023 Spluga Srl.pdf"  # Archivo PDF con las referencias
tablas = read_pdf(pdf, pages='all', multiple_tables=True)  # Leer todas las tablas
tablas_combined = pd.concat(tablas, ignore_index=True)  # Combinar todas las tablas en un DataFrame

# Convertir todas las celdas del DataFrame en una lista única
pdf_references = (tablas_combined.stack()
                  .astype(str)  # Convertir a cadenas
                  .unique())  # Obtener valores únicos

# Limpiar y normalizar las referencias
pdf_references = [
    ref.strip().replace("–", "").replace(":", " ").replace(".", " ").lower()
    for ref in pdf_references if ref != 'nan'
]

# Tokenizar las referencias
tokenized_references = [word_tokenize(ref) for ref in pdf_references]
flattened_references = [token for sublist in tokenized_references for token in sublist]

# Crear un DataFrame para referencias tokenizadas
tokenized_references_df = pd.DataFrame(flattened_references, columns=['reference'])
tokenized_references_df['reference'] = tokenized_references_df['reference'].str.replace("-", "", regex=True)
tokenized_references_df = tokenized_references_df.dropna().drop_duplicates()

# Normalizar las referencias del Excel
normalized_excel_references = [ref.replace("-", "").lower() for ref in references]
df_normalized_excel_references = pd.DataFrame(normalized_excel_references, columns=['reference'])
df_normalized_excel_references['reference'] = df_normalized_excel_references['reference'].str.strip()

# Encontrar elementos en común entre ambos DataFrames
coincidencias = pd.merge(tokenized_references_df, df_normalized_excel_references, on='reference')

# Eliminar duplicados en las coincidencias
df_sin_duplicados = coincidencias.drop_duplicates(subset='reference', keep='first')

# Exportar las coincidencias a Excel
excel_filename = os.path.splitext(os.path.basename(pdf))[0] + "_pedido.xlsx"
df_sin_duplicados.to_excel(excel_filename, index=False)

# Mostrar resultados
print(f"Coincidencias encontradas: {len(df_sin_duplicados)}")
print("DataFrame sin duplicados:")
print(df_sin_duplicados)

# Función para extraer tablas del PDF usando gmft
def extract_tables_to_dataframes(pdf_path):
    """
    Extrae tablas de un PDF usando gmft y las retorna como DataFrames de pandas.
    """
    doc = PyPDFium2Document(pdf_path)  # Abrir el documento PDF
    dataframes = []

    try:
        for page in doc:
            # Detectar tablas en la página
            tables = detector.extract(page)
            for table in tables:
                try:
                    # Formatear y convertir cada tabla a DataFrame
                    formatted_table = formatter.extract(table)
                    df = formatted_table.df()
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error formateando una tabla: {e}")
    finally:
        doc.close()

    return dataframes

# Extraer tablas del PDF
tables = extract_tables_to_dataframes(pdf_path = pdf)

# Verificar y combinar tablas
if tables and len(tables) > 0:
    tablas_2d = []
    for tabla in tables:
        if isinstance(tabla, pd.DataFrame):
            tablas_2d.append(tabla)
        elif isinstance(tabla, list):
            try:
                tablas_2d.append(pd.DataFrame(tabla))
            except Exception as e:
                print(f"Error al convertir tabla a DataFrame: {e}")

    if tablas_2d:
        tablas_combinadas = pd.concat(tablas_2d, ignore_index=True)

# Mostrar información de las tablas combinadas
print(tablas_combinadas.columns)

# Palabras clave relacionadas con "cantidad"
keywords = ["quantity", "qty", "cantidad", "cant", "quantité", "qté", "quant",
            "quantidade", "qtd", "quantità", "qtà"]

# Función para seleccionar el DataFrame con columnas relacionadas con palabras clave
def seleccionar_dataframe(variables):
    """
    Selecciona un DataFrame que contenga columnas relacionadas con palabras clave.
    """
    for nombre_variable, dataframe in variables.items():
        if isinstance(dataframe, pd.DataFrame):  # Verificar si es un DataFrame
            for col in dataframe.columns:
                if any(keyword.lower() in str(col).lower() for keyword in keywords):
                    print(f"Se encontró coincidencia en la variable '{nombre_variable}' en la columna '{col}'")
                    return dataframe, nombre_variable, col
        else:
            print(f"{nombre_va  riable} no es un DataFrame.")
    print("No se encontraron coincidencias en ninguna variable.")
    return None, None, None

# Diccionario con las variables a evaluar
variables = {
    "tablas_combinadas": tablas_combinadas,
    "tablas_combined": tablas_combined
}
print(tablas_combined)
# Seleccionar el DataFrame que contiene columnas relacionadas
df_seleccionado, variable_seleccionada, columna_seleccionada = seleccionar_dataframe(variables)
cantidades_ref = df_seleccionado[columna_seleccionada]
print(cantidades_ref)

if df_seleccionado is not None:
    print(f"DataFrame seleccionado: {variable_seleccionada}")
    print(f"Columna seleccionada: {columna_seleccionada}")
else:
    print("No se encontró ningún DataFrame con columnas relacionadas con 'cantidad'.")

cantidades_ref = df_seleccionado[columna_seleccionada]
#vamos a limpiar
cantidades_ref = cantidades_ref.dropna()
    
    # Función para limpiar una columna y extraer números
def limpiar_columna(column):
    def extraer_numeros(valor):
        if pd.isna(valor):
            return None
        valor_numerico = ''.join(char for char in str(valor) if char.isdigit() or char == ',')
        try:
            return float(valor_numerico.replace(',', '.')) if ',' in valor_numerico else int(valor_numerico)
        except ValueError:
            return None

    return column.apply(extraer_numeros)

cantidades_ref = limpiar_columna(cantidades_ref).reset_index(drop=True)
print('cantidad:', cantidades_ref)
    
# Reiniciar índices en df_sin_duplicados
df_sin_duplicados = df_sin_duplicados.reset_index(drop=True)

# Concatenar referencias únicas y cantidades
pedido = pd.concat([df_sin_duplicados, cantidades_ref], axis=1)
print(pedido)
    
# Exportar a Excel
excel_filename = os.path.splitext(os.path.basename(pdf))[0] + "_pedido.xlsx"
pedido.to_excel(excel_filename, index=False)
