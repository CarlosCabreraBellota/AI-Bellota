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
pdf = "ordine del 01-12-2023 Spluga Srl.pdf" #Pedido de compras CIA.ESPAÑOLA AISLAMIENTOS S.A. Nº 644506 #P-2302490 #a4034881output #ordine del 01-12-2023 Spluga Srl
tablas = read_pdf(pdf, pages='all', multiple_tables=True)
tablas_combined = pd.concat(tablas, ignore_index=True)

# Convertimos todas las celdas a una lista única, eliminando NaN y espacios innecesarios
pdf_references = tablas_combined.stack().astype(str).unique()  # Usar stack y convertir a str
pdf_references = [ref.strip().replace("–", "").replace(":", " ").replace(".", " ").lower() for ref in pdf_references if ref != 'nan']
#pdf_references = [ref.strip().replace("–", "").replace(" ", "").replace(":", " ").replace(".", " ").lower() for ref in pdf_references if ref != 'nan']
# Tokenizar cada referencia alfanumérica
tokenized_references = [word_tokenize(ref) for ref in pdf_references]
#print(tokenized_references)

tokenized_references_df = pd.DataFrame(tokenized_references)
#print(tokenized_references_df)
tokenized_references_df = pd.DataFrame(tokenized_references_df.values.flatten(), columns=['reference'])

tokenized_references_df["reference"] = tokenized_references_df["reference"].replace({'–':''}, regex=True)
tokenized_references_df = tokenized_references_df.dropna()
tokenized_references_df = tokenized_references_df.drop_duplicates()
tokenized_references_df = tokenized_references_df["reference"].replace({'-':''}, regex=True)
#print(tokenized_references_df)

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
df_flattened_references['reference'] = df_flattened_references['reference'].str.replace("-", "", regex=True)

# Encontrar elementos en común entre ambos DataFrames
coincidencias = pd.merge(df_flattened_references, df_normalized_excel_references, on='reference')

# Mostrar las coincidencias
print(f"Coincidencias encontradas: {coincidencias}")
#coincidencias = coincidencias.drop_duplicates(subset='reference', keep='first')
df_sin_duplicados = coincidencias.drop_duplicates(subset='reference', keep='first')
# Mostrar el DataFrame sin duplicados
print("DataFrame sin duplicados:")
print(df_sin_duplicados)

excel_filename = os.path.splitext(os.path.basename(pdf))[0] + "_pedido.xlsx"
df_sin_duplicados.to_excel(excel_filename, index=False)

#prueba con gmft
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

tables = extract_tables_to_dataframes(pdf_path = pdf)
print(tables)


# Verificar si se extrajeron tablas
if tables and len(tables) > 0:
    # Aplanar estructura si hay dimensiones adicionales
    tablas_2d = []
    for tabla in tables:
        # Si una tabla es un DataFrame, agregarla directamente
        if isinstance(tabla, pd.DataFrame):
            tablas_2d.append(tabla)
        # Si una tabla es una lista o matriz, convertirla en DataFrame
        elif isinstance(tabla, list):
            try:
                tablas_2d.append(pd.DataFrame(tabla))
            except Exception as e:
                print(f"Error al convertir tabla a DataFrame: {e}")

    # Combinar las tablas en un solo DataFrame
    if tablas_2d:
        tablas_combinadas = pd.concat(tablas_2d, ignore_index=True)
        
print(tablas_combinadas.columns)
print(tablas_combined.columns)
print(tablas_combinadas)
print(tablas_combined)

#%%


# Palabras clave relacionadas con "cantidad" en varios idiomas y abreviaturas
keywords = ["quantity", "qty", "cantidad", "cant", "quantité", "qté",
            "quantidade", "qtd", "quantità", "qtà"]

# Función para seleccionar el DataFrame que tiene alguna de las palabras clave
def seleccionar_dataframe(variables):
    for nombre_variable, dataframe in variables.items():
        if isinstance(dataframe, pd.DataFrame):  # Verificar si es un DataFrame
            for col in dataframe.columns:
                if any(keyword.lower() in str(col).lower() for keyword in keywords):
                    print(f"Se encontró coincidencia en la variable '{nombre_variable}' en la columna '{col}'")
                    return dataframe, nombre_variable, col
        else:
            print(f"{nombre_variable} no es un DataFrame.")
    print("No se encontraron coincidencias en ninguna variable.")
    return None, None, None

# Diccionario con las variables a evaluar
variables = {
    "tablas_combinadas": tablas_combinadas,
    "tablas_combined": tablas_combined
}

# Seleccionar el DataFrame que contiene columnas relacionadas
df_seleccionado, variable_seleccionada, columna_seleccionada = seleccionar_dataframe(variables)

if df_seleccionado is not None:
    print(f"DataFrame seleccionado: {variable_seleccionada}")
    print(f"Columna seleccionada: {columna_seleccionada}")
else:
    print("No se encontró ningún DataFrame con columnas relacionadas con 'cantidad'.")

cantidades_ref = df_seleccionado[columna_seleccionada]
#vamos a limpiar
cantidades_ref = cantidades_ref.dropna()


def limpiar_columna(column):
    # Función auxiliar para limpiar cada celda
    def extraer_numeros(valor):
        if pd.isna(valor):  # Si es NaN, mantener como NaN
            return None
        else:
            # Extraer solo números y convertir a float o int
            valor_numerico = ''.join(char for char in str(valor) if char.isdigit() or char == ',')
            try:
                # Reemplazar comas con puntos decimales si es necesario
                return float(valor_numerico.replace(',', '.')) if ',' in valor_numerico else int(valor_numerico)
            except ValueError:
                return None  # En caso de error, retornar None

    # Aplicar la limpieza a toda la columna
    return column.apply(extraer_numeros)

cantidades_ref = limpiar_columna(cantidades_ref)
cantidades_ref = cantidades_ref.dropna()
cantidades_ref = pd.DataFrame(cantidades_ref)
cantidades_ref = cantidades_ref.reset_index(drop='True')

df_sin_duplicados = df_sin_duplicados.reset_index(drop='True')

excel_filename = os.path.splitext(os.path.basename(pdf))[0] + "_pedido.xlsx"
pedido = pd.concat([df_sin_duplicados, cantidades_ref],axis=1)
pedido.to_excel(excel_filename, index=False)


