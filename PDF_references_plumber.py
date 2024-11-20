import pdfplumber
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
from thefuzz import fuzz, process
import re
import nltk
from nltk.tokenize import word_tokenize
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox

# Descargar el paquete necesario de NLTK
nltk.download('punkt')

# 1. Leer las referencias de Excel
references = pd.read_excel('Libro1.xlsx')
references_df = references['CODART']
references_df = references_df.astype(str)

# Normalizar y limpiar references_df cargado desde Excel
references_df = references_df.str.strip()  # Eliminar espacios al principio y al final
references_df = references_df.str.replace(" ", "").str.replace("-", "").str.upper()  # Eliminar espacios internos, guiones, y convertir a mayúsculas
references_df = references_df.dropna().drop_duplicates()  # Eliminar valores nulos y duplicados
references = references['CODART'].astype(str).tolist()

# 2. Leer y combinar las tablas del PDF usando pdfplumber
pdf_path = "ordine del 01-12-2023 Spluga Srl.pdf"  # Ruta del PDF

# Lista para almacenar todas las tablas extraídas del PDF
tables = []

# Abrir el PDF y extraer las tablas
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            # Convertir la tabla extraída en un DataFrame
            df = pd.DataFrame(table[1:], columns=table[0])  # Saltar la primera fila si es encabezado
            tables.append(df)

# Combinar todas las tablas en un único DataFrame
if tables:
    tablas_combined = pd.concat(tables, ignore_index=True)
else:
    tablas_combined = pd.DataFrame()

# Convertimos todas las celdas a una lista única, eliminando NaN y espacios innecesarios
pdf_references = tablas_combined.stack().astype(str).unique()  # Usar stack y convertir a str
pdf_references = [ref.strip().replace("–", "").replace(":", " ").replace(".", " ").lower() for ref in pdf_references if ref != 'nan']

# Tokenizar cada referencia alfanumérica
tokenized_references = [word_tokenize(ref) for ref in pdf_references]
print(tokenized_references)

# Convertir la lista tokenizada en un DataFrame
tokenized_references_df = pd.DataFrame(tokenized_references)
print(tokenized_references_df)
tokenized_references_df = pd.DataFrame(tokenized_references_df.values.flatten(), columns=['reference'])

# Limpiar datos en tokenized_references_df
tokenized_references_df["reference"] = tokenized_references_df["reference"].replace({'–': '', '-': ''}, regex=True)
tokenized_references_df = tokenized_references_df.dropna().drop_duplicates()

# Aplanar la lista de listas en una sola lista
flattened_references = [token for sublist in tokenized_references for token in sublist]

# Mostrar las primeras tokenizaciones como ejemplo
for i, tokens in enumerate(tokenized_references[:40]):  # Muestra las primeras 40 tokenizaciones
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

# Eliminar duplicados y mostrar el DataFrame sin duplicados
df_sin_duplicados = coincidencias.drop_duplicates(subset='reference', keep='first')
print("DataFrame sin duplicados:")
print(df_sin_duplicados)


def process_pdf(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    all_text = []

    for page in pdf.pages:
        filtered_page = page
        chars = filtered_page.chars

        for table in page.find_tables():
            first_table_char = page.crop(table.bbox).chars[0]
            filtered_page = filtered_page.filter(lambda obj:
                get_bbox_overlap(obj_to_bbox(obj), table.bbox) is None
            )
            chars = filtered_page.chars

            df = pd.DataFrame(table.extract())
            df.columns = df.iloc[0]
            markdown = df.drop(0).to_markdown(index=False)

            chars.append(first_table_char | {"text": markdown})

        page_text = extract_text(chars, layout=True)
        all_text.append(page_text)

    pdf.close()
    return "\n".join(all_text)


# Path to your PDF file
pdf_path = "ordine del 01-12-2023 Spluga Srl.pdf"
extracted_text = process_pdf(pdf_path)
print(extracted_text)

from unstract.llmwhisperer.client import LLMWhispererClient

client = LLMWhispererClient(base_url="https://llmwhisperer-api.unstract.com/v1", api_key="fb15dae601ce493aa4460f01852569b8")

# Get usage info
usage_info = client.get_usage_info()


result = client.whisper(file_path="ordine del 01-12-2023 Spluga Srl.pdf")
extracted_text = result["extracted_text"]
print(extracted_text)
print(isinstance(extracted_text, str))
df = pd.DataFrame(extracted_text)
print(df)

# # Función para encontrar valores en la columna de cantidades
# def extract_column_by_header(text, header="Quantità"):
#     # Dividir el texto en líneas
#     lines = text.splitlines()
#
#     # Buscar la línea con el encabezado
#     header_line = None
#     for i, line in enumerate(lines):
#         if header in line:
#             header_line = i
#             break
#
#     # Si no encontramos el encabezado, devolvemos una lista vacía
#     if header_line is None:
#         return []
#
#     # Dividir la línea del encabezado en columnas para determinar posiciones
#     header_columns = re.split(r'\s{2,}', lines[header_line])  # Separar por múltiples espacios
#     header_index = header_columns.index(header)
#
#     # Extraer la columna correspondiente de las líneas siguientes
#     quantities = []
#     for line in lines[header_line + 1:]:
#         # Dividir la línea en columnas
#         columns = re.split(r'\s{2,}', line)
#
#         # Asegurarse de que haya suficientes columnas y que la columna no sea vacía
#         if len(columns) > header_index and columns[header_index].strip().isdigit():
#             quantities.append(int(columns[header_index].strip()))
#
#     return quantities
#
# # Extraer las cantidades usando el encabezado "Quantità"
# quantities = extract_column_by_header(extracted_text, header="Quantità")
# print("Cantidades extraídas:", quantities)

#%%
import pdfplumber
import pandas as pd


def process_pdf_to_dataframe(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    all_tables = []  # Lista para almacenar todas las tablas como DataFrames

    for page in pdf.pages:
        filtered_page = page
        chars = filtered_page.chars

        # Iterar sobre las tablas detectadas en la página
        for table in page.find_tables():
            # Extraer la tabla y convertirla en un DataFrame
            df = pd.DataFrame(table.extract())

            # Configurar la primera fila como encabezados si corresponde
            df.columns = df.iloc[0]
            df = df[1:]  # Eliminar la fila de encabezados originales

            # Agregar el DataFrame a la lista de tablas
            all_tables.append(df)

    pdf.close()

    # Combinar todas las tablas en un único DataFrame
    combined_tables = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()

    return combined_tables


# Path to your PDF file
pdf_path = "ordine del 01-12-2023 Spluga Srl.pdf"

# Procesar el PDF y obtener un DataFrame con todas las tablas
extracted_tables_df = process_pdf_to_dataframe(pdf_path)

# Mostrar el DataFrame resultante
print(extracted_tables_df.head())
