###########
#%% Importar bibliotecas necesarias
import pandas as pd
from tabula.io import read_pdf
from nltk.tokenize import word_tokenize
import os
import nltk
from gmft.pdf_bindings import PyPDFium2Document
from gmft.auto import TableDetector, AutoTableFormatter, AutoFormatConfig
import os
import pypdfium2
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
pdfium_path = os.path.join(os.path.dirname(__file__), "pdfium.dll")
pypdfium2.raw.register_pdfium(pdfium_path)

# Descargar el tokenizador de nltk si es necesario
nltk.download('punkt')

#%% Funciones
def configure_table_tools():
    """
    Configura las herramientas para detección y formateo de tablas.
    """
    detector = TableDetector()
    config = AutoFormatConfig()
    config.semantic_spanning_cells = True
    config.enable_multi_header = True
    formatter = AutoTableFormatter(config)
    return detector, formatter

def read_and_normalize_excel(excel_path):
    """
    Lee y normaliza las referencias desde un archivo Excel.
    """
    try:
        references_df = pd.read_excel(excel_path)['CODART']
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo Excel no encontrado: {excel_path}")
    
    # Normalizar y limpiar el DataFrame
    references_df = (
        references_df.astype(str)
        .str.strip()
        .str.replace(" ", "")
        .str.replace("-", "")
        .str.upper()
    ).dropna().drop_duplicates()
    
    return references_df.tolist()

def read_and_combine_pdf(pdf_path):
    """
    Lee y combina las tablas de un PDF en un DataFrame.
    """
    try:
        tablas = read_pdf(pdf_path, pages='all', multiple_tables=True)
        tablas_combined = pd.concat(tablas, ignore_index=True)
    except Exception as e:
        raise ValueError(f"Error al leer el PDF: {e}")
    return tablas_combined

def normalize_and_tokenize_pdf_references(tablas_combined):
    """
    Normaliza y tokeniza las referencias extraídas del PDF.
    """
    pdf_references = (
        tablas_combined.stack()
        .astype(str)
        .unique()
    )
    pdf_references = [
        ref.strip().replace("–", "").replace(":", " ").replace(".", " ").lower()
        for ref in pdf_references if ref != 'nan'
    ]
    tokenized_references = [word_tokenize(ref) for ref in pdf_references]
    flattened_references = [token for sublist in tokenized_references for token in sublist]
    
    tokenized_references_df = pd.DataFrame(flattened_references, columns=['reference'])
    tokenized_references_df['reference'] = tokenized_references_df['reference'].str.replace("-", "", regex=True)
    return tokenized_references_df.dropna().drop_duplicates()

def find_common_references(tokenized_references_df, references):
    """
    Encuentra referencias en común entre las referencias del PDF y las del Excel.
    """
    normalized_excel_references = [ref.replace("-", "").lower() for ref in references]
    df_normalized_excel_references = pd.DataFrame(normalized_excel_references, columns=['reference'])
    df_normalized_excel_references['reference'] = df_normalized_excel_references['reference'].str.strip()
    return pd.merge(tokenized_references_df, df_normalized_excel_references, on='reference').drop_duplicates(subset='reference', keep='first')

def extract_tables_with_gmft(pdf_path, detector, formatter):
    """
    Extrae tablas de un PDF usando gmft y retorna como DataFrames de pandas.
    """
    doc = PyPDFium2Document(pdf_path)
    dataframes = []
    try:
        for page in doc:
            tables = detector.extract(page)
            for table in tables:
                try:
                    formatted_table = formatter.extract(table)
                    df = formatted_table.df()
                    dataframes.append(df)
                except Exception as e:
                    print(f"Error formateando una tabla: {e}")
    finally:
        doc.close()
    return dataframes

# Función para combinar múltiples tablas en un único DataFrame
def combine_tables(tables):
    """
    Combina múltiples tablas (listas de DataFrames) en un solo DataFrame.
    """
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
            return pd.concat(tablas_2d, ignore_index=True)
    return None

def seleccionar_dataframe(variables):
    """
    Selecciona un DataFrame que contenga columnas relacionadas con palabras clave.
    """
    keywords = ["quantity", "qty", "cantidad", "cant", "quantité", "qté", "quant",
            "quantidade", "qtd", "quantità", "qtà"]
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


def clean_column(column):
    """
    Limpia una columna y extrae solo los números.
    """
    def extract_numbers(value):
        if pd.isna(value):
            return None
        numeric_value = ''.join(char for char in str(value) if char.isdigit() or char == ',')
        try:
            return float(numeric_value.replace(',', '.')) if ',' in numeric_value else int(numeric_value)
        except ValueError:
            return None
    return column.apply(extract_numbers)

def get_embedded_excel_path():
    """
    Obtiene la ruta del archivo Excel embebido.
    """
    if hasattr(sys, '_MEIPASS'):  # Cuando se ejecuta como ejecutable
        base_path = sys._MEIPASS
    else:  # Cuando se ejecuta como script normal
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, 'resources', 'Libro1.xlsx')

def resolver_ruta(ruta_relativa):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, ruta_relativa)
    return os.path.join(os.path.abspath('.'), ruta_relativa)


#%%
def main():
    # Configurar rutas de entrada
    try:
        # Tu código aquí
        print("Todo está funcionando correctamente.")
        pdf_path = input("Por favor, ingrese la ruta completa del archivo PDF: ").strip()
        #excel_path = get_embedded_excel_path()
        excel_path = resolver_ruta("Libro1.xlsx")
        with open(excel_path, "r") as archivo:
            pass  # No hacemos nada con el archivo, solo comprobamos que se puede abrir
        input("Presiona Enter para salir")
                
        
        if not os.path.isfile(pdf_path):
            print(f"Archivo PDF no encontrado: {pdf_path}")
            return
        if not os.path.isfile(excel_path):
            print(f"Archivo Excel no encontrado: {excel_path}")
            return
        
        # Configurar herramientas de tablas
        detector, formatter = configure_table_tools()
        
        # Leer y procesar referencias del Excel
        references = read_and_normalize_excel(excel_path)
        
        # Leer y procesar tablas del PDF
        tablas_combined = read_and_combine_pdf(pdf_path)
        tokenized_references_df = normalize_and_tokenize_pdf_references(tablas_combined)
        
        # Encontrar coincidencias
        coincidencias = find_common_references(tokenized_references_df, references)
        coincidencias = coincidencias.reset_index(drop=True)
        
        #Encontrar cantidades por artículo
        tables = extract_tables_with_gmft(pdf_path, detector, formatter)
        print(tables)
        if not tables or len(tables) == 0:
            print("No se encontraron tablas en el PDF.")
            return
        
        # Combinar tablas
        tablas_combinadas = combine_tables(tables)
        if tablas_combinadas is None:
            print("No se pudieron combinar las tablas.")
            return

        # Mostrar columnas disponibles
        #print("Columnas en tablas combinadas:", tablas_combinadas.columns)

        # Palabras clave relacionadas con "cantidad"
        # Seleccionar el DataFrame que contiene columnas relacionadas
        variables = {
        "tablas_combinadas": tablas_combinadas,
        "tablas_combined": tablas_combined
        }
        print(variables)
        df_seleccionado, variable_seleccionada, columna_seleccionada = seleccionar_dataframe(variables)

        if df_seleccionado is None:
            print("No se encontró un DataFrame válido con columnas relacionadas con 'cantidad'.")
            return

        # Extraer y limpiar cantidades
        cantidades_ref = df_seleccionado[columna_seleccionada].dropna()
        cantidades_ref = clean_column(cantidades_ref).reset_index(drop=True)
        #print("Cantidad procesada:", cantidades_ref)
        
        pedido = pd.concat([coincidencias, cantidades_ref], axis = 1)
        
        # Exportar coincidencias a Excel
        output_excel_path = os.path.splitext(pdf_path)[0] + "_pedido.xlsx"
        pedido.to_excel(output_excel_path, index=False)
        print(f"Referencias y cantidades exportadas a: {output_excel_path}")
        
    except Exception as e:
        print(f"Se produjo un error: {e}")
        input("Presiona Enter para salir...")

if __name__ == "__main__":
    main()


# %%
