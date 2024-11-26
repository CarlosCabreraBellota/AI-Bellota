
import gmft
from gmft.pdf_bindings import PyPDFium2Document
from gmft.auto import TableDetector, AutoTableFormatter, AutoFormatConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
# Configure table detector and formatter
detector = TableDetector()
config = AutoFormatConfig()
config.semantic_spanning_cells = True
config.enable_multi_header = True
formatter = AutoTableFormatter(config)

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

def extract_reference(text):
    """
    Extrae y normaliza referencias en formatos como '8030 CI CF' o '72305-43 S1P'.
    """
    text = text.strip().upper()  # Normalizar a mayúsculas
    # Buscar referencias con patrones complejos (números, letras, guiones, espacios)
    match = re.search(r'[A-Z0-9\-]+(?:[\s\-][A-Z0-9]+)*(?:\sS1P)?', text)
    return match.group(0) if match else text

if __name__ == "__main__":
    # Path to your PDF
    pdf_path = "P-2302490.pdf"

    # Extract tables
    tables = extract_tables_to_dataframes(pdf_path)

    # Display and save each table
    for idx, df in enumerate(tables):
        print(f"Table {idx + 1}:")
        print(df)
        df = df.replace({'-':'',',00':'','un':'',':':' '}, regex=True)
        output_excel = f"Table_{idx + 1}.xlsx"
        df.to_excel(output_excel)
        # Save to CSV
        output_csv = f"Table_{idx + 1}.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved Table {idx + 1} to {output_csv}")


        # 1. Leer las referencias de Excel
        references = pd.read_excel('Libro1.xlsx')
        references_df = references['CODART']
        references_df = references_df.astype(str)
        # Normalizar y limpiar references_df cargado desde Excel
        references_df = references_df.str.strip()  # Eliminar espacios al principio y al final
        references_df = references_df.str.replace(" ", "").str.replace("-", "").str.upper()  # Eliminar espacios internos, guiones, y convertir a mayúsculas
        references_df = references_df.dropna().drop_duplicates()  # Eliminar valores nulos y duplicados

# Convertir referencias a cadenas
references_df = references_df.astype(str)

# Vectorizar todas las referencias
vectorizer = TfidfVectorizer().fit(references_df)
ref_vectors = vectorizer.transform(references_df)

# Lista para almacenar resultados
results = []

# Umbral de similitud
threshold = 0.2

# Recorrer cada celda de df
for row_idx, row in df.iterrows():
    for col_idx, cell in row.items():
        # Convertir la celda actual a cadena
        cell = str(cell) #extract_reference(str(cell)) str(cell)

        # Vectorizar la celda actual
        cell_vector = vectorizer.transform([cell])

        # Calcular similitudes con todas las referencias
        similarities = cosine_similarity(cell_vector, ref_vectors).flatten()

        # Encontrar todas las coincidencias con similitud >= umbral
        valid_matches = [
            {"Matched Reference": references_df.iloc[i], "Similarity Score": score}
            for i, score in enumerate(similarities)
            if score >= threshold
        ]

        # Añadir los resultados válidos con sus detalles
        for match in valid_matches:
            results.append({
                #'Row': row_idx,
                #'Column': col_idx,
                #'Original Value': cell,
                'Matched Reference': match["Matched Reference"],
                #'Similarity Score': match["Similarity Score"]
            })

# Convertir los resultados a un DataFrame
matched_df = pd.DataFrame(results)

# Eliminar duplicados
matched_df = matched_df.drop_duplicates()

# Mostrar y guardar los resultados
print(matched_df)
matched_df.to_csv("Matched_Results.csv", index=False)
print("Resultados guardados en 'Matched_Results.csv'")








