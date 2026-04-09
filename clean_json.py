import pandas as pd
import os
import re
import json

def load_dataset():
    LOCAL_PATH = "khanacademy.csv"
    if os.path.exists(LOCAL_PATH):
        df = pd.read_csv(LOCAL_PATH)
    else:
        df = pd.read_json("hf://datasets/iblai/ibl-khanacademy-transcripts/train.jsonl", lines=True)
        df.to_csv(LOCAL_PATH, index=False)
    return df

def clean_transcript(text):
    if not isinstance(text, str):
        return ""
    # Eliminar marcadores de tiempo [00:00:00]
    text = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', text)
    # Eliminar etiquetas tipo [música], [aplausos], [risas], etc.
    text = re.sub(r'\[.*?\]', '', text)
    # Eliminar etiquetas HTML si las hay
    text = re.sub(r'<[^>]+>', '', text)
    # Normalizar espacios y saltos de línea
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, max_words=300, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap
    return chunks

def main():
    df = load_dataset()

    if 'transcript' in df.columns:
        df = df.rename(columns={'transcript': 'content'})

    # Limpiar transcripciones
    df['content'] = df['content'].apply(clean_transcript)

    # Filtrar registros vacíos o demasiado cortos (< 50 palabras)
    df = df[df['content'].apply(lambda x: len(x.split()) >= 50)]

    # Eliminar duplicados
    df = df.drop_duplicates(subset=['content'])

    # Seleccionar solo campos relevantes para RAG
    relevant_cols = [c for c in ['title', 'content', 'url'] if c in df.columns]
    df = df[relevant_cols]

    # Chunking: dividir transcripciones largas
    records = []
    for _, row in df.iterrows():
        chunks = chunk_text(row['content'], max_words=300, overlap=50)
        for i, chunk in enumerate(chunks):
            record = {col: row[col] for col in relevant_cols if col != 'content'}
            record['content'] = chunk
            record['chunk_id'] = i
            records.append(record)

    df_clean = pd.DataFrame(records)
    df_clean.to_json("khanacademy_clean.json", orient='records', force_ascii=False, indent=2)

if __name__ == "__main__":
    main()