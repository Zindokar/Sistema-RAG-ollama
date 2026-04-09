import json
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def cargar_documentos(ruta: str):
    """
    Carga el JSON limpio y lo convierte en documentos de LangChain
    con metadatos (title, url, topic).
    """
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)

    documentos = []
    for item in datos:
        doc = Document(
            page_content=item["content"],
            metadata={
                "title": item.get("title", "Sin título"),
                "url": item.get("url", ""),
                "topic": item.get("topic", "General"),
            }
        )
        documentos.append(doc)

    return documentos


def dividir_en_chunks(documentos, chunk_size=500, chunk_overlap=50):
    """
    Divide los documentos en fragmentos más pequeños.
    
    - chunk_size: número máximo de caracteres por fragmento
    - chunk_overlap: caracteres de solapamiento entre fragmentos consecutivos
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documentos)
    return chunks


def crear_base_vectorial(chunks, persist_directory="./chroma_db"):
    """
    Genera embeddings con nomic-embed-text (Ollama) y los almacena en ChromaDB.
    """
    print("Generando embeddings con nomic-embed-text (esto puede tardar)...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:v1.5",
    )

    # Crear la base de datos vectorial
    # persist_directory permite que los datos sobrevivan entre sesiones
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="khan_academy",
    )

    return vectorstore


def main():
    print("Cargando dataset limpio...")
    documentos = cargar_documentos("khanacademy_clean.json")
    print(f"   Documentos cargados: {len(documentos)}")

    print("\nDividiendo en chunks...")
    chunks = dividir_en_chunks(documentos)
    print(f"   Chunks generados: {len(chunks)}")

    # Mostrar ejemplo de un chunk
    if chunks:
        print(f"\nEjemplo de chunk:")
        print(f"   Título: {chunks[0].metadata['title']}")
        print(f"   Topic:  {chunks[0].metadata['topic']}")
        print(f"   Texto:  {chunks[0].page_content[:200]}...")

    print("\nIndexando en ChromaDB...")
    vectorstore = crear_base_vectorial(chunks)
    print(f"   Base de datos creada en ./chroma_db")
    print(f"   Total de vectores indexados: {vectorstore._collection.count()}")

    print("\nIndexación completada.")


if __name__ == "__main__":
    main()