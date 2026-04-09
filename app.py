import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "khan_academy"
EMBEDDING_MODEL = "nomic-embed-text:v1.5"
LLM_MODEL = "mistral:latest"

def cargar_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def obtener_topics(vectorstore) -> list:
    try:
        resultados = vectorstore._collection.get(include=["metadatas"])
        topics = set()
        for meta in resultados["metadatas"]:
            if meta and "topic" in meta:
                topics.add(meta["topic"])
        return ["Todos"] + sorted(list(topics))
    except Exception:
        return ["Todos"]


def crear_chain(vectorstore, k=5, score_threshold=0.3, topic_filter="Todos"):
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)

    search_kwargs = {"k": k, "score_threshold": score_threshold}
    if topic_filter and topic_filter != "Todos":
        search_kwargs["filter"] = {"topic": topic_filter}

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs,
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Dado el historial de conversación y la última pregunta del usuario, "
         "reformula la pregunta para que sea independiente del historial. "
         "Si ya es independiente, devuélvela tal cual. Solo devuelve la pregunta, sin explicaciones."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """Eres un asistente educativo especializado en el contenido de Khan Academy.
            Tu tarea es responder preguntas basándote ÚNICAMENTE en el contexto proporcionado
            (fragmentos de transcripciones de vídeos educativos).
            
            Reglas:
            1. Basa tu respuesta exclusivamente en el contexto proporcionado.
            2. Si la información no está en el contexto, responde exactamente:
               "No dispongo de información sobre ese tema en las transcripciones disponibles."
            3. Cuando sea posible, menciona el título del vídeo de donde extraes la información.
            4. Responde en español de forma clara y didáctica.
            5. Si la pregunta es ambigua, pide aclaración.
            
            Contexto recuperado:
            {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    return rag_chain

vectorstore = cargar_vectorstore()
topics_disponibles = obtener_topics(vectorstore)
chain_cache: dict = {}
store: dict[str, BaseChatMessageHistory] = {}

def get_chain(k, score_threshold, topic_filter):
    key = (float(k), float(score_threshold), topic_filter)
    if key not in chain_cache:
        chain_cache[key] = crear_chain(vectorstore, int(k), float(score_threshold), topic_filter)
    return chain_cache[key]

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def responder(mensaje, historial, k, score_threshold, topic_filter):
    if not mensaje or not mensaje.strip():
        return historial or [], "### Fuentes\n\n*Escribe una pregunta primero.*"

    chain = get_chain(k, score_threshold, topic_filter)

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    try:
        resultado = chain_with_history.invoke(
            {"input": mensaje},
            config={"configurable": {"session_id": "default"}},
        )

        respuesta = resultado.get("answer", "No se pudo generar una respuesta.")
        fuentes = resultado.get("context", [])

        texto_fuentes = "### Fuentes utilizadas\n\n"
        if fuentes:
            for i, doc in enumerate(fuentes, 1):
                titulo = doc.metadata.get("title", "Sin título")
                topic = doc.metadata.get("topic", "N/A")
                url = doc.metadata.get("url", "")
                fragmento = doc.page_content[:300].replace("\n", " ")
                texto_fuentes += f"**{i}. {titulo}**\n"
                texto_fuentes += f"- Tema: `{topic}`\n"
                if url:
                    texto_fuentes += f"- [Ver vídeo]({url})\n"
                texto_fuentes += f"- *{fragmento}...*\n\n"
        else:
            texto_fuentes += "*No se encontraron fragmentos con similitud suficiente.*\n"

        historial = historial or []
        historial.append({"role": "user", "content": mensaje})
        historial.append({"role": "assistant", "content": respuesta})

        return historial, texto_fuentes

    except Exception as e:
        historial = historial or []
        historial.append({"role": "user", "content": mensaje})
        historial.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return historial, f"**Error:** {str(e)}"

def limpiar_historial():
    store.clear()
    return [], "### Fuentes\n\n*Realiza una consulta para ver las fuentes.*"

def aplicar_parametros(k, score_threshold, topic_filter):
    return f"Parámetros aplicados: k={int(k)}, umbral={score_threshold}, tema={topic_filter}"

def exportar_conversacion(historial):
    if not historial:
        return "La conversación está vacía."
    lineas = []
    for msg in historial:
        rol = "Usuario" if msg["role"] == "user" else "Asistente"
        lineas.append(f"**{rol}:** {msg['content']}\n")
    return "\n".join(lineas)

def crear_interfaz():
    with gr.Blocks(title="RAG Khan Academy", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# Sistema RAG — Khan Academy")
        gr.Markdown("Haz preguntas sobre el contenido educativo de Khan Academy. "
                    "El sistema busca en transcripciones de vídeos y genera respuestas fundamentadas.")

        with gr.Row():

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversación",
                    height=500,
                )

                with gr.Row():
                    entrada = gr.Textbox(
                        placeholder="Escribe tu pregunta aquí...",
                        label="Tu pregunta",
                        scale=4,
                        lines=1
                    )
                    btn_enviar = gr.Button("Enviar", variant="primary", scale=1)

                with gr.Row():
                    btn_limpiar = gr.Button("Limpiar historial", variant="secondary")
                    btn_exportar = gr.Button("Exportar conversación", variant="secondary")

                exportacion = gr.Textbox(
                    label="Conversación exportada",
                    visible=False,
                    lines=10,
                    interactive=False,
                )

            with gr.Column(scale=2):
                gr.Markdown("## Parámetros del Retriever")
                slider_k = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Fragmentos a recuperar (k)",
                    info="Cuántos fragmentos recuperar de la base de datos",
                )
                slider_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                    label="Umbral de similitud",
                    info="Similitud mínima requerida (0 = todo, 1 = exacto)",
                )
                dropdown_topic = gr.Dropdown(
                    choices=topics_disponibles,
                    value="Todos",
                    label="Filtrar por tema",
                    info="Limita la búsqueda a un área temática",
                )
                btn_aplicar = gr.Button("Aplicar parámetros", variant="primary")
                estado = gr.Textbox(
                    label="Estado",
                    value="Parámetros por defecto: k=5, umbral=0.3, tema=Todos",
                    interactive=False,
                )
                gr.Markdown("---")
                panel_fuentes = gr.Markdown(
                    value="### Fuentes\n\n*Realiza una consulta para ver las fuentes.*"
                )

        btn_enviar.click(
            fn=responder,
            inputs=[entrada, chatbot, slider_k, slider_threshold, dropdown_topic],
            outputs=[chatbot, panel_fuentes],
        ).then(fn=lambda: "", outputs=[entrada])

        entrada.submit(
            fn=responder,
            inputs=[entrada, chatbot, slider_k, slider_threshold, dropdown_topic],
            outputs=[chatbot, panel_fuentes],
        ).then(fn=lambda: "", outputs=[entrada])

        btn_limpiar.click(
            fn=limpiar_historial,
            outputs=[chatbot, panel_fuentes],
        )

        btn_aplicar.click(
            fn=aplicar_parametros,
            inputs=[slider_k, slider_threshold, dropdown_topic],
            outputs=[estado],
        )

        def toggle_exportar(historial):
            texto = exportar_conversacion(historial)
            return gr.update(visible=True, value=texto)

        btn_exportar.click(
            fn=toggle_exportar,
            inputs=[chatbot],
            outputs=[exportacion],
        )

    return demo

if __name__ == "__main__":
    print("Iniciando sistema RAG Khan Academy...")
    print(f"   LLM: {LLM_MODEL}")
    print(f"   Embeddings: {EMBEDDING_MODEL}")
    print(f"   Topics disponibles: {len(topics_disponibles)}")

    demo = crear_interfaz()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
