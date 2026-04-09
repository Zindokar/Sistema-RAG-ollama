# Sistema-RAG-ollama
El objetivo de este proyecto es construir un sistema de Retrieval-Augmented Generation (RAG) que permita a los usuarios interactuar en lenguaje natural con el contenido educativo de los cursos de Khan Academy, a través de sus transcripciones en texto.
# Sistema RAG — Khan Academy

Sistema de preguntas y respuestas basado en **Retrieval-Augmented Generation (RAG)** sobre las transcripciones de vídeos educativos de Khan Academy. Permite al usuario interactuar en lenguaje natural con el contenido del dataset a través de una interfaz web construida con Gradio.

El sistema ejecuta tanto el LLM como el modelo de embeddings de forma **totalmente local** mediante Ollama, sin depender de APIs externas.

---

## Características

- **Búsqueda semántica** sobre transcripciones usando embeddings vectoriales
- **Chat conversacional** con memoria de turnos anteriores
- **Visualización de fuentes** utilizadas en cada respuesta
- **Parámetros configurables** del retriever en tiempo real (k, umbral de similitud, filtro por tema)
- **100% local** — no se envían datos a servicios externos
- **Pipeline de limpieza** del dataset reproducible

---

## Estructura del proyecto

```
proyecto-rag-khan/
├── clean_json.py          # Fase 1: Descarga y limpieza del dataset
├── index_data.py          # Fase 2: Generación de embeddings e indexación
├── app.py                 # Fase 3+4: Pipeline RAG + interfaz Gradio
├── requirements.txt       # Dependencias de Python
├── khanacademy_clean.json # Dataset limpio (generado por clean_json.py)
├── chroma_db/             # Base de datos vectorial (generada por index_data.py)
└── README.md              # Este archivo
```

---

## Requisitos previos

Antes de empezar, asegúrate de tener instalado:

- **Python 3.10 o superior**
- **Ollama** — [https://ollama.com](https://ollama.com)
- **Git** (opcional, para clonar el repositorio)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd proyecto-rag-khan
```

### 2. Crear y activar el entorno virtual

Puedes usar el gestor de entornos estándar de Python (`venv`) o el moderno y rápido [`uv`](https://docs.astral.sh/uv/).

> ⚠️ **Requisito:** Python **3.11** o superior.

---

#### Opción A — `venv` (incluido con Python)

Asegúrate de tener Python 3.11 instalado y úsalo explícitamente:

**Linux / macOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

---

#### Opción B — `uv` (recomendado: más rápido y moderno)

##### Instalación de `uv`

**Linux / macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

O mediante `pip` (cualquier plataforma):

```bash
pip install uv
```

##### Crear y activar el entorno

Con `uv` puedes especificar la versión de Python directamente, e incluso descargarla automáticamente si no la tienes instalada:

**Linux / macOS:**

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
uv venv .venv --python 3.11
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
uv venv .venv --python 3.11
.venv\Scripts\activate.bat
```

> **Nota:** Si Python 3.11 no está instalado en el sistema, `uv` puede descargarlo y gestionarlo automáticamente. Con `venv` deberás instalarlo manualmente desde [python.org](https://www.python.org/downloads/).

Cuando el entorno esté activo verás `(.venv)` al principio de tu línea de comandos.

### 3. Instalar las dependencias

Con el entorno virtual activado, ejecuta:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Esto instalará, entre otras:

| Paquete | Uso |
|---------|-----|
| `langchain` | Framework de orquestación del pipeline RAG |
| `langchain-community` | Integraciones comunitarias de LangChain |
| `langchain-chroma` | Conector de LangChain con ChromaDB |
| `langchain-ollama` | Conector de LangChain con Ollama |
| `chromadb` | Base de datos vectorial |
| `datasets` | Descarga del dataset desde Hugging Face |
| `gradio` | Interfaz web del chatbot |

### 4. Instalar Ollama y descargar los modelos

Si aún no tienes Ollama instalado:

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS / Windows:** descarga el instalador desde [ollama.com/download](https://ollama.com/download).

Una vez instalado, descarga los modelos que usará el proyecto:

```bash
# Modelo de lenguaje (para generar las respuestas)
ollama pull llama3.2

# Modelo de embeddings (para la búsqueda semántica)
ollama pull nomic-embed-text
```

Verifica que ambos modelos están disponibles:

```bash
ollama list
```

Deberías ver algo como:

```
NAME                       SIZE      MODIFIED
llama3.2:latest            2.0 GB    1 minute ago
nomic-embed-text:latest    274 MB    2 minutes ago
```

> **Nota:** Ollama debe estar ejecutándose en segundo plano mientras uses la aplicación. En Linux normalmente arranca solo como servicio; en macOS/Windows se inicia al abrir la aplicación de Ollama.

---

## ▶️ Uso del sistema

Ejecuta los scripts en este orden la primera vez:

### Paso 1 — Limpiar el dataset

```bash
python clean_json.py
```

Este script descarga el dataset `iblai/ibl-khanacademy-transcripts` desde Hugging Face, elimina registros vacíos o demasiado cortos, limpia el ruido de las transcripciones (marcadores de tiempo, etiquetas tipo `[música]`, etc.) y guarda el resultado en `train_clean.json`.

### Paso 2 — Indexar en la base de datos vectorial

```bash
python index_data.py
```

Este script divide las transcripciones en fragmentos (chunks), genera los embeddings con `nomic-embed-text` y los almacena en una base de datos ChromaDB persistente en el directorio `chroma_db/`.

> Este paso puede tardar varios minutos dependiendo del número de registros y tu hardware.

### Paso 3 — Lanzar la aplicación

```bash
python app.py
```

Abre tu navegador en [http://localhost:7860](http://localhost:7860) y ya puedes empezar a hacer preguntas.

---

## Parámetros configurables desde la interfaz

La interfaz de Gradio incluye controles para ajustar el comportamiento del retriever en tiempo real:

| Parámetro | Rango | Descripción |
|-----------|-------|-------------|
| **k** | 1 – 10 | Número de fragmentos que el retriever devuelve por consulta |
| **Umbral de similitud** | 0.0 – 1.0 | Similitud mínima para que un fragmento sea considerado relevante |
| **Filtro por tema** | Dropdown | Limita la búsqueda a un área temática concreta |

También dispones de un botón **🗑️ Limpiar historial** para reiniciar la conversación y la memoria del sistema.

---

## Arquitectura del pipeline

```
Pregunta del usuario
        │
        ▼
 Embedding de la pregunta  (nomic-embed-text)
        │
        ▼
 Búsqueda en ChromaDB  (retriever con k + umbral + filtro)
        │
        ▼
 Construcción del prompt  (contexto + historial + pregunta)
        │
        ▼
 Generación con llama3.2  (vía Ollama)
        │
        ▼
 Respuesta + fuentes citadas
```

---

##  Solución de problemas

### `ConnectionError` o `connection refused` al arrancar `app.py`

Ollama no está en ejecución. Arráncalo con:

```bash
ollama serve
```

O abre la aplicación de Ollama si estás en macOS/Windows.

### El modelo no se encuentra

Asegúrate de haber descargado ambos modelos:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### La base de datos `chroma_db` está vacía o da errores

Borra el directorio y vuelve a ejecutar el paso de indexación:

```bash
rm -rf chroma_db/
python index_data.py
```

### Respuestas lentas

- El primer arranque del LLM siempre es más lento porque Ollama carga el modelo en memoria.
- Si tu máquina no tiene GPU, considera usar un modelo más pequeño como `llama3.2:1b` editando la constante `LLM_MODEL` en `app.py`.

### Quiero reducir el número de registros procesados

Edita la constante `MAX_REGISTROS` en `clean_json.py` y vuelve a ejecutar los pasos 1 y 2.

---

## Dependencias — contenido de `requirements.txt`

```
datasets
langchain
langchain-community
langchain-chroma
langchain-ollama
chromadb
gradio
```

---

## Tecnologías utilizadas

- **[LangChain](https://python.langchain.com/)** — Framework de orquestación del pipeline RAG
- **[Ollama](https://ollama.com/)** — Ejecución local de LLMs y modelos de embeddings
- **[ChromaDB](https://www.trychroma.com/)** — Base de datos vectorial
- **[Gradio](https://www.gradio.app/)** — Interfaz web del chatbot
- **[Hugging Face Datasets](https://huggingface.co/docs/datasets/)** — Descarga del dataset original

---

## Dataset

Este proyecto utiliza el dataset público [`iblai/ibl-khanacademy-transcripts`](https://huggingface.co/datasets/iblai/ibl-khanacademy-transcripts), que contiene transcripciones de vídeos educativos de Khan Academy. El uso del dataset se limita a fines educativos dentro del marco del proyecto final del curso.

---

## Licencia

Proyecto académico desarrollado en el contexto del curso. Uso libre con fines educativos. Licencia MIT.