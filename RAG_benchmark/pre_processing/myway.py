import re
import os
import chromadb
import time
import ollama
import argparse
import unicodedata

from langchain_ollama import OllamaLLM

# from langchain.chains import LLMChain  # Deprecated in newer versions
# from langchain.chains import RetrievalQA  # Deprecated in newer versions
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder


# Expresiones regulares para los títulos
TITLE_REGEX = re.compile(r"^#\s+\*\*(.*)\*\*$")
SUBTITLE_REGEX = re.compile(r"^##\s+\*\*(.*)\*\*$")
SUB_SUBTITLE_REGEX = re.compile(r"^###\s+\*\*(.*)\*\*$")
GENERIC_TITLE_REGEX = re.compile(r"^\*\*(.*?)\*\*$")

TITLE_REGEX_MATER = re.compile(r"^(#{1})\s\*\*(.*?)\*\*$")
SUBTITLE_REGEX_MATER = re.compile(r"^(#{2})\s\*\*(.+?)\*\*$")
SUB_SUBTITLE_REGEX_MATER = re.compile(r"^(#{3})\s\*\*(.?)\*\*$")
DEFINITION_REGEX_MATER = re.compile(r"^\*\*(.+?)\*\*:? (.+)$")

REGLAMENTO_REGEX_PORFAC = re.compile(r"^(#{1})\s\*\*(.+?)\*\*$")
TITLE_REGEX_PORFAC = re.compile(r"^(#{2})\s\*\*(.+?)\*\*$")
CAPITULO_REGEX_PORFAC = re.compile(r"^(#{3})\s\*\*(.+?)\*\*$")
ARTICULO_REGEX_PORFAC = re.compile(r"^(#{4})\s\*\*Art[íi]culo\s(\d+)\.\*\*")
PARAGRAFO_REGEX_PORFAC = re.compile(r"^\*\*(.+?)\*\*\s*(.*)$")
DEFINICION_REGEX_PORFAC = re.compile(r"^\*(.+?)\.\*\s(.+)$")

QUESTION_REGEX = re.compile(r"^###\s(.*)$")
REF_REGEX = re.compile(r"\*\*(.*?)\*\*")

# Texto que debe ser excluido
EXCLUDED_TEXTS = {
    "**2042**",
    "*2042*"
    "ESTRATEGIA PARA EL DESARROLLO AÉREO Y ESPACIAL DE LA FUERZA AÉREA COLOMBIANA",
    "HOJA DE ACTUALIZACIONES",
    "ÍNDICES DE TABLAS",
    "ÍNDICES DE FIGURAS",
    "TABLA DE CONTENIDO",
    "ÍNDICE",
    "INDICE",
    "ÍNDICE DE CONTENIDO",
    "ÍNDICE DE FIGURAS",
    "ÍNDICE DE TABLAS",
    "ÍNDICE DE ANEXOS",
    "ÍNDICE DE BIBLIOGRÁFICO",
    "ASÍ SE VA A LAS ESTRELLAS",
}

EXCLUDED_TITLES = {
    "HOJA DE ACTUALIZACIONES",
    "ÍNDICES DE TABLAS",
    "ÍNDICES DE FIGURAS",
    "TABLA DE CONTENIDO",
    "ÍNDICE",
    "INDICE",
    "ÍNDICE DE CONTENIDO",
    "ÍNDICE DE FIGURAS",
    "ÍNDICE DE TABLAS",
    "ÍNDICE DE ANEXOS",
    "ÍNDICE DE BIBLIOGRÁFICO",
}

parser = argparse.ArgumentParser(
    description="Configura el modelo y la temperatura para Ollama."
)
parser.add_argument(
    "--model", type=str, default="gemma3:4b", help="Nombre del modelo de Ollama a usar."
)
parser.add_argument(
    "--temperature", type=float, default=0.5, help="Temperatura del modelo (0.0 a 1.0)"
)

args = parser.parse_args()
model_name = args.model
temperature = args.temperature

# c_name = "pruebas"
c_name = "DELFOS"
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
persist_directory = "../data/chroma_db_v4"
re_ranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Reranker es el que reordenara eso documentos con el mas relevante de mayor a menor.
reranker = CrossEncoder(re_ranker_model)

prompt_template = """
Eres un experto asistente especializado en la Fuerza Aérea Colombiana (FAC). Tu misión es responder preguntas basándote ÚNICAMENTE en la información proporcionada.

INSTRUCCIONES DE RESPUESTA:
1. Responde DIRECTAMENTE a la pregunta. No divagues ni incluyas información no solicitada.
2. Identifica la respuesta en el texto proporcionado, sin importar si está en mayúsculas, minúsculas o sin acentuación (por ejemplo, "prevac" es lo mismo que "PREVAC").
3. Si en el texto dice literalmente "X es Y" o "que es X X es Y", extrae esa definición y preséntala bien redactada.
4. EVITA la frase "Según el documento...". Simplemente da la respuesta de forma clara y directa.
5. NO inventes definiciones. Si la información no está en el contexto, responde: "No tengo información suficiente en los documentos".

Pregunta: {pregunta}

Documentos disponibles: {contexto}

Respuesta:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["contexto", "pregunta"]
)

propmt_temp_rerank = """
        Eres un asistente que define si el contenido de un documento es relevante o no para la pregunta dada.
        Tu respuesta debe ser "Y" o "N".

        Documento: {documento}

        Pregunta: {pregunta}

        Respuesta: """

PROMPT_RERANK = PromptTemplate(
    template=propmt_temp_rerank, input_variables=["documento", "pregunta"]
)


def inicializar_modelo(model_name="mistral", temperature=0.5, prompt=PROMPT):
    # Aqui estamos creando el modelo en esta case deberas cambiar el nombre arriba del archivo o aqui mismo.
    llm = OllamaLLM(model=model_name, temperature=temperature)
    # Se crea el pipeline (reemplaza LLMChain que está deprecado)
    llm_chain = prompt | llm
    return llm_chain


LLM_CHAIN_RERANK = inicializar_modelo(temperature=0, prompt=PROMPT_RERANK)


def limpiar_string(texto):
    """
    Limpia un string convirtiéndolo a minúsculas, eliminando tildes y caracteres especiales.

    Args:
        texto (str): Texto a limpiar

    Returns:
        str: Texto limpio en minúsculas, sin tildes ni caracteres especiales
    """
    # Convertir a minúsculas
    texto = texto.lower()

    # Eliminar tildes y diacríticos
    texto = (
        unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    )

    # Eliminar caracteres especiales (manteniendo solo letras, números y espacios)
    texto = re.sub(r"[^a-z0-9\s]", "", texto)

    # Reemplazar múltiples espacios por uno solo
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def sanitize_filename(name):
    return re.sub(r'[.<>:"/\\|?*]', "_", name)


def re_rank_docs(query, docs, reranker):
    if not docs:
        return []

    # Extraer posibles términos clave (palabras en mayúscula o más de 4 letras que no sean comunes)
    import re

    query_upper = query.upper()
    keywords = [
        w
        for w in re.findall(r"\b[A-Za-z0-9]{4,}\b", query_upper)
        if w not in ["QUE", "COMO", "CUAL", "PARA", "ESTA", "CUALES"]
    ]

    texts = [doc.page_content for doc in docs]
    pairs = [[query, text] for text in texts]
    scores = reranker.predict(pairs)

    # Calcular nuevos scores combinando el reranker con coincidencia exacta
    boosted_scores = []
    for i, (score, text) in enumerate(zip(scores, texts)):
        text_upper = text.upper()
        boost = 0.0
        # Dar un bono masivo si el término exacto aparece (especialmente en acrónimos)
        for kw in keywords:
            if kw in text_upper:
                boost += 20.0  # Bono enorme para que suba al top
        boosted_scores.append(score + boost)

    sorted_docs = sorted(zip(docs, boosted_scores), key=lambda x: x[1], reverse=True)
    return [doc[0] for doc in sorted_docs][:20]


# def re_rank_docs(query, docs, llm_chain):
#     filter_docs = []
#     for doc in docs:
#         res = llm_chain.run(documento=doc.page_content, pregunta=query)
#         if res == "Y":
#             filter_docs.append(doc)
#             print(doc.page_content)
#         else:
#             continue
#     return filter_docs


# Función para procesar el Markdown y extraer jerarquía del MATER
def parse_markdown_mater(md_text, sections):

    lines = md_text.split("\n")
    stack = []

    x = 0
    for line in lines:
        title_match = TITLE_REGEX_MATER.match(line)
        subtitle_match = SUBTITLE_REGEX_MATER.match(line)
        sub_subtitle_match = SUB_SUBTITLE_REGEX_MATER.match(line)
        definition_match = DEFINITION_REGEX_MATER.match(line)
        # if line == "### **A**":
        #     print(title_match)
        #     print(subtitle_match)
        #     print(sub_subtitle_match)
        #     print(definition_match)

        def_text = ""
        if title_match:
            level, title = len(title_match.group(1)), title_match.group(2)
            section_type = "Título"
            stack.clear()
        elif subtitle_match:
            level, title = len(subtitle_match.group(1)), subtitle_match.group(2)
            section_type = "Subtítulo"
            stack = stack[:1]
        elif sub_subtitle_match:
            level, title = len(sub_subtitle_match.group(1)), sub_subtitle_match.group(2)
            section_type = "Sub-subtítulo"
            stack = stack[:2]
        elif definition_match:
            term = definition_match.group(1).strip()
            definition = definition_match.group(2).strip()

            # Crear chunk con la definición exacta
            exact_definition = f"**{term}:** {definition}"
            level = 4  # Nivel fijo para definiciones

            # Agregar la definición exacta a las secciones
            exact_node = {
                "id": f"chunk_{len(sections)}",
                "level": level,
                "title": term,
                "text": exact_definition,
                "type": "Definicion_Exacta",
                "parent": stack[-1]["id"] if stack else "None",
                "children": [],
                "origin": "Manual de términos FAC",
            }

            if stack:
                stack[-1]["children"].append(exact_node["id"])
            sections.append(exact_node)

            # También crear una versión expandida
            expanded_definition = f"¿QUÉ ES {term.upper()}?\n\n{term} es {definition}"
            expanded_node = {
                "id": f"chunk_{len(sections)}",
                "level": level,
                "title": f"Pregunta sobre {term}",
                "text": expanded_definition,
                "type": "Definicion_Expandida",
                "parent": stack[-1]["id"] if stack else "None",
                "children": [],
                "origin": "Manual de términos FAC",
            }

            if stack:
                stack[-1]["children"].append(expanded_node["id"])
            sections.append(expanded_node)
            continue
        else:
            if sections:
                sections[-1]["text"] += line + "\n"
            continue

        node = {
            "id": f"chunk_{len(sections)}",
            "level": level,
            "title": title.strip(),
            "text": limpiar_string(def_text),
            "type": section_type,
            "parent": stack[-1]["id"] if stack else "None",
            "children": [],
            "origin": "Manual de términos FAC",
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)
        stack.append(node)

    # return sections prueba = sections[-1]["text"]


# Función para procesar el Markdown y extraer jerarquía del EDAES
def parse_markdown_edaes(md_text, sections):
    """Parses a Markdown document, ensuring proper hierarchy."""
    lines = md_text.split("\n")
    stack = []

    for line in lines:
        title_match = TITLE_REGEX.match(line)
        subtitle_match = SUBTITLE_REGEX.match(line)
        sub_subtitle_match = SUB_SUBTITLE_REGEX.match(line)
        generic_title_match = GENERIC_TITLE_REGEX.match(line)

        if title_match:
            level = 1
            title = title_match.group(1)
            section_type = "Título"
            stack.clear()

        elif subtitle_match:
            level = 2
            title = subtitle_match.group(1)
            section_type = "Subtítulo"
            stack = stack[:1]

        elif sub_subtitle_match:
            level = 3
            title = sub_subtitle_match.group(1)
            section_type = "Sub-subtítulo"
            stack = stack[:2]

        elif generic_title_match:
            if not stack:
                # Los títulos genéricos deben tener un padre
                continue
            level = stack[-1]["level"] + 1
            title = generic_title_match.group(1)
            section_type = "Título genérico"
            stack = stack[: level - 1]

        else:
            # Si no se detectó como título, añadir texto al último nodo
            if sections:
                sections[-1]["text"] += limpiar_string(line) + "\n"
            continue

        # Construcción del texto con el nombre del padre (si existe)
        text = f"{title.strip()}"
        if stack:
            parent_title = stack[-1]["title"]
            text += f"({parent_title})"

        node = {
            "id": f"chunk_{len(sections)}",
            "level": level,
            "title": title.strip(),
            "text": limpiar_string(text) + "\n",
            "type": section_type,
            "parent": stack[-1]["id"] if stack else "None",
            "children": [],
            "origin": "EDAES",
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)

        # Solo los títulos principales pueden tener hijos
        if section_type != "Título genérico":
            stack.append(node)


def parse_markdown_Historia(md_text, sections):
    lines = md_text.split("\n")
    stack = []
    for line in lines:
        title_match = TITLE_REGEX.match(line)
        subtitle_match = SUBTITLE_REGEX.match(line)
        sub_subtitle_match = SUB_SUBTITLE_REGEX.match(line)
        generic_title_match = GENERIC_TITLE_REGEX.match(line)
        if title_match:
            level = 1
            title = title_match.group(1)
            section_type = "Título"
            stack.clear()
        elif subtitle_match:
            level = 2
            title = subtitle_match.group(1)
            section_type = "Subtítulo"
            stack = stack[:1]
        elif sub_subtitle_match:
            level = 3
            title = sub_subtitle_match.group(1)
            section_type = "Sub-subtítulo"
            stack = stack[:2]
        elif generic_title_match:
            if not stack:
                # Los títulos genéricos deben tener un padre
                continue
            level = stack[-1]["level"] + 1
            title = generic_title_match.group(1)
            section_type = "Título genérico"
            stack = stack[: level - 1]
        else:
            # Si no se detectó como título, añadir texto al último nodo
            if sections:
                sections[-1]["text"] += limpiar_string(line) + "\n"
            continue

        # Construcción del texto con el nombre del padre (si existe)
        text = f"{title.strip()}"
        if stack:
            parent_title = stack[-1]["title"]
            text += f"({parent_title})"

        node = {
            "id": f"chunk_{len(sections)}",
            "level": level,
            "title": title.strip(),
            "text": limpiar_string(text) + "\n",
            "type": section_type,
            "parent": stack[-1]["id"] if stack else "None",
            "children": [],
            "origin": "Historia",
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)

        # Solo los títulos principales pueden tener hijos
        if section_type != "Título genérico":
            stack.append(node)


# Función para procesar el Markdown y extraer jerarquía del PORFAC
def parse_markdown_porfac(md_text, sections):
    lines = md_text.split("\n")
    stack = []

    for line in lines:
        reg_match = REGLAMENTO_REGEX_PORFAC.match(line)
        tittle_match = TITLE_REGEX_PORFAC.match(line)
        cap_mathc = CAPITULO_REGEX_PORFAC.match(line)
        art_match = ARTICULO_REGEX_PORFAC.match(line)
        parag_match = PARAGRAFO_REGEX_PORFAC.match(line)
        def_match = DEFINICION_REGEX_PORFAC.match(line)

        if reg_match:
            level, title = len(reg_match.group(1)), reg_match.group(2)
            section_type = "Reglamento"
            stack.clear()

        elif tittle_match:
            level, title = len(tittle_match.group(1)), tittle_match.group(2)
            section_type = "Título"
            stack = stack[:1]

        elif cap_mathc:
            level, title = len(cap_mathc.group(1)), cap_mathc.group(2)
            section_type = "Capítulo"
            stack = stack[:2]

            # Descripción completa
            parent_titles = {"Título": None}
            for ancestor in reversed(stack):
                if ancestor["type"] == "Título":
                    parent_titles["Título"] = ancestor["title"]

            text = (
                f"{title} del {parent_titles['Título']}"
                if parent_titles["Título"]
                else title
            )

            node = {
                "id": f"chunk_{len(sections)}",
                "level": level,
                "title": title.strip(),
                "text": limpiar_string(text),
                "type": section_type,
                "parent": stack[-1]["id"] if stack else "None",
                "children": [],
                "origin": "Reglamento de las PORFAC",
            }

            if stack:
                stack[-1]["children"].append(node["id"])

            sections.append(node)
            stack.append(node)
            continue

        elif art_match:
            level, title = len(art_match.group(1)), f"Artículo {art_match.group(2)}"
            section_type = "Artículo"
            stack = stack[:3]

            # Descripción completa
            parent_titles = {"Capítulo": None, "Título": None}
            for ancestor in reversed(stack):
                if ancestor["type"] in parent_titles:
                    parent_titles[ancestor["type"]] = ancestor["title"]

            text = (
                f"{title} del {parent_titles['Capítulo']} del {parent_titles['Título']}"
                if all(parent_titles.values())
                else title
            )

            node = {
                "id": f"chunk_{len(sections)}",
                "level": level,
                "title": title.strip(),
                "text": limpiar_string(text),
                "type": section_type,
                "parent": stack[-1]["id"] if stack else "None",
                "children": [],
                "origin": "Reglamento de las PORFAC",
            }

            if stack:
                stack[-1]["children"].append(node["id"])

            sections.append(node)
            stack.append(node)
            continue

        elif parag_match:
            title = parag_match.group(1).strip()
            content = parag_match.group(2).strip()
            section_type = "Párrafo"
            level = 5  # Nivel fijo para párrafo

            # Descripción completa
            parent_titles = {"Artículo": None, "Capítulo": None, "Título": None}
            for ancestor in reversed(stack):
                if ancestor["type"] in parent_titles:
                    parent_titles[ancestor["type"]] = ancestor["title"]

            full_text = f"{title} del {parent_titles['Artículo']} del {parent_titles['Capítulo']} del {parent_titles['Título']}"
            text = f"{full_text}\n\n{content}"

            node = {
                "id": f"chunk_{len(sections)}",
                "level": level,
                "title": title,
                "text": limpiar_string(text),
                "type": section_type,
                "parent": stack[-1]["id"] if stack else "None",
                "children": [],
                "origin": "Reglamento de las PORFAC",
            }

            if stack:
                stack[-1]["children"].append(node["id"])

            sections.append(node)
            stack.append(node)
            continue

        elif def_match:
            # Definición de término específico (ejemplo: *Honor Militar.* Se entiende por...)
            title = def_match.group(1).strip()  # "Honor Militar"
            content = def_match.group(2).strip()  # "Se entiende por..."
            section_type = "definicion"
            level = 5  # Nivel fijo para definición

            # Descripción completa
            parent_titles = {"Artículo": None, "Capítulo": None, "Título": None}
            for ancestor in reversed(stack):
                if ancestor["type"] in parent_titles:
                    parent_titles[ancestor["type"]] = ancestor["title"]

            full_text = f"Definición de {title}"
            if parent_titles["Artículo"]:
                full_text += f" del {parent_titles['Artículo']}"
            if parent_titles["Capítulo"]:
                full_text += f" del {parent_titles['Capítulo']}"
            if parent_titles["Título"]:
                full_text += f" del {parent_titles['Título']}"

            text = f"{full_text}\n\n{title}: {content}"

            node = {
                "id": f"chunk_{len(sections)}",
                "level": level,
                "title": title,
                "text": limpiar_string(text),
                "type": section_type,
                "parent": stack[-1]["id"] if stack else "None",
                "children": [],
                "origin": "Reglamento de las PORFAC",
            }

            if stack:
                stack[-1]["children"].append(node["id"])

            sections.append(node)
            stack.append(node)
            continue

        else:
            if sections:
                sections[-1]["text"] += line + "\n"
            continue

        # Para reglamento y título (sin texto especial)
        node = {
            "id": f"chunk_{len(sections)}",
            "level": level,
            "title": title.strip(),
            "text": "",
            "type": section_type,
            "parent": stack[-1]["id"] if stack else "None",
            "children": [],
            "origin": "Reglamento de las PORFAC",
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)
        stack.append(node)


def parse_markdown_resumen_edaes(md_text, sections):

    lines = md_text.split("\n")

    referencias = ""

    for line in lines:
        question_match = QUESTION_REGEX.match(line)
        ref_match = REF_REGEX.match(line)

        if question_match:
            level = 1
            title = question_match.group(1)
            section_type = "Pregunta"
        elif ref_match:
            level = 2
            referencias = ref_match.group(1)
            sections[-1]["parent"] = referencias
            continue
        else:
            if sections:
                contenido = limpiar_string(line)
                sections[-1]["text"] += contenido + "\n"
            continue
        # print(referencias)
        node = {
            "id": f"chunk_{len(sections)}",
            "level": level,
            "title": title.strip(),
            "text": limpiar_string(title.strip()) + " ",
            "type": section_type,
            "parent": "",
            "children": [],
            "origin": "Resumen EDAES",
        }

        sections.append(node)


def prototipo_1(chunks_edaes, chunks_segmentados):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=512, chunk_overlap=20
    )
    i = 0
    for chunk in chunks_edaes:
        id_seg = i
        chunk_title = chunk["title"]
        split_text = text_splitter.split_text(chunk["text"])
        id_referencia = chunk["id"]
        origin = chunk["origin"] + " Segmentado"
        for text in split_text:
            chunks_segmentados.append(
                {
                    "id": f"chunk_{id_seg}",
                    "title": chunk_title,
                    "text": limpiar_string(chunk_title) + " " + text,
                    "origin": origin,
                    "id_referencia": id_referencia,
                }
            )
            id_seg += 1


def read_markdown_file(file_path):
    import os

    # Determinar el directorio base correcto
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Si estamos ejecutando desde el directorio pre_processing, subir un nivel
    if os.path.basename(current_dir) == "pre_processing":
        base_dir = os.path.dirname(current_dir)
    # Si estamos ejecutando desde RAG_benchmark, usar ese directorio
    elif "RAG_benchmark" in current_dir:
        # Encontrar la ruta hasta RAG_benchmark
        parts = current_dir.split(os.sep)
        if "RAG_benchmark" in parts:
            rag_benchmark_idx = parts.index("RAG_benchmark")
            if (
                rag_benchmark_idx + 1 < len(parts)
                and parts[rag_benchmark_idx + 1] == "RAG_benchmark"
            ):
                base_dir = os.sep.join(parts[: rag_benchmark_idx + 2])
            else:
                base_dir = os.sep.join(parts[: rag_benchmark_idx + 1])
        else:
            base_dir = current_dir
    else:
        # Usar el directorio del script como referencia
        base_dir = os.path.dirname(script_dir)

    # Construir la ruta completa
    full_path = os.path.join(base_dir, file_path)

    with open(full_path, "r", encoding="utf-8") as file:
        return file.read()


def store_in_chromadb(sections, persist_directory, c_name, embedding_model_name):
    """Stores parsed markdown sections into ChromaDB using Langchain and HuggingFaceEmbeddings."""

    # Inicializar el generador de embeddings con un modelo de Hugging Face
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Inicializar Chroma usando Langchain (almacenamiento persistente opcional)
    chroma_db = Chroma(
        persist_directory=persist_directory,
        collection_name=c_name,
        embedding_function=embedding_model,
    )

    # Convertir secciones a documentos de Langchain
    docs = []
    for section in sections:
        metadata = {
            "id": section.get("id", f"chunk_{len(docs)}"),
            "title": section.get("title", "Sin título"),
            "level": section.get("level", 0),
            "parent": section.get("parent", "None"),
            "children": (
                ", ".join(section["children"])
                if isinstance(section.get("children"), list)
                else section.get("children", "")
            ),
            "origin": section.get("origin", section.get("origen", "Desconocido")),
            "type": section.get("type", "unknown"),
        }
        docs.append(Document(page_content=section.get("text", ""), metadata=metadata))

    # Agregar documentos a Chroma
    chroma_db.add_documents(docs)
    return chroma_db


def store_in_chromadb_seg(sections, persist_directory, c_name, embedding_model_name):
    """Stores parsed markdown sections into ChromaDB using Langchain and HuggingFaceEmbeddings."""

    # Inicializar el generador de embeddings con un modelo de Hugging Face
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Inicializar Chroma usando Langchain (almacenamiento persistente opcional)
    chroma_db = Chroma(
        persist_directory=persist_directory,
        collection_name=c_name,
        embedding_function=embedding_model,
    )

    # Convertir secciones a documentos de Langchain
    docs = []
    for section in sections:
        metadata = {
            "id": section["id"],
            "title": section["title"],
            "origin": section.get("origin", section.get("origen", "Desconocido")),
            "id_referencia": section["id_referencia"],
        }
        docs.append(Document(page_content=section["text"], metadata=metadata))

    # Agregar documentos a Chroma
    chroma_db.add_documents(docs)
    return chroma_db


def get_full_context(vectorstore, doc):
    """Recupera el contexto completo de una sección, incluyendo su padre y todos sus hijos."""

    full_context = doc.page_content + "\n\n"

    # Extraer ID hijos y padre del metadata
    metadata = doc.metadata
    id_doc = metadata["id"]
    hijos = metadata["children"].split(", ")
    padre = metadata["parent"]

    if padre != "None":
        filtro_padre = vectorstore.get(where={"id": padre})
        ids_hijo_padre = filtro_padre["metadatas"][0]["children"].split(", ")

        if len(ids_hijo_padre) > 0:
            for id in ids_hijo_padre:
                if id != id_doc:
                    filtro_hijo_padre = vectorstore.get(where={"id": id})
                    doc_hijo_padre = filtro_hijo_padre["documents"][0]
                    titulo_hijo_padre = filtro_hijo_padre["metadatas"][0]["title"]
                    full_context += titulo_hijo_padre + "\n" + doc_hijo_padre + "\n\n"

        doc_padre = filtro_padre["documents"][0]
        titulo_padre = filtro_padre["metadatas"][0]["title"]
        full_context += titulo_padre + "\n" + doc_padre + "\n\n"
    print("metadata: ", metadata)
    print("hijos: ", hijos)
    if hijos != [""]:
        for id in hijos:
            filtro_hijo = vectorstore.get(where={"id": id})
            print(filtro_hijo)
            doc_hijo = filtro_hijo["documents"][0]
            titulo_hijo = filtro_hijo["metadatas"][0]["title"]
            full_context += titulo_hijo + "\n" + doc_hijo + "\n\n"

    return full_context


def get_full_chunk(vectorstore, doc):
    full_chunk_text = ""
    referencia = ""

    metadata = doc.metadata
    id_referencia = metadata["id_referencia"]

    if id_referencia != "None":
        filtro_referencia = vectorstore.get(where={"id": id_referencia})
        doc_referencia = filtro_referencia["documents"][0]
        full_chunk_text += doc_referencia
        referencia += (
            filtro_referencia["metadatas"][0]["origin"]
            + " - "
            + filtro_referencia["metadatas"][0]["title"]
            + "\n\n"
        )

    return full_chunk_text, referencia


def get_full_reference(vectorstore, doc, registro):
    full_chunk_text = ""
    referencia = ""

    metadata = doc.metadata
    id_referencias = metadata["parent"].split(", ")

    if id_referencias[0] != "None":
        for ref in id_referencias:
            if ref in registro:
                continue
            else:
                filtro_referencia = vectorstore.get(where={"id": ref})
                try:
                    doc_referencia = filtro_referencia["documents"][0]
                    full_chunk_text += doc_referencia
                    referencia += (
                        filtro_referencia["metadatas"][0]["origin"]
                        + " - "
                        + filtro_referencia["metadatas"][0]["title"]
                        + "\n\n"
                    )
                    registro.append(ref)
                except:
                    print("Error en al intentar recuperar el chunk con id: ", ref)

    return full_chunk_text, referencia


def inicializar_retriever_vectorstore(k=5):

    # Inicializar ChromaDB y HuggingFaceEmbeddings si esxiste la base de datos trae la info. Lo que carga es un archivo markdow al cual debes darle.
    # Yo te paso esos formatos. Aun no he probado meterle mas de un documento, por lo que estoy haciendo pruebas por separado pero de momento dejalo asi.
    # Si tu quieres probar el otro documento o los solo tiene que descomentar parse_markdown_edaes y/o parse_markdown_mater.

    # CARGAR O CREAR BASE DE DATOS COMPLETA
    if True and os.path.exists(
        persist_directory
    ):  # Base de datos completa con todos los documentos
        print("Cargando Base de datos completa... ")
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vectorstore_edaes = Chroma(
            persist_directory=persist_directory,
            collection_name="fac_documents_complete",
            embedding_function=embedding_model,
        )
        vectorstore_complete = (
            vectorstore_edaes  # Definir variable unificada en este scope
        )
        vectorstore_pruebas = vectorstore_edaes  # Misma colección para todos
        vectorstore_resumen_edaes = vectorstore_edaes
        vectorstore_edaes_seg = vectorstore_edaes
        vectorstore_newMater = vectorstore_edaes
        vectorstore_historia = vectorstore_edaes
    else:
        print("Creando Base de datos... ")
        chunks = []
        chunks_edaes = []
        chunk_resumen_edaes = []
        chunks_seg = []

        markdown_content_EDAES = read_markdown_file("FAC_Documents/rag_files/edaes.md")
        markdown_content_MATER = read_markdown_file(
            "FAC_Documents/rag_files/manual_de_terminos_fuerza_aerea_colombiana.md"
        )
        markdown_content_PORFAC = read_markdown_file(
            "FAC_Documents/rag_files/reglamento_porfac.md"
        )
        markdown_content_resumen_edaes = read_markdown_file(
            "FAC_Documents/rag_files/Preguntas_FAC.md"
        )
        markdown_contente_newMater = read_markdown_file(
            "FAC_Documents/rag_files/NuevoMATER.md"
        )
        markdown_content_Historia = read_markdown_file(
            "FAC_Documents/rag_files/Historia.md"
        )

        # Crear variable separada para chunks de NuevoMATER
        chunks_newMater = []
        create_specialized_mater_chunks(markdown_contente_newMater, chunks_newMater)

        # Crear variable separada para chunks de Historia
        chunks_Historia = []

        parse_markdown_edaes(markdown_content_EDAES, chunks_edaes)
        prototipo_1(chunks_edaes, chunks_seg)

        # Usar la función original para el MATER
        parse_markdown_mater(markdown_content_MATER, chunks)

        parse_markdown_porfac(markdown_content_PORFAC, chunks)
        parse_markdown_resumen_edaes(
            markdown_content_resumen_edaes, chunk_resumen_edaes
        )
        parse_markdown_Historia(markdown_content_Historia, chunks_Historia)

        # COMBINAR TODOS LOS DOCUMENTOS EN UNA SOLA COLECCIÓN
        print("📚 Combinando todos los documentos...")
        all_chunks = []

        # Agregar todos los chunks con verificación
        if chunks_edaes:
            all_chunks.extend(chunks_edaes)
            print(f"✅ EDAES: {len(chunks_edaes)} secciones")

        if chunks:
            all_chunks.extend(chunks)
            print(f"✅ MATER + PORFAC: {len(chunks)} secciones")

        if chunk_resumen_edaes:
            all_chunks.extend(chunk_resumen_edaes)
            print(f"✅ Resumen EDAES: {len(chunk_resumen_edaes)} secciones")

        if chunks_newMater:
            all_chunks.extend(chunks_newMater)
            print(f"✅ Nuevo MATER: {len(chunks_newMater)} secciones")

        if chunks_Historia:
            all_chunks.extend(chunks_Historia)
            print(f"✅ Historia: {len(chunks_Historia)} secciones")

        if chunks_seg:
            all_chunks.extend(chunks_seg)
            print(f"✅ EDAES Segmentado: {len(chunks_seg)} secciones")

        print(f"📊 Total documentos combinados: {len(all_chunks)}")

        # Verificar que Honor Militar esté incluido
        honor_militar_found = False
        for chunk in all_chunks:
            if (
                "honor militar" in chunk.get("text", "").lower()
                or "honor militar" in chunk.get("title", "").lower()
            ):
                honor_militar_found = True
                print(
                    f"✅ Honor Militar encontrado: '{chunk['title']}' (origen: {chunk.get('origin', 'N/A')})"
                )
                break

        if not honor_militar_found:
            print(
                "❌ ADVERTENCIA: Honor Militar NO encontrado en la colección combinada"
            )

        # CREAR UNA SOLA COLECCIÓN CON TODOS LOS DOCUMENTOS
        vectorstore_complete = store_in_chromadb(
            all_chunks,
            persist_directory,
            "fac_documents_complete",
            embedding_model_name,
        )

        # Usar la misma colección para todos los vectorstores
        vectorstore_edaes = vectorstore_complete
        vectorstore_pruebas = vectorstore_complete
        vectorstore_resumen_edaes = vectorstore_complete
        vectorstore_edaes_seg = vectorstore_complete
        vectorstore_newMater = vectorstore_complete
        vectorstore_historia = vectorstore_complete

    # prueba = vectorstore.get(where={"id": "chunk_5"})
    # print(prueba)

    # Se crea el retriever quien ser el encargado de traer los documento segun el query, la k es el numero de documento que se quiere recuperar.
    # En esta caso seria solo 5 documentos, pero si quieres porbar trae mas.

    # Validar que todos los vectorstores se crearon correctamente
    vectorstores = [
        ("vectorstore_edaes", vectorstore_edaes),
        ("vectorstore_pruebas", vectorstore_pruebas),
        ("vectorstore_newMater", vectorstore_newMater),
        ("vectorstore_resumen_edaes", vectorstore_resumen_edaes),
        ("vectorstore_edaes_seg", vectorstore_edaes_seg),
        ("vectorstore_historia", vectorstore_historia),
    ]

    for name, vs in vectorstores:
        if vs is None:
            raise ValueError(f"Error: {name} no se pudo crear correctamente")

    # Crear Retrievers estándar (Vector Search)
    retriever_edaes = vectorstore_edaes.as_retriever(search_kwargs={"k": k})
    retriever_pruebas = vectorstore_pruebas.as_retriever(search_kwargs={"k": k})
    retriever_newMater = vectorstore_newMater.as_retriever(search_kwargs={"k": k})
    retriever_historia = vectorstore_historia.as_retriever(search_kwargs={"k": k})
    retriever_resumen_edaes = vectorstore_resumen_edaes.as_retriever(
        search_kwargs={"k": k}
    )
    retriever_edaes_seg = vectorstore_edaes_seg.as_retriever(search_kwargs={"k": k})

    print(f"✅ Retrievers (VectorStore) inicializados correctamente.")

    return (
        retriever_edaes,  # 1
        retriever_pruebas,  # 2
        retriever_newMater,  # 3
        vectorstore_edaes,  # 4 (Se mantiene por compatibilidad, aunque no se use para búsqueda)
        retriever_resumen_edaes,  # 5
        retriever_edaes_seg,  # 6
        retriever_historia,  # 7
    )


# retriever, vectorstore = inicializar_retriever_vectorstore()
# llm_chain = inicializar_modelo(model_name, temperature)


def chatbot_response(
    query,
    usar_full_context,
    llm_chain,
    retriever_edaes,
    retriever_pruebas,
    reranker,
    vectorstore,
    retriever_resumen_edaes,
    retriever_edaes_seg,
    retriever_newMater,
    retriever_historia,
    re_type,
):

    # Se obtienen los documentos relevantes
    query_limpia = limpiar_string(query)

    # Aumentar la K del retriever para que la búsqueda pura vectorial traiga suficientes documentos
    # de toda la base de datos, y luego el Reranker escoja los 20 mejores exactos.
    # Dado que todos los retrievers son iguales según la configuración original de la rama, solo invocamos uno.
    old_k = retriever_edaes.search_kwargs.get("k", 5)
    retriever_edaes.search_kwargs["k"] = 600
    docs_globales = retriever_edaes.invoke(query)
    retriever_edaes.search_kwargs["k"] = old_k

    # Combinar TODOS los documentos
    docs = docs_globales

    # MEJORAR: Sistema de búsqueda inteligente
    query_keywords = query_limpia.lower().split()

    # Detectar tipo de consulta
    is_definition_query = any(
        word in query.lower()
        for word in [
            "que es",
            "qué es",
            "definición",
            "significa",
            "concepto",
            "qué",
            "que",
            "cuales son",
            "cuáles son",
            "principios",
            "como se define",
            "cómo se define",
            "explicame",
            "explícame",
        ]
    )

    # BÚSQUEDA UNIVERSAL: Sin restricciones de dominio, buscar en TODOS los documentos
    print(f"🔍 Búsqueda universal con VectorStore para: '{query}'")

    # Aumentamos k aquí para asegurar que el re-ranking tenga suficientes candidatos
    # Recuperamos documentos combinados de TODOS los orígenes
    docs_unique = []
    seen = set()

    for doc in docs_globales:
        doc_id = doc.metadata.get("id", doc.page_content[:30])
        if doc_id not in seen:
            seen.add(doc_id)
            docs_unique.append(doc)

    print(f"📄 Documentos recuperados por VectorStore: {len(docs_unique)}")

    # Pasamos directamente al re-ranking sin lógica vectorial/híbrida compleja
    final_docs = docs_unique

    # Re-ranking con CrossEncoder
    re_ranked_docs = re_rank_docs(query, final_docs, reranker)

    # Mostrar información de debug
    print(f"🏆 Top 3 documentos después de re-ranking:")
    for i, doc in enumerate(re_ranked_docs[:3]):
        origin = doc.metadata.get("origin", "Desconocido")
        title = doc.metadata.get("title", "Sin título")
        content_preview = doc.page_content[:100].replace("\n", " ")
        print(f"   {i+1}. {origin} - {title}")
        print(f"      {content_preview}...")

    # Construir contexto final
    fc = ""
    referencias_set = set()
    # Lista necesaria para la función get_full_reference si se usa
    registro = []

    # Solo enviar los top 5 al LLM para no confundirlo
    for doc in re_ranked_docs[:5]:
        origin = doc.metadata.get("origin", "Búsqueda híbrida")

        # MODIFICACIÓN: SOLO MENCIONAR EL DOCUMENTO DE ORIGEN (NO EL TÍTULO DEL CHUNK)
        referencias_set.add(origin)

        if usar_full_context and origin == "EDAES":
            try:
                fc += get_full_context(vectorstore, doc) + "\n\n"
            except Exception as e:
                # Fallback: usar contenido directo si falla el contexto completo
                fc += doc.page_content + "\n\n"
        elif (
            usar_full_context
            and origin == "EDAES Segmentado"
            and "id_referencia" in doc.metadata
        ):
            try:
                text, ref = get_full_chunk(vectorstore, doc)
                fc += text + "\n\n"
            except Exception as e:
                # Fallback: usar contenido directo
                fc += doc.page_content + "\n\n"
        elif (
            usar_full_context and origin == "Resumen EDAES" and "parent" in doc.metadata
        ):
            try:
                text, ref = get_full_reference(vectorstore, doc, registro)
                fc += text + "\n\n"
            except Exception as e:
                # Fallback: usar contenido directo
                fc += doc.page_content + "\n\n"
        else:
            fc += doc.page_content + "\n\n"

    # Convertir set a string sin duplicados
    referencias = "\n".join(referencias_set)

    start = time.time()
    respuesta = llm_chain.invoke({"contexto": fc, "pregunta": query})
    print("-" * 50)
    print(query)
    print("-" * 50)
    print(fc)
    stop = time.time()
    tiempo_res = stop - start

    return respuesta, tiempo_res, referencias


def create_specialized_mater_chunks(md_text, sections):
    """
    Función especializada SOLO para NuevoMATER.md
    optimizada para responder preguntas sobre términos aeronáuticos.
    """
    lines = md_text.split("\n")
    # CORREGIR: Regex generalizado para capturar múltiples variaciones (incluyendo **PREVAC***.*)
    term_regex_broad = re.compile(r"^\*\*(.+?)\*\*(?:[:.\*]*)\s*(.*)$")

    current_term = None
    current_definition = []

    for line in lines:
        line = line.strip()

        if not line or line.startswith("![]") or line.startswith("Figura"):
            continue

        # Intentar match con regex general
        term_match = term_regex_broad.match(line)

        if term_match:
            raw_term = term_match.group(1).strip()

            # Procesar término anterior si existe
            if current_term and current_definition:
                _process_nuevo_mater_term(current_term, current_definition, sections)

            # Iniciar nuevo término (limpiando asteriscos y demás del final)
            current_term = raw_term.rstrip(".:* ")
            definition_start = term_match.group(2).strip()
            current_definition = [definition_start] if definition_start else []

        elif current_term and line:
            # Agregar línea a definición actual
            if not (line.startswith("(") and "Manual" in line and line.endswith(")")):
                current_definition.append(line)

    # Procesar último término
    if current_term and current_definition:
        _process_nuevo_mater_term(current_term, current_definition, sections)


def _process_nuevo_mater_term(term, definition_lines, sections):
    """
    Procesa un término individual del NuevoMATER y crea múltiples chunks optimizados.
    """
    full_definition = " ".join(definition_lines).strip()

    if len(full_definition) < 20:
        return

    # Limpiar definición
    full_definition = re.sub(r"\s+", " ", full_definition)
    full_definition = re.sub(r"\*\*(.+?)\*\*", r"\1", full_definition)

    # Crear chunks especializados para mejor recuperación
    specialized_chunks = [
        {
            "pregunta": f"¿Qué es {term}?",
            "respuesta": f"{term} es {full_definition}",
            "tipo": "definicion_directa",
        },
        {
            "pregunta": f"¿Cuál es la definición de {term}?",
            "respuesta": f"La definición de {term} es: {full_definition}",
            "tipo": "definicion_formal",
        },
        {
            "pregunta": f"¿Qué significa {term}?",
            "respuesta": f"{term} significa {full_definition}",
            "tipo": "significado",
        },
        {
            "pregunta": f"Explícame qué es {term}",
            "respuesta": f"{term} se define como: {full_definition}",
            "tipo": "explicacion",
        },
    ]

    for idx, chunk_data in enumerate(specialized_chunks):
        combined_text = f"{chunk_data['pregunta']} {chunk_data['respuesta']}"

        node = {
            "id": f"chunk_nuevo_mater_{len(sections)}_{idx}",
            "level": 1,
            "title": f"{term} ({chunk_data['tipo']})",
            "text": combined_text,  # Conservar formato natural para LLM
            "type": "Definición NuevoMATER",
            "parent": "NuevoMATER FAC",
            "children": [],
            "origin": "NuevoMATER FAC",
            "termino": term,
            "pregunta_tipo": chunk_data["tipo"],
        }
        sections.append(node)
