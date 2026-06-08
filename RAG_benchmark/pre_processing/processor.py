import re
import os
import unicodedata

from numpy import rint
import chromadb
import time
from langchain_text_splitters import CharacterTextSplitter
import ollama
import argparse
import json
import csv

from langchain_ollama import OllamaLLM

from langchain_huggingface import HuggingFaceEmbeddings
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

QUESTION_REGEX = re.compile(r"^###\s(.*)$")
REF_REGEX = re.compile(r"\*\*(.*?)\*\*")

# Texto que debe ser excluido
EXCLUDED_TEXTS = {
    "**2042**",
    "*2042*",
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

# Variables de configuración globales (no dependen de args)
current_dir = os.path.dirname(os.path.abspath(__file__))  # Dir pre_processing
base_dir = os.path.dirname(current_dir)  # Dir RAG_benchmark

c_name = "pruebas"
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
persist_directory = os.path.join(base_dir, "data", "chroma_db")
re_ranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

input_dir = os.path.join(base_dir, "FAC_Documents", "rag_files")


# ============================================================================
# FUNCIONES UTILITARIAS
# ============================================================================


def load_goalset(ruta_json):
    with open(ruta_json, "r", encoding="utf-8") as f:
        return json.load(f)


def limpiar_string(texto):
    texto = texto.lower()
    texto = (
        unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    )
    texto = re.sub(r"[^a-z0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def sanitize_filename(name):
    return re.sub(r'[.<>:"/\\|?*]', "_", name)


def re_rank_docs(query, docs, reranker):
    texts = [doc.page_content for doc in docs]
    pairs = [[query, text] for text in texts]
    scores = reranker.predict(pairs)
    sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc[0] for doc in sorted_docs]


def eliminar_chain_of_thought(texto_respuesta):
    return re.sub(
        r"<tool_call>.*?</tool_call>",
        "",
        texto_respuesta,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# ============================================================================
# PARSERS DE MARKDOWN
# ============================================================================


def parse_markdown_mater(md_text, sections):
    lines = md_text.split("\n")
    stack = []

    for line in lines:
        title_match = TITLE_REGEX_MATER.match(line)
        subtitle_match = SUBTITLE_REGEX_MATER.match(line)
        sub_subtitle_match = SUB_SUBTITLE_REGEX_MATER.match(line)
        definition_match = DEFINITION_REGEX_MATER.match(line)

        if title_match:
            level = len(title_match.group(1))
            title = title_match.group(2)
            section_type = "Título"
            stack.clear()

        elif subtitle_match:
            level = len(subtitle_match.group(1))
            title = subtitle_match.group(2)
            section_type = "Subtítulo"
            stack = stack[:1]

        elif sub_subtitle_match:
            level = len(sub_subtitle_match.group(1))
            title = sub_subtitle_match.group(2)
            section_type = "Sub-subtítulo"
            stack = stack[:2]

        elif definition_match:
            level = 4
            title = definition_match.group(1)
            section_type = "Definición"
            def_text = f"QUÉ ES {definition_match.group(1)}? {definition_match.group(1)} es {definition_match.group(2)}"
            stack = stack[:3]

        else:
            if sections:
                sections[-1]["text"] += limpiar_string(line) + "\n"
            continue

        text_content = limpiar_string(def_text) if definition_match else ""

        node = {
            "id": f"chunk_{len(sections)}",
            "level": level,
            "title": title.strip(),
            "text": text_content,
            "type": section_type,
            "parent": stack[-1]["id"] if stack else "None",
            "children": [],
            "origen": "Manual de términos FAC",
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)
        stack.append(node)


def parse_markdown_edaes(md_text, sections):
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
                continue
            level = stack[-1]["level"] + 1
            title = generic_title_match.group(1)
            section_type = "Título genérico"
            stack = stack[: level - 1]

        else:
            if sections:
                sections[-1]["text"] += limpiar_string(line) + "\n"
            continue

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
            "origen": "EDAES",
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)

        if section_type != "Título genérico":
            stack.append(node)


def parse_markdown_porfac(md_text, sections):
    lines = md_text.split("\n")
    stack = []

    for line in lines:
        reg_match = REGLAMENTO_REGEX_PORFAC.match(line)
        tittle_match = TITLE_REGEX_PORFAC.match(line)
        cap_mathc = CAPITULO_REGEX_PORFAC.match(line)
        art_match = ARTICULO_REGEX_PORFAC.match(line)
        parag_match = PARAGRAFO_REGEX_PORFAC.match(line)

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
                "origen": "Reglamento de las PORFAC",
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
                "origen": "Reglamento de las PORFAC",
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
            level = 5

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
                "origen": "Reglamento de las PORFAC",
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

        node = {
            "id": f"chunk_{len(sections)}",
            "level": level,
            "title": title.strip(),
            "text": "",
            "type": section_type,
            "parent": stack[-1]["id"] if stack else "None",
            "children": [],
            "origen": "Reglamento de las PORFAC",
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)
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
                continue
            level = stack[-1]["level"] + 1
            title = generic_title_match.group(1)
            section_type = "Título genérico"
            stack = stack[: level - 1]
        else:
            if sections:
                sections[-1]["text"] += limpiar_string(line) + "\n"
            continue

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
            "origen": "Historia",
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)

        if section_type != "Título genérico":
            stack.append(node)


def parse_markdown_resumen_edaes(md_text, sections):
    lines = md_text.split("\n")

    for line in lines:
        question_match = QUESTION_REGEX.match(line)
        ref_match = REF_REGEX.match(line)

        if question_match:
            level = 1
            title = question_match.group(1)
            section_type = "Pregunta"
        elif ref_match:
            sections[-1]["parent"] = ref_match.group(1)
            continue
        else:
            if sections:
                sections[-1]["text"] += limpiar_string(line) + "\n"
            continue

        node = {
            "id": f"chunk_{len(sections)}",
            "level": level,
            "title": title.strip(),
            "text": limpiar_string(title.strip()) + " ",
            "type": section_type,
            "parent": "",
            "children": [],
            "origen": "Resumen EDAES",
        }

        sections.append(node)


# ============================================================================
# FUNCIONES DE CHROMADB
# ============================================================================


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
        origen = chunk["origen"] + " Segmentado"
        for text in split_text:
            chunks_segmentados.append(
                {
                    "id": f"chunk_{id_seg}",
                    "title": chunk_title,
                    "text": limpiar_string(chunk_title) + " " + text,
                    "origen": origen,
                    "id_referencia": id_referencia,
                }
            )
            id_seg += 1


def store_in_chromadb(sections, persist_directory, c_name, embedding_model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    chroma_db = Chroma(
        persist_directory=persist_directory,
        collection_name=c_name,
        embedding_function=embedding_model,
    )

    docs = []
    for section in sections:
        metadata = {
            "id": section["id"],
            "title": section["title"],
            "level": section["level"],
            "parent": section["parent"],
            "children": (
                ", ".join(section["children"])
                if isinstance(section.get("children"), list)
                else section.get("children", "")
            ),
            "origin": section["origen"],
        }
        docs.append(Document(page_content=section["text"], metadata=metadata))

    chroma_db.add_documents(docs)
    return chroma_db


def store_in_chromadb_seg(sections, persist_directory, c_name, embedding_model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    chroma_db = Chroma(
        persist_directory=persist_directory,
        collection_name=c_name,
        embedding_function=embedding_model,
    )

    docs = []
    for section in sections:
        metadata = {
            "id": section["id"],
            "title": section["title"],
            "origin": section["origen"],
            "id_referencia": section["id_referencia"],
        }
        docs.append(Document(page_content=section["text"], metadata=metadata))

    chroma_db.add_documents(docs)
    return chroma_db


def get_full_context(vectorstore, doc):
    full_context = doc.page_content + "\n\n"

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

    if hijos != "":
        for id in hijos:
            filtro_hijo = vectorstore.get(where={"id": id})
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


# ============================================================================
# FUNCIONES DE INICIALIZACIÓN
# ============================================================================


def inicializar_modelo(model_name="mistral", temperature=0.5, prompt=None):
    llm = OllamaLLM(model=model_name, temperature=temperature)
    llm_chain = prompt | llm
    return llm_chain


def inicializar_retriever_vectorstore(k=5):
    if os.path.exists(persist_directory):
        print("Cargando Base de datos... ")
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vectorstore_edaes = Chroma(
            persist_directory=persist_directory,
            collection_name="edaes",
            embedding_function=embedding_model,
        )
        vectorstore_pruebas = Chroma(
            persist_directory=persist_directory,
            collection_name="pruebas",
            embedding_function=embedding_model,
        )
        vectorstore_resumen_edaes = Chroma(
            persist_directory=persist_directory,
            collection_name="resumen_edaes",
            embedding_function=embedding_model,
        )
        vectorstore_edaes_seg = Chroma(
            persist_directory=persist_directory,
            collection_name="edaes_seg",
            embedding_function=embedding_model,
        )
        vectorstore_newMater = Chroma(
            persist_directory=persist_directory,
            collection_name="newMater",
            embedding_function=embedding_model,
        )
        vectorstore_Historia = Chroma(
            persist_directory=persist_directory,
            collection_name="Historia",
            embedding_function=embedding_model,
        )
    else:
        print(
            "ERROR: Database doesn't exist. Run processor.py first to create databases."
        )
        return None

    retriever_edaes = vectorstore_edaes.as_retriever(search_kwargs={"k": k})
    retriever_pruebas = vectorstore_pruebas.as_retriever(search_kwargs={"k": k})
    retriever_newMater = vectorstore_newMater.as_retriever(search_kwargs={"k": k})
    retriever_historia = vectorstore_Historia.as_retriever(search_kwargs={"k": k})
    retriever_resumen_edaes = vectorstore_resumen_edaes.as_retriever(
        search_kwargs={"k": k}
    )
    retriever_edaes_seg = vectorstore_edaes_seg.as_retriever(search_kwargs={"k": k})

    return (
        retriever_edaes,  # 1
        retriever_pruebas,  # 2
        retriever_newMater,  # 3
        vectorstore_edaes,  # 4
        retriever_resumen_edaes,  # 5
        retriever_edaes_seg,  # 6
        retriever_historia,  # 7
    )


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
    PROMPT,
):
    query_limpia = limpiar_string(query)

    docs_resumen_edaes = retriever_resumen_edaes.invoke(query_limpia)
    docs_pruebas = retriever_pruebas.invoke(query_limpia)
    docs_edaes = retriever_edaes.invoke(query_limpia)
    docs_historia = retriever_historia.invoke(query_limpia)
    docs_newMater = retriever_newMater.invoke(query_limpia)

    docs = (
        docs_pruebas + docs_edaes + docs_historia + docs_resumen_edaes + docs_newMater
    )

    re_ranked_docs = re_rank_docs(query_limpia, docs, reranker)

    fc = ""
    referencias_str = ""
    registro = []

    for doc in re_ranked_docs:
        if usar_full_context and doc.metadata["origin"] == "EDAES":
            fc += get_full_context(vectorstore, doc) + "\n\n"
            referencias_str += (
                doc.metadata["origin"] + " - " + doc.metadata["title"] + "\n\n"
            )
        elif usar_full_context and doc.metadata["origin"] == "Historia":
            fc += get_full_context(vectorstore, doc) + "\n\n"
            referencias_str += (
                doc.metadata["origin"] + " - " + doc.metadata["title"] + "\n\n"
            )
        else:
            fc += doc.page_content + "\n\n"
            referencias_str += (
                doc.metadata["origin"] + " - " + doc.metadata["title"] + "\n\n"
            )

    respuesta = llm_chain.invoke({"contexto": fc, "pregunta": query})

    return respuesta, referencias_str


# ============================================================================
# EJECUCIÓN PRINCIPAL — Todo lo que depende de args CLI va aquí
# ============================================================================

if __name__ == "__main__":

    # --- Parseo de argumentos ---
    parser = argparse.ArgumentParser(
        description="Configura el modelo y la temperatura para Ollama."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:4b",
        help="Nombre del modelo de Ollama a usar.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperatura del modelo (0.0 a 1.0)",
    )
    parser.add_argument(
        "--goalset",
        type=str,
        default=None,
        help="Ruta del archivo GoalSet JSON",
    )
    parser.add_argument(
        "--doc_name",
        type=str,
        default="DEFAULT",
        help="Nombre del documento para el archivo CSV",
    )
    args = parser.parse_args()

    model_name = args.model
    temperature = args.temperature
    doc_name = args.doc_name

    # --- Cargar GoalSet ---
    ruta_json = os.path.join(input_dir, "GoalSetNewMater.json")
    if args.goalset and os.path.exists(args.goalset):
        ruta_json = args.goalset

    preguntas_respuestas = load_goalset(ruta_json)

    # --- Sanitizar nombre del modelo (CORREGIDO: antes estaba fuera del bloque) ---
    sanitized_model_name = sanitize_filename(model_name)
    print(sanitized_model_name)

    # --- Cargar o crear base de datos ChromaDB ---
    if os.path.exists(persist_directory):
        print("Cargando Base de datos... ")
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vectorstore_edaes = Chroma(
            persist_directory=persist_directory,
            collection_name="edaes",
            embedding_function=embedding_model,
        )
        vectorstore_pruebas = Chroma(
            persist_directory=persist_directory,
            collection_name="pruebas",
            embedding_function=embedding_model,
        )
        vectorstore_resumen_edaes = Chroma(
            persist_directory=persist_directory,
            collection_name="resumen_edaes",
            embedding_function=embedding_model,
        )
        vectorstore_edaes_seg = Chroma(
            persist_directory=persist_directory,
            collection_name="edaes_seg",
            embedding_function=embedding_model,
        )
        vectorstore_newMater = Chroma(
            persist_directory=persist_directory,
            collection_name="newMater",
            embedding_function=embedding_model,
        )
        vectorstore_Historia = Chroma(
            persist_directory=persist_directory,
            collection_name="Historia",
            embedding_function=embedding_model,
        )
    else:
        print("Creando Base de datos... ")
        chunks = []
        chunks_edaes = []
        chunk_resumen_edaes = []
        chunks_seg = []
        chunks_Historia = []

        markdown_content_EDAES = read_markdown_file(os.path.join(input_dir, "edaes.md"))
        markdown_content_MATER = read_markdown_file(
            os.path.join(input_dir, "manual_de_terminos_fuerza_aerea_colombiana.md")
        )
        markdown_content_PORFAC = read_markdown_file(
            os.path.join(input_dir, "reglamento_porfac.md")
        )
        markdown_content_resumen_edaes = read_markdown_file(
            os.path.join(input_dir, "Preguntas_FAC.md")
        )
        markdown_content_newMater = read_markdown_file(
            os.path.join(input_dir, "NuevoMater.md")
        )
        markdown_content_Historia = read_markdown_file(
            os.path.join(input_dir, "Historia.md")
        )

        parse_markdown_edaes(markdown_content_EDAES, chunks_edaes)
        print(len(chunks_edaes))
        prototipo_1(chunks_edaes, chunks_seg)
        print(len(chunks_seg))
        parse_markdown_mater(markdown_content_MATER, chunks)
        print(len(chunks))
        parse_markdown_porfac(markdown_content_PORFAC, chunks)
        print(len(chunks))
        parse_markdown_resumen_edaes(
            markdown_content_resumen_edaes, chunk_resumen_edaes
        )
        print(len(chunk_resumen_edaes))
        parse_markdown_Historia(markdown_content_Historia, chunks_Historia)
        print(f"Historia chunks: {len(chunks_Historia)}")
        parse_markdown_mater(markdown_content_newMater, chunks)
        print(len(chunks))

        vectorstore_edaes = store_in_chromadb(
            chunks_edaes, persist_directory, "edaes", embedding_model_name
        )
        vectorstore_pruebas = store_in_chromadb(
            chunks, persist_directory, "pruebas", embedding_model_name
        )
        vectorstore_resumen_edaes = store_in_chromadb(
            chunk_resumen_edaes,
            persist_directory,
            "resumen_edaes",
            embedding_model_name,
        )
        vectorstore_edaes_seg = store_in_chromadb_seg(
            chunks_seg, persist_directory, "edaes_seg", embedding_model_name
        )
        vectorstore_newMater = store_in_chromadb(
            chunks, persist_directory, "newMater", embedding_model_name
        )
        vectorstore_Historia = store_in_chromadb(
            chunks_Historia, persist_directory, "Historia", embedding_model_name
        )

    # --- Crear retrievers ---
    k = 5
    retriever_edaes = vectorstore_edaes.as_retriever(search_kwargs={"k": k})
    retriever_pruebas = vectorstore_pruebas.as_retriever(search_kwargs={"k": k})
    retriever_resumen_edaes = vectorstore_resumen_edaes.as_retriever(
        search_kwargs={"k": k}
    )
    retriever_edaes_seg = vectorstore_edaes_seg.as_retriever(search_kwargs={"k": k})
    retriever_newMater = vectorstore_newMater.as_retriever(search_kwargs={"k": k})
    retriever_historia = vectorstore_Historia.as_retriever(search_kwargs={"k": k})

    # --- Modelo LLM y Prompt ---
    llm = OllamaLLM(model=model_name, temperature=temperature)

    prompt_template = """
            Eres un asistente especializado en responder preguntas sobre la Fuerza Aérea Colombiana, o sus siglas FAC, basadas en
            documentos proporcionados. Tu respuesta debe basarse únicamente en la información de los documentos recuperados.
            Evita estrictamente frases como 'no sé', 'no encontré información al respecto', 'no puedo responder', 'la información no está disponible' o similares.
            Si la información para responder una parte de la pregunta no se encuentra explícitamente en el contexto,
            céntrate en responder las partes de la pregunta que sí puedes abordar con el contexto.
            Proporciona una respuesta concisa, directa y sin conversación innecesaria.

            Contexto: {contexto}

            Pregunta: {pregunta}

            Respuesta: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["contexto", "pregunta"]
    )

    # --- Reranker ---
    reranker = CrossEncoder(re_ranker_model)

    # --- Configuración del proceso ---
    num_iterations = 10
    usar_full_context = False

    # --- Proceso principal ---
    print(f"\n{'='*81}")
    print(f"INICIANDO PROCESO CON TEMPERATURA: {temperature}, DOCUMENTO: {doc_name}")
    print(f"{'='*81}\n")

    results_dir = os.path.join(base_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    temp_str = f"{temperature:.1f}".replace(".", "_")
    output_filename = os.path.join(
        results_dir, f"{sanitized_model_name}_temperatura_{temp_str}_{doc_name}.csv"
    )
    output_summary_filename = os.path.join(
        results_dir,
        f"{sanitized_model_name}_temperatura_{temp_str}_{doc_name}_summary.csv",
    )

    csv_headers = [
        "Pregunta_ID",
        "Iteracion_Num",
        "Pregunta",
        "Respuesta_Esperada",
        "Respuesta_Limpia",
        "Num_Chunks",
        "Tiempo_Segundos",
    ]

    todos_los_tiempos_para_resumen_global = []
    datos_resumen_por_pregunta_lista = []

    with open(output_filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

        for idx, item in enumerate(preguntas_respuestas, start=1):
            pregunta = item["pregunta"]
            respuesta_esperada = item["respuesta"]
            tiempos_iteraciones_pregunta_actual = []

            print(
                f"\nProcesando pregunta {idx} de {len(preguntas_respuestas)} (Temp: {temperature})"
            )
            print(f"{pregunta}\n")

            for iter_num_actual in range(1, num_iterations + 1):
                print(
                    f"Ejecutando iteracion {iter_num_actual} de {num_iterations} para pregunta {idx}"
                )

                query_limpia = limpiar_string(pregunta)
                docs_resumen_edaes = retriever_resumen_edaes.invoke(query_limpia)
                docs_pruebas = retriever_pruebas.invoke(query_limpia)
                docs_seg = retriever_edaes_seg.invoke(query_limpia)
                docs_edaes = retriever_edaes.invoke(query_limpia)
                docs_historia = retriever_historia.invoke(query_limpia)

                docs = docs_pruebas + docs_edaes + docs_historia

                re_ranked_docs = re_rank_docs(query_limpia, docs, reranker)

                retrieved_doc_ids_for_question = [
                    doc.metadata.get("id", "id_desconocido") for doc in re_ranked_docs
                ]
                num_chunks_retrieved = len(retrieved_doc_ids_for_question)

                fc = ""
                referencias_str = ""
                registro = []
                for doc in re_ranked_docs:
                    if usar_full_context and doc.metadata["origin"] == "EDAES":
                        fc += get_full_context(vectorstore_edaes, doc) + "\n\n"
                        referencias_str += (
                            doc.metadata["origin"]
                            + " - "
                            + doc.metadata["title"]
                            + "\n\n"
                        )
                    elif (
                        usar_full_context
                        and doc.metadata["origin"] == "EDAES Segmentado"
                    ):
                        text, ref = get_full_chunk(vectorstore_edaes, doc)
                        fc += text + "\n\n"
                        referencias_str += ref
                    elif (
                        usar_full_context and doc.metadata["origin"] == "Resumen EDAES"
                    ):
                        text, ref = get_full_reference(vectorstore_edaes, doc, registro)
                        fc += text + "\n\n"
                        referencias_str += ref
                    else:
                        fc += doc.page_content + "\n\n"
                        referencias_str += (
                            doc.metadata["origin"]
                            + " - "
                            + doc.metadata["title"]
                            + "\n\n"
                        )

                final_prompt = PROMPT.format(contexto=fc, pregunta=pregunta)

                start_time = time.time()
                output = ollama.generate(
                    model=model_name,
                    prompt=final_prompt,
                    options={"temperature": temperature},
                )
                end_time = time.time()
                tiempo_segundos = end_time - start_time

                respuesta_bruta = output.get("response", "No se obtuvo respuesta.")
                respuesta_limpia_final = eliminar_chain_of_thought(respuesta_bruta)

                tiempos_iteraciones_pregunta_actual.append(tiempo_segundos)
                todos_los_tiempos_para_resumen_global.append(tiempo_segundos)

                writer.writerow(
                    {
                        "Pregunta_ID": idx,
                        "Iteracion_Num": iter_num_actual,
                        "Pregunta": pregunta,
                        "Respuesta_Esperada": respuesta_esperada,
                        "Respuesta_Limpia": respuesta_limpia_final,
                        "Num_Chunks": num_chunks_retrieved,
                        "Tiempo_Segundos": round(tiempo_segundos, 2),
                    }
                )

            # Resumen por pregunta
            if tiempos_iteraciones_pregunta_actual:
                promedio_q = sum(tiempos_iteraciones_pregunta_actual) / len(
                    tiempos_iteraciones_pregunta_actual
                )
                tiempo_max_q = max(tiempos_iteraciones_pregunta_actual)
                tiempo_min_q = min(tiempos_iteraciones_pregunta_actual)
                iteracion_max_q_num = (
                    tiempos_iteraciones_pregunta_actual.index(tiempo_max_q) + 1
                )
                iteracion_min_q_num = (
                    tiempos_iteraciones_pregunta_actual.index(tiempo_min_q) + 1
                )

                datos_resumen_por_pregunta_lista.append(
                    {
                        "Tipo_Resumen": "Pregunta",
                        "Pregunta_ID": idx,
                        "Num_Iteraciones": len(tiempos_iteraciones_pregunta_actual),
                        "Tiempo_Promedio_s": round(promedio_q, 2),
                        "Tiempo_Max_s": round(tiempo_max_q, 2),
                        "Iteracion_Mas_Lenta": iteracion_max_q_num,
                        "Tiempo_Min_s": round(tiempo_min_q, 2),
                        "Iteracion_Mas_Rapida": iteracion_min_q_num,
                    }
                )
                print(
                    f"Resumen pregunta {idx}: Promedio={promedio_q:.2f}s, "
                    f"Max={tiempo_max_q:.2f}s (Iter {iteracion_max_q_num}), "
                    f"Min={tiempo_min_q:.2f}s (Iter {iteracion_min_q_num})"
                )

    # --- Archivo de resúmenes ---
    summary_csv_headers = [
        "Tipo_Resumen",
        "Pregunta_ID",
        "Num_Iteraciones",
        "Tiempo_Promedio_s",
        "Tiempo_Max_s",
        "Iteracion_Mas_Lenta",
        "Tiempo_Min_s",
        "Iteracion_Mas_Rapida",
    ]

    with open(
        output_summary_filename, "w", encoding="utf-8", newline=""
    ) as csvfile_summary:
        writer_summary = csv.DictWriter(csvfile_summary, fieldnames=summary_csv_headers)
        writer_summary.writeheader()

        if datos_resumen_por_pregunta_lista:
            writer_summary.writerows(datos_resumen_por_pregunta_lista)

        if todos_los_tiempos_para_resumen_global:
            promedio_global = sum(todos_los_tiempos_para_resumen_global) / len(
                todos_los_tiempos_para_resumen_global
            )
            tiempo_max_global = max(todos_los_tiempos_para_resumen_global)
            tiempo_min_global = min(todos_los_tiempos_para_resumen_global)
            iteracion_max_global_num = (
                todos_los_tiempos_para_resumen_global.index(tiempo_max_global) + 1
            )
            iteracion_min_global_num = (
                todos_los_tiempos_para_resumen_global.index(tiempo_min_global) + 1
            )

            writer_summary.writerow(
                {
                    "Tipo_Resumen": "Global",
                    "Pregunta_ID": "N/A",
                    "Num_Iteraciones": len(todos_los_tiempos_para_resumen_global),
                    "Tiempo_Promedio_s": round(promedio_global, 2),
                    "Tiempo_Max_s": round(tiempo_max_global, 2),
                    "Iteracion_Mas_Lenta": iteracion_max_global_num,
                    "Tiempo_Min_s": round(tiempo_min_global, 2),
                    "Iteracion_Mas_Rapida": iteracion_min_global_num,
                }
            )
            print("\n========== RESUMEN GLOBAL ==========")
            print(
                f"Total iteraciones (LLM calls): {len(todos_los_tiempos_para_resumen_global)}"
            )
            print(f"Tiempo promedio global: {promedio_global:.2f} s")
            print(
                f"Iteracion global mas lenta: {iteracion_max_global_num} ({tiempo_max_global:.2f} s)"
            )
            print(
                f"Iteracion global mas rapida: {iteracion_min_global_num} ({tiempo_min_global:.2f} s)"
            )
            print("=" * 60 + "\n")
        else:
            print(
                "\nNo se ejecutaron iteraciones, no se pudo generar el resumen global."
            )

    print(
        f"\nProceso finalizado para Temperatura {temperature}. Resultados: '{output_filename}'."
    )
    if todos_los_tiempos_para_resumen_global:
        print(f"Resumen de rendimiento: '{output_summary_filename}'.")

    # --- Actualizar config.py ---
    config_path = os.path.join(current_dir, "config.py")
    results_path_abs = os.path.join(base_dir, "results").replace("\\", "/")

    # Usar la misma convención de nombres que usamos arriba (temp_str)
    out_csv = (
        f"{results_path_abs}/{sanitized_model_name}_temperatura_{temp_str}_{doc_name}.csv"
    )
    metrics_csv = f"{results_path_abs}/{sanitized_model_name}_temperatura_{temp_str}_{doc_name}_metrics.csv"
    summary_csv = f"{results_path_abs}/{sanitized_model_name}_temperatura_{temp_str}_{doc_name}_metrics_summary.csv"

    with open(config_path, "w", encoding="utf-8") as config_file:
        config_file.write(f'model_name = "{model_name}"\n')
        config_file.write(f'temperature = {temperature}\n')
        config_file.write(f'doc_name = "{doc_name}"\n')
        config_file.write(f'output_filename = "{out_csv}"\n')
        config_file.write(f'metrics_output = "{metrics_csv}"\n')
        config_file.write(f'metrics_output_summary = "{summary_csv}"\n')

    print(f"config.py actualizado para T={temperature}.")
