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
from langchain_community.retrievers import BM25Retriever  # Importar BM25

# from langchain.chains import RetrievalQA  # Deprecated, will use direct approach
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


def load_goalset(ruta_json):
    with open(ruta_json, "r", encoding="utf-8") as f:
        return json.load(f)


# Solo ejecutar al correr directamente, no al importar
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configura el modelo, temperatura y GoalSet para Ollama."
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
        default="FAC_Documents/rag_files/GoalSetNewMater.json",
        help="Ruta al archivo GoalSet JSON",
    )
    parser.add_argument(
        "--doc_name",
        type=str,
        default="",
        help="Nombre del documento (para tracking)",
    )
    args = parser.parse_args()
    model_name = args.model
    temperature = args.temperature
    ruta_json = args.goalset
    doc_name = args.doc_name

    # Cargar GoalSet desde la ruta especificada
    preguntas_respuestas = load_goalset(ruta_json)

# Variables de configuración (disponibles al importar)
c_name = "pruebas"
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
persist_directory = "../data/chroma_db"
re_ranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def limpiar_string(texto):
    texto = texto.lower()
    texto = (
        unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    )
    texto = re.sub(r"[^a-z0-9\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def re_rank_docs(query, docs, reranker):
    texts = [doc.page_content for doc in docs]
    pairs = [[query, text] for text in texts]
    scores = reranker.predict(pairs)
    sorted_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc[0] for doc in sorted_docs]


def parse_markdown_mater(md_text, sections):
    lines = md_text.split("\n")
    stack = []

    for line in lines:
        title_match = TITLE_REGEX_MATER.match(line)
        subtitle_match = SUBTITLE_REGEX_MATER.match(line)
        sub_subtitle_match = SUB_SUBTITLE_REGEX_MATER.match(line)
        definition_match = DEFINITION_REGEX_MATER.match(line)

        # Si es título principal
        if title_match:
            level = len(title_match.group(1))
            title = title_match.group(2)
            section_type = "Título"
            stack.clear()

        # Si es subtítulo
        elif subtitle_match:
            level = len(subtitle_match.group(1))
            title = subtitle_match.group(2)
            section_type = "Subtítulo"
            stack = stack[:1]

        # Si es sub-subtítulo
        elif sub_subtitle_match:
            level = len(sub_subtitle_match.group(1))
            title = sub_subtitle_match.group(2)
            section_type = "Sub-subtítulo"
            stack = stack[:2]

        # Si es definición (formato **Termino**: definición)
        elif definition_match:
            level = 4
            title = definition_match.group(1)
            section_type = "Definición"
            # generar una frase tipo "QUÉ ES <Termino>? <Termino> es <Definición>"
            def_text = f"QUÉ ES {definition_match.group(1)}? {definition_match.group(1)} es {definition_match.group(2)}"
            stack = stack[:3]

        else:
            # Si no es ninguno, añadir el texto limpiado al último nodo creado
            if sections:
                sections[-1]["text"] += limpiar_string(line) + "\n"
            continue

        # Construir el texto asociado al nodo (solo para definiciones se rellena)
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
            "origen": "EDAES",
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

        # Para reglamento y título (sin texto especial)
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
            "origen": "Historia",
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)

        # Solo los títulos principales pueden tener hijos
        if section_type != "Título genérico":
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
            "origen": "Resumen EDAES",
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


def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


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

    if hijos != "":
        for id in hijos:
            filtro_hijo = vectorstore.get(where={"id": id})
            doc_hijo = filtro_hijo["documents"][0]
            titulo_hijo = filtro_hijo["metadatas"][0]["title"]
            full_context += titulo_hijo + "\n" + doc_hijo + "\n\n"

    return full_context


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


# Inicializar ChromaDB y HuggingFaceEmbeddings si esxiste la base de datos trae la info. Lo que carga es un archivo markdow al cual debes darle.
# Yo te paso esos formatos. Aun no he probado meterle mas de un documento, por lo que estoy haciendo pruebas por separado pero de momento dejalo asi.
# Si tu quieres probar el otro documento o los solo tiene que descomentar parse_markdown_edaes y/o parse_markdown_mater.

# Ejecución principal del procesamiento
if __name__ == "__main__":
    # Continuar con la ejecución principal solo cuando se ejecute directamente
    if os.path.exists(persist_directory):
        print("Cargando Base de datos... ")
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vectorstore_edaes = Chroma(
            persist_directory=persist_directory,
            collection_name="edaes",
            embedding_function=embedding_model_name,
        )
        vectorstore_pruebas = Chroma(
            persist_directory=persist_directory,
            collection_name="pruebas",
            embedding_function=embedding_model_name,
        )
    vectorstore_resumen_edaes = Chroma(
        persist_directory=persist_directory,
        collection_name="resumen_edaes",
        embedding_function=embedding_model_name,
    )
    vectorstore_edaes_seg = Chroma(
        persist_directory=persist_directory,
        collection_name="edaes_seg",
        embedding_function=embedding_model_name,
    )
    vectorstore_newMater = Chroma(
        persist_directory=persist_directory,
        collection_name="newMater",
        embedding_function=embedding_model_name,
    )
    vectorstore_Historia = Chroma(
        persist_directory=persist_directory,
        collection_name="Historia",
        embedding_function=embedding_model_name,
    )
else:
    print("Creando Base de datos... ")
    chunks = []
    chunks_edaes = []
    chunk_resumen_edaes = []
    chunks_seg = []
    chunks_Historia = []

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
    markdown_content_newMater = read_markdown_file(
        "FAC_Documents/rag_files/NuevoMater.md"
    )
    markdown_content_Historia = read_markdown_file(
        "FAC_Documents/rag_files/Historia.md"
    )

    parse_markdown_edaes(markdown_content_EDAES, chunks_edaes)
    print(len(chunks_edaes))
    prototipo_1(chunks_edaes, chunks_seg)
    print(len(chunks_seg))
    parse_markdown_mater(markdown_content_MATER, chunks)
    print(len(chunks))
    parse_markdown_porfac(markdown_content_PORFAC, chunks)
    print(len(chunks))
    parse_markdown_resumen_edaes(markdown_content_resumen_edaes, chunk_resumen_edaes)
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

k = 5

# NOTA: Las viejas líneas que creaban vectorstores han sido removidas.
# Ahora usamos la función inicializar_retriever_vectorstore() que carga BM25 de la base de datos v7.

# Reranker es el que reordenará los documentos con el más relevante de mayor a menor.
reranker = CrossEncoder(re_ranker_model)

# Esto simplemente es un template de prompt que se le pasará al modelo de lenguaje para que genere la respuesta.
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


def inicializar_modelo(model_name="mistral", temperature=0.5, prompt=PROMPT):
    # Aqui estamos creando el modelo en esta caso deberas cambiar el nombre arriba del archivo o aqui mismo.
    llm = OllamaLLM(
        model=model_name, temperature=temperature, base_url="http://localhost:11434"
    )
    # Se crea el pipeline (reemplaza LLMChain que está deprecado)
    llm_chain = prompt | llm
    return llm_chain


def inicializar_retriever_vectorstore(k=5):
    """
    Inicializa los retrievers usando BM25.
    Busca la base de datos en las ubicaciones conocidas.
    """
    print("Inicializando BM25Retriever desde la base de datos...")

    # Intentar encontrar la base de datos en diferentes ubicaciones
    possible_paths = [
        "../../data/chroma_db_v7",  # Ubicación correcta (2 niveles arriba)
        "../data/chroma_db_v7",  # v7 unificada (1 nivel arriba)
        "../data/chroma_db",  # Ubicación original
        "chroma_db_persistente_v3",  # Local en pre_processing
        "../pre_processing/chroma_db_persistente_v3",  # Relativa desde raíz
    ]

    persist_directory_v7 = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Base de datos encontrada en: {path}")
            persist_directory_v7 = path
            break

    if persist_directory_v7 is None:
        print(f"ERROR: No se encontró base de datos en ninguna ubicación.")
        print(f"Ubicaciones buscadas: {possible_paths}")
        return None

    try:
        print("✅ Cargando base de datos...")

        # Cargar el vectorstore
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vectorstore_complete = Chroma(
            persist_directory=persist_directory_v7,
            collection_name="fac_documents_complete",
            embedding_function=embedding_model,
        )

        # Extraer todos los documentos para crear el índice BM25
        print("Generando índice BM25...")
        all_data = vectorstore_complete.get()
        all_docs_bm25 = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(all_data["documents"], all_data["metadatas"])
        ]

        if not all_docs_bm25:
            raise ValueError(
                "No se encontraron documentos en ChromaDB para construir BM25."
            )

        # Crear el Retriever BM25 global
        bm25_retriever_global = BM25Retriever.from_documents(all_docs_bm25)
        bm25_retriever_global.k = k

        print(f"✅ BM25 inicializado con {len(all_docs_bm25)} documentos.")

        # Asignar el mismo BM25 a todos los retrievers
        retriever_edaes = bm25_retriever_global
        retriever_pruebas = bm25_retriever_global
        retriever_newMater = bm25_retriever_global
        retriever_historia = bm25_retriever_global
        retriever_resumen_edaes = bm25_retriever_global
        retriever_edaes_seg = bm25_retriever_global

        return (
            retriever_edaes,  # 1
            retriever_pruebas,  # 2
            retriever_newMater,  # 3
            vectorstore_complete,  # 4 (Se mantiene para compatibilidad)
            retriever_resumen_edaes,  # 5
            retriever_edaes_seg,  # 6
            retriever_historia,  # 7
        )
    except Exception as e:
        print(f"❌ ERROR inicializando BM25: {str(e)}")
        print("Por favor, asegúrate de que la base de datos existe.")
        return None


# INICIALIZAR COMPONENTES DEL SISTEMA
print("\n🚀 INICIALIZANDO SISTEMA RAG CON BM25...\n")
llm_chain = inicializar_modelo(model_name, temperature)
print(f"✅ Modelo LLM inicializado: {model_name}")
print(f"🌡️  Temperatura configurada: {temperature}")

(
    retriever_edaes,
    retriever_pruebas,
    retriever_newMater,
    vectorstore_edaes,
    retriever_resumen_edaes,
    retriever_edaes_seg,
    retriever_historia,
) = inicializar_retriever_vectorstore(k)
print(f"✅ BM25Retriever inicializado con k={k}\n")

re_type = "BASE"  # Tipo de recuperación por defecto


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
    """
    Función de chatbot_response usando BM25Retriever.
    Como todos los retrievers ahora apuntan al mismo BM25 global,
    invocamos uno solo para obtener resultados del corpus completo.
    """
    # Limpiar consulta
    query_limpia = limpiar_string(query)

    print(f"🔍 Búsqueda BM25 para: '{query}'")

    # Usar el BM25 global (todos los retrievers apuntan al mismo índice)
    # Invocamos solo uno ya que todos son idénticos
    docs = retriever_edaes.invoke(query_limpia)

    print(f"📄 Documentos encontrados por BM25: {len(docs)}")

    # Re-ranking con CrossEncoder
    re_ranked_docs = re_rank_docs(query_limpia, docs[:15], reranker)

    print(f"🏆 Top {len(re_ranked_docs)} documentos después de re-ranking")

    # Construir contexto
    fc = ""
    referencias_set = set()

    for doc in re_ranked_docs:
        origin = doc.metadata.get("origin", "Desconocido")
        referencias_set.add(origin)

        # Simplemente agregar el contenido del documento
        fc += doc.page_content + "\n\n"

    referencias_str = "\n".join(referencias_set)

    # Generar respuesta
    print("🤖 Invocando LLM...")
    start = time.time()
    respuesta = llm_chain.invoke({"contexto": fc, "pregunta": query})
    tiempo_res = time.time() - start

    print(f"✅ Respuesta generada en {tiempo_res:.2f}s")

    return respuesta, referencias_str, tiempo_res


def sanitize_filename(name):
    return re.sub(r'[.<>:"/\\|?*]', "_", name)


sanitized_model_name = sanitize_filename(model_name)
print(sanitized_model_name)

# Crear nombre del CSV incluyendo modelo, temperatura y documento
# Formato: granite4_3b_temperatura_0.0_EDAES.csv
temp_str = str(temperature).replace(".", "_")
sanitized_doc_name = sanitize_filename(doc_name) if doc_name else "documento"

csv_base_name = f"{sanitized_model_name}_temperatura_{temp_str}_{sanitized_doc_name}"

# Configuración para la automatización:
num_iterations = 10
usar_full_context = False
output_filename = "../results/" + csv_base_name + ".csv"
output_summary_filename = "../results/" + csv_base_name + "_summary.csv"

# ============================================================================
# ACTUALIZAR config.py CON LOS VALORES USADOS (para que metrics.py los use)
# ============================================================================
config_content = f"""model_name = "{model_name}"
temperature = {temperature}
doc_name = "{doc_name}"
output_filename = "{output_filename}"
metrics_output = "../results/{csv_base_name}_metrics.csv"
metrics_output_summary = "../results/{csv_base_name}_metrics_summary.csv"
"""

with open("config.py", "w", encoding="utf-8") as config_file:
    config_file.write(config_content)

print(f"✅ config.py actualizado")
print(f"   Modelo: {model_name}")
print(f"   Temperatura: {temperature}")
print(f"   Documento: {doc_name}")
print(f"   CSV: {csv_base_name}.csv")

tiempos_iteraciones = []
tiempos_globales = []

csv_headers = [
    "Pregunta_ID",
    "Iteracion_Num",
    "Pregunta",
    "Respuesta_Esperada",
    "Respuesta_Limpia",
    "Num_Chunks",
    "Tiempo_Segundos",
]

# Listas para almacenar datos para los resúmenes
todos_los_tiempos_para_resumen_global = []
datos_resumen_por_pregunta_lista = []


def eliminar_chain_of_thought(texto_respuesta):
    return re.sub(
        r"<think>.*?</think>", "", texto_respuesta, flags=re.DOTALL | re.IGNORECASE
    ).strip()


# Crear carpeta de resultados si no existe
import os
from pathlib import Path

results_dir = Path("../results")
results_dir.mkdir(parents=True, exist_ok=True)

with open(output_filename, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()  # Escribir la fila de encabezados

    for idx, item in enumerate(preguntas_respuestas, start=1):
        pregunta = item["pregunta"]
        respuesta_esperada = item["respuesta"]
        tiempos_iteraciones_pregunta_actual = (
            []
        )  # Almacena tiempos para la pregunta actual

        print(f"\n🟦 Procesando pregunta {idx} de {len(preguntas_respuestas)}")
        print(f"➡️  {pregunta}\n")

        for iter_num_actual in range(1, num_iterations + 1):
            print(
                f"Ejecutando iteración {iter_num_actual} de {num_iterations} para pregunta {idx}"
            )

            # Usar la función chatbot_response que ya tiene BM25 implementado
            respuesta_bruta, referencias_str, tiempo_segundos = chatbot_response(
                query=pregunta,
                usar_full_context=usar_full_context,
                llm_chain=llm_chain,
                retriever_edaes=retriever_edaes,
                retriever_pruebas=retriever_pruebas,
                reranker=reranker,
                vectorstore=vectorstore_edaes,
                retriever_resumen_edaes=retriever_resumen_edaes,
                retriever_edaes_seg=retriever_edaes_seg,
                retriever_newMater=retriever_newMater,
                retriever_historia=retriever_historia,
                re_type=re_type,
            )

            # Contar documentos recuperados (para estadísticas)
            query_limpia = limpiar_string(pregunta)
            docs_retrieved = retriever_edaes.invoke(query_limpia)
            num_chunks_retrieved = len(docs_retrieved)

            # Limpiar la respuesta
            respuesta_limpia_final = eliminar_chain_of_thought(respuesta_bruta)

            # Guardar tiempos para resúmenes
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

        # --- Resumen por pregunta (después de todas sus iteraciones) ---
        if tiempos_iteraciones_pregunta_actual:
            promedio_q = sum(tiempos_iteraciones_pregunta_actual) / len(
                tiempos_iteraciones_pregunta_actual
            )
            tiempo_max_q = max(tiempos_iteraciones_pregunta_actual)
            tiempo_min_q = min(tiempos_iteraciones_pregunta_actual)
            # +1 porque index() es base 0 y las iteraciones son base 1
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
                f"Resumen para pregunta {idx}: Promedio={promedio_q:.2f}s, Max={tiempo_max_q:.2f}s (Iter {iteracion_max_q_num}), Min={tiempo_min_q:.2f}s (Iter {iteracion_min_q_num})"
            )

# --- Escritura del archivo CSV de Resúmenes (después de procesar todas las preguntas) ---
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

    # Escribir resúmenes por pregunta
    if datos_resumen_por_pregunta_lista:
        writer_summary.writerows(datos_resumen_por_pregunta_lista)

    # Calcular y escribir resumen global
    if todos_los_tiempos_para_resumen_global:
        promedio_global = sum(todos_los_tiempos_para_resumen_global) / len(
            todos_los_tiempos_para_resumen_global
        )
        tiempo_max_global = max(todos_los_tiempos_para_resumen_global)
        tiempo_min_global = min(todos_los_tiempos_para_resumen_global)
        # La iteración más lenta/rápida global se refiere al índice en la lista aplanada de todos los tiempos
        iteracion_max_global_num = (
            todos_los_tiempos_para_resumen_global.index(tiempo_max_global) + 1
        )
        iteracion_min_global_num = (
            todos_los_tiempos_para_resumen_global.index(tiempo_min_global) + 1
        )

        resumen_global_dict = {
            "Tipo_Resumen": "Global",
            "Pregunta_ID": "N/A",  # No aplica para el resumen global
            "Num_Iteraciones": len(
                todos_los_tiempos_para_resumen_global
            ),  # Total de iteraciones LLM
            "Tiempo_Promedio_s": round(promedio_global, 2),
            "Tiempo_Max_s": round(tiempo_max_global, 2),
            "Iteracion_Mas_Lenta": iteracion_max_global_num,  # Número de iteración global (1 hasta N_total_iteraciones)
            "Tiempo_Min_s": round(tiempo_min_global, 2),
            "Iteracion_Mas_Rapida": iteracion_min_global_num,  # Número de iteración global
        }
        writer_summary.writerow(resumen_global_dict)
        print("\n========== RESUMEN GLOBAL ==========")
        print(
            f"Total de iteraciones (LLM calls): {len(todos_los_tiempos_para_resumen_global)}"
        )
        print(f"Tiempo promedio global: {promedio_global:.2f} s")
        print(
            f"Iteración global más lenta: {iteracion_max_global_num} ({tiempo_max_global:.2f} s)"
        )
        print(
            f"Iteración global más rápida: {iteracion_min_global_num} ({tiempo_min_global:.2f} s)"
        )
        print("=" * 60 + "\n")

    else:
        print(
            "\n⚠️ No se ejecutaron iteraciones, no se pudo generar el resumen global."
        )

print(
    f"\n✅ Proceso finalizado. Resultados detallados guardados en '{output_filename}'."
)
if todos_los_tiempos_para_resumen_global:
    print(f"📊 Resumen de rendimiento guardado en '{output_summary_filename}'.")
