import re
import unicodedata

QUESTION_REGEX = re.compile(r"^###\s(.*)$")
REF_REGEX = re.compile(r"\*\*(.*?)\*\*")

referencia = "**chunk_0**"
referencia2 = "**chunk_1**"
pregunta = "### 83. Como se describen los objetivos estrategicos de la fuerza aerea colombiana desde la perspectiva de partes interesadas?"


def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

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
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    
    # Eliminar caracteres especiales (manteniendo solo letras, números y espacios)
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    
    # Reemplazar múltiples espacios por uno solo
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

pregunta = "### 35. Cuales son las veinte politicas constitucionales de la Fuerza Aerea Colombiana (FAC)?"
# print(QUESTION_REGEX.match(pregunta))

def parse_markdown_goldset(md_text, sections):
    
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
            "children": []
        }
        
        # print("Antes: ", len(sections))
        sections.append(node)
        # print("Despues: ", len(sections))

            


# print("Creando Base de datos... ")
sections = []
markdown_content  = read_markdown_file("../../Referencias/rag_files/Preguntas_FAC.md")
parse_markdown_goldset(markdown_content, sections)
print(len(sections))
i = 0
for chunk in sections:
    if i < len(sections):
        print(chunk['id'], chunk['parent'])
        i += 1
    pass

def get_full_chunk_resume(vectorstore, doc, registro):
    full_chunk_text = ""
    referencia = ""
    
    metadata = doc.metadata
    id_referencias = metadata['parent'].split(", ")

    if id_referencias[0] != 'None':
        for ref in id_referencias:
            if ref in  registro:
                continue
            else:
                filtro_referencia = vectorstore.get(where={"id": ref})
                doc_referencia = filtro_referencia['documents'][0]
                full_chunk_text +=  doc_referencia
                referencia += filtro_referencia['metadatas'][0]['origin'] + " - " + filtro_referencia['metadatas'][0]['title'] + "\n\n"
                registro.append(ref)

    return full_chunk_text, referencia