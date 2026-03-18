import re
import unicodedata

from langchain.text_splitter import CharacterTextSplitter

TITLE_REGEX = re.compile(r"^#\s+\*\*(.*)\*\*$")
SUBTITLE_REGEX = re.compile(r"^##\s+\*\*(.*)\*\*$")
SUB_SUBTITLE_REGEX = re.compile(r"^###\s+\*\*(.*)\*\*$")
GENERIC_TITLE_REGEX = re.compile(r"^\*\*(.*?)\*\*$")

def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

tittle = "# **CAPÍTULO 1 MARCO ESTRATÉGICO**"
subtittlw = "## **1.1. MISIÓN**"
sub_subtittle = "### **1.6.2. MISIONALIDAD**"
generic = "**RANDOM**"


print(TITLE_REGEX.match(tittle))
print(SUBTITLE_REGEX.match(subtittlw))
print(SUB_SUBTITLE_REGEX.match(sub_subtittle))
print(GENERIC_TITLE_REGEX.match(generic))

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
            stack = stack[:level - 1]

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
            "origen": "EDAES"
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)

        # Solo los títulos principales pueden tener hijos
        if section_type != "Título genérico":
            stack.append(node)

def prototipo_1(chunks_edaes, chunks_segmentados):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=512, chunk_overlap=20
    )
    i = 0
    for chunk in chunks_edaes:
        id_seg = i
        chunk_title = chunk['title']
        split_text = text_splitter.split_text(chunk["text"])
        id_referencia = chunk["id"]
        origen = chunk["origen"]
        for text in split_text:
            chunks_segmentados.append({
                "id": f"chunk_{id_seg}",
                "title": chunk_title,
                "text": limpiar_string(chunk_title) + " " + text,
                "origen": origen,
                "id_referencia": id_referencia
            })
            id_seg+=1
            
print("Creando Base de datos... ")
sections = []
markdown_content  = read_markdown_file("../../Referencias/rag_files/edaes/edaes.md")
parse_markdown_edaes(markdown_content, sections)
c_s =[]
# prototipo_1(sections, c_s)
with open ("bloques_ref.txt", "w", encoding="utf-8") as f:
    for i in sections:
        f.write(f"{i['id']}: {i['title']}\n")
        f.write("-"*200 + "\n")
        f.write(f"{i['text']}\n")
        f.write("-"*200 + "\n")
        f.write("\n\n")
x = 0
for i in sections:
    if x < 10:
        print(f"{i['id']}: {i['title']}")
        print("-"*200)
        print(f"{i['text']}")
        print("-"*200)
        print("\n\n")
        x += 1
    else:
        break

# i = 0

# t = 'presentacion del edaes y lema de la fac\n\n"Sic itur ad astra", es el lema que sostiene en sus garras el águila de gules e inspira a la Fuerza Aérea Colombiana (FAC); ha sido entendido como "así se va a las alturas", no obstante, tiene un significado literal más ambicioso: "así se va a las estrellas". Se trata de evolucionar de forma permanente, volando más alto, más lejos y más rápido en la protección de los colombianos y el Estado de Derecho, siempre volando, entrenando y combatiendo para vencer. "\n\nLa oportunidad, precisión, efectividad y contundencia del poder aéreo ha consolidado esta Institución como un factor decisivo para enfrentar los retos del Estado. Su versatilidad ha permitido disuadir amenazas externas, derrotar a los que desafían el imperio de la Ley, apoyar a la población civil mediante el transporte de ayuda humanitaria, atender y prevenir emergencias y evacuar o trasladar colombianos desde las regiones más apartadas del país, en el marco del desarrollo de operaciones conjuntas, coordinadas e interagenciales con el Ejército, la Armada, la Policía Nacional y las demás entidades estatales.\n\nLa Estrategia para el desarrollo aéreo y espacial de la Fuerza Aérea Colombiana 2042 se soporta en el espíritu de evolución, innovación y transformación permanente, teniendo como propósito la consolidación de una Fuerza polivalente e interoperable que cumpla los más altos estándares internacionales y se consolide como preferente y líder regional. Tal proyecto se constituye en la hoja de ruta de la Institución para los próximos años y será la base en la toma de decisiones y el direccionamiento del alto mando en el corto, mediano y largo plazo.'
# t = limpiar_string(t)
# print(t)