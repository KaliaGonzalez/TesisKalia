import re

# REGLAMENTO_REGEX_PORFAC = re.compile(r"^(#{1})\s\*\*(.*?)\*\*$")
# TITLE_REGEX_PORFAC = re.compile(r"^(#{2})\s\*\*(T[ÍI]TULO\s[IVXLCDM]+.*?)\*\*$|^##\s\*\*(CERTIFICADO\s\d+)\*\*$")
# CAPITULO_REGEX_PORFAC = re.compile(r"^(#{3})\s\*\*(CAP[IÍ]TULO\s(ÚNICO|[IVXLCDM]+).*?)\*\*$")
# # ARTICULO_REGEX_PORFAC = re.compile(r"^(#{4})\s\*\*Art[íi]culo\s(\d+)\.\*\*$")
# ARTICULO_REGEX_PORFAC = re.compile(r"^####\s\*\*Art[íi]culo\s(\d+)\.\*\*")
# PARAGRAFO_REGEX_PORFAC = re.compile(r"^\*\*(.+?)\*\*\s*(.*)$")
# REGLAMENTO_REGEX_PORFAC = re.compile(r"^(#{1})\s\*\*(.*?)\*\*$")
# TITLE_REGEX_PORFAC = re.compile(r"^(#{2})\s\*\*(T[ÍI]TULO\s[IVXLCDM]+.*?)\*\*$|^(#{2})\s\*\*(CERTIFICADO\s\d+)\*\*$")
# CAPITULO_REGEX_PORFAC = re.compile(r"^(#{3})\s\*\*(CAP[IÍ]TULO\s(ÚNICO|[IVXLCDM]+).*?)\*\*$")
# ARTICULO_REGEX_PORFAC = re.compile(r"^(#{4})\s\*\*Art[íi]culo\s(\d+)\.\*\*")
# PARAGRAFO_REGEX_PORFAC = re.compile(r"^\*\*(.+?)\*\*\s*(.*)$")
REGLAMENTO_REGEX_PORFAC = re.compile(r"^(#{1})\s\*\*(.+?)\*\*$")
TITLE_REGEX_PORFAC = re.compile(r"^(#{2})\s\*\*(.+?)\*\*$")
CAPITULO_REGEX_PORFAC = re.compile(r"^(#{3})\s\*\*(.+?)\*\*$")
ARTICULO_REGEX_PORFAC = re.compile(r"^(#{4})\s\*\*Art[íi]culo\s(\d+)\.\*\*")
PARAGRAFO_REGEX_PORFAC = re.compile(r"^\*\*(.+?)\*\*\s*(.*)$")

reg = "# **REGLAMENTO DEL CUERPO DE 'PROFESIONALES OFICIALES DE RESERVA DE LAS FUERZAS MILITARES'**"
title1 = "## **CERTIFICADO 1**"
title2 = "## **TÍTULO I Disposiciones preliminares**"
capitulo1 = "### **CAPÍTULO ÚNICO Generalidades**"
capitulo2 =  "### **CAPÍTULO I Del ingreso, ascenso y formación de los Profesionales Oficiales de Reserva**"
articulo1 = "#### **Artículo 40.** "
paragrafo1 = "**Parágrafo 2.**"
paragrafo2 = "**A. Generales**"

def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
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
            parent_titles = {
                "Título": None
            }
            for ancestor in reversed(stack):
                if ancestor["type"] == "Título":
                    parent_titles["Título"] = ancestor["title"]

            text = f"{title} del {parent_titles['Título']}" if parent_titles["Título"] else title

            node = {
                "id": f"chunk_{len(sections)}",
                "level": level,
                "title": title.strip(),
                "text": text,
                "type": section_type,
                "parent": stack[-1]["id"] if stack else "None",
                "children": []
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
            parent_titles = {
                "Capítulo": None,
                "Título": None
            }
            for ancestor in reversed(stack):
                if ancestor["type"] in parent_titles:
                    parent_titles[ancestor["type"]] = ancestor["title"]

            text = f"{title} del {parent_titles['Capítulo']} del {parent_titles['Título']}" if all(parent_titles.values()) else title

            node = {
                "id": f"chunk_{len(sections)}",
                "level": level,
                "title": title.strip(),
                "text": text,
                "type": section_type,
                "parent": stack[-1]["id"] if stack else "None",
                "children": []
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
            parent_titles = {
                "Artículo": None,
                "Capítulo": None,
                "Título": None
            }
            for ancestor in reversed(stack):
                if ancestor["type"] in parent_titles:
                    parent_titles[ancestor["type"]] = ancestor["title"]

            full_text = f"{title} del {parent_titles['Artículo']} del {parent_titles['Capítulo']} del {parent_titles['Título']}"
            text = f"{full_text}\n\n{content}"

            node = {
                "id": f"chunk_{len(sections)}",
                "level": level,
                "title": title,
                "text": text,
                "type": section_type,
                "parent": stack[-1]["id"] if stack else "None",
                "children": []
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
            "children": []
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)
        stack.append(node)

print("Creando Base de datos... ")
sections = []
markdown_content  = read_markdown_file("../../Referencias/rag_files/reglamento_porfac/reglamento_porfac/reglamento_porfac.md")
chunks = parse_markdown_porfac(markdown_content, sections)
print("Datos Parseados... ")
i = 0

# print(REGLAMENTO_REGEX_PORFAC.match(reg))
# print(TITLE_REGEX_PORFAC.match(title1))
# print(TITLE_REGEX_PORFAC.match(title2))
# print(CAPITULO_REGEX_PORFAC.match(capitulo1))
# print(CAPITULO_REGEX_PORFAC.match(capitulo2))
# print(ARTICULO_REGEX_PORFAC.match(articulo1))
# print(ARTICULO_REGEX_PORFAC.match(reg))
# print(ARTICULO_REGEX_PORFAC.match(title1))
# print(ARTICULO_REGEX_PORFAC.match(title2))
# print(ARTICULO_REGEX_PORFAC.match(capitulo1))
# print(ARTICULO_REGEX_PORFAC.match(capitulo2))
# print(ARTICULO_REGEX_PORFAC.match(articulo1))
# print(ARTICULO_REGEX_PORFAC.match(paragrafo1))
# print(ARTICULO_REGEX_PORFAC.match(paragrafo2))

for chunk in sections:
    if chunk['level'] == 5:
        print(f"ID: {chunk['id']}, Título: {chunk['title']}, Texto: {chunk['text']}")