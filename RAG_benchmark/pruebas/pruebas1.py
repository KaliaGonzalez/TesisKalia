import re
import string

# Expresiones regulares para los títulos
TITLE_REGEX_MATER = re.compile(r"^(#{1})\s\*\*(.*?)\*\*$")
SUBTITLE_REGEX_MATER = re.compile(r"^(#{2})\s\*\*(.+?)\*\*$")
SUB_SUBTITLE_REGEX_MATER = re.compile(r"^(#{3})\s\*\*(.?)\*\*$")
DEFINITION_REGEX_MATER = re.compile(r"^\*\*(.+?)\*\*:? (.+)$")


def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def parse_markdown_mater(md_text):
    
    lines = md_text.split("\n")
    sections = []
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
            level = len(definition_match.group(1))
            title = definition_match.group(1) 
            section_type = "Definicion"
            stack = stack[:3]
            def_text = definition_match.group(1) + " ES " + definition_match.group(2)
        else:
            if sections:
                sections[-1]["text"] += line + "\n"
            continue
        
        node = {
            "id": f"chunk_{len(sections)}",
            "level": level,
            "title": title.strip(),
            "text": def_text,
            "type": section_type,
            "parent": stack[-1]["id"] if stack else "None",
            "children": []
        }

        if stack:
            stack[-1]["children"].append(node["id"])

        sections.append(node)
        stack.append(node)

    return sections

print("Creando Base de datos... ")
markdown_content  = read_markdown_file("../../Referencias/rag_files/manual_de_terminos_fuerza_aerea_colombiana/manual_de_terminos_fuerza_aerea_colombiana.md")
chunks = parse_markdown_mater(markdown_content)
i = 0
for chunk in chunks:
    if chunk['title'] == "POLÍGONO AÉREO":
      print(f"{chunk}\n")