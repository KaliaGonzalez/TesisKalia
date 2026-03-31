import re

line = "**PREVAC***.* Prevención de Accidentes Aéreos, conjunto de actividades orientadas a detectar y suprimir los peligros, así como disminuir los riesgos propios de la operación de la FAC."

term_regex_broad = re.compile(r"^\*\*(.+?)\*\*(?:[:.\*]*)\s+(.+)$")

match = term_regex_broad.match(line)
if match:
    print("Match found!")
    print(f"Group 1 (Term): '{match.group(1)}'")
    print(f"Group 2 (Def): '{match.group(2)}'")
else:
    print("No match found.")
