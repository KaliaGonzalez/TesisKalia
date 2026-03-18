import streamlit as st
import tkinter as tk
import re
from tkinter import filedialog
from pre_processing.chatbotcontroller import ChatbotController

# Inicializar controlador solo una vez
st.set_page_config(page_title="Chatbot RAG FAC")


@st.cache_resource
def init_controller():
    return ChatbotController()


@st.dialog("BENCHMARK")
def benchmark():
    st.write("Configura el benchmark antes de continuar.")
    goal_set_path = st.file_uploader("Sube el goalset", type=["json"])
    iteraciones = st.slider(
        "Cantidad de iteraciones", min_value=10, max_value=100, value=50, step=10
    )
    st.write("Selecciona la carpeta donde se guardarán los resultados.")
    selected_folder_path = st.session_state.get("folder_path", None)
    folder_select_button = st.button("Select Folder")
    if folder_select_button:
        selected_folder_path = abrir_carpeta()
        st.session_state.folder_path = selected_folder_path
    st.write(f"Carpeta seleccionada: {selected_folder_path}")
    st.checkbox("Usar contexto completo", value=False)
    st.button(
        "Iniciar",
        on_click=lambda: controller.iniciar_benchmark(
            selected_folder_path, goal_set_path, iteraciones
        ),
    )


def abrir_carpeta():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path


controller = init_controller()


st.title("DELFOS")
st.markdown(
    "Soy un asistente que te puede ayudar a responder preguntas de la Fuerza Aereoespacial Colombiana."
)

# Estado de sesión para guardar mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar de ajustes
st.sidebar.title("Ajustes")
st.sidebar.markdown("## Ajustes para la búsqueda y respuesta")
# Valor fijo para cantidad de documentos (eliminamos el slider)
cant_documentos = 5
re_type = st.sidebar.selectbox(
    "Tipo de Recuperacion de Datos",
    ["BASE", "RESUMEN", "SEGMENTADO"],
    key="tipo_referencia",
)
st.sidebar.markdown("## Ajustes del modelo")
temperatura = st.sidebar.slider(
    "Temperatura", min_value=0.0, max_value=1.0, value=0.5, step=0.1
)
modelo = st.sidebar.selectbox(
    "Modelo", ["llama3.2:3b", "gemma3:4b", "mistral", "qwen3:4b", "deepseek-r1:7b"]
)
st.sidebar.button(
    "Guardar ajustes",
    on_click=lambda: controller.guardar_ajustes(
        cant_documentos, temperatura, modelo, re_type
    ),
)
st.sidebar.markdown("## Benchmark")
st.sidebar.button("Iniciar benchmark", on_click=benchmark)


# Mostrar mensajes anteriores
for message in st.session_state["messages"]:
    role = message.get("role")
    content = message.get("content", "")
    if role in ("user", "assistant"):
        with st.chat_message(role):
            st.markdown(content)
    else:
        st.warning(f"Mensaje con rol inválido o vacío: {message}")

usar_contexto_completo = st.checkbox("Usar contexto completo", value=True)

# Entrada del usuario
if query := st.chat_input("Escribe tu pregunta:"):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.spinner("Pensando..."):
        respuesta, tiempo, referecias = controller.obtener_respuesta(
            query, usar_contexto_completo=usar_contexto_completo
        )

    # MOSTRAR DIRECTAMENTE EN LA PÁGINA - SIMPLE Y DIRECTO
    st.success("🤖 RESPUESTA:")
    st.write(respuesta)

    st.info("📚 FUENTES:")
    if referecias and str(referecias).strip():
        unique_refs = []
        seen = set()
        for line in str(referecias).splitlines():
            entry = line.strip()
            if not entry:
                continue
            if entry in seen:
                continue
            seen.add(entry)
            unique_refs.append(entry)

        if unique_refs:
            for ref in unique_refs:
                st.write(f"• {ref}")
        else:
            st.write("No hay referencias válidas")
    else:
        st.write("Las referencias están vacías")

    st.caption(f"⏱️ Tiempo: {tiempo:.2f} segundos")

    # Guardar solo en el historial de sesión (sin mostrar de nuevo)
    st.session_state["messages"].append({"role": "assistant", "content": respuesta})
