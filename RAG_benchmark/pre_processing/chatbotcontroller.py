import sys
import os
from datetime import datetime
from pathlib import Path

# Agregar el directorio padre al path para importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pre_processing.myway import (
        inicializar_retriever_vectorstore,
        inicializar_modelo,
        chatbot_response,
        CrossEncoder,
        re_ranker_model,
    )
except ImportError:
    # Fallback para cuando se ejecuta directamente desde el directorio pre_processing
    from myway import (
        inicializar_retriever_vectorstore,
        inicializar_modelo,
        chatbot_response,
        CrossEncoder,
        re_ranker_model,
    )


class ChatbotController:
    def __init__(self, model_name="llama3.2:latest", temperature=0.5, k=5):
        print(f"[INIT] Inicializando ChatbotController con k={k}")

        # Crear carpeta de logs si no existe
        self.logs_dir = Path("../chat_logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Crear archivo de log con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"conversacion_{timestamp}.txt"

        # Escribir encabezado del log
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"{'='*80}\n")
            f.write(f"REGISTRO DE CONVERSACIÓN - CHATBOT RAG\n")
            f.write(f"{'='*80}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modelo: {model_name}\n")
            f.write(f"Temperatura: {temperature}\n")
            f.write(f"{'='*80}\n\n")

        print(f"[LOG] Registro de conversación: {self.log_file}\n")

        # Obtener los componentes del RAG
        components = inicializar_retriever_vectorstore(k)

        print(f"[INFO] RECIBIDO - Tipos en orden:")
        for i, comp in enumerate(components, 1):
            print(f"  {i}. {type(comp).__name__}")

        # Asignar en el orden exacto que se devuelve
        (
            self.retriever_edaes,  # 1 - VectorStoreRetriever
            self.retriever_pruebas,  # 2 - VectorStoreRetriever
            self.retriever_newMater,  # 3 - VectorStoreRetriever
            self.vectorstore,  # 4 - Chroma
            self.retriever_resumen_edaes,  # 5 - VectorStoreRetriever
            self.retriever_edaes_seg,  # 6 - VectorStoreRetriever
            self.retriever_historia,  # 7 - VectorStoreRetriever
        ) = components

        # Validar asignaciones
        retrievers_check = [
            ("self.retriever_edaes", self.retriever_edaes),
            ("self.retriever_pruebas", self.retriever_pruebas),
            ("self.retriever_newMater", self.retriever_newMater),
            ("self.retriever_resumen_edaes", self.retriever_resumen_edaes),
            ("self.retriever_edaes_seg", self.retriever_edaes_seg),
            ("self.retriever_historia", self.retriever_historia),
        ]

        for name, obj in retrievers_check:
            if not hasattr(obj, "invoke"):
                raise AttributeError(
                    f"ERROR: {name} no tiene método 'invoke'. Tipo: {type(obj)}"
                )
            print(f"[OK] {name}: {type(obj).__name__}")

        print(f"[OK] vectorstore: {type(self.vectorstore).__name__}")

        self.llm_chain = inicializar_modelo(model_name, temperature)
        self.reranker = CrossEncoder(re_ranker_model)
        self.re_type = "BASE"

        print(f"[OK] ChatbotController inicializado exitosamente")

    def obtener_respuesta(self, pregunta, usar_contexto_completo=True):
        respuesta, tiempo, referencias = chatbot_response(
            pregunta,  # query
            usar_contexto_completo,  # usar_full_context
            self.llm_chain,  # llm_chain
            self.retriever_edaes,  # retriever_edaes (4to arg)
            self.retriever_pruebas,  # retriever_pruebas (5to arg)
            self.reranker,  # reranker
            self.vectorstore,  # vectorstore
            self.retriever_resumen_edaes,  # retriever_resumen_edaes
            self.retriever_edaes_seg,  # retriever_edaes_seg
            self.retriever_newMater,  # retriever_newMater
            self.retriever_historia,  # retriever_historia
            self.re_type,  # re_type
        )

        # Guardar en el log de conversación
        self._guardar_en_log(pregunta, respuesta, tiempo)

        return respuesta, tiempo, referencias

    def _guardar_en_log(self, pregunta, respuesta, tiempo):
        """Guarda pregunta, respuesta y tiempo en el archivo de log"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"Pregunta: {pregunta}\n")
                f.write(f"Respuesta: {respuesta}\n")
                f.write(f"Tiempo_respuesta: {tiempo:.2f}s\n")
                f.write(f"{'-'*80}\n\n")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar en log: {e}")

    def guardar_ajustes(self, cant_documentos, temperatura, modelo, re_type):
        (
            self.retriever_edaes,
            self.retriever_pruebas,
            self.vectorstore,
            self.retriever_resumen_edaes,
            self.retriever_newMater,
            self.retriever_edaes_seg,
            self.retriever_historia,
        ) = inicializar_retriever_vectorstore(cant_documentos)
        self.llm_chain = inicializar_modelo(modelo, temperatura)
        self.re_type = re_type

    def iniciar_benchmark(self, folder_path, goal_set_path, iteraciones):
        print(folder_path)
        print(goal_set_path)
        print(iteraciones)
