import sys
import os

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
        print(f"🚀 Inicializando ChatbotController con k={k}")

        # Obtener los componentes del RAG
        components = inicializar_retriever_vectorstore(k)

        print(f"📥 RECIBIDO - Tipos en orden:")
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
            print(f"✅ {name}: {type(obj).__name__}")

        print(f"✅ vectorstore: {type(self.vectorstore).__name__}")

        self.llm_chain = inicializar_modelo(model_name, temperature)
        self.reranker = CrossEncoder(re_ranker_model)
        self.re_type = "BASE"

        print(f"🎯 ChatbotController inicializado exitosamente")

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
        return respuesta, tiempo, referencias

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
