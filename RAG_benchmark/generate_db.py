#!/usr/bin/env python3
"""
Script simplificado para generar solo la base de datos vectorial usando myway.py
"""

import sys
import os
sys.path.append('pre_processing')

from myway import (
    parse_markdown_porfac, parse_markdown_mater, parse_markdown_edaes, 
    parse_markdown_Historia, create_specialized_mater_chunks,
    store_in_chromadb, inicializar_retriever_vectorstore
)

def generate_vector_db():
    """Genera la base de datos vectorial con todos los documentos"""
    
    print("🔄 Generando base de datos vectorial...")
    print("=" * 50)
    
    try:
        # Lista para almacenar todas las secciones
        all_sections = []
        
        # Documentos a procesar
        documents = [
            ("FAC_Documents/rag_files/reglamento_porfac.md", parse_markdown_porfac, "PORFAC"),
            ("FAC_Documents/rag_files/manual_de_terminos_fuerza_aerea_colombiana.md", parse_markdown_mater, "MATER"),
            ("FAC_Documents/rag_files/edaes.md", parse_markdown_edaes, "EDAES"),
            ("FAC_Documents/rag_files/Historia.md", parse_markdown_Historia, "Historia"),
            ("FAC_Documents/rag_files/NuevoMATER.md", create_specialized_mater_chunks, "NuevoMATER")
        ]
        
        # Procesar cada documento
        for file_path, parser_func, doc_name in documents:
            if os.path.exists(file_path):
                print(f"📄 Procesando {doc_name}...")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                sections = []
                parser_func(content, sections)
                all_sections.extend(sections)
                
                print(f"   ✅ {doc_name}: {len(sections)} secciones procesadas")
            else:
                print(f"   ⚠️  {doc_name}: Archivo no encontrado - {file_path}")
        
        # Almacenar en ChromaDB
        print(f"\n�️  Almacenando {len(all_sections)} secciones en ChromaDB...")
        
        persist_directory = "chroma_db_4"
        collection_name = "rag_collection_v4"
        embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        store_in_chromadb(
            all_sections, 
            persist_directory, 
            collection_name, 
            embedding_model_name
        )
        
        # Verificar el contenido usando el retriever
        print(f"\n🔍 Verificando contenido...")
        result = inicializar_retriever_vectorstore(k=5)
        
        # El vectorstore está en la posición 3 (cuarto elemento de la tupla)
        vectorstore = result[3]  # vectorstore_edaes
        
        # Buscar "honor militar"
        honor_docs = vectorstore.similarity_search("honor militar", k=3)
        
        if honor_docs:
            print(f"✅ ¡Honor Militar encontrado en la base de datos!")
            print(f"   Documentos encontrados: {len(honor_docs)}")
            for i, doc in enumerate(honor_docs[:2]):
                print(f"   Doc {i+1}: {doc.page_content[:100]}...")
        else:
            print("❌ Honor Militar NO encontrado en la base de datos")
            
        return True
        
    except Exception as e:
        print(f"❌ Error al generar la base de datos: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🗄️  GENERACIÓN DE BASE DE DATOS VECTORIAL")
    print("=" * 50)
    
    success = generate_vector_db()
    
    if success:
        print("\n✅ BASE DE DATOS GENERADA EXITOSAMENTE")
        print("💡 Ahora puedes ejecutar: streamlit run rag.py")
    else:
        print("\n❌ ERROR AL GENERAR LA BASE DE DATOS")