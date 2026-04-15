#!/usr/bin/env python3
"""
Regenerar base de datos completa usando myway.py
"""

import sys
import os

# FIX: Set encoding for Windows console output
if sys.stdout.encoding != "utf-8":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.append("pre_processing")

from myway import *


def regenerar_base_datos():
    """Regenera la base de datos completa"""

    print("[INFO] REGENERANDO BASE DE DATOS COMPLETA")
    print("=" * 50)

    try:
        # 1. Procesar todos los documentos
        print("📚 Procesando documentos...")

        # Procesar EDAES
        edaes_path = "FAC_Documents/rag_files/edaes.md"
        if os.path.exists(edaes_path):
            print(f"   📄 Procesando EDAES...")
            edaes_content = read_markdown_file(edaes_path)
            edaes_sections = []
            parse_markdown_edaes(edaes_content, edaes_sections)
            print(f"   ✅ EDAES: {len(edaes_sections)} secciones")

        # Procesar PORFAC
        porfac_path = "FAC_Documents/rag_files/reglamento_porfac.md"
        if os.path.exists(porfac_path):
            print(f"   📄 Procesando PORFAC...")
            porfac_content = read_markdown_file(porfac_path)
            porfac_sections = []
            parse_markdown_porfac(porfac_content, porfac_sections)
            print(f"   ✅ PORFAC: {len(porfac_sections)} secciones")

            # Verificar que Honor Militar esté en las secciones
            honor_encontrado = False
            for section in porfac_sections:
                if (
                    "honor militar" in section.get("text", "").lower()
                    or "honor militar" in section.get("title", "").lower()
                ):
                    honor_encontrado = True
                    print(
                        f"   ✅ Honor Militar encontrado en PORFAC: '{section['title']}'"
                    )
                    break

            if not honor_encontrado:
                print(f"   ❌ Honor Militar NO encontrado en las secciones de PORFAC")

        # Procesar MATER
        mater_path = "FAC_Documents/rag_files/NuevoMATER.md"
        if os.path.exists(mater_path):
            print(f"   📄 Procesando MATER...")
            mater_content = read_markdown_file(mater_path)
            mater_sections = []
            create_specialized_mater_chunks(mater_content, mater_sections)
            print(f"   ✅ MATER: {len(mater_sections)} secciones")

        # Procesar Historia
        historia_path = "FAC_Documents/rag_files/Historia.md"
        if os.path.exists(historia_path):
            print(f"   📄 Procesando Historia...")
            historia_content = read_markdown_file(historia_path)
            historia_sections = []
            parse_markdown_Historia(historia_content, historia_sections)
            print(f"   ✅ Historia: {len(historia_sections)} secciones")

        # 2. Combinar todas las secciones
        all_sections = []
        if "edaes_sections" in locals():
            all_sections.extend(edaes_sections)
        if "porfac_sections" in locals():
            all_sections.extend(porfac_sections)
        if "mater_sections" in locals():
            all_sections.extend(mater_sections)
        if "historia_sections" in locals():
            all_sections.extend(historia_sections)

        print(f"\n📊 Total de secciones: {len(all_sections)}")

        # 3. Almacenar en ChromaDB
        print("💾 Almacenando en ChromaDB...")
        persist_directory = "../data/chroma_db_v5"
        collection_name = "fac_documents_complete"
        embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        store_in_chromadb(
            all_sections, persist_directory, collection_name, embedding_model
        )
        print("✅ Base de datos almacenada")

        # 4. Verificar que funciona
        print("\n🔍 Verificando Honor Militar...")
        resultado = inicializar_retriever_vectorstore(k=5)
        vectorstore = resultado[3]  # vectorstore_edaes

        collection = vectorstore._collection
        results = collection.query(
            query_texts=["honor militar"],
            n_results=5,
            include=["documents", "metadatas"],
        )

        if results["documents"] and len(results["documents"][0]) > 0:
            print(f"✅ ¡Honor Militar encontrado en la base final!")
            for i, (doc, metadata) in enumerate(
                zip(results["documents"][0][:2], results["metadatas"][0][:2])
            ):
                print(f"   Doc {i+1}: {metadata.get('origin', 'N/A')} - {doc[:100]}...")
        else:
            print("❌ Honor Militar NO encontrado en la base final")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = regenerar_base_datos()

    if success:
        print("\n🎉 BASE DE DATOS REGENERADA EXITOSAMENTE")
        print("💡 Ahora puedes probar: streamlit run rag.py")
    else:
        print("\n💥 ERROR AL REGENERAR LA BASE DE DATOS")
