#!/usr/bin/env python3
"""
Script para verificar el contenido de las bases de datos vectoriales ChromaDB
"""

import os
import chromadb
from chromadb.config import Settings


def verificar_chromadb():
    """Verifica el contenido de las bases de datos vectoriales"""

    # Ruta de la base de datos
    persist_directory = "./data/chroma_db"

    # Verificar si existe el directorio
    if not os.path.exists(persist_directory):
        print(f"❌ El directorio {persist_directory} no existe")
        return

    # Verificar archivos en el directorio
    archivos = os.listdir(persist_directory)
    print(f"📁 Archivos en {persist_directory}: {archivos}")

    if not archivos:
        print("❌ La base de datos está vacía")
        return

    try:
        # Conectar a ChromaDB
        client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        # Listar todas las colecciones
        collections = client.list_collections()
        print(f"\n📊 Colecciones encontradas: {len(collections)}")

        for collection in collections:
            print(f"\n🗃️  Colección: {collection.name}")
            print(f"   ID: {collection.id}")

            # Obtener información de la colección
            count = collection.count()
            print(f"   📄 Documentos: {count}")

            if count > 0:
                # Obtener algunos documentos de ejemplo
                results = collection.peek(limit=3)
                print(f"   📝 Primeros documentos:")

                if results and "documents" in results:
                    for i, doc in enumerate(results["documents"][:3]):
                        preview = doc[:100] + "..." if len(doc) > 100 else doc
                        print(f"      {i+1}. {preview}")

                # Verificar metadatos
                if results and "metadatas" in results and results["metadatas"]:
                    print(f"   🏷️  Metadatos de ejemplo: {results['metadatas'][0]}")

    except Exception as e:
        print(f"❌ Error al acceder a ChromaDB: {e}")


def verificar_documentos_especificos():
    """Verifica documentos específicos por colección"""

    persist_directory = "./data/chroma_db"

    # Nombres de las colecciones esperadas
    colecciones_esperadas = [
        "edaes",
        "pruebas",
        "newMater",
        "resumen_edaes",
        "edaes_seg",
    ]

    try:
        client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        print("\n🔍 Verificación detallada por colección:")

        for nombre_coleccion in colecciones_esperadas:
            try:
                collection = client.get_collection(name=nombre_coleccion)
                count = collection.count()
                print(f"\n✅ {nombre_coleccion}: {count} documentos")

                if count > 0:
                    # Hacer una búsqueda de prueba
                    results = collection.query(
                        query_texts=["información"], n_results=min(2, count)
                    )

                    if results and "documents" in results:
                        for i, doc in enumerate(results["documents"][0]):
                            preview = doc[:80] + "..." if len(doc) > 80 else doc
                            print(f"   📄 Doc {i+1}: {preview}")

            except Exception as e:
                print(f"❌ {nombre_coleccion}: No encontrada o error - {e}")

    except Exception as e:
        print(f"❌ Error general: {e}")


if __name__ == "__main__":
    print("🔍 VERIFICACIÓN DE BASE DE DATOS VECTORIAL CHROMADB")
    print("=" * 50)

    verificar_chromadb()
    verificar_documentos_especificos()

    print("\n" + "=" * 50)
    print("✅ Verificación completada")
