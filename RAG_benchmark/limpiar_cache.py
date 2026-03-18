#!/usr/bin/env python3
"""
Script para limpiar el caché y reiniciar Streamlit completamente
"""

import streamlit as st
import shutil
import os
import sys


def limpiar_cache_streamlit():
    """Limpia completamente el caché de Streamlit"""

    # Limpiar caché de Streamlit
    if hasattr(st, "cache_data"):
        st.cache_data.clear()
    if hasattr(st, "cache_resource"):
        st.cache_resource.clear()

    # Limpiar archivos de caché
    cache_dirs = [
        os.path.expanduser("~/.streamlit"),
        ".streamlit",
        "__pycache__",
        "pre_processing/__pycache__",
    ]

    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Caché eliminado: {cache_dir}")
            except:
                print(f"❌ No se pudo eliminar: {cache_dir}")

    print("🧹 Caché de Streamlit limpiado")


def reiniciar_modulos():
    """Recargar módulos de Python"""

    modulos_a_recargar = [
        "pre_processing.myway",
        "pre_processing.chatbotcontroller",
        "pre_processing.config",
    ]

    for modulo in modulos_a_recargar:
        if modulo in sys.modules:
            del sys.modules[modulo]
            print(f"🔄 Módulo recargado: {modulo}")


if __name__ == "__main__":
    print("🧹 LIMPIANDO CACHÉ COMPLETO")
    print("=" * 30)

    limpiar_cache_streamlit()
    reiniciar_modulos()

    print("✅ Limpieza completada")
    print("\n💡 Ahora ejecuta: streamlit run rag.py")
