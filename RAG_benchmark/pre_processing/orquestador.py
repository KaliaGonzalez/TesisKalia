import subprocess
import sys
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MODELOS = ["gemma3:12b", "granite4:3b", "qwen3:14b"]
TEMPERATURAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

DOCUMENTOS = {
    "MATER": "../FAC_Documents/rag_files/Goalset_FAC_MATER.json",
    "PORFAC": "../FAC_Documents/rag_files/Goalset_FAC_PORFAC.json",
    "EDAES": "../FAC_Documents/rag_files/Goalset_FAC_EDAES.json",
    "EDAES_SEGMENTADO": "../FAC_Documents/rag_files/Goalset_FAC_EDAES.json",
    "EDAES_MARKDOWN": "../FAC_Documents/rag_files/Goalset_FAC_EDAES.json",
    "EDAES_RESUMIDO": "../FAC_Documents/rag_files/Goalset_FAC_EDAES.json",
    "HISTORIA": "../FAC_Documents/rag_files/GoalSet_Historia.json",
    "NUEVO_MATER": "../FAC_Documents/rag_files/GoalSet_NuevoMATER.json",
}

# ============================================================================
# LOGGING
# ============================================================================


def log(msg, level="INFO"):
    """Log message with level indicator"""
    icons = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "RUNNING": "🔄",
    }
    print(f"{icons.get(level, '')} {msg}")


# ============================================================================
# MAIN LOOP
# ============================================================================

log(f"🚀 INICIANDO ORQUESTADOR DE PRUEBAS", "RUNNING")
log(f"Documentos: {len(DOCUMENTOS)}", "INFO")
log(f"Modelos: {len(MODELOS)}", "INFO")
log(f"Temperaturas: {len(TEMPERATURAS)}", "INFO")
log(
    f"Total de combinaciones: {len(DOCUMENTOS) * len(MODELOS) * len(TEMPERATURAS)}",
    "INFO",
)
print("=" * 80)

contador_total = 0
contador_exito = 0
contador_error = 0

try:
    for doc_name, goalset_path in DOCUMENTOS.items():
        for modelo in MODELOS:
            for temp in TEMPERATURAS:
                contador_total += 1

                print(f"\n{'='*80}")
                log(
                    f"[{contador_total}] Documento={doc_name} | Modelo={modelo} | Temp={temp}",
                    "RUNNING",
                )
                print(f"{'='*80}")

                # ============================================================
                # PASO 1: EJECUTAR PROCESSOR.PY
                # ============================================================
                log("Ejecutando processor.py...", "INFO")
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "processor.py",
                            "--model",
                            modelo,
                            "--temperature",
                            str(temp),
                            "--goalset",
                            goalset_path,
                            "--doc_name",
                            doc_name,
                        ],
                        check=True,
                        capture_output=False,
                    )
                    log(f"✅ processor.py completado", "SUCCESS")
                except subprocess.CalledProcessError as e:
                    log(f"❌ processor.py falló: {e}", "ERROR")
                    contador_error += 1
                    continue

                # ============================================================
                # PASO 2: EJECUTAR METRICS.PY
                # ============================================================
                log("Ejecutando metrics.py...", "INFO")
                try:
                    result = subprocess.run(
                        [sys.executable, "metrics.py"],
                        check=True,
                        capture_output=False,
                    )
                    log(f"✅ metrics.py completado", "SUCCESS")
                    contador_exito += 1
                except subprocess.CalledProcessError as e:
                    log(f"❌ metrics.py falló: {e}", "ERROR")
                    contador_error += 1

except KeyboardInterrupt:
    log("\n⚠️  Orquestador interrumpido por el usuario", "ERROR")
    sys.exit(1)
except Exception as e:
    log(f"❌ Error general: {e}", "ERROR")
    sys.exit(1)

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print(f"\n{'='*80}")
log("🎉 ORQUESTADOR FINALIZADO", "SUCCESS")
print(f"{'='*80}")
log(f"Total de pruebas: {contador_total}", "INFO")
log(f"Exitosas: {contador_exito}", "SUCCESS")
log(f"Con error: {contador_error}", "ERROR")
print(f"{'='*80}\n")

if contador_error > 0:
    sys.exit(1)
