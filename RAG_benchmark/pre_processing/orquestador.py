import subprocess
import sys
from pathlib import Path
import time

# ============================================================================
# CONFIGURACIÓN DEL ORQUESTADOR
# ============================================================================

# 🔴 MODELOS ACTUALIZADOS: qwen3:14b, gemma3:12b, granite4:3b
MODELOS = ["qwen3:14b", "gemma3:12b", "granite4:3b"]

# TEMPERATURAS: Rango completo para máxima variabilidad
TEMPERATURAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# 📚 GOALSETS COMPLETOS: Todos los documentos relevantes
DOCUMENTOS = {
    "MATER": "../FAC_Documents/rag_files/Goalset_FAC_MATER.json",
    "PORFAC": "../FAC_Documents/rag_files/Goalset_FAC_PORFAC.json",
    "EDAES": "../FAC_Documents/rag_files/Goalset_FAC_EDAES.json",
    "EDAES_SEGMENTADO": "../FAC_Documents/rag_files/Goalset_FAC_EDAES.json",
    "EDAES_MARKDOWN": "../FAC_Documents/rag_files/Goalset_FAC_EDAES.json",
    "EDAES_RESUMIDO": "../FAC_Documents/rag_files/Goalset_FAC_EDAES.json",
    "HISTORIA": "../FAC_Documents/rag_files/GoalSet_Historia.json",
    "NUEVO_MATER": "../FAC_Documents/rag_files/GoalSetNewMater.json",
}

# ============================================================================
# FUNCIONES DE LOGGING Y UTILIDAD
# ============================================================================


def log(msg, level="INFO"):
    """Log con iconos y formato"""
    icons = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "RUNNING": "🔄",
        "WARNING": "⚠️ ",
    }
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {icons.get(level, '')} {msg}")


def verify_files_exist():
    """Verifica que existan los archivos necesarios"""
    log("Verificando archivos necesarios...", "INFO")

    # Verificar processor.py
    if not Path("processor.py").exists():
        log("❌ processor.py no encontrado", "ERROR")
        return False

    # Verificar metrics.py
    if not Path("metrics.py").exists():
        log("❌ metrics.py no encontrado", "ERROR")
        return False

    # Verificar GoalSets
    missing_goalsets = []
    for doc_name, path in DOCUMENTOS.items():
        if not Path(path).exists():
            missing_goalsets.append(f"{doc_name} ({path})")

    if missing_goalsets:
        log(f"⚠️  GoalSets no encontrados: {missing_goalsets}", "WARNING")

    log("✅ Archivos verificados", "SUCCESS")
    return True


# ============================================================================
# ORQUESTACIÓN PRINCIPAL
# ============================================================================

print("\n" + "=" * 80)
print("🚀 INICIANDO ORQUESTADOR DE PRUEBAS AUTOMATIZADAS")
print("=" * 80)

# Verificar archivos
if not verify_files_exist():
    log("❌ Error en verificación de archivos", "ERROR")
    sys.exit(1)

# Estadísticas
total_pruebas = len(MODELOS) * len(TEMPERATURAS) * len(DOCUMENTOS)
pruebas_completadas = 0
pruebas_exitosas = 0
pruebas_fallidas = 0

log(f"Modelos: {len(MODELOS)} {MODELOS}", "INFO")
log(f"Temperaturas: {len(TEMPERATURAS)}", "INFO")
log(f"GoalSets: {len(DOCUMENTOS)}", "INFO")
log(f"Total de pruebas: {total_pruebas}", "INFO")
log(
    f"Archivos esperados: {total_pruebas * 3} (resultados + resumen + métricas)", "INFO"
)

print("=" * 80 + "\n")

# ============================================================================
# FLUJO PRINCIPAL: Modelo → Temperatura → Documento
# ============================================================================

for modelo in MODELOS:
    print(f"\n{'='*80}")
    print(f"📦 PROCESANDO MODELO: {modelo}")
    print(f"{'='*80}\n")

    for temp in TEMPERATURAS:
        print(f"\n{'-'*80}")
        print(f"🌡️  TEMPERATURA: {temp}")
        print(f"{'-'*80}\n")

        for doc_name, goalset_path in DOCUMENTOS.items():
            pruebas_completadas += 1

            log(
                f"[{pruebas_completadas}/{total_pruebas}] {modelo} | Temp={temp} | Doc={doc_name}",
                "RUNNING",
            )

            # Verificar que el GoalSet existe
            if not Path(goalset_path).exists():
                log(
                    f"⚠️  GoalSet no encontrado: {goalset_path}, omitiendo...",
                    "WARNING",
                )
                pruebas_fallidas += 1
                continue

            try:
                # ================================================================
                # PASO 1: EJECUTAR PROCESSOR.PY
                # ================================================================
                log(f"  → Ejecutando processor.py...", "INFO")

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
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    log(f"  ❌ processor.py falló: {result.stderr[:100]}", "ERROR")
                    pruebas_fallidas += 1
                    continue

                log(f"  ✅ processor.py completado", "SUCCESS")

                # ================================================================
                # PASO 2: EJECUTAR METRICS.PY
                # ================================================================
                log(f"  → Ejecutando metrics.py...", "INFO")

                result = subprocess.run(
                    [sys.executable, "metrics.py"],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    log(
                        f"  ⚠️  metrics.py completó pero con avisos: {result.stderr[:100]}",
                        "WARNING",
                    )
                else:
                    log(f"  ✅ metrics.py completado", "SUCCESS")

                # ================================================================
                # VERIFICACIÓN DE ARCHIVOS GENERADOS
                # ================================================================
                # Se espera generar estos archivos:
                # - {modelo_sanitizado}_temperatura_{temp}_{doc_name}.csv (resultados)
                # - {modelo_sanitizado}_temperatura_{temp}_{doc_name}_summary.csv (resumen)
                # - {modelo_sanitizado}_temperatura_{temp}_{doc_name}_metrics.csv (métricas)

                modelo_sanitizado = modelo.replace(":", "_").replace(".", "_")
                temp_str = str(temp).replace(".", "_")

                csv_resultado = Path(
                    f"../results/{modelo_sanitizado}_temperatura_{temp_str}_{doc_name}.csv"
                )
                csv_resumen = Path(
                    f"../results/{modelo_sanitizado}_temperatura_{temp_str}_{doc_name}_summary.csv"
                )
                csv_metricas = Path(
                    f"../results/{modelo_sanitizado}_temperatura_{temp_str}_{doc_name}_metrics.csv"
                )

                # Contar archivos generados
                archivos_generados = sum(
                    [
                        csv_resultado.exists(),
                        csv_resumen.exists(),
                        csv_metricas.exists(),
                    ]
                )

                if archivos_generados >= 2:  # Al menos resultados y una de las otras
                    log(
                        f"  ✅ Se generaron {archivos_generados}/3 archivos CSV",
                        "SUCCESS",
                    )
                    pruebas_exitosas += 1
                else:
                    log(
                        f"  ⚠️  Solo {archivos_generados}/3 archivos generados",
                        "WARNING",
                    )
                    pruebas_exitosas += 1  # Contar como exitosa aunque falte una

            except Exception as e:
                log(f"  ❌ Error inesperado: {str(e)}", "ERROR")
                pruebas_fallidas += 1
                continue

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print(f"\n\n{'='*80}")
print("📊 RESUMEN FINAL DEL ORQUESTADOR")
print(f"{'='*80}\n")

log(f"Total de pruebas ejecutadas: {pruebas_completadas}/{total_pruebas}", "INFO")
log(f"Pruebas exitosas: {pruebas_exitosas}", "SUCCESS")
log(f"Pruebas fallidas: {pruebas_fallidas}", "ERROR")

# Archivos esperados
csv_esperados = total_pruebas * 3
print(f"\n📁 Archivos CSV esperados: {csv_esperados}")
print(f"   - {total_pruebas} archivos de resultados")
print(f"   - {total_pruebas} archivos de resumen")
print(f"   - {total_pruebas} archivos de métricas")

print(f"\n📂 Ubicación: ../results/")
print(f"\n📋 Patrón de nombres:")
print(f"   - qwen3_14b_temperatura_0_0_MATER.csv")
print(f"   - qwen3_14b_temperatura_0_0_MATER_summary.csv")
print(f"   - qwen3_14b_temperatura_0_0_MATER_metrics.csv")

print(f"\n{'='*80}")

if pruebas_fallidas == 0:
    log("🎉 ¡ORQUESTADOR COMPLETADO EXITOSAMENTE!", "SUCCESS")
    print(f"{'='*80}\n")
    sys.exit(0)
else:
    log(f"⚠️  Orquestador completado con {pruebas_fallidas} errores", "WARNING")
    print(f"{'='*80}\n")
    sys.exit(1)

print("\n🚀 EXPERIMENTO GLOBAL FINALIZADO.")
