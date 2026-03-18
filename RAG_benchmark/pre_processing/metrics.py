import re
import time
import unicodedata
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
import csv
from collections import defaultdict

# Leer la configuración desde config.py
from config import (
    model_name,
    output_filename, # ENTRADA para este script
    metrics_output,           # SALIDA 1 (métricas detalladas)
    metrics_output_summary             # SALIDA 2 (resumen de métricas)
)

# Lista ordenada de nombres de métricas para cabeceras y cálculos consistentes
ORDERED_METRIC_NAMES = [
    'EM', 'TokenOverlap', 'LengthRatio',
    'BLEU-1', 'BLEU-2', 'BLEU-4',
    'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR',
    'BERT-P', 'BERT-R', 'BERT-F1', 'SemanticSim'
]

# 1. Configuración inicial y carga de modelos (sin cambios)
def cargar_recursos():
    print("Cargando recursos para métricas...")
    recursos = {
        'rouge': load('rouge'),
        'meteor': load('meteor'),
        'bertscore': load('bertscore'),
        'smoother': SmoothingFunction().method2, # Usar un método específico, e.g., method2
        'sbert': SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    }
    print("Recursos cargados.")
    return recursos

# 2. Funciones de preprocesamiento (sin cambios)
def normalizar_texto(texto):
    if texto is None: return ""
    texto = unicodedata.normalize('NFKD', str(texto).lower())
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# 3. Extracción de datos del archivo CSV de resultados (sin cambios)
def extraer_datos_desde_csv(filepath_csv):
    print(f"Extrayendo datos desde el archivo CSV de entrada: {filepath_csv}")
    datos_agrupados_por_pregunta = {}
    try:
        with open(filepath_csv, "r", encoding="utf-8", newline='') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader):
                try:
                    preg_id = int(row["Pregunta_ID"])
                    # Corregido para usar la columna correcta si es "Pregunta" o "Pregunta_Texto"
                    pregunta_texto = row.get("Pregunta_Texto", row.get("Pregunta")) 
                    respuesta_esperada = row["Respuesta_Esperada"]
                    iter_num = int(row["Iteracion_Num"])
                    respuesta_limpia_generada = row["Respuesta_Limpia"]
                    tiempo_segundos = float(row["Tiempo_Segundos"])

                    if preg_id not in datos_agrupados_por_pregunta:
                        datos_agrupados_por_pregunta[preg_id] = {
                            "pregunta_texto": pregunta_texto,
                            "respuesta_esperada": respuesta_esperada, # Aún se extrae para el cálculo
                            "iteraciones_data": []
                        }
                    datos_agrupados_por_pregunta[preg_id]["iteraciones_data"].append(
                        (iter_num, respuesta_limpia_generada, tiempo_segundos)
                    )
                except KeyError as e:
                    print(f"Error de clave procesando fila {row_num + 1} del CSV: Falta la columna {e}. Fila: {row}")
                    continue
                except ValueError as e:
                    print(f"Error de valor procesando fila {row_num + 1} del CSV: {e}. Fila: {row}")
                    continue
        
        preguntas_formateadas = []
        for preg_id, data in sorted(datos_agrupados_por_pregunta.items()):
            iteraciones_ordenadas = sorted(data["iteraciones_data"], key=lambda x: x[0])
            preguntas_formateadas.append((
                preg_id,
                data["pregunta_texto"],
                data["respuesta_esperada"], # Se pasa para calcular_metricas_rag
                iteraciones_ordenadas
            ))
        print(f"Datos extraídos para {len(preguntas_formateadas)} preguntas.")
        return preguntas_formateadas
    except FileNotFoundError:
        print(f"Error: El archivo de entrada '{filepath_csv}' no fue encontrado.")
        return []
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer el CSV '{filepath_csv}': {e}")
        return []

# 4. Cálculo de métricas (sin cambios)
def calcular_metricas_rag(ref, pred, recursos):
    ref_norm = normalizar_texto(ref)
    pred_norm = normalizar_texto(pred)
    metricas = {'EM': 1.0 if ref_norm == pred_norm else 0.0}
    try:
        metricas['TokenOverlap'] = len(set(ref_norm.split()) & set(pred_norm.split())) / max(len(set(ref_norm.split()) | set(pred_norm.split())), 1)
    except ZeroDivisionError: metricas['TokenOverlap'] = 0.0
    try:
        metricas['LengthRatio'] = len(pred_norm.split()) / max(len(ref_norm.split()), 1) if len(ref_norm.split()) > 0 else 0.0
    except ZeroDivisionError: metricas['LengthRatio'] = 0.0
    
    for n in [1, 2, 4]:
        try:
            metricas[f'BLEU-{n}'] = sentence_bleu([ref_norm.split()], pred_norm.split(), weights=tuple([1/n]*n), smoothing_function=recursos['smoother'])
        except ZeroDivisionError: metricas[f'BLEU-{n}'] = 0.0
    
    if pred_norm and ref_norm:
        try:
            rouge_scores = recursos['rouge'].compute(predictions=[pred_norm], references=[ref_norm], use_stemmer=True)
            metricas.update({'ROUGE-1': rouge_scores['rouge1'], 'ROUGE-2': rouge_scores['rouge2'], 'ROUGE-L': rouge_scores['rougeL']})
        except Exception: metricas.update({'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0})
    else: metricas.update({'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0})

    if pred_norm and ref_norm:
        try: metricas['METEOR'] = recursos['meteor'].compute(predictions=[pred_norm], references=[ref_norm])['meteor']
        except Exception: metricas['METEOR'] = 0.0
    else: metricas['METEOR'] = 0.0

    if pred_norm and ref_norm:
        try:
            bert_scores = recursos['bertscore'].compute(predictions=[pred_norm], references=[ref_norm], lang="es", model_type="bert-base-multilingual-cased", device='cuda' if hasattr(util, 'torch') and util.torch.cuda.is_available() else 'cpu')
            metricas.update({'BERT-P': bert_scores['precision'][0], 'BERT-R': bert_scores['recall'][0], 'BERT-F1': bert_scores['f1'][0]})
        except Exception: metricas.update({'BERT-P': 0.0, 'BERT-R': 0.0, 'BERT-F1': 0.0})
    else: metricas.update({'BERT-P': 0.0, 'BERT-R': 0.0, 'BERT-F1': 0.0})
    
    if ref and pred:
        try:
            emb_ref = recursos['sbert'].encode(ref, convert_to_tensor=True)
            emb_pred = recursos['sbert'].encode(pred, convert_to_tensor=True)
            metricas['SemanticSim'] = util.pytorch_cos_sim(emb_ref, emb_pred).item()
        except Exception: metricas['SemanticSim'] = 0.0
    else: metricas['SemanticSim'] = 0.0
    return metricas

# 5. Función principal para procesar y generar archivos CSV de métricas
def procesar_y_generar_archivos_metricas(filepath_input_csv):
    inicio_total_script = time.time()
    recursos = cargar_recursos()
    datos_de_preguntas = extraer_datos_desde_csv(filepath_input_csv)

    if not datos_de_preguntas:
        print("No se extrajeron datos del archivo de entrada. Terminando script de métricas.")
        return

    # --- Preparación para el CSV de métricas detalladas ---
    cabeceras_csv_detallado = [
        "Pregunta_ID", "Iteracion_Num", "Pregunta_Texto",
        "Respuesta_Limpia_Generada", "Tiempo_Segundos_Generacion"
    ] + ORDERED_METRIC_NAMES
    
    # --- Preparación para el CSV de resumen de métricas ---
    cabeceras_csv_resumen = [
        "Tipo_Resumen", "Pregunta_ID", "Pregunta_Texto"
    ] + ORDERED_METRIC_NAMES + ["Tiempo_Promedio_Generacion_s"]

    # Almacenar datos para resúmenes
    metricas_por_pregunta = defaultdict(lambda: {"textos_pregunta": None, "metricas_lista": [], "tiempos_lista": []})
    todas_las_metricas_globales = []
    todos_los_tiempos_globales = []

    print(f"Abriendo archivo para métricas detalladas: {metrics_output}")
    with open(metrics_output, "w", encoding="utf-8", newline='') as f_detallado_csv:
        writer_detallado = csv.DictWriter(f_detallado_csv, fieldnames=cabeceras_csv_detallado)
        writer_detallado.writeheader()

        for preg_id, pregunta_texto, respuesta_esperada, iteraciones in datos_de_preguntas:
            if metricas_por_pregunta[preg_id]["textos_pregunta"] is None:
                 metricas_por_pregunta[preg_id]["textos_pregunta"] = pregunta_texto
            
            # print(f"Procesando Pregunta ID: {preg_id}") # Opcional
            for iter_num, respuesta_generada, tiempo_generacion in iteraciones:
                # print(f"  Calculando métricas para Iteración: {iter_num}") # Opcional
                metricas_iteracion = calcular_metricas_rag(respuesta_esperada, respuesta_generada, recursos)
                
                # Almacenar para resumen
                metricas_por_pregunta[preg_id]["metricas_lista"].append(metricas_iteracion)
                metricas_por_pregunta[preg_id]["tiempos_lista"].append(tiempo_generacion)
                todas_las_metricas_globales.append(metricas_iteracion)
                todos_los_tiempos_globales.append(tiempo_generacion)

                # Escribir al CSV detallado
                fila_detallada = {
                    "Pregunta_ID": preg_id,
                    "Iteracion_Num": iter_num,
                    "Pregunta_Texto": pregunta_texto,
                    "Respuesta_Limpia_Generada": respuesta_generada,
                    "Tiempo_Segundos_Generacion": round(tiempo_generacion, 2)
                }
                for nombre_metrica in ORDERED_METRIC_NAMES:
                    fila_detallada[nombre_metrica] = round(metricas_iteracion.get(nombre_metrica, 0.0), 4)
                writer_detallado.writerow(fila_detallada)
    print(f"Métricas detalladas guardadas en: {metrics_output}")

    # --- Generar el CSV de resumen de métricas ---
    print(f"Abriendo archivo para resumen de métricas: {metrics_output_summary}")
    with open(metrics_output_summary, "w", encoding="utf-8", newline='') as f_resumen_csv:
        writer_resumen = csv.DictWriter(f_resumen_csv, fieldnames=cabeceras_csv_resumen)
        writer_resumen.writeheader()

        # Resúmenes por pregunta
        print("Calculando resúmenes por pregunta...")
        for preg_id, data_pregunta in sorted(metricas_por_pregunta.items()):
            fila_resumen_pregunta = {
                "Tipo_Resumen": "Pregunta",
                "Pregunta_ID": preg_id,
                "Pregunta_Texto": data_pregunta["textos_pregunta"]
            }
            if data_pregunta["tiempos_lista"]:
                promedio_tiempo_gen = sum(data_pregunta["tiempos_lista"]) / len(data_pregunta["tiempos_lista"])
                fila_resumen_pregunta["Tiempo_Promedio_Generacion_s"] = round(promedio_tiempo_gen, 4)
            else:
                fila_resumen_pregunta["Tiempo_Promedio_Generacion_s"] = "N/A"

            for nombre_metrica in ORDERED_METRIC_NAMES:
                valores_metrica = [m.get(nombre_metrica, 0.0) for m in data_pregunta["metricas_lista"] if isinstance(m.get(nombre_metrica), (float, int))]
                if valores_metrica:
                    promedio_val = sum(valores_metrica) / len(valores_metrica)
                    fila_resumen_pregunta[nombre_metrica] = round(promedio_val, 4)
                else:
                    fila_resumen_pregunta[nombre_metrica] = 0.0 # o "N/A"
            writer_resumen.writerow(fila_resumen_pregunta)
        
        # Resumen global
        print("Calculando resumen global...")
        if todas_las_metricas_globales:
            fila_resumen_global = {
                "Tipo_Resumen": "Global",
                "Pregunta_ID": "GLOBAL_PROMEDIO",
                "Pregunta_Texto": "N/A"
            }
            if todos_los_tiempos_globales:
                promedio_tiempo_gen_global = sum(todos_los_tiempos_globales) / len(todos_los_tiempos_globales)
                fila_resumen_global["Tiempo_Promedio_Generacion_s"] = round(promedio_tiempo_gen_global, 4)
            else:
                fila_resumen_global["Tiempo_Promedio_Generacion_s"] = "N/A"

            for nombre_metrica in ORDERED_METRIC_NAMES:
                valores_metrica_global = [m.get(nombre_metrica, 0.0) for m in todas_las_metricas_globales if isinstance(m.get(nombre_metrica), (float, int))]
                if valores_metrica_global:
                    promedio_val_global = sum(valores_metrica_global) / len(valores_metrica_global)
                    fila_resumen_global[nombre_metrica] = round(promedio_val_global, 4)
                else:
                    fila_resumen_global[nombre_metrica] = 0.0 # o "N/A"
            writer_resumen.writerow(fila_resumen_global)
    print(f"Resumen de métricas guardado en: {metrics_output_summary}")

    tiempo_total_script_seg = time.time() - inicio_total_script
    print(f"\n{'=' * 60}")
    print(f"Evaluación de Métricas Completada")
    print(f"Modelo Evaluado: {model_name}")
    print(f"Entrada: {filepath_input_csv}")
    print(f"Salida Detallada: {metrics_output}")
    print(f"Salida Resumen: {metrics_output_summary}")
    if todos_los_tiempos_globales: print(f"Respuestas evaluadas: {len(todos_los_tiempos_globales)}")
    print(f"Tiempo total del script de métricas: {tiempo_total_script_seg:.2f}s")
    print(f"{'=' * 60}")

# 6. Configuración y ejecución
if __name__ == "__main__":
    print(f"Iniciando script de métricas para el modelo: {model_name}")
    print(f"Leyendo resultados detallados de (ENTRADA): {output_filename}")
    print(f"Métricas detalladas se guardarán en (SALIDA 1): {metrics_output}")
    print(f"Resumen de métricas se guardará en (SALIDA 2): {metrics_output_summary}")
    
    procesar_y_generar_archivos_metricas(output_filename)