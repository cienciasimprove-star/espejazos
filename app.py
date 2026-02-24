import streamlit as st
from PIL import Image
import io
import base64
import pandas as pd
from docx import Document
from docx.shared import Inches
# Importa la librería de Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image as VertexImage, GenerationConfig
import json
import re
import random # Necesario para la clave aleatoria
from google.cloud import storage
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# --- IMPORTACIÓN CLAVE ---
# Importamos las TRES funciones que necesitamos
try:
    from graficos_plugins import (
        crear_grafico, 
        generar_grafico_desde_texto, 
        build_visual_json_with_llm,
        generar_imagen_artistica
    )
    GRAFICOS_DISPONIBLES = True
except ImportError:
    st.error("Advertencia: No se encontró el archivo 'graficos_plugins.py'. La previsualización de gráficos no funcionará.")
    GRAFICOS_DISPONIBLES = False
    # Definir funciones placeholder si falla la importación
    def crear_grafico(*args, **kwargs):
        return None
    def generar_grafico_desde_texto(*args, **kwargs):
        return None, None
    def build_visual_json_with_llm(*args, **kwargs):
        return None
    def generar_imagen_artistica(*args, **kwargs):
        return None
# --- Configuración de Google Cloud (hacer al inicio) ---
# Usa variables de entorno para el proyecto y la región, con valores por defecto
GCP_PROJECT = os.environ.get("GCP_PROJECT", "espejazos")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global") # <--- Cambiado a 'global' para soporte Gemini 3
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

# --- 1. FUNCIONES DEL NÚCLEO LLM (MULTIMODO) ---

def limpiar_json_robustez(raw_text):
    """
    Extrae el primer JSON objeto válido de una respuesta de texto crudo.
    Maneja: bloques markdown, strings con llaves, objetos anidados.
    """
    if not raw_text:
        return None

    # 1. Eliminar bloques de código markdown (```json ... ``` o ``` ... ```)
    text = re.sub(r'```(?:json)?\s*', '', raw_text).strip()

    # 2. Intentar parsear el texto completo directamente
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Extractor por balance de llaves: maneja strings con llaves y anidamiento
    i = 0
    while i < len(text):
        if text[i] != '{':
            i += 1
            continue
        depth = 0
        in_string = False
        escape_next = False
        for j in range(i, len(text)):
            ch = text[j]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except (json.JSONDecodeError, ValueError):
                            break  # Este { no abrió un JSON válido, probar el siguiente
        i += 1

    return None

def analizar_adn_llm(inputs, model_name):
    """
    Analiza una o varias fuentes (imágenes/texto) para extraer el ADN psicométrico.
    """
    model = GenerativeModel(model_name)
    partes = ["Eres un experto psicómetra. Tu tarea es analizar los siguientes ítems para extraer su ADN estructural."]
    
    if inputs.get("imagenes"):
        for img in inputs["imagenes"]:
            img_bytes = img.getvalue() if hasattr(img, 'getvalue') else img
            partes.append(VertexImage.from_bytes(img_bytes))
    
    if inputs.get("texto"):
        partes.append(f"Texto de referencia: {inputs['texto']}")
        
    partes.append("""
    Analiza y devuelve en formato JSON:
    1. ESTRUCTURAS: Patrones comunes en el enunciado y opciones (ej. negaciones, tablas, comparaciones).
    2. QUÉ EVALÚA: Competencia profunda y evidencia detectada.
    3. PROCESO COGNITIVO: Pasos mentales (identificar, relacionar, inferir) para resolverlo.
    4. ESQUELETO SINTÁCTICO: Estructura lógica de las oraciones.
    
    Responde solo con el JSON.
    """)
    
    try:
        response = model.generate_content(partes, generation_config={"response_mime_type": "application/json"})
        return response.text
    except Exception as e:
        st.error(f"Error analizando ADN: {e}")
        return None

def generar_ideas_llm(taxonomia_dict, model_name):
    """
    Genera 3 ideas creativas basadas en la taxonomía y las devuelve en formato JSON.
    """
    model = GenerativeModel(model_name)
    tax_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])
    prompt = f"""
    Basado en esta taxonomía:
    {tax_texto}
    
    Propón 3 contextos creativos y situacionales para crear una pregunta de alto impacto.
    Responde ÚNICAMENTE con un objeto JSON que tenga una lista llamada "ideas" con 3 strings cortos y potentes.
    Ejemplo: {{"ideas": ["Idea 1...", "Idea 2...", "Idea 3..."]}}
    """
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text).get("ideas", [])
    except Exception as e:
        return [f"Error generando ideas: {e}"]

def generar_item_llm(modo, inputs, taxonomia_dict, model_name, similitud="Alta", feedback_auditor=""):
    """
    GENERADOR UNIVERSAL: Maneja Nuevo, Inspirado y Espejo.
    Genera el ítem, pidiendo descripciones de gráficos en LENGUAJE NATURAL PURO.
    """
    model = GenerativeModel(model_name)
    taxonomia_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])
    clave_aleatoria = random.choice(['A', 'B', 'C', 'D'])

    # Preparar partes multimedia
    partes = []
    if inputs.get("imagenes"):
        for img in inputs["imagenes"]:
            try:
                img_bytes = img.getvalue() if hasattr(img, 'getvalue') else img
                partes.append(VertexImage.from_bytes(img_bytes))
            except Exception as e:
                st.warning(f"No se pudo procesar una de las imágenes: {e}")
    
    contexto_ref = inputs.get("texto", "")

    # Sección de feedback del auditor
    seccion_feedback = ""
    if feedback_auditor:
        seccion_feedback = f"""
        --- RETROALIMENTACIÓN DE AUDITORÍA (Error a corregir) ---
        El intento anterior fue rechazado. DEBES corregir los siguientes errores:
        {feedback_auditor}
        --- VUELVE A GENERAR EL ÍTEM CORRIGIENDO ESTO ---
        """

    # SECCIONES DE PROMPT DINÁMICAS SEGÚN MODO
    instrucciones_modo = ""
    if modo == "✨ Nuevo Ítem":
        instrucciones_modo = f"""
        Tu objetivo es crear un ítem DESDE CERO basado en este concepto inicial: "{contexto_ref}".
        Usa tu creatividad pedagógica para diseñar una situación auténtica.

        --- ANÁLISIS COGNITIVO OBLIGATORIO (Tu paso 1) ---
        Basado en la taxonomía (Evidencia, Afirmación, Competencia), define la Tarea Cognitiva exacta que el ítem debe evaluar.
        
        --- CONSTRUCCIÓN DEL ÍTEM (Tu paso 2) ---
        - ENUNCIADO: Debe ser claro y **NO** usar jerarquías ("más", "mejor", "principalmente").
        - CLAVE: La respuesta correcta DEBE ser la opción **{clave_aleatoria}**.
        - DISTRACTORES: Plausibles, basados en errores comunes. Deben tener la redacción "El estudiante podría escoger la opción XX porque... Sin embargo esto es incorrecto porque..."
        """
    elif modo == "🎨 Ítem Inspirado":
        instrucciones_modo = f"""
        Tu objetivo es crear un ítem INSPIRADO en los ejemplos adjuntos (imágenes y/o texto).
        1. Analiza el ADN psicométrico de las fuentes (Estructura, Tarea Cognitiva, Esqueleto Sintáctico).
        2. Identifica la habilidad cognitiva (ej., inferencia, comprensión literal, vocabulario).
        3. Observa el formato (cita breve, expresión subrayada, pregunta abierta, etc.).
        4. Crea un ítem TOTALMENTE NUEVO que herede esa estructura y profundidad cognitiva, pero con un tema diferente.
        - CLAVE: La respuesta correcta DEBE ser la opción **{clave_aleatoria}**.
        - DISTRACTORES: Plausibles, basados en errores comunes de la Tarea Cognitiva. Redacción: "El estudiante podría escoger la opción XX porque... Sin embargo esto es incorrecto porque..."
        """
    elif modo == "🪞 Ítem Espejo":
        filtro_similitud = {
            "Alta": "CLONACIÓN ESTRICTA: Cambia solo datos numéricos y nombres propios. Mantén estructura y orden exacto.",
            "Media": "CAMBIO CONTEXTUAL: Cambia el escenario y representación de datos. Mantén la lógica de resolución.",
            "Baja": "REINVENCIÓN ESENCIAL: El ítem debe lucir visualmente muy diferente. Puedes variar el orden de las ideas, pero DEBE evaluar la misma habilidad y usar el mismo esqueleto sintáctico profundo."
        }
        instrucciones_modo = f"""
        Eres un experto en evaluación educativa, con especialización en el diseño de ítems para pruebas estandarizadas de alto impacto, como la prueba Saber 11 en Colombia. Tu misión es crear una pregunta espejo que sea un clon psicométrico de la pregunta original. Esto significa que, aunque se aplique a un contexto nuevo, debe evaluar exactamente la misma habilidad, con el mismo formato y nivel de dificultad.

        **Shell Cognitivo (Pregunta Original):**
        Analiza la estructura lógica y la "Tarea Cognitiva" de la pregunta en la IMAGEN ADJUNTA o el texto de referencia.
        - Si la pregunta original usa una tabla o gráfico, tu ítem espejo también debería usar uno.
        - **¡IMPORTANTE!** Si las *opciones de respuesta* en la imagen original son gráficas o tablas, debes replicar esa estructura para las opciones del ítem espejo.
        2. Análisis de la pregunta modelo

        Identifica la habilidad cognitiva (p. ej., inferencia, comprensión literal, vocabulario).
        Observa el formato (cita breve, expresión subrayada, pregunta abierta, etc.).
        Revisa el tipo de distractores (antónimos, conceptos afines, distractores temáticos).
          
        **¡INSTRUCCIÓN CRÍTICA DE SIMILITUD! ({similitud})**
            Regla: {filtro_similitud.get(similitud)}
            1.  **NO CAMBIES LA ESTRUCTURA**: Si la pregunta usa una tabla, tu ítem espejo debe usar una tabla con la MISMA ESTRUCTURA (mismas columnas y filas).
            2.  **DEBES CAMBIAR**:
                - Los **valores numéricos** 
                - Los **nombres ficticios**
                - Los **contextos**
            3. Creación de la pregunta espejo
            Sobre el contexto nuevo, elabora una pregunta que:
            Evalúe la misma competencia y evidencia, con igual nivel de dificultad.
            Repita el formato estructural.
            Garantice una respuesta correcta única y clara; los distractores deben ser plausibles, pero inequívocamente incorrectos.

        **Contexto Adicional del Usuario (Tema del ítem nuevo):**
        {contexto_ref}

        --- ANÁLISIS COGNITIVO OBLIGATORIO (Tu paso 1) ---
        Basado en la taxonomía (Evidencia, Afirmación, Competencia), define la Tarea Cognitiva exacta que el ítem espejo debe evaluar.
        
        --- CONSTRUCCIÓN DEL ÍTEM (Tu paso 2) ---
        Basado en tu análisis, construye el ítem.
        - ENUNCIADO: Debe ser claro y **NO** usar jerarquías ("más", "mejor", "principalmente").
        - CLAVE: La respuesta correcta DEBE ser la opción **{clave_aleatoria}**.
        - DISTRACTORES: Plausibles, basados en errores comunes de la Tarea Cognitiva. Deben tener la redacción "El estudiante podría escoger la opción XX porque... Sin embargo esto es incorrecto porque..."
        - DIFERENCIAS CON EL ITEM INICIAL: *CRITICO* NO se puede usar ninguno de los valores numéricos del ítem inicial. Deben ser totalmente diferentes.
        """

    prompt_final = f"""
    Eres un experto en psicometría educativa (estilo Saber 11).
    {instrucciones_modo}
    {seccion_feedback}
    
    **Taxonomía Requerida (Tu Guía):**
    {taxonomia_texto}
    
    --- INSTRUCCIONES DE SALIDA PARA GRÁFICO (ENUNCIADO Y OPCIONES) ---
    ¡INSTRUCCIÓN CRÍTICA! Para los gráficos, NO debes generar el JSON.
    En su lugar, proporciona una descripción detallada en LENGUAJE NATURAL de lo que el gráfico debe mostrar.
    
    Si el elemento (enunciado u opción) NO necesita un gráfico, usa "NO" y "N/A".
    Si SÍ necesita un gráfico, usa "SÍ" y escribe la descripción.
    
    Ejemplo de descripción: "Una tabla de 3 columnas y 2 filas. Las columnas son 'País', 'Capital', 'Población'. La primera fila es 'Colombia', 'Bogotá', '8M'. La segunda es 'Argentina', 'Buenos Aires', '3M'."
    Otro ejemplo: "Un gráfico de barras verticales simple con 3 barras. El eje X tiene las etiquetas 'A', 'B', 'C'. El eje Y (valores) tiene '10', '20', '15'."

    --- FORMATO DE SALIDA OBLIGATORIO (JSON VÁLIDO) ---
    Responde ÚNICAMENTE con el objeto JSON. No incluyas ```json.
    {{
      "pregunta_espejo": "Texto completo del enunciado/stem...",
      "clave": "{clave_aleatoria}",
      "justificacion_clave": "Razón por la que la clave es correcta...",
      
      "grafico_necesario_enunciado": "SÍ",
      "descripcion_texto_grafico_enunciado": "Una tabla simple. La primera fila es el encabezado con 'País' y 'Capital'. La segunda fila tiene 'Colombia' y 'Bogotá'.",
      
      "opciones": {{
        "A": {{
          "texto": "Ver gráfico A",
          "grafico_necesario": "SÍ",
          "descripcion_texto_grafico": "Un gráfico de barras verticales simple. El eje X tiene dos categorías: 'X' y 'Y'. Los valores del eje Y son 5 para 'X' y 10 para 'Y'."
        }},
        "B": {{
          "texto": "Texto de la Opción B (sin gráfico)",
          "grafico_necesario": "NO",
          "descripcion_texto_grafico": "N/A"
        }},
        "C": {{
          "texto": "Texto de la Opción C",
          "grafico_necesario": "NO",
          "descripcion_texto_grafico": "N/A"
        }},
        "D": {{
          "texto": "Texto de la Opción D",
          "grafico_necesario": "NO",
          "descripcion_texto_grafico": "N/A"
        }}
      }},
      
      "justificaciones_distractores": [
        {{ "opcion": "A", "justificacion": "Justificación para A..." }},
        {{ "opcion": "B", "justificacion": "Justificación para B..." }},
        {{ "opcion": "C", "justificacion": "Justificación para C..." }},
        {{ "opcion": "D", "justificacion": "Justificación para D..." }}
      ]
    }}
    """
    partes.append(prompt_final)

    try:
        response = model.generate_content(partes, generation_config={"response_mime_type": "application/json"})
        return limpiar_json_robustez(response.text)
    except Exception as e:
        st.error(f"Error en generación: {e}")
        return None


def refinar_item_llm(item_json_actual, feedback_usuario, taxonomia_dict, model_name):
    """
    REFINADOR: Toma un ítem existente y lo mejora basándose en feedback.
    """
    model = GenerativeModel(model_name)
    tax_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])
    
    prompt = f"""
    Eres un experto en psicometría educativa. Tu tarea es REFInAR el siguiente ítem basándote en el FEEDBACK del usuario.
    
    **Ítem Actual (JSON):**
    {item_json_actual}
    
    **Feedback del Usuario (Instrucciones de Mejora):**
    {feedback_usuario}
    
    **Taxonomía que debe mantenerse (OBLIGATORIO):**
    {tax_texto}
    
    --- REGLAS DE REFINAMIENTO ---
    1. Mantén la estructura general del JSON.
    2. Aplica los cambios solicitados en el feedback de forma creativa y profesional.
    3. Asegúrate de que el ítem siga siendo técnicamente correcto y alineado con la taxonomía.
    4. NO cambies la letra de la clave a menos que el feedback lo pida explícitamente.
    
    Responde ÚNICAMENTE con el nuevo JSON del ítem refinado.
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return limpiar_json_robustez(response.text)
    except Exception as e:
        st.error(f"Error en refinamiento: {e}")
        return None

def cargar_item_en_editor(item_json_texto):
    """
    Parsea un JSON de ítem y lo carga en los campos editables de session_state.
    """
    try:
        datos_obj = json.loads(item_json_texto)
        st.session_state['resultado_json_obj'] = datos_obj
        
        st.session_state.editable_pregunta = datos_obj.get("pregunta_espejo", "")
        st.session_state.editable_clave = datos_obj.get("clave", "")
        st.session_state.editable_just_clave = datos_obj.get("justificacion_clave", "")

        # Gráfico del Enunciado
        st.session_state.editable_grafico_nec_enunciado = datos_obj.get("grafico_necesario_enunciado", "NO")
        st.session_state.editable_grafico_texto_enunciado = datos_obj.get("descripcion_texto_grafico_enunciado", "N/A")
        st.session_state.editable_grafico_json_enunciado = "[]"
        st.session_state['img_buffer_enunciado'] = None

        # Opciones
        opciones = datos_obj.get("opciones", {})
        for letra in ["A", "B", "C", "D"]:
            opcion_obj = opciones.get(letra, {}) 
            st.session_state[f"editable_opcion_{letra.lower()}_texto"] = opcion_obj.get("texto", "")
            st.session_state[f"editable_opcion_{letra.lower()}_grafico_nec"] = opcion_obj.get("grafico_necesario", "NO")
            st.session_state[f"editable_opcion_{letra.lower()}_grafico_texto"] = opcion_obj.get("descripcion_texto_grafico", "N/A")
            st.session_state[f"editable_opcion_{letra.lower()}_grafico_json"] = "[]"
            st.session_state[f'img_buffer_op_{letra}'] = None

        # Justificaciones
        justifs_list = datos_obj.get("justificaciones_distractores", [])
        justifs_map = {j.get('opcion'): j.get('justificacion') for j in justifs_list}
        st.session_state.editable_just_a = justifs_map.get("A", "N/A")
        st.session_state.editable_just_b = justifs_map.get("B", "N/A")
        st.session_state.editable_just_c = justifs_map.get("C", "N/A")
        st.session_state.editable_just_d = justifs_map.get("D", "N/A")
        
        st.session_state.show_editor = True
    except Exception as e:
        st.error(f"Error al cargar el ítem en el editor: {e}")

# --- GENERADOR DE ESPECIFICACIÓN COMPLETA DE GRÁFICO (MODO LÓGICO) ---
def generar_descripcion_grafico_logico_llm(pregunta, opciones_texto, seccion, taxonomia_dict, model_name):
    """
    Genera una descripción data-completa del gráfico para modo lógico (JSON/Plot).
    Incluye: tipo de gráfico, todos los datos numéricos, etiquetas, ejes y título.
    Esta descripción se pasa a build_visual_json_with_llm para generar el JSON real.
    """
    model = GenerativeModel(model_name)
    tax_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])

    prompt = f"""
    Eres un experto en psicometría educativa y visualización de datos.
    El siguiente ítem de evaluación requiere un gráfico en la sección: {seccion}.

    **Contexto del Ítem:**
    - Enunciado: {pregunta}
    - Opciones: {opciones_texto}
    - Taxonomía: {tax_texto}

    Tu tarea es determinar el gráfico más adecuado para apoyar el ítem y generar una descripción
    COMPLETAMENTE ESPECIFICADA que incluya TODOS los datos necesarios para construirlo.

    --- REGLAS OBLIGATORIAS ---
    1. Elige el tipo de gráfico más apropiado para el concepto evaluado.
    2. Inventa datos COHERENTES con el contexto del ítem (datos verosímiles, no arbitrarios).
    3. La descripción debe incluir: tipo exacto, título, todos los valores numéricos con sus etiquetas,
       nombres de los ejes, unidades de medida y cualquier detalle necesario para construir el gráfico
       sin tomar ninguna decisión adicional.
    4. Escribe en español. Sé específico y completo.

    --- FORMATO DE SALIDA (JSON ÚNICAMENTE) ---
    {{
      "descripcion_completa": "Gráfico de barras verticales titulado 'Producción agrícola por departamento 2023'. Eje X: Departamento (Antioquia, Cundinamarca, Valle, Boyacá, Nariño). Eje Y: Toneladas producidas (miles). Valores: Antioquia 45, Cundinamarca 38, Valle 52, Boyacá 29, Nariño 33. El gráfico compara la producción agrícola total por departamento en el año 2023.",
      "tipo_grafico": "grafico_barras_verticales",
      "razon": "Justificación de por qué este gráfico y estos datos apoyan la pregunta."
    }}
    """

    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        data = json.loads(limpiar_json_robustez(response.text))
        return data.get("descripcion_completa", ""), data.get("razon", "")
    except Exception as e:
        print(f"Error generando descripción lógica del gráfico: {e}")
        return "", ""


# --- FUNCIÓN GENERADORA DE PROMPTS DE IMAGEN (NUEVA) ---
def generar_prompt_imagen_llm(pregunta, opciones_texto, seccion, taxonomia_dict, model_name):
    """
    Genera un prompt estructurado y optimizado para la creación de imágenes
    educativas acordes al tipo de ítem psicométrico.
    seccion: 'enunciado' o 'opcion_A/B/C/D'
    """
    model = GenerativeModel(model_name)
    tax_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])
    
    prompt = f"""
    Eres un experto en pedagogía visual y generación de prompts para IA de imágenes (estilo Imagen 3.0 / Midjourney).
    Tu tarea es generar un prompt de imagen ESTRUCTURADO para apoyar la siguiente pregunta de evaluación educativa.
    
    **Contexto del Ítem:**
    - Enunciado: {pregunta}
    - Opciones: {opciones_texto}
    - Sección que necesita imagen: {seccion}
    - Taxonomía (área y grado): {tax_texto}
    
    --- REGLAS PARA EL PROMPT ---
    1. El prompt debe describir una imagen CLARA, EDUCATIVA y PERTINENTE.
    2. Debe ser visualmente preciso: especifica colores, composición y estilo.
    3. Usa estilo "ilustración educativa limpia, vector art, fondo blanco".
    4. Adapta la complejidad visual al grado escolar indicado en la taxonomía.
    5. NO menciones texto, etiquetas ni palabras dentro de la imagen.
    6. El prompt debe estar en INGLÉS (Imagen 3.0 funciona mejor así).
    
    --- FORMATO DE SALIDA (JSON ÚNICAMENTE) ---
    {{
      "prompt_imagen": "Educational illustration of... [descripción detallada en inglés]",
      "razon": "Por qué este visual apoya el aprendizaje del concept evaluado."
    }}
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        data = json.loads(limpiar_json_robustez(response.text))
        return data.get("prompt_imagen", ""), data.get("razon", "")
    except Exception as e:
        print(f"Error generando prompt de imagen: {e}")
        return "", ""

# --- 2. FUNCIÓN DEL AUDITOR (ACTUALIZADA CON LIMPIEZA DE JSON) ---
def auditar_item_llm(item_json_texto, taxonomia_dict, model_name):
    """
    AUDITOR: Audita el ítem Y la coherencia de los gráficos (enunciado Y opciones).
    """
    
    # Modelo de Gemini (ahora dinámico)
    model = GenerativeModel(model_name)
    taxonomia_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])

    prompt_auditor = f"""
    Eres un auditor psicométrico ELITE y extremadamente RIGUROSO. Tu misión es asegurar que el ítem cumpla con los estándares de calidad psicométrica y la taxonomía educativa.

    **Taxonomía de Referencia (INNEGOCIABLE):**
    {taxonomia_texto}

    **Ítem Generado (JSON a Auditar):**
    {item_json_texto}

    --- CRITERIOS DE AUDITORÍA ---
    Evalúa cada criterio de forma independiente con "✅ CUMPLE", "⚠️ OBSERVACIÓN" o "❌ NO CUMPLE".

    1. **Alineación Taxonómica**: ¿El ítem evalúa exactamente la EVIDENCIA y AFIRMACIÓN solicitadas? Una desviación temática clara es "❌ NO CUMPLE". Una alineación parcial es "⚠️ OBSERVACIÓN".
    2. **Cero Jerarquización**: El enunciado no debe inducir la respuesta con palabras como "mejor", "más importante", "principalmente", "siempre". Si aparecen, es "❌ NO CUMPLE".
    3. **Nivel Cognitivo Real**: ¿El proceso mental requerido para resolver el ítem corresponde al nivel taxonómico? Un ítem de "análisis" que solo exige memoria es "❌ NO CUMPLE".
    4. **Univocidad de la Clave**: ¿Solo hay UNA respuesta definitivamente correcta? ¿Podría un experto defender otra opción? Si hay ambigüedad, es "❌ NO CUMPLE".
    5. **Plausibilidad de Distractores**: ¿Todos los distractores son plausibles para un estudiante con conocimiento parcial? Un distractor absurdo o trivialmente descartable es "⚠️ OBSERVACIÓN" o "❌ NO CUMPLE".
    6. **Justificaciones Educativas**: Cada justificación de distractor debe explicar la LÓGICA DEL ERROR del estudiante (qué confundió, qué malentendió), no solo por qué la opción es incorrecta.
    7. **Coherencia Visual**: Si se solicita un gráfico, la descripción debe ser inequívoca y suficiente para construirlo sin decisiones adicionales.

    --- REGLAS DE DICTAMEN ---
    - "✅ CUMPLE": Todos los criterios en "✅ CUMPLE" o "⚠️ OBSERVACIÓN", y ninguno en "❌ NO CUMPLE".
    - "⚠️ CUMPLE CON OBSERVACIONES": Al menos un "⚠️ OBSERVACIÓN" pero ningún "❌ NO CUMPLE". El ítem es usable pero mejorable.
    - "❌ RECHAZADO": Al menos un criterio en "❌ NO CUMPLE". El ítem debe regenerarse.

    --- FORMATO DE SALIDA (JSON ÚNICAMENTE) ---
    {{
      "criterios": [
        {{ "criterio": "Alineación Taxonómica", "estado": "✅ CUMPLE", "comentario": "..." }},
        {{ "criterio": "Cero Jerarquización", "estado": "✅ CUMPLE", "comentario": "..." }},
        {{ "criterio": "Nivel Cognitivo Real", "estado": "✅ CUMPLE", "comentario": "..." }},
        {{ "criterio": "Univocidad de la Clave", "estado": "✅ CUMPLE", "comentario": "..." }},
        {{ "criterio": "Plausibilidad de Distractores", "estado": "✅ CUMPLE", "comentario": "..." }},
        {{ "criterio": "Justificaciones Educativas", "estado": "✅ CUMPLE", "comentario": "..." }},
        {{ "criterio": "Coherencia Visual", "estado": "✅ CUMPLE", "comentario": "..." }}
      ],
      "dictamen_final": "✅ CUMPLE" o "⚠️ CUMPLE CON OBSERVACIONES" o "❌ RECHAZADO",
      "correcciones": [
        {{ "criterio": "nombre del criterio fallido", "prioridad": "ALTA" o "MEDIA", "accion": "Instrucción concreta y específica para corregirlo." }}
      ],
      "observaciones_finales": "Resumen ejecutivo del estado del ítem. Si se rechaza, explica el problema principal."
    }}
    """
    
    config_generacion = GenerationConfig(
        response_mime_type="application/json"
    )

    try:
        response = model.generate_content(
            prompt_auditor, 
            generation_config=config_generacion
        )
        
        raw_text = response.text
        return limpiar_json_robustez(raw_text)

    except Exception as e:
        st.error(f"Error al contactar Vertex AI (Auditor): {e}")
        return None

# --- 3. FUNCIONES DE EXPORTACIÓN (ACTUALIZADAS) ---

# --- 3. FUNCIONES DE EXPORTACIÓN (ACTUALIZADAS) ---

def _insertar_tabla_en_celda(celda, spec_tabla):
    """
    Inserta una tabla Word editable dentro de una celda del template,
    en lugar de un PNG. spec_tabla es el dict del tipo_elemento 'tabla'.
    """
    from docx.shared import Pt
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    datos = spec_tabla.get("datos", {})
    config = spec_tabla.get("configuracion", {})

    # Normalizar encabezados y filas (distintos formatos posibles de plugin_tabla)
    encabezados = (
        datos.get("encabezados")
        or datos.get("headers")
        or datos.get("columnas")
        or []
    )
    filas = (
        datos.get("filas")
        or datos.get("rows")
        or datos.get("data")
        or []
    )
    titulo = config.get("titulo", "")

    if not encabezados and not filas:
        return

    num_cols = len(encabezados) if encabezados else (len(filas[0]) if filas else 1)
    has_header = bool(encabezados)
    num_rows = (1 if has_header else 0) + len(filas)

    # Añadir tabla a la celda (queda debajo del párrafo vacío)
    tabla_word = celda.add_table(rows=num_rows, cols=num_cols)
    tabla_word.style = 'Table Grid'

    row_idx = 0
    # Fila de encabezados
    if has_header:
        for col_i, header in enumerate(encabezados[:num_cols]):
            cell = tabla_word.rows[row_idx].cells[col_i]
            cell.text = str(header)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.bold = True
        row_idx += 1

    # Filas de datos
    for fila in filas:
        for col_i, valor in enumerate(list(fila)[:num_cols]):
            tabla_word.rows[row_idx].cells[col_i].text = str(valor)
        row_idx += 1

    # Añadir título como párrafo si existe
    if titulo:
        p_titulo = celda.add_paragraph(titulo)
        p_titulo.runs[0].bold = True if p_titulo.runs else False

    return tabla_word


def reemplazar_texto_en_doc(doc, reemplazos, imagenes_reemplazo=None, tablas_reemplazo=None):
    """
    Recorre todos los párrafos y tablas en un documento y reemplaza los placeholders.
    - imagenes_reemplazo: {placeholder: BytesIO} → inserta PNG.
    - tablas_reemplazo: {placeholder: spec_dict} → inserta tabla Word editable.
    """
    if imagenes_reemplazo is None:
        imagenes_reemplazo = {}
    if tablas_reemplazo is None:
        tablas_reemplazo = {}

    def _reemplazar_en_parrafo(p, reemplazos, imagenes_reemplazo, img_width):
        """
        Reemplaza placeholders en un párrafo de forma robusta.
        Problema conocido: Word puede dividir {{placeholder}} entre varios runs.
        Solución: unir todo el texto, reemplazar, luego poner el resultado en
        el primer run y vaciar el resto (preserva el formato del primer run).
        """
        # --- TEXTO ---
        # 1. Verificar si hay algún placeholder en el texto completo del párrafo
        texto_completo = "".join(run.text for run in p.runs)
        texto_modificado = texto_completo
        hubo_cambio = False
        for clave, valor in reemplazos.items():
            if clave in texto_modificado:
                texto_modificado = texto_modificado.replace(clave, str(valor))
                hubo_cambio = True

        if hubo_cambio and p.runs:
            # Poner el texto completo en el primer run y vaciar los demás
            p.runs[0].text = texto_modificado
            for run in p.runs[1:]:
                run.text = ""

        # --- IMAGEN ---
        # Verificar en el texto recalculado (puede haber cambiado tras el paso anterior)
        texto_actual = "".join(run.text for run in p.runs)
        for clave, buf in imagenes_reemplazo.items():
            if clave in texto_actual and buf:
                # Limpiar todos los runs
                for run in p.runs:
                    run.text = run.text.replace(clave, "")
                # Añadir imagen en un run nuevo
                new_run = p.add_run()
                buf.seek(0)
                new_run.add_picture(buf, width=img_width)

    # 1. PÁRRAFOS
    for p in doc.paragraphs:
        _reemplazar_en_parrafo(p, reemplazos, imagenes_reemplazo, Inches(5.0))

    # 2. TABLAS DEL TEMPLATE
    # Placeholders "media" = aquellos que pueden llevar texto + imagen + tabla Word.
    # En la misma pasada: reemplazamos el texto, insertamos PNG (si hay) y tabla (si es tabla).
    media_phs = set(list(imagenes_reemplazo.keys()) + list(tablas_reemplazo.keys()))

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    # Texto completo del párrafo (maneja split-runs)
                    texto_p = "".join(r.text for r in p.runs)

                    # Buscar si hay un placeholder "media" en este párrafo
                    ph_media = None
                    for ph in media_phs:
                        if ph in texto_p:
                            ph_media = ph
                            break

                    if ph_media:
                        # 1. Reemplazar el placeholder con el texto de descripción
                        texto_nuevo = texto_p.replace(ph_media, reemplazos.get(ph_media, ""))
                        if p.runs:
                            p.runs[0].text = texto_nuevo
                            for r in p.runs[1:]:
                                r.text = ""

                        # 2. Insertar PNG si existe
                        buf = imagenes_reemplazo.get(ph_media)
                        if buf:
                            buf.seek(0)
                            new_run = p.add_run()
                            new_run.add_picture(buf, width=Inches(3.5))

                        # 3. Insertar tabla Word editable si es tipo tabla
                        spec = tablas_reemplazo.get(ph_media)
                        if spec:
                            _insertar_tabla_en_celda(cell, spec)
                    else:
                        # Párrafo normal: solo reemplazo de texto
                        _reemplazar_en_parrafo(p, reemplazos, {}, Inches(3.5))

    return doc

# --- 3. FUNCIONES DE EXPORTACIÓN (EXCEL REESCRITO CON AFIRMACIÓN) ---

def crear_excel(datos_generados, taxonomia_seleccionada, oportunidad_mejora):
    """
    Crea un Excel en formato HORIZONTAL (una fila por ítem) con las columnas
    específicas solicitadas.
    """
    
    # 1. Mapear las justificaciones a un diccionario para fácil acceso
    justificaciones = datos_generados.get("justificaciones_distractores", [])
    justifs_map = {j.get('opcion'): j.get('justificacion') for j in justificaciones}
    
    # La justificación de la clave correcta va en su columna correspondiente
    clave = datos_generados.get("clave", "").strip().upper()
    justificacion_clave = datos_generados.get("justificacion_clave", "")
    if clave and justificacion_clave:
        justifs_map[clave] = justificacion_clave
    
    # 2. Crear el diccionario de datos para la única fila
    data_dict = {
        # --- Columnas de Taxonomía ---
        "Área": taxonomia_seleccionada.get("Área", "N/A"),
        "RESPONSABLE": "IA ESPEJAZOS",
        "COMPONENTE": taxonomia_seleccionada.get("Componente_Estructura", "N/A"),
        "Competencia": taxonomia_seleccionada.get("Competencia", "N/A"),
        "Afirmación": taxonomia_seleccionada.get("Afirmación", "N/A"),
        "Evidencia": taxonomia_seleccionada.get("Evidencia", "N/A"),
        "Temática": taxonomia_seleccionada.get("Ref. Temática", "N/A"),
        "Nivel (curso)": taxonomia_seleccionada.get("Grado", "N/A"),
        
        # --- Columnas de Metadatos (Fijas) ---
        "PASTILLA": "NA",
        "Dificultad estimada": "NA",
        "Estándar": "NA",
        "Estado": "Espejo",
        "Número en el PDF": "NA",
        "ID ÍTEM": "NA",
        "ID CONTEXTO": "NA",

        # --- Columnas de Contenido del Ítem ---
        "Guía (Primeras palabras del ítem)": datos_generados.get("pregunta_espejo", "N/A"),
        "Opción A": datos_generados.get("opciones", {}).get("A", {}).get("texto", "N/A"),
        "Opción B": datos_generados.get("opciones", {}).get("B", {}).get("texto", "N/A"),
        "Opción C": datos_generados.get("opciones", {}).get("C", {}).get("texto", "N/A"),
        "Opción D": datos_generados.get("opciones", {}).get("D", {}).get("texto", "N/A"),
        "Oportunidad de mejora": oportunidad_mejora,
        "Justificación de la respuesta A": justifs_map.get("A", "N/A"),
        "Justificación de la respuesta B": justifs_map.get("B", "N/A"),
        "Justificación de la respuesta C": justifs_map.get("C", "N/A"),
        "Justificación de la respuesta D": justifs_map.get("D", "N/A"),
        "Clave": datos_generados.get("clave", "N/A")
    }

    # 3. Crear el DataFrame
    df = pd.DataFrame([data_dict])
    
    # 4. Forzar el orden de columnas exacto
    columnas_ordenadas = [
        "Área", "RESPONSABLE", "COMPONENTE", "Competencia", "Afirmación", "Evidencia",
        "PASTILLA", "Temática", "Dificultad estimada", "Estándar", "Estado", "Nivel (curso)",
        "Número en el PDF", "ID ÍTEM", "ID CONTEXTO", "Guía (Primeras palabras del ítem)",
        "Opción A", "Opción B", "Opción C", "Opción D",
        "Oportunidad de mejora",
        "Justificación de la respuesta A", "Justificación de la respuesta B",
        "Justificación de la respuesta C", "Justificación de la respuesta D",
        "Clave"
    ]
    
    # Filtra solo las columnas que existen en el df para evitar errores
    columnas_finales = [col for col in columnas_ordenadas if col in df.columns]
    df = df[columnas_finales]

    # 5. Guardar en el buffer de Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Item Generado')
    return output.getvalue()

@st.cache_data(show_spinner=False)
def _descargar_template_word(bucket_name, template_name):
    """Descarga el template Word desde GCS una sola vez por sesión."""
    try:
        storage_client = storage.Client(project=GCP_PROJECT)
        blob = storage_client.bucket(bucket_name).blob(template_name)
        if not blob.exists():
            return None
        return blob.download_as_bytes()
    except Exception:
        return None

def crear_word(datos_editados, taxonomia_seleccionada, oportunidad_mejora):
    """
    Genera un documento Word rellenando una plantilla desde GCS.
    """
    try:
        # 1. Obtener bytes del template (cacheado por sesión)
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "bucket-espejos1")
        template_name = os.environ.get("WORD_TEMPLATE_NAME", "formato_limpio.docx")
        template_bytes = _descargar_template_word(bucket_name, template_name)

        if template_bytes is None:
            st.error(f"Error: La plantilla '{template_name}' no se encontró en el bucket '{bucket_name}'.")
            return None

        doc = Document(io.BytesIO(template_bytes))
        
        # 2. Oportunidad de mejora (Ahora se recibe como argumento)
        # (La llamada a la IA se eliminó de aquí)
        
        # 3. Preparar los distractores
        clave = datos_editados.get("clave", "")
        distractores_texto = []
        for just in datos_editados.get("justificaciones_distractores", []):
            opcion = just.get("opcion")
            if opcion and opcion != clave:
                distractores_texto.append(f"Opción {opcion}: {just.get('justificacion', 'N/A')}")
        analisis_distractores = "\n".join(distractores_texto)

        # 4. Preparar las instrucciones de gráficos (convierte JSON a string)
        def get_grafico_json(data):
            if not data or data == "[]" or data == []:
                return "N/A"
            # Usamos ensure_ascii=False para que no escape tildes (ej. \u00f3)
            return json.dumps(data, ensure_ascii=False, indent=2)

        inst_enunciado = get_grafico_json(datos_editados.get("descripcion_grafico_enunciado", []))
        inst_a = get_grafico_json(datos_editados.get("opciones", {}).get("A", {}).get("descripcion_grafico", []))
        inst_b = get_grafico_json(datos_editados.get("opciones", {}).get("B", {}).get("descripcion_grafico", []))
        inst_c = get_grafico_json(datos_editados.get("opciones", {}).get("C", {}).get("descripcion_grafico", []))
        inst_d = get_grafico_json(datos_editados.get("opciones", {}).get("D", {}).get("descripcion_grafico", []))

        # 5. Definir todos los reemplazos (¡TODOS CON str()!)
        reemplazos = {
            "{{ItemPruebaId}}": str(taxonomia_seleccionada.get("Área", "N/A")),
            "{{ItemGradoId}}": str(taxonomia_seleccionada.get("Grado", "N/A")), 
            "{{CompetenciaNombre}}": str(taxonomia_seleccionada.get("Competencia", "N/A")),
            "{{ComponenteNombre}}": str(taxonomia_seleccionada.get("Componente_Estructura", "N/A")),
            "{{AfirmacionNombre}}": str(taxonomia_seleccionada.get("Afirmación", "N/A")),
            "{{EvidenciaNombre}}": str(taxonomia_seleccionada.get("Evidencia", "N/A")),
            "{{ItemContexto}}": "", 
            "{{ItemEnunciado}}": str(datos_editados.get("pregunta_espejo", "N/A")),
            "{{Opción A}}": str(datos_editados.get("opciones", {}).get("A", {}).get("texto", "N/A")),
            "{{Opción B}}": str(datos_editados.get("opciones", {}).get("B", {}).get("texto", "N/A")),
            "{{Opción C}}": str(datos_editados.get("opciones", {}).get("C", {}).get("texto", "N/A")),
            "{{Opción D}}": str(datos_editados.get("opciones", {}).get("D", {}).get("texto", "N/A")),
            "{{  Clave}}": str(clave).strip().upper(),
            "{{Justificacion_Correcta}}": str(datos_editados.get("justificacion_clave", "N/A")),
            "{{Analisis_Distractores}}": str(analisis_distractores),
            "{{Oportunidad_mejora}}": str(oportunidad_mejora)
        }

        # 5b. Instrucciones de gráfico como texto (o "No aplica" si está vacío)
        def _texto_instruccion(texto, grafico_nec):
            """Retorna el texto de instrucción o 'No aplica' si no hay gráfico o texto."""
            t = (texto or "").strip()
            if grafico_nec == "SÍ" and t and t.lower() not in ("n/a", ""):
                return t
            return "No aplica"

        reemplazos["{{Instrucciones_enuncuado}}"] = _texto_instruccion(
            datos_editados.get("texto_grafico_enunciado", ""),
            datos_editados.get("grafico_necesario_enunciado", "NO")
        )
        for letra in ["A", "B", "C", "D"]:
            op = datos_editados.get("opciones", {}).get(letra, {})
            reemplazos[f"{{{{Instrucciones_{letra}}}}}"] = _texto_instruccion(
                op.get("texto_grafico", ""),
                op.get("grafico_necesario", "NO")
            )

        # 6. Clasificar gráficos: tabla Word editable vs PNG
        def _es_tabla_word(spec_list):
            """Retorna el spec si es tipo tabla, None si no."""
            if isinstance(spec_list, list) and spec_list:
                s = spec_list[0]
                if isinstance(s, dict) and "tabla" in s.get("tipo_elemento", "").lower():
                    return s
            return None

        fuentes = [
            ("{{Instrucciones_enuncuado}}",
             datos_editados.get("descripcion_grafico_enunciado", []),
             datos_editados.get("img_buffer_enunciado")),
            ("{{Instrucciones_A}}",
             datos_editados.get("opciones", {}).get("A", {}).get("descripcion_grafico", []),
             datos_editados.get("opciones", {}).get("A", {}).get("img_buffer")),
            ("{{Instrucciones_B}}",
             datos_editados.get("opciones", {}).get("B", {}).get("descripcion_grafico", []),
             datos_editados.get("opciones", {}).get("B", {}).get("img_buffer")),
            ("{{Instrucciones_C}}",
             datos_editados.get("opciones", {}).get("C", {}).get("descripcion_grafico", []),
             datos_editados.get("opciones", {}).get("C", {}).get("img_buffer")),
            ("{{Instrucciones_D}}",
             datos_editados.get("opciones", {}).get("D", {}).get("descripcion_grafico", []),
             datos_editados.get("opciones", {}).get("D", {}).get("img_buffer")),
        ]

        imagenes_reemplazo = {}
        tablas_reemplazo = {}
        for ph, spec_list, img_buf in fuentes:
            tabla_spec = _es_tabla_word(spec_list)
            if tabla_spec:
                # Es tabla: poner en AMBOS — PNG (referencia visual) + tabla Word editable
                tablas_reemplazo[ph] = tabla_spec
                if img_buf:
                    imagenes_reemplazo[ph] = img_buf   # también el PNG
            elif img_buf:
                imagenes_reemplazo[ph] = img_buf       # solo PNG

        # 7. Ejecutar los reemplazos (texto, imágenes y tablas Word)
        doc = reemplazar_texto_en_doc(doc, reemplazos, imagenes_reemplazo, tablas_reemplazo)

        # 7. Guardar el documento final en un nuevo buffer
        final_buffer = io.BytesIO()
        doc.save(final_buffer)
        final_buffer.seek(0)
        return final_buffer

    except Exception as e:
        st.error(f"Error al crear el documento Word: {e}")
        return None


# --- NUEVA FUNCIÓN PARA LEER EXCEL DESDE GCS ---
@st.cache_data
def leer_excel_desde_gcs(bucket_name, file_path):
    """
    Lee un archivo Excel (con todas sus hojas) directamente desde GCS.
    """
    try:
        storage_client = storage.Client(project=GCP_PROJECT) # Usa el proyecto ya inicializado
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        
        if not blob.exists():
            st.error(f"Error: El archivo '{file_path}' no se encontró en el bucket '{bucket_name}'.")
            return None
            
        file_bytes = blob.download_as_bytes()
        data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        return data
    except Exception as e:
        st.error(f"Error al leer Excel desde GCS: {e}")
        st.info("Asegúrate de que la cuenta de servicio de Streamlit tenga permisos de 'Storage Object Viewer' en el bucket 'bucket-espejos1'.")
        return None
# --- FIN DE LA NUEVA FUNCIÓN ---


# --- 2b. NUEVA FUNCIÓN: GENERADOR DE OPORTUNIDAD DE MEJORA ---
def generar_oportunidad_mejora_llm(taxonomia_data, justificacion_clave, model_name):
    """
    Genera una breve recomendación académica basada en la habilidad evaluada.
    """
    try:
        model = GenerativeModel(model_name)
        
        # Extraemos los datos clave para el prompt
        evidencia = taxonomia_data.get("Evidencia", "la habilidad evaluada")
        competencia = taxonomia_data.get("Competencia", "la competencia general")
        
        prompt = f"""
        Eres un tutor académico experto. Tu tarea es escribir una breve recomendación (1-2 frases)
        para un estudiante o profesor.
        
        HABILIDAD EVALUADA (Evidencia): {evidencia}
        COMPETENCIA: {competencia}
        JUSTIFICACIÓN DE LA RESPUESTA CORRECTA: {justificacion_clave}

        Basado en esta información, escribe una recomendación fácil de aplicar durante clase.
        Habla en tercera persona y manten un tono formal pero sencillo
        NO uses más de 50 palabras.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        print(f"Error al generar oportunidad de mejora: {e}")
        return "Para mejorar en esta habilidad, repasa los conceptos clave de la competencia y practica con ejercicios similares."



# --- 4. INTERFAZ DE STREAMLIT (UI) ---

st.set_page_config(layout="wide")
st.title("🤖 Suite de Creación Psicométrica: Espejazos")

# --- NAVEGACIÓN LATERAL ---
with st.sidebar:
    st.header("Modo de Creación")
    modo_creacion = st.radio(
        "¿Qué deseas hacer hoy?",
        options=["✨ Nuevo Ítem", "🎨 Ítem Inspirado", "🪞 Ítem Espejo"],
        index=2 # Espejo por defecto para no romper el flujo previo
    )
    
    st.divider()
    st.header("⚙️ Configuración Global")
    # --- Selección de Modelos (Verificados Feb 2026) ---
    modelos_disponibles = [
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-exp"
    ]
    modelo_generador_sel = st.selectbox(
        "Modelo para Generar Ítem",
        options=modelos_disponibles,
        index=4  # gemini-2.5-flash por defecto
    )
    modelo_auditor_sel = st.selectbox(
        "Modelo para Auditar Ítem",
        options=modelos_disponibles,
        index=0,  # gemini-3.1-pro-preview: más capaz e independiente del generador
        help="Se recomienda usar un modelo diferente (y más potente) que el generador para una auditoría independiente."
    )

    st.divider()
    modo_visual = st.radio(
        "🎭 Modo de Generación Visual",
        options=["Lógico (JSON/Plot)", "Creativo (Nano Banana / AI)"],
        index=0,
        help="El modo Lógico es mejor para gráficos de datos. El modo Creativo (Nano Banana) es ideal para ilustraciones y escenas complejas."
    )
    st.session_state["modo_visual_preferido"] = "creativo" if "Creativo" in modo_visual else "logico"
    
# --- 1. ENTRADA DE DATOS SEGÚN MODO ---
col1, col2 = st.columns(2)

# --- COLUMNA 2 (Taxonomía) ---
with col2:
    st.header("2. Cascada de Taxonomía")
    
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "bucket_espejos")
    excel_file_path = os.environ.get("EXCEL_TAXONOMY_PATH", "Estructura privados1.xlsx")
    data = leer_excel_desde_gcs(bucket_name, excel_file_path)
    
    # Inicialización silenciosa de variables en el ámbito de la página
    grado_sel = area_sel = comp1_sel = comp2_sel = ref_sel = competen_sel = afirm_sel = evid_sel = None
    
    if data is not None:
        try:
            if 'df1' not in st.session_state or 'df2' not in st.session_state:
                sheet_names = list(data.keys())
                st.session_state.df1 = data[sheet_names[0]]
                st.session_state.df2 = data[sheet_names[1]]
            
            df1 = st.session_state.df1
            df2 = st.session_state.df2

            # Función auxiliar para mostrar estado visual
            def mark(val): return "✅" if val else "❌"

            st.write(f"{mark(st.session_state.get('tax_grado'))} **Grado y Área**")
            grado_sel = st.selectbox("Grado", options=df1['Grado'].unique(), key="tax_grado")
            
            df_grado = df1[df1['Grado'] == grado_sel]
            area_sel = st.selectbox("Área", options=df_grado['Área'].unique(), key="tax_area")

            st.write(f"{mark(st.session_state.get('tax_comp1'))} **Estructura**")
            df_area_h1 = df_grado[df_grado['Área'] == area_sel]
            comp1_sel = st.selectbox("Componente (Estructura)", options=df_area_h1['Componente1'].unique(), key="tax_comp1") 

            df_comp1 = df_area_h1[df_area_h1['Componente1'] == comp1_sel]
            competen_sel = st.selectbox("Competencia", options=df_comp1['Competencia'].unique(), key="tax_competen")

            df_competen = df_comp1[df_comp1['Competencia'] == competen_sel]
            df_af_base = df_competen[df_competen['Componente1'] == comp1_sel] if area_sel == 'Ciencias Naturales' else df_competen
            afirm_sel = st.selectbox("Afirmación", options=df_af_base['Afirmación'].unique(), key="tax_afirm")

            df_afirm = df_af_base[df_af_base['Afirmación'] == afirm_sel]
            evid_sel = st.selectbox("Evidencia", options=df_afirm['Evidencia'].unique(), key="tax_evid")

            st.write(f"{mark(st.session_state.get('tax_comp2'))} **Temática**")
            df_area_h2 = df2[(df2['Grado'] == grado_sel) & (df2['Área'] == area_sel)]
            comp2_sel = st.selectbox("Componente (Temática)", options=df_area_h2['Componente2'].unique(), key="tax_comp2")

            df_comp2 = df_area_h2[df_area_h2['Componente2'] == comp2_sel]
            refs = df_comp2['Ref. Temática'].unique() if not df_comp2.empty else ["N/A"]
            ref_sel = st.selectbox("Ref. Temática", options=refs, key="tax_ref")

        except Exception as e:
            st.error(f"Error en taxonomía: {e}")
            data = None

# --- COLUMNA 1 (Entrada de Datos) ---
with col1:
    contexto_adicional = ""
    imagen_subida = None
    
    if modo_creacion == "✨ Nuevo Ítem":
        st.header("1. Definir Nuevo Ítem")
        
        # Botones para generar propuestas
        if st.button("💡 Ayúdame a generar ideas", use_container_width=True):
            tax_keys = {
                "Grado": "tax_grado", "Área": "tax_area", 
                "Competencia": "tax_competen", "Afirmación": "tax_afirm", 
                "Evidencia": "tax_evid", "Ref. Temática": "tax_ref"
            }
            faltantes = [label for label, key in tax_keys.items() if not st.session_state.get(key)]
            
            if not faltantes:
                with st.spinner("Generando propuestas creativas..."):
                    tax_temp = {label: st.session_state.get(key) for label, key in tax_keys.items()}
                    st.session_state.ideas_propuestas = generar_ideas_llm(tax_temp, modelo_generador_sel)
            else:
                st.error(f"⚠️ **Faltan campos:** {', '.join(faltantes)}")

        # Mostrar ideas generadas como tarjetas interactivas
        if "ideas_propuestas" in st.session_state and st.session_state.ideas_propuestas:
            st.write("---")
            st.subheader("Selecciona una idea para personalizar:")
            for i, idea in enumerate(st.session_state.ideas_propuestas):
                with st.container(border=True):
                    st.write(idea)
                    if st.button(f"Seleccionar Opción {i+1}", key=f"btn_idea_{i}", use_container_width=True):
                        st.session_state.contexto_nuevo = idea
                        st.rerun()
            st.write("---")

        contexto_adicional = st.text_area(
            "Describe o edita el concepto del ítem",
            placeholder="Ej: Un problema sobre la ley de la gravitación universal...",
            height=200,
            key="contexto_nuevo"
        )
            
    elif modo_creacion == "🎨 Ítem Inspirado":
        st.header("1. Fuentes de Inspiración")
        metodo_inspiracion = st.radio("Método de entrada", ["🖼️ Imágenes", "📝 Texto/Links"], horizontal=True)
        
        if metodo_inspiracion == "🖼️ Imágenes":
            imagenes_inspiracion = st.file_uploader(
                "Sube uno o varios pantallazos", 
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True
            )
            imagen_subida = imagenes_inspiracion[0] if imagenes_inspiracion else None
            if imagenes_inspiracion:
                for img in imagenes_inspiracion:
                    st.image(img, use_container_width=True)
        else:
            contexto_adicional = st.text_area(
                "Pega aquí los ítems o descripciones",
                placeholder="Pega textos de preguntas o links de interés...",
                height=300,
                key="contexto_inspirado"
            )
            
    elif modo_creacion == "🪞 Ítem Espejo":
        st.header("1. Cargar Ítem Base")
        base_input_type = st.radio("Entrada base", ["🖼️ Imagen", "📝 Texto"], horizontal=True)
        
        if base_input_type == "🖼️ Imagen":
            imagen_subida = st.file_uploader(
                "Sube el pantallazo de la pregunta", 
                type=["png", "jpg", "jpeg"]
            )
            if imagen_subida:
                st.image(imagen_subida, caption="Ítem cargado", use_container_width=True)
        else:
            contexto_adicional = st.text_area(
                "Pega el texto de la pregunta original",
                placeholder="Enunciado y opciones...",
                height=250,
                key="contexto_espejo"
            )
            
        st.divider()
        st.subheader("Nivel de Similitud")
        similitud_sel = st.select_slider(
            "Grado de parecido al original",
            options=["Baja", "Media", "Alta"],
            value="Alta"
        )
        if similitud_sel == "Alta":
            st.caption("✅ Solo cambian datos y nombres. Estructura idéntica.")
        elif similitud_sel == "Media":
            st.caption("🔄 Cambios contextuales y de representación de datos.")
        else:
            st.caption("🧬 Misma esencia evaluativa, pero apariencia y orden distintos.")

# --- 5. LÓGICA DEL BOTÓN (Bucle Generador-Auditor) ---
st.divider()
if st.button(f"🚀 Generar {modo_creacion.split()[-1]} (con Auditoría)", use_container_width=True, type="primary"):
    
    # Validaciones según modo
    input_valido = False
    inputs_llm = {"imagenes": [], "texto": ""}
    
    if modo_creacion == "✨ Nuevo Ítem":
        if contexto_adicional:
            input_valido = True
            inputs_llm["texto"] = contexto_adicional
        else:
            st.warning("Escribe un concepto o idea para el ítem.")
            
    elif modo_creacion == "🎨 Ítem Inspirado":
        if metodo_inspiracion == "🖼️ Imágenes" and imagenes_inspiracion:
            input_valido = True
            inputs_llm["imagenes"] = imagenes_inspiracion
        elif metodo_inspiracion == "📝 Texto/Links" and contexto_adicional:
            input_valido = True
            inputs_llm["texto"] = contexto_adicional
        else:
            st.warning("Carga al menos una fuente de inspiración.")
            
    elif modo_creacion == "🪞 Ítem Espejo":
        if base_input_type == "🖼️ Imagen" and imagen_subida:
            input_valido = True
            inputs_llm["imagenes"] = [imagen_subida]
        elif base_input_type == "📝 Texto" and contexto_adicional:
            input_valido = True
            inputs_llm["texto"] = contexto_adicional
        else:
            st.warning("Carga un ítem base (imagen o texto).")

    if input_valido:
        # Verificación de taxonomía usando session_state
        tax_keys = {
            "Grado": "tax_grado", "Área": "tax_area", 
            "Competencia": "tax_competen", "Afirmación": "tax_afirm", 
            "Evidencia": "tax_evid", "Ref. Temática": "tax_ref",
            "Componente_Estructura": "tax_comp1", "Componente_Tematica": "tax_comp2"
        }
        faltantes_tax = [label for label, key in tax_keys.items() if not st.session_state.get(key)]

        if data is None:
            st.warning("El archivo Excel de taxonomía no se pudo cargar.")
        elif faltantes_tax:
            st.error(f"⚠️ **Taxonomía incompleta:** Faltan {', '.join(faltantes_tax)}")
            st.info("Por favor, completa todas las selecciones en la Columna 2.")
        else:
            taxonomia_seleccionada = {label: st.session_state.get(key) for label, key in tax_keys.items()}
            st.session_state['taxonomia_actual'] = taxonomia_seleccionada
            
            max_intentos = 3
            intento_actual = 0
            feedback_auditor = ""
            item_final_json = None

            with st.status("Procesando...", expanded=True) as status:
                while intento_actual < max_intentos:
                    intento_actual += 1

                    status.update(label=f"Intento {intento_actual}/{max_intentos}: Generando ítem...")
                    item_json_str = generar_item_llm(
                        modo_creacion,
                        inputs_llm,
                        taxonomia_seleccionada,
                        modelo_generador_sel,
                        similitud_sel if modo_creacion == "🪞 Ítem Espejo" else "Media",
                        feedback_auditor
                    )

                    if item_json_str is None:
                        status.update(label=f"⚠️ Error en generación (Intento {intento_actual}). Reintentando...", state="running")
                        continue

                    status.update(label=f"Intento {intento_actual}/{max_intentos}: Auditando ítem con {modelo_auditor_sel}...")
                    audit_json_str = auditar_item_llm(item_json_str, taxonomia_seleccionada, modelo_auditor_sel)

                    if audit_json_str is None:
                        status.update(label=f"⚠️ Error en auditoría (Intento {intento_actual}). Reintentando...", state="running")
                        continue

                    try:
                        audit_data = json.loads(audit_json_str)
                        dictamen = audit_data.get("dictamen_final", "")

                        if dictamen in ("✅ CUMPLE", "⚠️ CUMPLE CON OBSERVACIONES"):
                            estado_label = "¡Auditoría Aprobada!" if dictamen == "✅ CUMPLE" else "Ítem aprobado con observaciones menores."
                            status.update(label=estado_label, state="complete")
                            item_final_json = item_json_str

                            # Mostrar observaciones si las hay
                            if dictamen == "⚠️ CUMPLE CON OBSERVACIONES":
                                observaciones = audit_data.get("correcciones", [])
                                if observaciones:
                                    with st.expander("⚠️ Observaciones del auditor (el ítem es usable, pero considera estos puntos)"):
                                        for obs in observaciones:
                                            st.warning(f"**{obs.get('criterio')}** [{obs.get('prioridad', '')}]: {obs.get('accion', '')}")
                            break

                        else:  # ❌ RECHAZADO
                            # Construir feedback estructurado desde la lista de correcciones
                            correcciones = audit_data.get("correcciones", [])
                            if correcciones:
                                partes_feedback = [
                                    f"- [{c.get('prioridad', 'ALTA')}] {c.get('criterio')}: {c.get('accion', '')}"
                                    for c in correcciones
                                ]
                                feedback_auditor = "Correcciones requeridas:\n" + "\n".join(partes_feedback)
                            else:
                                feedback_auditor = audit_data.get("observaciones_finales", "Rechazado sin observaciones.")

                            status.update(label=f"Intento {intento_actual} Rechazado. Preparando re-intento...")
                            st.expander(f"Detalles del Rechazo (Intento {intento_actual})").json(audit_data)

                    except json.JSONDecodeError:
                        st.error(f"Error al leer respuesta JSON del auditor: {audit_json_str}")
                        feedback_auditor = "La respuesta del auditor no fue un JSON válido."

            if item_final_json is None:
                status.update(label=f"No se pudo generar un ítem de alta calidad después de {max_intentos} intentos.", state="error")
                st.error(f"Último feedback del auditor: {feedback_auditor}")
            
        if item_final_json:
            st.success("¡Ítem generado y auditado con éxito! Puedes editarlo abajo.")
            cargar_item_en_editor(item_final_json)



# --- 6. EDITOR DE ÍTEMS Y DESCARGA (LÓGICA DE BOTONES SEPARADA) ---
if 'show_editor' in st.session_state and st.session_state.show_editor:
    st.divider()
    st.header("3. Edita el Ítem Generado")
    
    # --- ENUNCIADO Y GRÁFICO DEL ENUNCIADO ---
    st.subheader("Enunciado")
    st.text_area("Texto del Enunciado", key="editable_pregunta", height=150)
    st.selectbox(
        "¿Enunciado necesita un gráfico/tabla?", 
        options=["NO", "SÍ"], 
        key="editable_grafico_nec_enunciado"
    )
    
    if st.session_state.editable_grafico_nec_enunciado == "SÍ":
        # --- AUTO-GENERACIÓN DEL PROMPT SI NO EXISTE AUN ---
        if not st.session_state.get("editable_grafico_texto_enunciado", "").strip() or \
           st.session_state.get("editable_grafico_texto_enunciado") in ["", "N/A", None]:
            opciones_txt = " | ".join([
                f"{l}: {st.session_state.get(f'editable_opcion_{l.lower()}_texto', '')}"
                for l in ["A", "B", "C", "D"]
            ])
            modo_visual = st.session_state.get("modo_visual_preferido", "logico")
            if modo_visual == "logico":
                with st.spinner("🧠 Generando especificación completa del gráfico..."):
                    prompt_gen, razon = generar_descripcion_grafico_logico_llm(
                        pregunta=st.session_state.get("editable_pregunta", ""),
                        opciones_texto=opciones_txt,
                        seccion="el enunciado de la pregunta",
                        taxonomia_dict=st.session_state.get('taxonomia_actual', {}),
                        model_name=modelo_generador_sel
                    )
            else:
                with st.spinner("🧠 Generando prompt optimizado para Nano Banana..."):
                    prompt_gen, razon = generar_prompt_imagen_llm(
                        pregunta=st.session_state.get("editable_pregunta", ""),
                        opciones_texto=opciones_txt,
                        seccion="el enunciado de la pregunta",
                        taxonomia_dict=st.session_state.get('taxonomia_actual', {}),
                        model_name=modelo_generador_sel
                    )
            if prompt_gen:
                st.session_state.editable_grafico_texto_enunciado = prompt_gen
                st.session_state["prompt_razon_enunciado"] = razon
        
        if st.session_state.get("prompt_razon_enunciado"):
            st.info(f"💡 **Racional del visual:** {st.session_state.prompt_razon_enunciado}")
        
        st.text_area(
            "📸 Prompt de Imagen (en inglés, optimizado para IA)", 
            key="editable_grafico_texto_enunciado", 
            height=100
        )
        
        # --- LÓGICA HÍBRIDA: LÓGICO VS CREATIVO ---
        if st.session_state.get("modo_visual_preferido") == "creativo":
            st.text_input("💡 Sugerencias de refinamiento", key="refine_feedback_enunciado", placeholder="Ej: 'Hazlo más minimalista' o 'Cambia el color de fondo'")
            if st.button("🎨 Generar/Refinar Imagen con Nano Banana (Enunciado) 🍌", key="btn_nano_enunciado"):
                if GRAFICOS_DISPONIBLES:
                    with st.spinner("Nano Banana está imaginando tu escena..."):
                        texto_desc = st.session_state.editable_grafico_texto_enunciado
                        feedback = st.session_state.get("refine_feedback_enunciado", "")
                        prompt_completo = f"{texto_desc}. Refinamiento: {feedback}" if feedback else texto_desc
                        buffer_imagen = generar_imagen_artistica(prompt_completo)
                        if buffer_imagen:
                            st.session_state['img_buffer_enunciado'] = buffer_imagen
                            st.success("¡Imagen generada con éxito!")
                        else:
                            st.error("No se pudo generar.")
                else:
                    st.warning("El módulo de gráficos no está disponible.")
        else:
            # --- Botones Modo Lógico ---
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Generar JSON desde Texto 🤖", key="btn_gen_json_enunciado"):
                    if GRAFICOS_DISPONIBLES:
                        with st.spinner("Llamando a IA para generar JSON..."):
                            texto_desc = st.session_state.editable_grafico_texto_enunciado
                            spec = build_visual_json_with_llm(texto_desc)
                            if spec:
                                st.session_state.editable_grafico_json_enunciado = json.dumps([spec], indent=2)
                                st.session_state['img_buffer_enunciado'] = None
                                st.success("¡JSON generado!")
                    else:
                        st.warning("Módulo no disponible.")
            with col_b:
                if st.button("Renderizar desde JSON 🖼️", key="btn_render_enunciado"):
                    if GRAFICOS_DISPONIBLES:
                        try:
                            json_data = json.loads(st.session_state.editable_grafico_json_enunciado)
                            if json_data and isinstance(json_data, list):
                                spec = json_data[0]
                                buffer_imagen = crear_grafico(
                                    tipo_grafico=spec.get("tipo_elemento"),
                                    datos=spec.get("datos", {}),
                                    configuracion=spec.get("configuracion", {})
                                )
                                if buffer_imagen:
                                    st.session_state['img_buffer_enunciado'] = buffer_imagen
                                    st.success("Gráfico renderizado.")
                        except Exception as e:
                            st.error(f"Error: {e}")

            st.text_area(
                "Datos del Gráfico (JSON) - (Editable)", 
                key="editable_grafico_json_enunciado", 
                height=150
            )
        
        # Mostramos la imagen si existe en el estado
        if 'img_buffer_enunciado' in st.session_state and st.session_state.img_buffer_enunciado:
            st.image(st.session_state.img_buffer_enunciado, caption="Previsualización generada")


    # --- OPCIONES Y SUS GRÁFICOS ---
    st.subheader("Opciones")
    
    for letra in ["A", "B", "C", "D"]:
        st.markdown(f"--- \n**Opción {letra}**")
        st.text_input(f"Texto Opción {letra}", key=f"editable_opcion_{letra.lower()}_texto")
        st.selectbox(
            f"¿Gráfico en Opción {letra}?", 
            options=["NO", "SÍ"], 
            key=f"editable_opcion_{letra.lower()}_grafico_nec"
        )
        
        if st.session_state[f"editable_opcion_{letra.lower()}_grafico_nec"] == "SÍ":
            # --- AUTO-GENERACIÓN DEL PROMPT SI NO EXISTE AUN ---
            texto_op_key = f"editable_opcion_{letra.lower()}_grafico_texto"
            if not st.session_state.get(texto_op_key, "").strip() or \
               st.session_state.get(texto_op_key) in ["", "N/A", None]:
                opciones_txt = " | ".join([
                    f"{l}: {st.session_state.get(f'editable_opcion_{l.lower()}_texto', '')}".strip()
                    for l in ["A", "B", "C", "D"]
                ])
                modo_visual = st.session_state.get("modo_visual_preferido", "logico")
                if modo_visual == "logico":
                    with st.spinner(f"🧠 Generando especificación completa del gráfico (Opción {letra})..."):
                        prompt_gen, razon = generar_descripcion_grafico_logico_llm(
                            pregunta=st.session_state.get("editable_pregunta", ""),
                            opciones_texto=opciones_txt,
                            seccion=f"la Opción {letra} de la pregunta",
                            taxonomia_dict=st.session_state.get('taxonomia_actual', {}),
                            model_name=modelo_generador_sel
                        )
                else:
                    with st.spinner(f"🧠 Generando prompt Nano Banana (Opción {letra})..."):
                        prompt_gen, razon = generar_prompt_imagen_llm(
                            pregunta=st.session_state.get("editable_pregunta", ""),
                            opciones_texto=opciones_txt,
                            seccion=f"la Opción {letra} de la pregunta",
                            taxonomia_dict=st.session_state.get('taxonomia_actual', {}),
                            model_name=modelo_generador_sel
                        )
                if prompt_gen:
                    st.session_state[texto_op_key] = prompt_gen
                    st.session_state[f"prompt_razon_op_{letra}"] = razon

            if st.session_state.get(f"prompt_razon_op_{letra}"):
                st.info(f"💡 **Racional del visual ({letra}):** {st.session_state[f'prompt_razon_op_{letra}']}")

            st.text_area(
                f"📸 Prompt de Imagen Opción {letra} (inglés, optimizado para IA)", 
                key=f"editable_opcion_{letra.lower()}_grafico_texto", 
                height=100
            )
            
            # --- LÓGICA HÍBRIDA: LÓGICO VS CREATIVO PARA OPCIONES ---
            if st.session_state.get("modo_visual_preferido") == "creativo":
                st.text_input(f"💡 Sugerencias de refinamiento ({letra})", key=f"refine_feedback_op_{letra}", placeholder="Ej: 'Más contraste'")
                if st.button(f"🎨 Generar/Refinar Imagen con Nano Banana (Opción {letra}) 🍌", key=f"btn_nano_op_{letra}"):
                    if GRAFICOS_DISPONIBLES:
                        with st.spinner(f"Nano Banana imaginando Opción {letra}..."):
                            texto_desc = st.session_state[f"editable_opcion_{letra.lower()}_grafico_texto"]
                            feedback = st.session_state.get(f"refine_feedback_op_{letra}", "")
                            prompt_completo = f"{texto_desc}. Refinamiento: {feedback}" if feedback else texto_desc
                            buffer_imagen = generar_imagen_artistica(prompt_completo)
                            if buffer_imagen:
                                st.session_state[f'img_buffer_op_{letra}'] = buffer_imagen
                                st.success(f"¡Imagen Opción {letra} generada!")
                            else:
                                st.error(f"Error en Nano Banana para Opción {letra}.")
            else:
                # --- Botones Modo Lógico ---
                col_c, col_d = st.columns(2)
                with col_c:
                    if st.button(f"Generar JSON 🤖", key=f"btn_gen_json_op_{letra}"):
                        if GRAFICOS_DISPONIBLES:
                            with st.spinner(f"Generando JSON para Opción {letra}..."):
                                texto_desc = st.session_state[f"editable_opcion_{letra.lower()}_grafico_texto"]
                                spec = build_visual_json_with_llm(texto_desc)
                                if spec:
                                    st.session_state[f"editable_opcion_{letra.lower()}_grafico_json"] = json.dumps([spec], indent=2)
                                    st.session_state[f'img_buffer_op_{letra}'] = None
                                    st.success(f"¡JSON Opción {letra} generado!")
                with col_d:
                    if st.button(f"Renderizar 🖼️", key=f"btn_render_op_{letra}"):
                        if GRAFICOS_DISPONIBLES:
                            try:
                                json_data = json.loads(st.session_state[f"editable_opcion_{letra.lower()}_grafico_json"])
                                if json_data and isinstance(json_data, list):
                                    spec = json_data[0]
                                    buffer_imagen = crear_grafico(
                                        tipo_grafico=spec.get("tipo_elemento"),
                                        datos=spec.get("datos", {}),
                                        configuracion=spec.get("configuracion", {})
                                    )
                                    if buffer_imagen:
                                        st.session_state[f'img_buffer_op_{letra}'] = buffer_imagen
                                        st.success(f"Opción {letra} renderizada.")
                            except Exception as e:
                                st.error(f"Error: {e}")

                st.text_area(
                    f"Datos Gráfico Opción {letra} (JSON) - (Editable)", 
                    key=f"editable_opcion_{letra.lower()}_grafico_json", 
                    height=150
                )

            # Mostramos la imagen si existe en el estado
            if f'img_buffer_op_{letra}' in st.session_state and st.session_state[f'img_buffer_op_{letra}']:
                st.image(st.session_state[f'img_buffer_op_{letra}'], caption=f"Previsualización Opción {letra}")

    st.subheader("Clave")
    st.text_input("Clave (Respuesta Correcta)", key="editable_clave")

    st.subheader("Justificaciones")
    st.text_area("Justificación Clave", key="editable_just_clave", height=100)
    st.text_area("Justificación A", key="editable_just_a", height=100)
    st.text_area("Justificación B", key="editable_just_b", height=100)
    st.text_area("Justificación C", key="editable_just_c", height=100)
    st.text_area("Justificación D", key="editable_just_d", height=100)

    st.subheader("🔄 Ajustes Mayores")
    feedback_mejora = st.text_area("Retroalimentación de Ajustes Generales", key="ajustes_mayores_it", height=120, placeholder="Escribe aquí si el ítem requiere un cambio profundo de concepto o estructura...")

    if st.button("🚀 Mejorar con feedback", use_container_width=True):
        if not feedback_mejora.strip():
            st.warning("Por favor, escribe el feedback para mejorar el ítem.")
        else:
            with st.spinner("Refinando ítem con tu feedback..."):
                # Re-ensamblar JSON actual para enviarlo al LLM
                item_actual = {
                    "pregunta_espejo": st.session_state.editable_pregunta,
                    "clave": st.session_state.editable_clave,
                    "justificacion_clave": st.session_state.editable_just_clave,
                    "grafico_necesario_enunciado": st.session_state.editable_grafico_nec_enunciado,
                    "descripcion_texto_grafico_enunciado": st.session_state.editable_grafico_texto_enunciado,
                    "opciones": {
                        letra: {
                            "texto": st.session_state[f"editable_opcion_{letra.lower()}_texto"],
                            "grafico_necesario": st.session_state[f"editable_opcion_{letra.lower()}_grafico_nec"],
                            "descripcion_texto_grafico": st.session_state[f"editable_opcion_{letra.lower()}_grafico_texto"]
                        } for letra in ["A", "B", "C", "D"]
                    },
                    "justificaciones_distractores": [
                        {"opcion": "A", "justificacion": st.session_state.get("editable_just_a")},
                        {"opcion": "B", "justificacion": st.session_state.get("editable_just_b")},
                        {"opcion": "C", "justificacion": st.session_state.get("editable_just_c")},
                        {"opcion": "D", "justificacion": st.session_state.get("editable_just_d")},
                    ]
                }
                
                nuevo_json = refinar_item_llm(
                    json.dumps(item_actual, ensure_ascii=False),
                    feedback_mejora,
                    st.session_state['taxonomia_actual'],
                    modelo_generador_sel
                )
                
                if nuevo_json:
                    cargar_item_en_editor(nuevo_json)
                    st.success("¡Ítem refinado! Los cambios se han cargado en el editor.")
                    st.rerun()

    # --- SECCIÓN DE DESCARGA (SINCRONIZADA DINÁMICAMENTE) ---
    st.divider()
    st.header("4. Descargar Resultados")
    
    # --- 4.1 RE-ENSAMBLE DINÁMICO DE TAXONOMÍA ---
    # Capturamos los valores actuales de los selectores (Columna 2)
    taxonomia_actual = {
        "Grado": st.session_state.get("tax_grado"),
        "Área": st.session_state.get("tax_area"),
        "Componente_Estructura": st.session_state.get("tax_comp1"),
        "Componente_Tematica": st.session_state.get("tax_comp2"),
        "Ref. Temática": st.session_state.get("tax_ref"),
        "Competencia": st.session_state.get("tax_competen"),
        "Afirmación": st.session_state.get("tax_afirm"),
        "Evidencia": st.session_state.get("tax_evid")
    }
    st.session_state['taxonomia_actual'] = taxonomia_actual

    # --- 4.2 RE-ENSAMBLE DINÁMICO DEL ÍTEM EDITADO ---
    datos_editados = {
        "pregunta_espejo": st.session_state.get("editable_pregunta", ""),
        "clave": st.session_state.get("editable_clave", ""),
        "justificacion_clave": st.session_state.get("editable_just_clave", ""),
        "grafico_necesario_enunciado": st.session_state.get("editable_grafico_nec_enunciado", "NO"),
        "opciones": {},
        "justificaciones_distractores": [
            {"opcion": "A", "justificacion": st.session_state.get("editable_just_a", "N/A")},
            {"opcion": "B", "justificacion": st.session_state.get("editable_just_b", "N/A")},
            {"opcion": "C", "justificacion": st.session_state.get("editable_just_c", "N/A")},
            {"opcion": "D", "justificacion": st.session_state.get("editable_just_d", "N/A")},
        ]
    }
    
    # Gráficos Enunciado (texto natural + JSON + buffer)
    datos_editados["texto_grafico_enunciado"] = st.session_state.get("editable_grafico_texto_enunciado", "")
    try:
        json_str_en = st.session_state.get("editable_grafico_json_enunciado", "[]")
        datos_editados["descripcion_grafico_enunciado"] = json.loads(json_str_en)
    except Exception:
        datos_editados["descripcion_grafico_enunciado"] = st.session_state.get("editable_grafico_json_enunciado", "[]")
    
    # Gráficos Enunciado (Buffer)
    datos_editados["img_buffer_enunciado"] = st.session_state.get('img_buffer_enunciado')
    
    # Gráficos Opciones (Buffer)
    for letra in ["A", "B", "C", "D"]:
        op_data = {
            "texto": st.session_state.get(f"editable_opcion_{letra.lower()}_texto", ""),
            "grafico_necesario": st.session_state.get(f"editable_opcion_{letra.lower()}_grafico_nec", "NO"),
            "img_buffer": st.session_state.get(f'img_buffer_op_{letra}')
        }
        try:
            json_str_op = st.session_state.get(f"editable_opcion_{letra.lower()}_grafico_json", "[]")
            op_data["descripcion_grafico"] = json.loads(json_str_op)
        except Exception:
            op_data["descripcion_grafico"] = st.session_state.get(f"editable_opcion_{letra.lower()}_grafico_json", "[]")
        op_data["texto_grafico"] = st.session_state.get(f"editable_opcion_{letra.lower()}_grafico_texto", "")
        
        datos_editados["opciones"][letra] = op_data

    # --- 4.3 GENERAR OPORTUNIDAD DE MEJORA (con caché por sesión) ---
    # Solo se regenera si cambia la evidencia, competencia o justificación de la clave.
    _oport_key = (
        taxonomia_actual.get("Evidencia", ""),
        taxonomia_actual.get("Competencia", ""),
        datos_editados.get("justificacion_clave", "")[:80]
    )
    if st.session_state.get("_oport_cache_key") != _oport_key:
        with st.spinner("Generando oportunidad de mejora..."):
            oportunidad_mejora = generar_oportunidad_mejora_llm(
                taxonomia_actual,
                datos_editados.get("justificacion_clave", ""),
                modelo_auditor_sel
            )
        st.session_state["_oport_mejora"] = oportunidad_mejora
        st.session_state["_oport_cache_key"] = _oport_key
    else:
        oportunidad_mejora = st.session_state["_oport_mejora"]

    col_word, col_excel = st.columns(2)
    
    with col_word:
        # 2. Pasar la oportunidad_mejora a la función de Word
        archivo_word = crear_word(datos_editados, taxonomia_actual, oportunidad_mejora)
        st.download_button(
            label="Descargar en Word (.docx)",
            data=archivo_word,
            file_name="item_espejo_auditado.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        
    with col_excel:
        # 3. Pasar los datos a la nueva función de Excel
        archivo_excel = crear_excel(
            datos_editados, 
            taxonomia_actual, 
            oportunidad_mejora
        )
        st.download_button(
            label="Descargar en Excel (.xlsx)",
            data=archivo_excel,
            file_name="item_espejo_auditado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
