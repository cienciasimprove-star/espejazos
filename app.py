import streamlit as st
import io
import base64
import pandas as pd
from docx import Document
from docx.shared import Inches
import PyPDF2
# Importa la librería de Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel, Image as VertexImage
import json
import re
import random # Necesario para la clave aleatoria
import openai
import anthropic
import google.generativeai as genai
from google.cloud import storage
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()
# --- VALIDACIÓN DE CONTRASEÑA Y SELECCIÓN DE MODO ---
def check_password():
    """Muestra selector de modo y valida contraseña. Retorna True si autenticado."""
    PASSWORDS = {
        "privados": os.environ.get("APP_PASSWORD_PRIVADOS", "Armado#pr1vad0s$2026"),
        "simulacros": os.environ.get("APP_PASSWORD_SIMULACROS", "Armado#S1mulacros$2026"),
    }

    if st.session_state.get("password_correct", False):
        return True

    st.title("🤖 Suite de Creación Psicométrica")
    st.markdown("---")
    modo = st.radio(
        "Selecciona el módulo:",
        options=["privados", "simulacros"],
        format_func=lambda x: "🏫 Colegios Privados" if x == "privados" else "📝 Super Simulacros",
        key="_modo_seleccionado_radio",
        horizontal=True,
    )

    def password_entered():
        pwd = st.session_state.get("password", "")
        modo_sel = st.session_state.get("_modo_seleccionado_radio", "privados")
        if pwd == PASSWORDS[modo_sel]:
            st.session_state["password_correct"] = True
            st.session_state["modo_producto"] = modo_sel
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    st.warning("⚠️ Esta aplicación es de uso exclusivo.")
    st.text_input(
        "Clave de acceso:", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Contraseña incorrecta")
    return False

if not check_password():
    st.stop()

# --- IMPORTACIÓN CLAVE ---
# Importamos las TRES funciones que necesitamos
try:
    from graficos_plugins import (
        crear_grafico,
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
    def build_visual_json_with_llm(*args, **kwargs):
        return None
    def generar_imagen_artistica(*args, **kwargs):
        return None
# --- Configuración de Google Cloud (hacer al inicio) ---
# Usa variables de entorno para el proyecto y la región, con valores por defecto
GCP_PROJECT = os.environ.get("GCP_PROJECT", "espejazos")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global") # <--- Cambiado a 'global' para soporte Gemini 3
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

# --- ENRUTADOR MULTI-MODELO (Vertex, Gemini API, OpenAI) ---
def call_llm_router(partes_texto, imagenes, model_name, json_mode=False):
    """Router universal para enviar peticiones."""
    proveedor = st.session_state.get("proveedor_llm", "🏢 Vertex AI (Nativo)")
    prompt_completo = "\n\n".join(partes_texto)
    
    if proveedor == "🏢 Vertex AI (Nativo)":
        model = GenerativeModel(model_name)
        partes = list(partes_texto)
        for img in imagenes:
            img_bytes = img.getvalue() if hasattr(img, 'getvalue') else img
            partes.append(VertexImage.from_bytes(img_bytes))
        config = {"response_mime_type": "application/json"} if json_mode else {}
        response = model.generate_content(partes, generation_config=config)
        return response.text
        
    elif proveedor == "🔑 Google Gemini (API Key)":
        api_key = (st.session_state.get("api_key_gemini", "") or st.session_state.get("_default_gemini_loaded", "")).strip()
        if not api_key: return '{"error": "Falta la API Key de Gemini"}'
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        partes = list(partes_texto)
        if imagenes:
            from PIL import Image as PILImage
            for img in imagenes:
                img_bytes = img.getvalue() if hasattr(img, 'getvalue') else img
                partes.append(PILImage.open(io.BytesIO(img_bytes)))
        config = genai.types.GenerationConfig(response_mime_type="application/json") if json_mode else genai.types.GenerationConfig()
        response = model.generate_content(partes, generation_config=config)
        return response.text
        
    elif proveedor == "🔑 OpenAI (API Key)":
        api_key = st.session_state.get("api_key_openai", "").strip()
        if not api_key: return '{"error": "Falta la API Key de OpenAI"}'
        client = openai.OpenAI(api_key=api_key)
        
        content = [{"type": "text", "text": prompt_completo}]
        for img in imagenes:
            img_bytes = img.getvalue() if hasattr(img, 'getvalue') else img
            b64_img = base64.b64encode(img_bytes).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
            
        messages = [{"role": "user", "content": content}]
        kwargs = {"model": model_name, "messages": messages}
        # Para OpenAI, o1-mini o o1-preview no soportan json_object de forma nativa en todo momento, omitiremos si es o1.
        if json_mode and "o1" not in model_name:
            kwargs["response_format"] = {"type": "json_object"}
            
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
        
    elif proveedor == "🤖 Anthropic Claude (Nativo)":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key: return '{"error": "ANTHROPIC_API_KEY no configurada en el servidor"}'
        client = anthropic.Anthropic(api_key=api_key)

        content = []
        for img in imagenes:
            img_bytes = img.getvalue() if hasattr(img, 'getvalue') else img
            b64_img = base64.b64encode(img_bytes).decode('utf-8')
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_img}
            })
        content.append({"type": "text", "text": prompt_completo})
        response = client.messages.create(
            model=model_name,
            max_tokens=8000,
            messages=[{"role": "user", "content": content}]
        )
        return response.content[0].text

    elif proveedor == "🔑 Anthropic Claude (API Key)":
        api_key = (st.session_state.get("api_key_anthropic", "") or st.session_state.get("_default_claude_loaded", "")).strip()
        if not api_key: return '{"error": "Falta la API Key de Anthropic"}'
        client = anthropic.Anthropic(api_key=api_key)
        
        content = []
        for img in imagenes:
            img_bytes = img.getvalue() if hasattr(img, 'getvalue') else img
            b64_img = base64.b64encode(img_bytes).decode('utf-8')
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64_img,
                }
            })
        content.append({"type": "text", "text": prompt_completo})
            
        messages = [{"role": "user", "content": content}]
        
        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            messages=messages
        )
        return response.content[0].text
        
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

# --- FUNCIONES DE USUARIOS Y PROGRESO ---

@st.cache_data(ttl=3600)
def cargar_usuarios(modo_producto):
    """Carga el listado de usuarios desde GCS según el modo (privados/simulacros)."""
    try:
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "sumun-pruebas")
        storage_client = storage.Client(project=GCP_PROJECT)
        filename = "Usuarios_simulacros.xlsx" if modo_producto == "simulacros" else "Usuarios.xlsx"
        blob = storage_client.bucket(bucket_name).blob(filename)
        data = blob.download_as_bytes()
        return pd.read_excel(io.BytesIO(data))
    except Exception as e:
        return pd.DataFrame(columns=["Área", "Usuario"])

def guardar_progreso_gcs(area, usuario, item_json_str, taxonomia, modo_creacion):
    """Guarda el ítem generado en GCS bajo progreso/{modo}/{area}/{usuario}/{timestamp}.json"""
    try:
        from datetime import datetime
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "sumun-pruebas")
        storage_client = storage.Client(project=GCP_PROJECT)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modo_producto = st.session_state.get("modo_producto", "privados")
        area_slug = area.replace(" ", "_").replace("/", "-")
        usuario_slug = usuario.replace(" ", "_").replace("/", "-")
        blob_name = f"progreso/{modo_producto}/{area_slug}/{usuario_slug}/{timestamp}.json"
        payload = {
            "timestamp": timestamp,
            "area": area,
            "usuario": usuario,
            "modo_creacion": modo_creacion,
            "taxonomia": taxonomia,
            "item_json": item_json_str,
            "contexto_texto": st.session_state.get("contexto_compartido_texto", "")
        }
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_string(json.dumps(payload, ensure_ascii=False), content_type="application/json")
        return blob_name
    except Exception as e:
        return None

def listar_progreso_gcs(area, usuario):
    """Lista los ítems guardados de un usuario ordenados por fecha desc."""
    try:
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "sumun-pruebas")
        storage_client = storage.Client(project=GCP_PROJECT)
        modo_producto = st.session_state.get("modo_producto", "privados")
        area_slug = area.replace(" ", "_").replace("/", "-")
        usuario_slug = usuario.replace(" ", "_").replace("/", "-")
        prefix = f"progreso/{modo_producto}/{area_slug}/{usuario_slug}/"
        blobs = list(storage_client.bucket(bucket_name).list_blobs(prefix=prefix))
        items = []
        for blob in sorted(blobs, key=lambda b: b.name, reverse=True):
            try:
                data = json.loads(blob.download_as_bytes())
                item_obj = json.loads(data.get("item_json", "{}")) if isinstance(data.get("item_json"), str) else {}
                preview = item_obj.get("pregunta_espejo", "")[:80] or "Sin título"
                items.append({
                    "blob_name": blob.name,
                    "timestamp": data.get("timestamp", ""),
                    "modo": data.get("modo_creacion", ""),
                    "preview": preview,
                    "payload": data
                })
            except Exception:
                continue
        return items
    except Exception as e:
        return []

def cargar_item_desde_progreso(payload):
    """Restaura un ítem guardado al editor y la taxonomía al session_state."""
    try:
        item_json_str = payload.get("item_json", "{}")
        taxonomia = payload.get("taxonomia", {})
        if taxonomia:
            st.session_state["taxonomia_seleccionada"] = taxonomia
        contexto = payload.get("contexto_texto", "")
        if contexto:
            st.session_state["contexto_compartido_texto"] = contexto
        cargar_item_en_editor(item_json_str)
        return True
    except Exception:
        return False

# --- SELECCIÓN DE USUARIO ---
def seleccionar_usuario():
    if st.session_state.get("_usuario_confirmado"):
        return True
    modo_producto = st.session_state.get("modo_producto", "privados")
    titulo = "🏫 Colegios Privados — Suite de Creación Psicométrica" if modo_producto == "privados" else "📝 Super Simulacros — Suite de Creación Psicométrica"
    st.title(titulo)
    st.subheader("Identifícate para continuar")
    df_usuarios = cargar_usuarios(modo_producto)
    areas = sorted(df_usuarios["Área"].dropna().unique().tolist()) if not df_usuarios.empty else []
    area_sel = st.selectbox("Área", ["— Selecciona —"] + areas, key="_login_area")
    if area_sel != "— Selecciona —":
        usuarios_area = df_usuarios[df_usuarios["Área"] == area_sel]["Usuario"].dropna().tolist()
        usuario_sel = st.selectbox("Usuario", ["— Selecciona —"] + usuarios_area, key="_login_usuario")
        if usuario_sel != "— Selecciona —":
            if st.button("Entrar", type="primary"):
                st.session_state["_usuario_confirmado"] = True
                st.session_state["_usuario_area"] = area_sel
                st.session_state["_usuario_nombre"] = usuario_sel
                st.rerun()
    return False

if not seleccionar_usuario():
    st.stop()

@st.cache_data(ttl=3600)
def cargar_base_improve():
    """Carga y cachea la base de datos Excel de Improve desde GCS."""
    try:
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "sumun-pruebas")
        blob_name = os.environ.get("BD_IMPROVE_BLOB", "202603041401_contextos_reindexados.xlsx")
        storage_client = storage.Client(project=GCP_PROJECT)
        blob = storage_client.bucket(bucket_name).blob(blob_name)
        data = blob.download_as_bytes()
        return pd.read_excel(io.BytesIO(data))
    except Exception as e:
        print(f"Error cargando BD Improve desde GCS: {e}")
        return None

def analizar_adn_llm(inputs, model_name):
    """
    Analiza una o varias fuentes (imágenes/texto) para extraer el ADN psicométrico.
    """
    partes_texto = ["Eres un experto psicómetra. Tu tarea es analizar los siguientes ítems para extraer su ADN estructural."]
    if inputs.get("texto"):
        partes_texto.append(f"Texto de referencia: {inputs['texto']}")
        
    partes_texto.append("""
    Analiza y devuelve en formato JSON:
    1. ESTRUCTURAS: Patrones comunes en el enunciado y opciones (ej. negaciones, tablas, comparaciones).
    2. QUÉ EVALÚA: Competencia profunda y evidencia detectada.
    3. PROCESO COGNITIVO: Pasos mentales (identificar, relacionar, inferir) para resolverlo.
    4. ESQUELETO SINTÁCTICO: Estructura lógica de las oraciones.
    
    Responde solo con el JSON.
    """)
    
    imagenes = inputs.get("imagenes", [])
    
    try:
        response_text = call_llm_router(partes_texto, imagenes, model_name, json_mode=True)
        return response_text
    except Exception as e:
        st.error(f"Error analizando ADN: {e}")
        return None

def generar_ideas_llm(taxonomia_dict, model_name):
    """
    Genera 3 ideas creativas basadas en la taxonomía y las devuelve en formato JSON.
    """
    tax_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])
    prompt = f"""
    Basado en esta taxonomía:
    {tax_texto}
    
    Propón 3 contextos creativos y situacionales para crear una pregunta de alto impacto.
    Responde ÚNICAMENTE con un objeto JSON que tenga una lista llamada "ideas" con 3 strings cortos y potentes.
    Ejemplo: {{"ideas": ["Idea 1...", "Idea 2...", "Idea 3..."]}}
    """
    try:
        response_text = call_llm_router([prompt], [], model_name, json_mode=True)
        return json.loads(response_text).get("ideas", [])
    except Exception as e:
        return [f"Error generando ideas: {e}"]

def generar_contexto_base_llm(taxonomia_dict, idea_usuario, model_name):
    """
    Genera un texto de contexto base para un bloque de preguntas.
    """
    tax_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])
    prompt = f"""
    Eres un experto redactor de evaluaciones educativas.
    
    Basado en los siguientes parámetros:
    {tax_texto}
    
    Y en esta idea o tema propuesto:
    "{idea_usuario}"
    
    Redacta un CONTEXTO BASE (una lectura, caso de estudio, situación o texto informativo) 
    que servirá como base para crear entre 2 y 7 preguntas de selección múltiple.
    El texto debe ser riguroso, interesante, adecuado para el grado especificado y contener 
    suficiente información para formular múltiples preguntas.
    
    Responde ÚNICAMENTE con el texto del contexto. No incluyas introducciones ni las preguntas.
    """
    try:
        response_text = call_llm_router([prompt], [], model_name, json_mode=False)
        return response_text.strip()
    except Exception as e:
        return f"Error generando contexto: {e}"

def generar_item_llm(modo, inputs, taxonomia_dict, model_name, similitud="Alta", feedback_auditor=""):
    """
    GENERADOR UNIVERSAL: Maneja Nuevo, Inspirado y Espejo.
    """
    taxonomia_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])
    clave_aleatoria = random.choice(['A', 'B', 'C', 'D'])

    imagenes = inputs.get("imagenes", [])
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
            "Baja": """REINVENCIÓN TOTAL — Lee con atención estas reglas, son OBLIGATORIAS e INNEGOCIABLES:
        1. ESCENARIO COMPLETAMENTE DIFERENTE: Si el original habla de ventas/sucursales, el nuevo debe tratar un tema distinto (ej. temperaturas, alturas, tiempos, puntajes, poblaciones). NUNCA uses el mismo dominio.
        2. ENTIDADES DIFERENTES: Si el original usa etiquetas P/Q/R/S o nombres de personas/ciudades/empresas, usa etiquetas y nombres completamente distintos.
        3. VALORES NUMÉRICOS COMPLETAMENTE DIFERENTES: Ningún valor numérico del original (ni en el enunciado, ni en las opciones, ni en el gráfico) puede aparecer en el nuevo ítem. Los valores de las opciones deben calcularse desde cero a partir del nuevo escenario inventado.
        4. REDACCIÓN DIFERENTE: No copies ni parafrasees frases del original. El enunciado debe estar redactado de forma completamente independiente.
        5. LO ÚNICO QUE SE CONSERVA: La habilidad cognitiva evaluada (ej. calcular rango estadístico) y el esqueleto lógico abstracto (ej. leer datos de una gráfica y aplicar una operación). Todo lo demás cambia."""
        }
        instrucciones_modo = f"""
        Eres un experto en evaluación educativa.
        Se te entrega el ítem original (en texto y/o imagen adjunta).
        **Contexto Adicional del Usuario (Tema sugerido para el nuevo ítem, si aplica):**
        {contexto_ref if contexto_ref else "No se especificó tema — inventa uno completamente diferente al original."}
        **Regla de Similitud — SIGUE ESTO AL PIE DE LA LETRA:**
        {filtro_similitud.get(similitud)}
        - CLAVE: La respuesta correcta DEBE ser la opción **{clave_aleatoria}**.
        - DISTRACTORES: Plausibles, basados en errores comunes de la Tarea Cognitiva. Deben tener la redacción "El estudiante podría escoger la opción XX porque... Sin embargo esto es incorrecto porque..."
        """
    elif modo == "🧩 Generación en Contexto (Bloques)":
        instrucciones_modo = f"""
        Estás en la modalidad de GENERACIÓN DE BLOQUES EN CONTEXTO.
        A continuación, se te entrega todo el material de lectura (CONTEXTO BASE) y la directriz para redactar la pregunta (INSTRUCCIÓN ESPECÍFICA).

        *** INICIO DEL MATERIAL DE TRABAJO ***
        {contexto_ref}
        *** FIN DEL MATERIAL DE TRABAJO ***

        Tu misión es redactar ÚNICAMENTE el Enunciado y las 4 Opciones de Respuesta para UNA sola pregunta que dependa enteramente del Contexto Base entregado arriba.
        
        --- REGLAS DE ORO ---
        1. **DEPENDE DEL CONTEXTO ESTRICTAMENTE:** La pregunta formulada DEBE derivarse de la lectura o caso entregado en el Contexto Base. Evalúa lo que dice la INSTRUCCIÓN ESPECÍFICA aplicándolo a esa lectura.
        2. **NO REPITAS EL CONTEXTO:** Asume que el estudiante ya tiene el Contexto Base impreso arriba de la pregunta. TU ENUNCIADO DEBE IR DIRECTO A LA INTERROGANTE (ej. "Teniendo en cuenta el caso anterior, ¿qué sucedería si...?"). NUNCA transcribas ni repitas el texto del contexto.
        3. **CLAVE:** La respuesta correcta DEBE ser la opción **{clave_aleatoria}**.
        4. **DISTRACTORES:** Deben basarse en errores plausibles o malas interpretaciones de la lectura específica. Redacción: "El estudiante podría escoger la opción XX porque... Sin embargo esto es incorrecto porque..."
        """

    prompt_final = f"""
    Eres un experto en psicometría educativa (estilo Saber 11).
    {instrucciones_modo}
    {seccion_feedback}

    **Taxonomía Requerida (Tu Guía):**
    {taxonomia_texto}

    --- REGLAS DE CONSTRUCCIÓN OBLIGATORIAS (APLICAN A TODOS LOS MODOS) ---
    Estas reglas son INNEGOCIABLES. Incumplir cualquiera causará rechazo automático.

    1. SIN CONTENIDO SENSIBLE: Prohíbido abordar temas de religión, política, sexualidad, género, etnicidad, violencia, ideologías o cualquier tema que pueda generar controversia o afectar la sensibilidad del estudiante.
    2. NEGACIONES CON FORMATO: Si el enunciado requiere una negación, escríbela obligatoriamente en MAYÚSCULA Y NEGRITA. Ejemplo correcto: "¿Cuál de las siguientes afirmaciones **NO** es correcta?". Nunca uses "no" en minúscula dentro de una pregunta si es la negación principal.
    3. SIN COMBINACIONES PROHIBIDAS EN OPCIONES: Las opciones NO pueden incluir enunciados tipo "Todas las anteriores", "Ninguna de las anteriores", "A y B son correctas", "Tanto A como C" ni variantes similares.
    4. OPCIONES SIN PISTAS FORMALES — La clave NO debe ser identificable por su forma. Verifica que:
       a) Todas las opciones tengan longitud similar (ninguna opción puede ser notoriamente más larga que las demás).
       b) Todas las opciones usen el mismo registro de lenguaje (todas técnicas o todas coloquiales, nunca mezclado).
       c) Todas las opciones tengan la misma estructura gramatical (si una empieza con verbo, todas deben empezar con verbo).
       d) Ninguna opción repita frases tomadas directamente del enunciado.
       e) Ninguna opción use adverbios absolutos como "siempre", "nunca", "jamás", "completamente", "todos", "ninguno".

    --- INSTRUCCIONES DE SALIDA PARA GRÁFICO (ENUNCIADO Y OPCIONES) ---
    ¡INSTRUCCIÓN CRÍTICA! Para los gráficos, NO debes generar el JSON.
    En su lugar, proporciona una descripción detallada en LENGUAJE NATURAL de lo que el gráfico debe mostrar.
    
    Si el elemento (enunciado u opción) NO necesita un gráfico, usa "NO" y "N/A".
    Si SÍ necesita un gráfico, usa "SÍ" y escribe la descripción.
    
    --- FORMATO DE SALIDA OBLIGATORIO (JSON VÁLIDO) ---
    Responde ÚNICAMENTE con el objeto JSON. No incluyas ```json.
    {{
      "pregunta_espejo": "Texto completo del enunciado/stem...",
      "clave": "{clave_aleatoria}",
      "justificacion_clave": "Razón por la que la clave es correcta...",
      
      "grafico_necesario_enunciado": "SÍ",
      "descripcion_texto_grafico_enunciado": "Una tabla simple. La primera fila es el encabezado con 'País' y 'Capital'.",
      
      "opciones": {{
        "A": {{
          "texto": "Ver gráfico A",
          "grafico_necesario": "SÍ",
          "descripcion_texto_grafico": "Un gráfico de barras verticales simple."
        }},
        "B": {{"texto": "B", "grafico_necesario": "NO", "descripcion_texto_grafico": "N/A"}},
        "C": {{"texto": "C", "grafico_necesario": "NO", "descripcion_texto_grafico": "N/A"}},
        "D": {{"texto": "D", "grafico_necesario": "NO", "descripcion_texto_grafico": "N/A"}}
      }},
      "justificaciones_distractores": [
        {{ "opcion": "A", "justificacion": "A..." }},
        {{ "opcion": "B", "justificacion": "B..." }},
        {{ "opcion": "C", "justificacion": "C..." }},
        {{ "opcion": "D", "justificacion": "D..." }}
      ]
    }}
    """

    try:
        response_text = call_llm_router([prompt_final], imagenes, model_name, json_mode=True)
        return limpiar_json_robustez(response_text)
    except Exception as e:
        st.error(f"Error en generación: {e}")
        return None


def refinar_item_llm(item_json_actual, feedback_usuario, taxonomia_dict, model_name):
    """
    REFINADOR: Toma un ítem existente y lo mejora basándose en feedback.
    """
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
        response_text = call_llm_router([prompt], [], model_name, json_mode=True)
        return limpiar_json_robustez(response_text)
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
        
        pregunta_generada = datos_obj.get("pregunta_espejo", "")
        # Purgar el contexto si el LLM lo repitió accidentalmente en el enunciado
        ctx = st.session_state.get("contexto_compartido_texto", "")
        if ctx and len(ctx) > 20 and ctx in pregunta_generada:
            pregunta_generada = pregunta_generada.replace(ctx, "").strip()
            
        st.session_state.editable_pregunta = pregunta_generada
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
    """
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
       - Si es geográfico, prefiere `mapa` (indicando posiciones x,y relativas o marcadores).
       - Si es un proceso biológico/científico estándar (ej. ciclo del agua), prefiere `infografia` e indica las palabras clave a reemplazar.
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
        response_text = call_llm_router([prompt], [], model_name, json_mode=True)
        data = json.loads(limpiar_json_robustez(response_text))
        return data.get("descripcion_completa", ""), data.get("razon", "")
    except Exception as e:
        print(f"Error generando descripción lógica del gráfico: {e}")
        return "", ""


# --- FUNCIÓN GENERADORA DE PROMPTS DE IMAGEN (NUEVA) ---
def generar_prompt_imagen_llm(pregunta, opciones_texto, seccion, taxonomia_dict, model_name):
    """
    Genera un prompt estructurado y optimizado para la creación de imágenes educativas.
    """
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
    6. El prompt debe estar en INGLÉS.
    
    --- FORMATO DE SALIDA (JSON ÚNICAMENTE) ---
    {{
      "prompt_imagen": "Educational illustration of... [descripción detallada en inglés]",
      "razon": "Por qué este visual apoya el aprendizaje del concept evaluado."
    }}
    """
    
    try:
        response_text = call_llm_router([prompt], [], model_name, json_mode=True)
        data = json.loads(limpiar_json_robustez(response_text))
        return data.get("prompt_imagen", ""), data.get("razon", "")
    except Exception as e:
        print(f"Error generando prompt de imagen: {e}")
        return "", ""

# --- 2. FUNCIÓN DEL AUDITOR (ACTUALIZADA CON LIMPIEZA DE JSON) ---
def auditar_item_llm(item_json_texto, taxonomia_dict, model_name, item_original=None, nivel_similitud=None, imagenes_original=None):
    """
    AUDITOR: Audita el ítem Y la coherencia de los gráficos (enunciado Y opciones).
    Cuando se proveen item_original/imagenes_original y nivel_similitud (modo Espejo), agrega criterio de fidelidad al nivel.
    """
    taxonomia_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])
    imagenes_original = imagenes_original or []

    # Bloque condicional: criterio de similitud solo para modo Espejo
    tiene_original = bool(item_original) or bool(imagenes_original)
    if tiene_original and nivel_similitud:
        if nivel_similitud == "Alta":
            regla_similitud = """Solo deben haber cambiado datos numéricos y nombres propios respecto al original.
    Si el escenario, la estructura de las opciones, la lógica de resolución o el orden de ideas fue alterado significativamente, es "❌ NO CUMPLE"."""
        elif nivel_similitud == "Media":
            regla_similitud = """Debe haberse cambiado el escenario o contexto narrativo y la representación de datos respecto al original.
    Si el escenario es idéntico o muy similar al original, es "❌ NO CUMPLE".
    Si la lógica de resolución cambió radicalmente (se evalúa una habilidad diferente), es "❌ NO CUMPLE"."""
        else:  # Baja
            regla_similitud = """El ítem DEBE ser RADICALMENTE diferente al original en todos los aspectos superficiales. Verifica cada punto — cualquiera es causa de "❌ NO CUMPLE":
    - ¿El escenario o dominio temático es igual o similar? (ej. ambos hablan de ventas, ambos de sucursales, ambos de temperaturas) → "❌ NO CUMPLE"
    - ¿Las etiquetas o entidades son iguales o similares? (ej. ambos usan P/Q/R/S, mismo tipo de empresa, mismas categorías) → "❌ NO CUMPLE"
    - ¿Algún valor numérico del original aparece en las opciones o en el enunciado del nuevo ítem? → "❌ NO CUMPLE"
    - ¿Hay frases, expresiones o fragmentos de redacción copiados o parafraseados del original? → "❌ NO CUMPLE"
    - ¿Las opciones del nuevo ítem tienen los mismos valores o son derivados directos de los valores del original? → "❌ NO CUMPLE"
    RECUERDA: las opciones del nuevo ítem deben ser calculadas desde cero a partir del nuevo escenario inventado, no derivadas del original.
    Lo ÚNICO permitido: misma habilidad cognitiva (ej. calcular rango) y mismo tipo de operación abstracta. Absolutamente todo lo demás debe ser diferente."""

        ref_texto = f"\n    **ÍTEM ORIGINAL EN TEXTO (referencia):**\n    {item_original}" if item_original else "\n    (El ítem original fue entregado como imagen — compara visualmente con las imágenes adjuntas)"
        criterio_similitud_prompt = f"""
    8. **Fidelidad al Nivel de Similitud ({nivel_similitud})**: Compara el ítem generado contra el ÍTEM ORIGINAL.
    {regla_similitud}
    {ref_texto}
    """
        criterio_similitud_json = f'{{ "criterio": "Fidelidad al Nivel de Similitud ({nivel_similitud})", "estado": "✅ CUMPLE", "comentario": "..." }},'
    else:
        criterio_similitud_prompt = ""
        criterio_similitud_json = ""

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
    8. **Pistas Formales en Opciones**: Revisa las 4 opciones en busca de pistas que delaten la clave. Es "❌ NO CUMPLE" si: alguna opción es notoriamente más larga que las demás; las opciones mezclan registros (técnico vs. coloquial); la estructura gramatical es diferente entre opciones; alguna opción repite frases del enunciado; alguna opción usa adverbios absolutos ("siempre", "nunca", "jamás", "completamente", "todos", "ninguno").
    9. **Negaciones con Formato**: Si el enunciado contiene una negación principal (NO, EXCEPTO, etc.), debe estar escrita en MAYÚSCULA Y NEGRITA (ej. **NO**). Si aparece en minúscula o sin negritas, es "❌ NO CUMPLE".
    10. **Sin Combinaciones Prohibidas**: Las opciones no pueden contener enunciados tipo "Todas las anteriores", "Ninguna de las anteriores", "A y B son correctas" o cualquier variante. Si aparecen, es "❌ NO CUMPLE".
    11. **Sin Contenido Sensible**: El ítem no debe abordar temas de religión, política, sexualidad, género, etnicidad, violencia o ideologías. Si lo hace, es "❌ NO CUMPLE".
    {criterio_similitud_prompt}

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
        {{ "criterio": "Coherencia Visual", "estado": "✅ CUMPLE", "comentario": "..." }},
        {{ "criterio": "Pistas Formales en Opciones", "estado": "✅ CUMPLE", "comentario": "..." }},
        {{ "criterio": "Negaciones con Formato", "estado": "✅ CUMPLE", "comentario": "..." }},
        {{ "criterio": "Sin Combinaciones Prohibidas", "estado": "✅ CUMPLE", "comentario": "..." }},
        {{ "criterio": "Sin Contenido Sensible", "estado": "✅ CUMPLE", "comentario": "..." }},
        {criterio_similitud_json}
      ],
      "dictamen_final": "✅ CUMPLE" o "⚠️ CUMPLE CON OBSERVACIONES" o "❌ RECHAZADO",
      "correcciones": [
        {{ "criterio": "nombre del criterio fallido", "prioridad": "ALTA" o "MEDIA", "accion": "Instrucción concreta y específica para corregirlo." }}
      ],
      "observaciones_finales": "Resumen ejecutivo del estado del ítem. Si se rechaza, explica el problema principal."
    }}
    """

    try:
        response_text = call_llm_router([prompt_auditor], imagenes_original, model_name, json_mode=True)
        return limpiar_json_robustez(response_text)
    except Exception as e:
        st.error(f"Error al contactar IA (Auditor): {e}")
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
        "ID CONTEXTO": str(st.session_state.get("contexto_compartido_texto", "") or "NA"),

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

def _cargar_template_word():
    """Carga el template Word: primero local, luego GCS."""
    # 1. Intentar desde archivo local (incluido en la imagen Docker)
    for local_path in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "formato_limpio.docx"),
        "/app/formato_limpio.docx",
        "formato_limpio.docx",
    ]:
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                return f.read()
    # 2. Intentar desde GCS
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "sumun-pruebas")
    template_name = os.environ.get("WORD_TEMPLATE_NAME", "formato_limpio.docx")
    return _descargar_template_word(bucket_name, template_name)

def crear_word(datos_editados, taxonomia_seleccionada, oportunidad_mejora):
    """
    Genera un documento Word rellenando una plantilla desde GCS o local.
    """
    try:
        # 1. Obtener bytes del template
        template_bytes = _cargar_template_word()

        if template_bytes is None:
            st.error("Error: No se encontró la plantilla 'formato_limpio.docx' (ni local ni en GCS).")
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

        # 5. Definir todos los reemplazos (¡TODOS CON str()!)
        reemplazos = {
            "{{ItemPruebaId}}": str(taxonomia_seleccionada.get("Área", "N/A")),
            "{{ItemGradoId}}": str(taxonomia_seleccionada.get("Grado", "N/A")), 
            "{{CompetenciaNombre}}": str(taxonomia_seleccionada.get("Competencia", "N/A")),
            "{{ComponenteNombre}}": str(taxonomia_seleccionada.get("Componente_Estructura", "N/A")),
            "{{AfirmacionNombre}}": str(taxonomia_seleccionada.get("Afirmación", "N/A")),
            "{{EvidenciaNombre}}": str(taxonomia_seleccionada.get("Evidencia", "N/A")),
            "{{ItemContexto}}": str(st.session_state.get("contexto_compartido_texto", "") or ""),
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
        return final_buffer.getvalue()

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
_col_titulo, _col_usuario, _col_historial = st.columns([5, 2, 1])
with _col_titulo:
    _modo_prod = st.session_state.get("modo_producto", "privados")
    _titulo_app = "🏫 Colegios Privados — Espejazos" if _modo_prod == "privados" else "📝 Super Simulacros — Espejazos"
    st.title(_titulo_app)
with _col_usuario:
    st.markdown(f"**👤 {st.session_state.get('_usuario_nombre', '')}** — {st.session_state.get('_usuario_area', '')}")
with _col_historial:
    if st.button("📂 Mi historial"):
        st.session_state["_mostrar_historial"] = not st.session_state.get("_mostrar_historial", False)

if st.session_state.get("_mostrar_historial"):
    st.subheader("📂 Mis ítems guardados")
    _area_u = st.session_state.get("_usuario_area", "")
    _nombre_u = st.session_state.get("_usuario_nombre", "")
    _items_guardados = listar_progreso_gcs(_area_u, _nombre_u)
    if not _items_guardados:
        st.info("Aún no tienes ítems guardados.")
    else:
        for _ig in _items_guardados:
            _ts = _ig["timestamp"].replace("_", " ")
            _label = f"🕐 {_ts} | {_ig['modo']} | {_ig['preview']}..."
            if st.button(_label, key=f"_cargar_{_ig['blob_name']}"):
                cargar_item_desde_progreso(_ig["payload"])
                st.session_state["_mostrar_historial"] = False
                st.success("Ítem cargado en el editor.")
                st.rerun()
    st.divider()

# --- INICIALIZACIÓN MÁQUINA DE ESTADOS PARA CONTEXTOS ---
if "etapa_contexto" not in st.session_state:
    st.session_state.etapa_contexto = "CREACION_CONTEXTO"
if "contexto_compartido_texto" not in st.session_state:
    st.session_state.contexto_compartido_texto = ""
if "contexto_compartido_imagen" not in st.session_state:
    st.session_state.contexto_compartido_imagen = None
if "contexto_area" not in st.session_state:
    st.session_state.contexto_area = None
if "contexto_grado" not in st.session_state:
    st.session_state.contexto_grado = None
if "total_items_contexto" not in st.session_state:
    st.session_state.total_items_contexto = 2
if "item_actual_contexto" not in st.session_state:
    st.session_state.item_actual_contexto = 1

# --- NAVEGACIÓN LATERAL ---
with st.sidebar:
    st.header("Modo de Creación")
    _es_ingles = st.session_state.get("_usuario_area", "") == "Inglés"
    if _es_ingles:
        _modos = ["🎨 Ítem Inspirado", "🪞 Ítem Espejo", "🧩 Generación en Contexto (Partes)"]
        _modo_idx = 2
    else:
        _modos = ["✨ Nuevo Ítem", "🎨 Ítem Inspirado", "🪞 Ítem Espejo", "🧩 Generación en Contexto (Bloques)"]
        _modo_idx = 2
    modo_creacion = st.radio(
        "¿Qué deseas hacer hoy?",
        options=_modos,
        index=_modo_idx
    )
    
    st.divider()
    st.header("🔐 Conexión LLM / API Keys")
    proveedor_llm = st.radio(
        "Proveedor de Inteligencia Artificial",
        options=["🏢 Vertex AI (Nativo)", "🤖 Anthropic Claude (Nativo)", "🔑 Google Gemini (API Key)", "🔑 OpenAI (API Key)", "🔑 Anthropic Claude (API Key)"],
        index=0,
        help="Los modos Nativos usan credenciales del servidor. Las API Keys te permiten traer tu propio motor de pago."
    )
    st.session_state["proveedor_llm"] = proveedor_llm
    
    # Renderizamos los inputs condicionalmente y determinamos la lista de modelos
    if proveedor_llm == "🔑 Google Gemini (API Key)":
        st.text_input("Ingresa tu Gemini API Key:", type="password", key="api_key_gemini")
        if st.session_state.get("_default_gemini_loaded"):
            st.success("✅ Clave Gemini por defecto activa.")
        with st.expander("Cargar clave por defecto"):
            pwd_gemini = st.text_input("Contraseña para clave Gemini:", type="password", key="_pwd_gemini_default")
            if st.button("Cargar clave Gemini", key="_btn_gemini_default"):
                if pwd_gemini == os.environ.get("DEFAULT_KEY_PASSWORD_GEMINI", ""):
                    st.session_state["_default_gemini_loaded"] = os.environ.get("DEFAULT_GEMINI_KEY", "")
                    st.rerun()
                else:
                    st.error("Contraseña incorrecta.")
        modelos_disponibles = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]
    elif proveedor_llm == "🤖 Anthropic Claude (Nativo)":
        st.success("✅ Claude conectado vía credenciales del servidor.")
        modelos_disponibles = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
    elif proveedor_llm == "🔑 OpenAI (API Key)":
        st.text_input("Ingresa tu OpenAI API Key (sk-...):", type="password", key="api_key_openai")
        modelos_disponibles = ["gpt-4o", "gpt-4o-mini", "o3-mini", "o1", "o1-mini"]
    elif proveedor_llm == "🔑 Anthropic Claude (API Key)":
        st.text_input("Ingresa tu Anthropic API Key (sk-ant-...):", type="password", key="api_key_anthropic")
        if st.session_state.get("_default_claude_loaded"):
            st.success("✅ Clave Claude por defecto activa.")
        with st.expander("Cargar clave por defecto"):
            pwd_claude = st.text_input("Contraseña para clave Claude:", type="password", key="_pwd_claude_default")
            if st.button("Cargar clave Claude", key="_btn_claude_default"):
                if pwd_claude == os.environ.get("DEFAULT_KEY_PASSWORD_CLAUDE", ""):
                    st.session_state["_default_claude_loaded"] = os.environ.get("DEFAULT_CLAUDE_KEY", "")
                    st.rerun()
                else:
                    st.error("Contraseña incorrecta.")
        modelos_disponibles = ["claude-opus-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"]
    else:
        # Vertex nativo
        modelos_disponibles = [
            "gemini-3.1-pro-preview",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-exp"
        ]
        
    st.divider()
    st.header("⚙️ Configuración Global")
    
    modelo_generador_sel = st.selectbox(
        "Modelo para Generar Ítem",
        options=modelos_disponibles,
        index=0 if proveedor_llm != "🏢 Vertex AI (Nativo)" else 4
    )
    modelo_auditor_sel = st.selectbox(
        "Modelo para Auditar Ítem",
        options=modelos_disponibles,
        index=0,
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

    st.divider()
    _bd_blob = os.environ.get("BD_IMPROVE_BLOB", "")
    try:
        _ts = _bd_blob.split("_")[0]  # "202604080803"
        _fecha_bd = f"{_ts[6:8]}/{_ts[4:6]}/{_ts[0:4]}  {_ts[8:10]}:{_ts[10:12]}"
    except Exception:
        _fecha_bd = "desconocida"
    st.caption(f"📦 Base de ítems: **{_fecha_bd}**")

# --- 1. ENTRADA DE DATOS SEGÚN MODO ---
col1, col2 = st.columns(2)

# --- COLUMNA 2 (Taxonomía) ---
with col2:
    st.header("2. Cascada de Taxonomía")
    if _es_ingles:
        st.markdown(
            """
            <div style="opacity:0.35; pointer-events:none; user-select:none; border:1px solid #ccc;
                        border-radius:8px; padding:16px; background:#f5f5f5;">
            <p><strong>🔒 No disponible para Inglés</strong></p>
            <p>La taxonomía se toma automáticamente de la MATRIZ_INGLES según la parte seleccionada.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Si BD Improve ya importó la taxonomía, mostrar resumen y no renderizar los selectores
        _bd_improve_activo = bool(st.session_state.get("taxonomia_importada"))
        if _bd_improve_activo:
            _ti = st.session_state["taxonomia_importada"]
            st.success("✅ Taxonomía importada desde BD Improve")
            st.markdown(f"""
- **Área:** {_ti.get('Área', '—')}
- **Grado:** {_ti.get('Grado', '—')}
- **Competencia:** {_ti.get('Competencia', '—')}
- **Afirmación:** {_ti.get('Afirmación', '—')}
- **Evidencia:** {_ti.get('Evidencia', '—')}
            """)
        # Inicialización silenciosa de variables en el ámbito de la página
        grado_sel = area_sel = comp1_sel = comp2_sel = ref_sel = competen_sel = afirm_sel = evid_sel = None

        if not _bd_improve_activo:
            bucket_name = os.environ.get("GCS_BUCKET_NAME", "bucket_espejos")
            excel_file_path = os.environ.get("EXCEL_TAXONOMY_PATH", "Estructura privados11.xlsx")
            data = leer_excel_desde_gcs(bucket_name, excel_file_path)

            if data is not None:
                try:
                    if 'df1' not in st.session_state or 'df2' not in st.session_state:
                        sheet_names = list(data.keys())
                        st.session_state.df1 = data[sheet_names[0]]
                        st.session_state.df2 = data[sheet_names[1]]

                    df1 = st.session_state.df1
                    df2 = st.session_state.df2

                    def mark(val): return "✅" if val else "❌"

                    grado_sel = 11
                    st.session_state['tax_grado'] = grado_sel
                    st.write(f"✅ **Grado:** {grado_sel} (Fijo)")

                    df_grado = df1[df1['Grado'] == grado_sel]

                    if modo_creacion == "🧩 Generación en Contexto (Bloques)" and st.session_state.etapa_contexto != "CREACION_CONTEXTO":
                        area_sel = st.session_state.contexto_area
                        st.write(f"✅ **Área:** {area_sel} (Fijo por bloque)")
                        st.session_state['tax_area'] = area_sel
                    else:
                        st.write(f"{mark(st.session_state.get('tax_area'))} **Área**")
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
        base_input_type = st.radio("Entrada base", ["🖼️ Imagen", "📝 Texto", "📄 PDF", "🔍 Buscar en BD Improve"], horizontal=True)
        
        # Variable especial para taxonomía importada de DB
        st.session_state["taxonomia_importada"] = None
        
        if base_input_type == "🖼️ Imagen":
            imagen_subida = st.file_uploader(
                "Sube el pantallazo de la pregunta", 
                type=["png", "jpg", "jpeg"]
            )
            if imagen_subida:
                st.image(imagen_subida, caption="Ítem cargado", use_container_width=True)
        elif base_input_type == "📝 Texto":
            contexto_adicional = st.text_area(
                "Pega el texto de la pregunta original",
                placeholder="Enunciado y opciones...",
                height=250,
                key="contexto_espejo"
            )
        elif base_input_type == "📄 PDF":
            pdf_subido = st.file_uploader("Sube el archivo PDF", type=["pdf"])
            contexto_adicional = ""
            if pdf_subido:
                try:
                    pdf_reader = PyPDF2.PdfReader(pdf_subido)
                    texto_extraido = ""
                    for page in pdf_reader.pages:
                        texto_extraido += page.extract_text() + "\n"
                    contexto_adicional = st.text_area("Texto extraído del PDF (puedes editarlo)", value=texto_extraido, height=250, key="contexto_pdf")
                except Exception as e:
                    st.error(f"Error al leer el PDF: {e}")
        elif base_input_type == "🔍 Buscar en BD Improve":
            item_id_input = st.text_input("Ingresa el ID del Ítem (Ej: 12345)")
            contexto_adicional = ""
            if item_id_input:
                try:
                    df_improve = cargar_base_improve()
                    if df_improve is None:
                        st.error("No se pudo cargar la base de datos Improve local.")
                    else:
                        # Filtrar por ItemId. Convertimos asumiendo numérico o texto
                        item_id_num = float(item_id_input) if item_id_input.isnumeric() else item_id_input
                        
                        df_item = df_improve[df_improve['ItemId'] == item_id_num]
                        if df_item.empty:
                            # Fallback a string si no lo encontró como número
                            df_item = df_improve[df_improve['ItemId'].astype(str) == str(item_id_input)]
                            
                        if df_item.empty:
                            st.warning(f"No se encontró el ítem con ID {item_id_input} en la base de datos.")
                        else:
                            st.success("Ítem encontrado exitosamente")
                            
                            # Tomar la primera fila para datos comunes
                            row = df_item.iloc[0]
                            
                            # Extraer componentes del ítem
                            contexto = str(row.get('ItemContexto', ''))
                            enunciado = str(row.get('ItemEnunciado', ''))
                            
                            if contexto == "nan": contexto = ""
                            if enunciado == "nan": enunciado = ""
                            
                            # Construir texto amigable para el usuario y el LLM
                            texto_armado = f"**Contexto:**\n{contexto}\n\n**Enunciado:**\n{enunciado}\n\n**Opciones:**\n"
                            
                            opciones_str = []
                            for index, opt_row in df_item.iterrows():
                                texto_opt = str(opt_row.get('AlternativaTexto', ''))
                                # Usamos AlternativaCorrecta o IsCorrect (si es 1, True o 'Clave')
                                es_correcta = str(opt_row.get('AlternativaCorrecta', '')).lower() in ['1', '1.0', 'true', 'sí', 'si', 'clave']
                                # AlternativaClave suele ser "A", "B", etc.
                                letra = str(opt_row.get('AlternativaClave', '?'))
                                
                                if es_correcta:
                                    opciones_str.append(f"{letra}: <span style='color:red; font-weight:bold;'>{texto_opt} (CORRECTA)</span>")
                                else:
                                    opciones_str.append(f"{letra}: {texto_opt}")
                                    
                            texto_armado += "<br>".join(opciones_str)
                            
                            # Mostrar el item en pantalla
                            st.markdown(texto_armado, unsafe_allow_html=True)
                            
                            # Convertimos a markdown plano para enviarlo al LLM
                            contexto_adicional = texto_armado.replace("<span style='color:red; font-weight:bold;'>", "").replace("</span>", "").replace("<br>", "\n")
                            
                            # Extraer taxonomía silenciosamente
                            tax_importada = {
                                "Grado": str(row.get("ItemGradoNombre", "")),
                                "Área": str(row.get("BloqueAreaNombre", "")),
                                "Competencia": str(row.get("CompetenciaNombre", "")),
                                "Afirmación": str(row.get("AfirmacionNombre", "")),
                                "Evidencia": str(row.get("EvidenciaNombre", "")),
                                "Ref. Temática": str(row.get("TematicaNombre", "")),
                                "Componente_Estructura": str(row.get("ComponenteNombre", "")),
                                "Componente_Tematica": str(row.get("TematicaNombre", "")) # Fallback
                            }
                            
                            # Limpiar NaN o campos vacíos
                            tax_importada = {k: ("No aplica" if pd.isna(v) or v == "nan" else v) for k,v in tax_importada.items()}
                            st.session_state["taxonomia_importada"] = tax_importada
                            
                            st.write("**Taxonomía Importada Automáticamente:**")
                            st.json(tax_importada)
                                
                except Exception as e:
                    st.error(f"Error procesando la base de datos: {e}")
            
            
    elif modo_creacion == "🧩 Generación en Contexto (Bloques)":
        if st.session_state.etapa_contexto == "CREACION_CONTEXTO":
            st.header("1. Definir o Cargar el Contexto Base")
            metodo_contexto = st.radio("¿Cómo deseas crear el contexto?", ["💡 Generar con IA", "📝 Escribirlo / Pegarlo", "🖼️ Subir Imagen"], horizontal=True)
            
            if metodo_contexto == "💡 Generar con IA":
                idea_base = st.text_input("¿Sobre qué tema quieres el contexto? (Ej: Los hoyos negros)", key="idea_ctx")
                if st.button("Generar Contexto Base con IA 🚀", type="primary"):
                    if idea_base and st.session_state.get('tax_grado') and st.session_state.get('tax_area'):
                        tax_temp = {
                            "Grado": st.session_state.get('tax_grado'),
                            "Área": st.session_state.get('tax_area')
                        }
                        with st.spinner("Generando un texto base robusto (Lectura/Caso)..."):
                            ctx_generado = generar_contexto_base_llm(tax_temp, idea_base, modelo_generador_sel)
                            st.session_state.contexto_generado_temp = ctx_generado
                    else:
                        st.warning("Escribe una idea y asegúrate de seleccionar Grado y Área en la Col. 2")
                
                # Paso intermedio de revisión de IA
                if st.session_state.get("contexto_generado_temp"):
                    st.write("---")
                    st.subheader("Revisa y Edita el Contexto")
                    contexto_editado = st.text_area(
                        "Contexto generado por IA:", 
                        value=st.session_state.contexto_generado_temp, 
                        height=250
                    )
                    st.write("---")
                    st.subheader("¿Añadir gráfico o tabla al contexto? (opcional)")
                    st.selectbox("¿El contexto incluye un gráfico?", ["NO", "SÍ"], key="_ctx_base_grafico_nec")
                    if st.session_state.get("_ctx_base_grafico_nec") == "SÍ":
                        st.text_area("Describe el gráfico:", placeholder="Ej: Tabla de precipitaciones mensuales por ciudad", height=70, key="_ctx_base_grafico_desc")
                        if st.button("🎨 Generar gráfico del contexto"):
                            _desc_ctx = st.session_state.get("_ctx_base_grafico_desc", "").strip()
                            if _desc_ctx and GRAFICOS_DISPONIBLES:
                                try:
                                    with st.spinner("Generando gráfico..."):
                                        spec = build_visual_json_with_llm(_desc_ctx)
                                        buf = crear_grafico(spec.get("tipo_elemento"), spec.get("datos", {}), spec.get("configuracion", {})) if spec else None
                                    if buf:
                                        st.session_state["_ctx_base_img_buffer"] = buf
                                        st.success("¡Gráfico generado!")
                                    else:
                                        st.error("No se pudo generar el gráfico.")
                                except Exception as _e:
                                    st.error(f"Error: {_e}")
                            else:
                                st.warning("Describe el gráfico primero.")
                        if st.session_state.get("_ctx_base_img_buffer"):
                            st.image(st.session_state["_ctx_base_img_buffer"], caption="Gráfico del contexto")
                    else:
                        st.session_state["_ctx_base_img_buffer"] = None

                    if st.button("✅ Aprobar Contexto IA y Continuar", type="primary"):
                        st.session_state.contexto_compartido_texto = contexto_editado
                        st.session_state.contexto_compartido_imagen = st.session_state.get("_ctx_base_img_buffer")
                        st.session_state.contexto_area = st.session_state.get('tax_area')
                        st.session_state.contexto_grado = st.session_state.get('tax_grado')
                        st.session_state.contexto_generado_temp = None
                        st.session_state.etapa_contexto = "SETEO_CANTIDAD"
                        st.rerun()
                        
            elif metodo_contexto == "📝 Escribirlo / Pegarlo":
                contexto_txt = st.text_area("Pega aquí el texto base que usarán todas las preguntas:", height=250)
                if st.button("Aprobar Contexto y Continuar", type="primary"):
                    if contexto_txt and st.session_state.get('tax_grado') and st.session_state.get('tax_area'):
                        st.session_state.contexto_compartido_texto = contexto_txt
                        st.session_state.contexto_area = st.session_state.get('tax_area')
                        st.session_state.contexto_grado = st.session_state.get('tax_grado')
                        st.session_state.etapa_contexto = "SETEO_CANTIDAD"
                        st.rerun()
                    else:
                        st.warning("Escribe un contexto válido y asegúrate de seleccionar Grado y Área en la Col. 2")
            else:
                imagen_subida_ctx = st.file_uploader("Sube el pantallazo de la lectura/gráfico", type=["png", "jpg", "jpeg"])
                if imagen_subida_ctx:
                    st.image(imagen_subida_ctx, use_container_width=True)
                    if st.button("Aprobar Imagen y Continuar", type="primary"):
                        if st.session_state.get('tax_grado') and st.session_state.get('tax_area'):
                            st.session_state.contexto_compartido_imagen = imagen_subida_ctx
                            st.session_state.contexto_area = st.session_state.get('tax_area')
                            st.session_state.contexto_grado = st.session_state.get('tax_grado')
                            st.session_state.etapa_contexto = "SETEO_CANTIDAD"
                            st.rerun()
                        else:
                            st.warning("Asegúrate de seleccionar Grado y Área en la Col. 2")
                            
        elif st.session_state.etapa_contexto == "SETEO_CANTIDAD":
            st.header("2. Diseño del Bloque")
            n_items = st.number_input("¿Cuántos ítems deseas generar para este contexto?", min_value=2, max_value=7, value=3)
            if st.button("Comenzar a Diseñar Ítems 🚀", type="primary", use_container_width=True):
                st.session_state.total_items_contexto = int(n_items)
                st.session_state.item_actual_contexto = 1
                st.session_state.etapa_contexto = "GENERACION_ITEMS"
                st.rerun()
                
        elif st.session_state.etapa_contexto == "GENERACION_ITEMS":
            st.header(f"3. Diseñando Ítem {st.session_state.item_actual_contexto} de {st.session_state.total_items_contexto}")
            with st.expander("Ver Contexto Compartido Activo", expanded=False):
                if st.session_state.contexto_compartido_texto:
                    st.write(st.session_state.contexto_compartido_texto)
                if st.session_state.contexto_compartido_imagen:
                    st.image(st.session_state.contexto_compartido_imagen)
            
            contexto_adicional = st.text_area(
                 "Escribe qué debe evaluar específicamente esta pregunta:",
                 placeholder="Ej: Formula una pregunta sobre el segundo párrafo que evalúe inferencia local...",
                 height=100
            )
            st.selectbox(
                "¿Este ítem requiere un gráfico o tabla?",
                options=["NO", "SÍ"],
                key="_ctx_grafico_nec"
            )
            if st.session_state.get("_ctx_grafico_nec") == "SÍ":
                st.selectbox(
                    "¿Dónde va el gráfico?",
                    options=["Enunciado", "Opción A", "Opción B", "Opción C", "Opción D"],
                    key="_ctx_grafico_loc"
                )
                st.text_area(
                    "Describe qué debe mostrar el gráfico:",
                    placeholder="Ej: Tabla comparativa de exportaciones por año y país",
                    height=80,
                    key="_ctx_grafico_desc"
                )

    elif modo_creacion == "🧩 Generación en Contexto (Partes)":
        # ── MODO INGLÉS: GENERACIÓN POR PARTES ──────────────────────────────
        st.header("🧩 Generación en Contexto (Partes) — Inglés")

        # Inicializar sesión
        if "ingles_matriz" not in st.session_state:
            st.session_state.ingles_matriz = None
        if "ingles_resultado" not in st.session_state:
            st.session_state.ingles_resultado = None
        if "ingles_filas_excel" not in st.session_state:
            st.session_state.ingles_filas_excel = None

        # Cargar matriz si no está en sesión
        if st.session_state.ingles_matriz is None:
            with st.spinner("Cargando matriz de taxonomía..."):
                try:
                    from ingles_partes import cargar_matriz_ingles
                    st.session_state.ingles_matriz = cargar_matriz_ingles()
                except Exception as e:
                    st.error(f"Error cargando matriz: {e}")

        if st.session_state.ingles_matriz:
            from ingles_partes import (
                REGLAS_PARTES, partes_disponibles,
                max_items_parte, obtener_taxonomia_parte,
                construir_prompt_parte, parsear_respuesta_parte,
                items_a_filas_excel, generar_excel_parte,
                construir_words_parte, empaquetar_zip,
                validar_vocabulario, validar_repeticiones,
            )

            matriz = st.session_state.ingles_matriz

            col_par1, col_par2, col_par3 = st.columns([2, 2, 2])

            with col_par1:
                _grado_sel = 11
                st.write(f"✅ **Grado:** {_grado_sel} (Fijo)")

            with col_par2:
                _partes_disp = partes_disponibles(matriz, _grado_sel)
                _parte_opciones = [f"Parte {p} – {REGLAS_PARTES.get(p, {}).get('nombre', '').split('–')[-1].strip()}" for p in _partes_disp]
                _parte_idx = st.selectbox("Parte", range(len(_partes_disp)), format_func=lambda i: _parte_opciones[i], key="ingles_parte_idx")
                _parte_num = _partes_disp[_parte_idx] if _partes_disp else 1

            with col_par3:
                _max_items = max_items_parte(matriz, _grado_sel, _parte_num)
                _n_items = st.number_input("N° de preguntas", min_value=1, max_value=_max_items, value=min(5, _max_items), key="ingles_n_items")

            _tema = st.text_input("Tema sugerido (opcional)", placeholder="Ej: sports, animals, technology", key="ingles_tema")

            # Info de la parte seleccionada
            if _parte_num in REGLAS_PARTES:
                with st.expander(f"📋 Reglas de {REGLAS_PARTES[_parte_num]['nombre']}", expanded=False):
                    st.markdown(REGLAS_PARTES[_parte_num]["reglas"])

            _taxonomias_parte = obtener_taxonomia_parte(matriz, _grado_sel, _parte_num)
            if _taxonomias_parte:
                _niveles_str = ", ".join(dict.fromkeys(t["nivel_mcer"] for t in _taxonomias_parte[:int(_n_items)]))
                st.caption(f"Niveles MCER para esta selección: **{_niveles_str}**")

            st.divider()

            if st.button("🚀 Generar Parte Completa", type="primary", use_container_width=True, key="btn_generar_parte"):
                _tax_sel = _taxonomias_parte[:int(_n_items)]
                if not _tax_sel:
                    st.error("No hay taxonomía disponible para este grado/parte en la matriz.")
                else:
                    _prompt = construir_prompt_parte(_parte_num, int(_n_items), _tax_sel, _tema, grado=_grado_sel)
                    with st.spinner(f"Generando {_parte_num} con {int(_n_items)} ítems… (puede tardar 1-3 min con modelos avanzados)"):
                        try:
                            _respuesta_raw = call_llm_router([_prompt], [], modelo_generador_sel)
                            _resultado = parsear_respuesta_parte(_respuesta_raw, _parte_num)
                            st.session_state.ingles_resultado = _resultado
                            _nivel_obj = _tax_sel[0]["nivel_mcer"] if _tax_sel else ""
                            _alertas = validar_vocabulario(_resultado, _parte_num, _nivel_obj)
                            _rep = validar_repeticiones(_resultado, _parte_num)
                            st.session_state.ingles_filas_excel = items_a_filas_excel(
                                _resultado, _parte_num, _grado_sel, _tax_sel,
                                alertas=_alertas, rep=_rep,
                            )
                            st.session_state.ingles_meta = {
                                "grado": _grado_sel,
                                "parte": _parte_num,
                                "n_items": int(_n_items),
                                "taxonomias": _tax_sel,
                            }
                            st.success(f"✅ Parte {_parte_num} generada correctamente.")
                        except Exception as e:
                            st.error(f"Error generando: {e}")
                            st.code(str(_respuesta_raw) if '_respuesta_raw' in dir() else "Sin respuesta")

            # Mostrar resultado si existe
            if st.session_state.ingles_resultado:
                _res = st.session_state.ingles_resultado
                _meta = st.session_state.get("ingles_meta", {})

                # Estímulo compartido
                _estimulo = _res.get("estimulo", {})
                if _estimulo:
                    with st.expander("📖 Estímulo / Contexto compartido", expanded=True):
                        if isinstance(_estimulo, dict):
                            if _estimulo.get("titulo"):
                                st.markdown(f"**{_estimulo['titulo']}**")
                            if "texto" in _estimulo:
                                st.write(_estimulo["texto"])
                            elif "texto_con_blancos" in _estimulo:
                                st.write(_estimulo["texto_con_blancos"])
                            elif "opciones" in _estimulo:
                                ops = _estimulo["opciones"]
                                for k, v in ops.items():
                                    st.write(f"**{k}.** {v}")

                # Ítems generados
                st.subheader("Ítems generados")
                for idx, _item in enumerate(_res.get("items", []), 1):
                    with st.expander(f"Ítem {idx}", expanded=idx == 1):
                        # Estímulo individual (Parte 2 y 3)
                        if _item.get("estimulo"):
                            st.info(f"📢 **Aviso/Anuncio:** {_item['estimulo']}")
                        elif _item.get("turno_p"):
                            st.info(f"💬 **P:** {_item.get('turno_p', '')}  \n**Q:** {_item.get('turno_q_incompleto', '')}")
                        enun = _item.get("enunciado", _item.get("descripcion", ""))
                        st.markdown(f"**{enun}**")
                        # Parte 1: opciones A-H vienen del banco del estímulo
                        _meta_parte = _meta.get("parte", 0)
                        if _meta_parte == 1:
                            _banco = _res.get("estimulo", {}).get("opciones", {})
                            for letra, texto in _banco.items():
                                clave = _item.get("clave", "")
                                _bold = "**" if letra.upper() == clave else ""
                                st.write(f"{_bold}{letra}. {texto}{_bold}")
                        else:
                            for letra in ["a", "b", "c", "d"]:
                                v = _item.get(f"opcion_{letra}")
                                if v:
                                    clave = _item.get("clave", "")
                                    _bold = "**" if letra.upper() == clave else ""
                                    st.write(f"{_bold}{letra.upper()}. {v}{_bold}")
                        st.caption(f"Clave: {_item.get('clave', '')}")

                # Descargas
                st.divider()
                st.subheader("⬇️ Descargar")
                _col_dl1, _col_dl2 = st.columns(2)

                with _col_dl1:
                    _filas = st.session_state.ingles_filas_excel
                    if _filas:
                        _excel_bytes = generar_excel_parte(_filas)
                        st.download_button(
                            "📊 Descargar Excel",
                            data=_excel_bytes,
                            file_name=f"parte{_meta.get('parte', '')}_grado{str(_meta.get('grado', '')).replace('°','')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )

                with _col_dl2:
                    try:
                        _tmpl = _cargar_template_word()
                        _words = construir_words_parte(
                            _res,
                            _meta.get("parte", 1),
                            _meta.get("grado", ""),
                            _meta.get("taxonomias", []),
                            template_bytes=_tmpl,
                            reemplazar_fn=reemplazar_texto_en_doc,
                        )
                        _zip_bytes = empaquetar_zip(_words)
                        st.download_button(
                            "📦 Descargar ZIP (Word)",
                            data=_zip_bytes,
                            file_name=f"parte{_meta.get('parte', '')}_grado{str(_meta.get('grado', '')).replace('°','')}_words.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Error generando Word: {e}")

    if modo_creacion == "🪞 Ítem Espejo":
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
    else:
        similitud_sel = "Media"

# --- 5. LÓGICA DEL BOTÓN (Bucle Generador-Auditor) ---
st.divider()
mostrar_boton_generar = True
if modo_creacion == "🧩 Generación en Contexto (Bloques)" and st.session_state.etapa_contexto != "GENERACION_ITEMS":
    mostrar_boton_generar = False
if modo_creacion == "🧩 Generación en Contexto (Partes)":
    mostrar_boton_generar = False

if mostrar_boton_generar and st.button(f"🚀 Generar {modo_creacion.split()[-1]} (con Auditoría)", use_container_width=True, type="primary"):
    
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
        elif base_input_type in ["📝 Texto", "📄 PDF", "🔍 Buscar en BD Improve"] and contexto_adicional:
            input_valido = True
            inputs_llm["texto"] = contexto_adicional
        else:
            st.warning("Carga un ítem base válido (imagen, texto, PDF o un ID de Improve que exista).")
            
    elif modo_creacion == "🧩 Generación en Contexto (Bloques)":
        if contexto_adicional:
            input_valido = True
            _ctx_grafico_instruccion = ""
            if st.session_state.get("_ctx_grafico_nec") == "SÍ":
                _loc = st.session_state.get("_ctx_grafico_loc", "Enunciado")
                _desc = st.session_state.get("_ctx_grafico_desc", "").strip()
                if _desc:
                    if _loc == "Enunciado":
                        _ctx_grafico_instruccion = (
                            f'\n\nGRÁFICO REQUERIDO EN ENUNCIADO: Este ítem DEBE incluir un gráfico/tabla. '
                            f'En el JSON generado establece "grafico_necesario_enunciado": "SÍ" y '
                            f'"descripcion_texto_grafico_enunciado": "{_desc}".'
                        )
                    else:
                        _letra = _loc.split()[-1]
                        _ctx_grafico_instruccion = (
                            f'\n\nGRÁFICO REQUERIDO EN OPCIÓN {_letra}: La opción {_letra} DEBE incluir un gráfico/tabla. '
                            f'En el JSON generado establece opciones.{_letra}.grafico_necesario: "SÍ" y '
                            f'opciones.{_letra}.descripcion_texto_grafico: "{_desc}".'
                        )
            texto_compilado = (
                f"CONTEXTO BASE:\n{st.session_state.contexto_compartido_texto}\n\n"
                f"INSTRUCCIÓN ESPECÍFICA PARA ESTA PREGUNTA:\n{contexto_adicional}\n\n"
                f"IMPORTANTE: NO RE-ESCRIBAS EL CONTEXTO BASE. SOLO GENERA EL ENUNCIADO Y LAS 4 OPCIONES "
                f"DE RESPUESTA BASADO ÚNICAMENTE ESTRICTAMENTE EN EL CONTEXTO PROVEÍDO. TU ENUNCIADO DEBE "
                f"SER AUTOSUFICIENTE Y ENLAZAR AL CONTEXTO INDIRECTAMENTE SIN COPIAR SU TEXTO."
                f"{_ctx_grafico_instruccion}"
            )
            inputs_llm["texto"] = texto_compilado
            if st.session_state.contexto_compartido_imagen:
                inputs_llm["imagenes"] = [st.session_state.contexto_compartido_imagen]
        else:
            st.warning("Escribe una instrucción específica sobre qué evaluar en esta iteración.")

    if input_valido:
        # Verificación de taxonomía usando session_state (solo si NO se importó por BD Improve)
        taxonomia_seleccionada = st.session_state.get("taxonomia_importada", None)
        
        if not taxonomia_seleccionada:
            tax_keys = {
                "Grado": "tax_grado", "Área": "tax_area", 
                "Competencia": "tax_competen", "Afirmación": "tax_afirm", 
                "Evidencia": "tax_evid", "Ref. Temática": "tax_ref",
                "Componente_Estructura": "tax_comp1", "Componente_Tematica": "tax_comp2"
            }
            faltantes_tax = [label for label, key in tax_keys.items() if not st.session_state.get(key)]
    
            if data is None:
                st.warning("El archivo Excel de taxonomía no se pudo cargar.")
                input_valido = False
            elif faltantes_tax:
                st.error(f"⚠️ **Taxonomía incompleta:** Faltan {', '.join(faltantes_tax)}")
                st.info("Por favor, completa todas las selecciones en la Columna 2.")
                input_valido = False
            else:
                taxonomia_seleccionada = {label: st.session_state.get(key) for label, key in tax_keys.items()}
        
        if input_valido and taxonomia_seleccionada:
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
                    item_original_para_auditor = inputs_llm.get("texto", "") if modo_creacion == "🪞 Ítem Espejo" else None
                    imagenes_original_para_auditor = inputs_llm.get("imagenes", []) if modo_creacion == "🪞 Ítem Espejo" else []
                    nivel_similitud_para_auditor = similitud_sel if modo_creacion == "🪞 Ítem Espejo" else None
                    audit_json_str = auditar_item_llm(item_json_str, taxonomia_seleccionada, modelo_auditor_sel, item_original_para_auditor, nivel_similitud_para_auditor, imagenes_original_para_auditor)

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
                                    st.write("**⚠️ Observaciones del auditor (considera estos puntos):**")
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
                            st.write(f"**Detalles del Rechazo (Intento {intento_actual})**")
                            st.json(audit_data)

                    except json.JSONDecodeError:
                        st.error(f"Error al leer respuesta JSON del auditor: {audit_json_str}")
                        feedback_auditor = "La respuesta del auditor no fue un JSON válido."

            if item_final_json is None:
                status.update(label=f"No se pudo generar un ítem de alta calidad después de {max_intentos} intentos.", state="error")
                st.warning(f"⚠️ Este ítem no superó la auditoría tras {max_intentos} intentos. Revísalo y corrígelo manualmente antes de usarlo.")
                st.caption(f"Feedback del auditor: {feedback_auditor}")
                if item_json_str:
                    cargar_item_en_editor(item_json_str)
                    st.session_state["_item_requiere_revision"] = True

        if item_final_json:
            st.session_state["_item_requiere_revision"] = False
            st.success("¡Ítem generado y auditado con éxito! Puedes editarlo abajo.")
            cargar_item_en_editor(item_final_json)

            # Auto-generar gráfico si fue solicitado en modo Contexto
            if (modo_creacion == "🧩 Generación en Contexto (Bloques)"
                    and st.session_state.get("_ctx_grafico_nec") == "SÍ"
                    and GRAFICOS_DISPONIBLES):
                _desc = st.session_state.get("_ctx_grafico_desc", "").strip()
                _loc  = st.session_state.get("_ctx_grafico_loc", "Enunciado")
                if _desc:
                    try:
                        with st.spinner("🎨 Generando gráfico del ítem..."):
                            spec = build_visual_json_with_llm(_desc)
                            buf  = crear_grafico(
                                spec.get("tipo_elemento"),
                                spec.get("datos", {}),
                                spec.get("configuracion", {})
                            ) if spec else None
                        if spec and buf:
                            if _loc == "Enunciado":
                                st.session_state.editable_grafico_nec_enunciado    = "SÍ"
                                st.session_state.editable_grafico_texto_enunciado  = _desc
                                st.session_state.editable_grafico_json_enunciado   = json.dumps([spec], indent=2)
                                st.session_state["img_buffer_enunciado"]           = buf
                            else:
                                _letra = _loc.split()[-1]
                                st.session_state[f"editable_opcion_{_letra.lower()}_grafico_nec"]   = "SÍ"
                                st.session_state[f"editable_opcion_{_letra.lower()}_grafico_texto"] = _desc
                                st.session_state[f"editable_opcion_{_letra.lower()}_grafico_json"]  = json.dumps([spec], indent=2)
                                st.session_state[f"img_buffer_op_{_letra}"]                        = buf
                    except Exception as _e_ctx_graf:
                        st.warning(f"No se pudo generar el gráfico automáticamente: {_e_ctx_graf}")

            guardar_progreso_gcs(
                st.session_state.get("_usuario_area", "General"),
                st.session_state.get("_usuario_nombre", "Desconocido"),
                item_final_json,
                taxonomia_seleccionada,
                modo_creacion
            )



# --- 6. EDITOR DE ÍTEMS Y DESCARGA (LÓGICA DE BOTONES SEPARADA) ---
if 'show_editor' in st.session_state and st.session_state.show_editor:
    st.divider()
    st.header("3. Edita el Ítem Generado")
    if st.session_state.get("_item_requiere_revision"):
        st.error("🚨 **Este ítem NO superó la auditoría automática.** Debes revisarlo y corregirlo manualmente antes de usarlo. Los problemas detectados aparecen arriba.")
    
    if modo_creacion == "🧩 Generación en Contexto (Bloques)" and st.session_state.get('contexto_compartido_texto'):
        with st.expander("📖 Ver Contexto Base del Bloque", expanded=False):
            st.write(st.session_state.contexto_compartido_texto)
            if st.session_state.get('contexto_compartido_imagen'):
                st.image(st.session_state.contexto_compartido_imagen)
    
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
            try:
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
            except Exception as _e_autogen:
                st.warning(f"No se pudo auto-generar la descripción del gráfico: {_e_autogen}")
        
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
                                # Auto-render
                                try:
                                    buf = crear_grafico(spec.get("tipo_elemento"), spec.get("datos", {}), spec.get("configuracion", {}))
                                    st.session_state['img_buffer_enunciado'] = buf
                                except Exception:
                                    st.session_state['img_buffer_enunciado'] = None
                                st.success("¡JSON y Gráfico generados!")
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
                try:
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
                except Exception as _e_autogen:
                    st.warning(f"No se pudo auto-generar la descripción del gráfico (Opción {letra}): {_e_autogen}")

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
                                    json_op_key = f"editable_opcion_{letra.lower()}_grafico_json"
                                    st.session_state[json_op_key] = json.dumps([spec], indent=2)
                                # Auto-render
                                try:
                                    buf = crear_grafico(spec.get("tipo_elemento"), spec.get("datos", {}), spec.get("configuracion", {}))
                                    st.session_state[f'img_buffer_op_{letra}'] = buf
                                except Exception:
                                    st.session_state[f'img_buffer_op_{letra}'] = None
                                st.success(f"¡JSON y Gráfico generados!")
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
    # Prioridad: 1) BD Improve importada  2) UI selectores  3) Guardada desde historial
    _tax_saved = st.session_state.get("taxonomia_seleccionada", {}) or {}
    taxonomia_actual = {
        "Grado": st.session_state.get("tax_grado") or _tax_saved.get("Grado"),
        "Área": st.session_state.get("tax_area") or _tax_saved.get("Área"),
        "Componente_Estructura": st.session_state.get("tax_comp1") or _tax_saved.get("Componente_Estructura"),
        "Componente_Tematica": st.session_state.get("tax_comp2") or _tax_saved.get("Componente_Tematica"),
        "Ref. Temática": st.session_state.get("tax_ref") or _tax_saved.get("Ref. Temática"),
        "Competencia": st.session_state.get("tax_competen") or _tax_saved.get("Competencia"),
        "Afirmación": st.session_state.get("tax_afirm") or _tax_saved.get("Afirmación"),
        "Evidencia": st.session_state.get("tax_evid") or _tax_saved.get("Evidencia")
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
        if archivo_word is not None:
            st.download_button(
                label="Descargar en Word (.docx)",
                data=archivo_word,
                file_name="item_espejo_auditado.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        else:
            st.button("Descargar en Word (.docx)", disabled=True, use_container_width=True)
        
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

    # --- 7. CONTROL DE FLUJO PARA BLOQUES EN CONTEXTO ---
    if modo_creacion == "🧩 Generación en Contexto (Bloques)":
        st.divider()
        if st.session_state.item_actual_contexto < st.session_state.total_items_contexto:
            st.info(f"Has terminado el ítem {st.session_state.item_actual_contexto} de {st.session_state.total_items_contexto}.")
            if st.button("Siguiente Ítem ➡️", type="primary", use_container_width=True):
                st.session_state.item_actual_contexto += 1
                st.session_state.show_editor = False # Ocultar el editor
                st.rerun()
        else:
            st.success(f"¡Has completado el bloque de {st.session_state.total_items_contexto} ítems!")
            if st.button("Finalizar Bloque y Crear Nuevo 🏁", type="primary", use_container_width=True):
                st.session_state.etapa_contexto = "CREACION_CONTEXTO"
                st.session_state.contexto_compartido_texto = ""
                st.session_state.contexto_compartido_imagen = None
                st.session_state.contexto_area = None
                st.session_state.contexto_grado = None
                st.session_state.item_actual_contexto = 1
                st.session_state.show_editor = False
                st.rerun()
