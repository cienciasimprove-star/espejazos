import streamlit as st
from PIL import Image
import io
import base64
import pandas as pd
from docx import Document
# Importa la librería de Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image as VertexImage

# --- Configuración de Google Cloud (hacer al inicio) ---
# vertexai.init(project="tu-proyecto-gcp", location="tu-region")

# --- Función Placeholder para llamar a Vertex AI ---
# Esta es la función central (Punto 4)
def generar_item_espejo(imagen_cargada, taxonomia, contexto_adicional):
    """
    Llama a Vertex AI (Gemini) para analizar la imagen y el texto
    y generar el nuevo ítem y las justificaciones.
    """
    
    # 1. Inicializar el modelo multimodal (ej. Gemini 1.5 Pro)
    model = GenerativeModel("gemini-2.5-flash-lite") 

    # 2. Cargar la imagen y convertirla para la API
    # imagen_cargada es el objeto de st.file_uploader
    img_pil = Image.open(imagen_cargada)
    
    # Convertir PIL Image a bytes
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    # Crear el objeto de imagen para Vertex AI
    vertex_img = VertexImage.from_bytes(img_bytes)

    # 3. Diseño del Prompt (La parte más importante)
    # Aquí integramos la idea de "Shells Cognitivos"
    prompt_texto = f"""
    Eres un experto en psicometría y diseño de ítems educativos.
    Tu tarea es analizar una pregunta de selección múltiple (presentada como imagen)
    y generar una "pregunta espejo" basada en el concepto de 'shell cognitivo' de Shavelson.

    **Shell Cognitivo (Pregunta Original):**
    Analiza la estructura lógica, el tipo de habilidad cognitiva evaluada y el formato
    de la pregunta en la imagen adjunta.

    **Taxonomía Requerida:**
    La nueva pregunta debe alinearse con esta taxonomía: {taxonomia}

    **Contexto Adicional del Usuario:**
    {contexto_adicional}

    **Instrucciones de Generación:**

    1.  **Generar Pregunta Espejo (Punto 4.1):**
        Crea una nueva pregunta que mantenga la misma estructura cognitiva (el 'shell')
        que la pregunta original, pero utiliza un contenido temático diferente.
        Asegúrate de que la dificultad y la habilidad medida sean equivalentes.
        Presenta la pregunta completa con sus opciones (A, B, C, D...).

    2.  **Descripción de Imagen (Punto 4.2):**
        Si la pregunta original usaba una imagen, genera una descripción textual 
        detallada de esa imagen y describe qué tipo de imagen se necesitaría
        para la nueva "pregunta espejo" (si aplica). Si no hay imagen, indica "N/A".

    3.  **Justificaciones (Punto 4.3):**
        Para la NUEVA pregunta espejo que generaste:
        * Identifica la clave (respuesta correcta).
        * Escribe una justificación detallada de por qué la clave es correcta.
        * Escribe justificaciones detalladas para CADA una de las opciones no válidas
            (distractores), explicando el error conceptual que representa cada una.

    **Formato de Salida (JSON):**
    Responde ÚNICAMENTE con un objeto JSON válido con la siguiente estructura:
    {{
      "pregunta_espejo": "Texto completo de la nueva pregunta...",
      "opciones": {{
        "A": "Texto de la opción A",
        "B": "Texto de la opción B",
        "C": "Texto de la opción C",
        "D": "Texto de la opción D"
      }},
      "clave": "A",
      "descripcion_imagen_original": "Descripción de la imagen en la pregunta de entrada...",
      "justificacion_clave": "Razón por la que la clave es correcta...",
      "justificaciones_distractores": [
        {{ "opcion": "A", "justificacion": "Por qué A es incorrecta..." }},
        {{ "opcion": "B", "justificacion": "Por qué B es incorrecta..." }},
        {{ "opcion": "C", "justificacion": "Por qué C es incorrecta..." }},
        {{ "opcion": "D", "justificacion": "Por qué D es incorrecta..." }}
      ]
    }}
    """

    # 4. Realizar la llamada multimodal
    st.info("Generando ítem... esto puede tardar un momento.")
    
    try:
        # Combinar la imagen y el prompt de texto
        response = model.generate_content([vertex_img, prompt_texto])
        
        # Asumiendo que la respuesta es un JSON como se solicitó
        # Es crucial limpiar el 'markdown' que a veces añade el modelo
        respuesta_texto = response.text.strip().replace("```json", "").replace("```", "")
        
        # Aquí deberías parsear el JSON (import json)
        # Por simplicidad, aquí solo devolvemos el texto
        return respuesta_texto 

    except Exception as e:
        st.error(f"Error al contactar Vertex AI: {e}")
        return None

# --- Funciones de Exportación (Punto 5) ---

def crear_excel(datos_generados):
    # Aquí 'datos_generados' debería ser el JSON parseado
    # Esto es un ejemplo simplificado
    df = pd.DataFrame({
        'Componente': ['Pregunta Espejo', 'Clave', 'Justificación Clave'],
        'Contenido': [
            datos_generados.get("pregunta_espejo", ""),
            datos_generados.get("clave", ""),
            datos_generados.get("justificacion_clave", "")
        ]
    })
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Item Generado')
    
    return output.getvalue()

def crear_word(datos_generados):
    # Aquí 'datos_generados' debería ser el JSON parseado
    document = Document()
    document.add_heading('Ítem Espejo Generado', level=1)
    
    document.add_heading('Pregunta Espejo', level=2)
    document.add_paragraph(datos_generados.get("pregunta_espejo", "N/A"))
    
    document.add_heading('Opciones', level=3)
    opciones = datos_generados.get("opciones", {})
    for letra, texto in opciones.items():
        document.add_paragraph(f"**{letra}:** {texto}")

    document.add_heading('Justificaciones', level=2)
    document.add_paragraph(f"**Clave:** {datos_generados.get('clave', 'N/A')}")
    document.add_paragraph(f"**Justificación de la Clave:** {datos_generados.get('justificacion_clave', 'N/A')}")
    
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()

# --- Interfaz de Streamlit ---

st.set_page_config(layout="wide")
st.title("🤖 Generador de Ítems Espejo (Basado en Shells Cognitivos)")

# --- Columnas para la entrada ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Cargar Ítem Original")
    # (Punto 1)
    imagen_subida = st.file_uploader(
        "Sube el pantallazo de la pregunta", 
        type=["png", "jpg", "jpeg"]
    )
    
    if imagen_subida:
        st.image(imagen_subida, caption="Ítem cargado", use_column_width=True)

with col2:
    st.header("2. Configurar Generación")
    
    # (Punto 2)
    # Debes pre-cargar tu lista de taxonomías
    TAXONOMIAS_PRECARGADAS = [
        "Recordar (Bloom)",
        "Comprender (Bloom)",
        "Aplicar (Bloom)",
        "Analizar (Bloom)",
        "Evaluar (Bloom)",
        "Crear (Bloom)",
        "Otro Nivel Taxonómico"
    ]
    taxonomia_sel = st.selectbox(
        "Selecciona la taxonomía del ítem", 
        options=TAXONOMIAS_PRECARGADAS
    )
    
    # (Punto 3)
    info_adicional = st.text_area(
        "Información adicional (ej. tema específico, contexto)",
        height=150,
        placeholder="Ej: 'Usar el tema de fotosíntesis', 'Enfocar en estudiantes de grado 10'"
    )

# --- Botón de Generación ---
st.divider()
if st.button("🚀 Generar Ítem Espejo", use_container_width=True, type="primary"):
    if imagen_subida is not None:
        # (Punto 4)
        # Aquí se llama a la función de Vertex AI
        resultado_generado = generar_item_espejo(
            imagen_subida, 
            taxonomia_sel, 
            info_adicional
        )
        
        if resultado_generado:
            st.success("¡Ítem generado con éxito!")
            # Guardamos el resultado en el estado de la sesión
            # Asumiendo que 'resultado_generado' es el texto JSON
            # En un caso real, aquí deberías parsear el JSON
            st.session_state['resultado_json_texto'] = resultado_generado
            st.session_state['resultado_json_obj'] = pd.io.json.loads(resultado_generado) # Parsear
            
            # Mostrar la salida
            st.json(st.session_state['resultado_json_obj'])

    else:
        st.warning("Por favor, sube una imagen primero.")

# --- Sección de Descarga (Punto 5) ---
if 'resultado_json_obj' in st.session_state:
    st.divider()
    st.header("3. Descargar Resultados")
    
    datos_obj = st.session_state['resultado_json_obj']
    
    col_word, col_excel = st.columns(2)
    
    with col_word:
        archivo_word = crear_word(datos_obj)
        st.download_button(
            label="Descargar en Word (.docx)",
            data=archivo_word,
            file_name="item_espejo.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        
    with col_excel:
        archivo_excel = crear_excel(datos_obj)
        st.download_button(
            label="Descargar en Excel (.xlsx)",
            data=archivo_excel,
            file_name="item_espejo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
