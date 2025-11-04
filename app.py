import streamlit as st
from PIL import Image
import io
import base64
import pandas as pd
from docx import Document
from docx.shared import Inches
# Importa la librería de Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image as VertexImage
import json

# --- Configuración de Google Cloud (hacer al inicio) ---
# Descomenta esta línea y configúrala con tu proyecto y región
# vertexai.init(project="tu-proyecto-gcp", location="tu-region")

# --- Función Placeholder para llamar a Vertex AI ---
def generar_item_espejo(imagen_cargada, taxonomia, contexto_adicional):
    """
    Llama a Vertex AI (Gemini) para analizar la imagen y el texto
    y generar el nuevo ítem y las justificaciones.
    """
    
    # 1. Inicializar el modelo multimodal
    # Nota: "gemini-2.5-flash-lite" puede ser un nombre de modelo no final.
    # Asegúrate de usar un modelo multimodal disponible en tu proyecto,
    # como "gemini-1.5-pro-001" o "gemini-1.5-flash-001"
    model = GenerativeModel("gemini-1.5-flash-001") 

    # 2. Cargar la imagen y convertirla para la API
    img_pil = Image.open(imagen_cargada)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    vertex_img = VertexImage.from_bytes(img_bytes)

    # 3. Diseño del Prompt
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
        response = model.generate_content([vertex_img, prompt_texto])
        respuesta_texto = response.text.strip().replace("```json", "").replace("```", "")
        return respuesta_texto 

    except Exception as e:
        st.error(f"Error al contactar Vertex AI: {e}")
        return None

# --- Funciones de Exportación (Punto 5) ---
# --- ACTUALIZADAS PARA INCLUIR TODOS LOS CAMPOS ---

def crear_excel(datos_generados):
    # 'datos_generados' es el diccionario con los datos (posiblemente editados)
    
    # Crear una lista de filas para el DataFrame
    data_rows = []
    
    data_rows.append({"Componente": "Pregunta Espejo", "Contenido": datos_generados.get("pregunta_espejo", "")})
    
    opciones = datos_generados.get("opciones", {})
    for letra, texto in opciones.items():
        data_rows.append({"Componente": f"Opción {letra}", "Contenido": texto})
        
    data_rows.append({"Componente": "Clave", "Contenido": datos_generados.get("clave", "")})
    data_rows.append({"Componente": "Justificación Clave", "Contenido": datos_generados.get("justificacion_clave", "")})
    
    justificaciones = datos_generados.get("justificaciones_distractores", [])
    for just in justificaciones:
        data_rows.append({"Componente": f"Justificación {just.get('opcion')}", "Contenido": just.get('justificacion')})

    df = pd.DataFrame(data_rows)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Item Generado')
    
    return output.getvalue()

def crear_word(datos_generados):
    # 'datos_generados' es el diccionario con los datos (posiblemente editados)
    document = Document()
    document.add_heading('Ítem Espejo Generado', level=1)
    
    document.add_heading('Pregunta Espejo (Enunciado)', level=2)
    document.add_paragraph(datos_generados.get("pregunta_espejo", "N/A"))
    
    document.add_heading('Opciones', level=3)
    opciones = datos_generados.get("opciones", {})
    for letra, texto in opciones.items():
        document.add_paragraph(f"**{letra}:** {texto}")

    document.add_heading('Clave', level=2)
    document.add_paragraph(datos_generados.get('clave', 'N/A'))
    
    document.add_heading('Justificaciones', level=2)
    document.add_paragraph(f"**Justificación de la Clave:** {datos_generados.get('justificacion_clave', 'N/A')}")
    
    document.add_heading('Justificaciones de Distractores', level=3)
    justificaciones = datos_generados.get("justificaciones_distractores", [])
    for just in justificaciones:
        # No justificar la clave dos veces
        if just.get('opcion') != datos_generados.get('clave'):
            document.add_paragraph(f"**Justificación {just.get('opcion')}:** {just.get('justificacion')}")
    
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
    imagen_subida = st.file_uploader(
        "Sube el pantallazo de la pregunta", 
        type=["png", "jpg", "jpeg"]
    )
    
    if imagen_subida:
        st.image(imagen_subida, caption="Ítem cargado", use_container_width=True)

with col2:
    st.header("2. Configurar Generación")
    
    TAXONOMIAS_PRECARGADAS = [
        "Recordar (Bloom)", "Comprender (Bloom)", "Aplicar (Bloom)",
        "Analizar (Bloom)", "Evaluar (Bloom)", "Crear (Bloom)",
        "Otro Nivel Taxonómico"
    ]
    taxonomia_sel = st.selectbox(
        "Selecciona la taxonomía del ítem", 
        options=TAXONOMIAS_PRECARGADAS
    )
    
    info_adicional = st.text_area(
        "Información adicional (ej. tema específico, contexto)",
        height=150,
        placeholder="Ej: 'Usar el tema de fotosíntesis', 'Enfocar en estudiantes de grado 10'"
    )

# --- Botón de Generación ---
st.divider()
if st.button("🚀 Generar Ítem Espejo", use_container_width=True, type="primary"):
    if imagen_subida is not None:
        resultado_generado_texto = generar_item_espejo(
            imagen_subida, 
            taxonomia_sel, 
            info_adicional
        )
        
        if resultado_generado_texto:
            st.success("¡Ítem generado con éxito! Puedes editarlo abajo.")
            try:
                # --- LÓGICA DE INICIALIZACIÓN ---
                datos_obj = json.loads(resultado_generado_texto)
                
                # Guardar el objeto original por si acaso
                st.session_state['resultado_json_obj'] = datos_obj
                
                # Inicializar el estado para cada campo editable
                st.session_state.editable_pregunta = datos_obj.get("pregunta_espejo", "")
                
                opciones = datos_obj.get("opciones", {})
                st.session_state.editable_opcion_a = opciones.get("A", "")
                st.session_state.editable_opcion_b = opciones.get("B", "")
                st.session_state.editable_opcion_c = opciones.get("C", "")
                st.session_state.editable_opcion_d = opciones.get("D", "")
                
                st.session_state.editable_clave = datos_obj.get("clave", "")
                st.session_state.editable_just_clave = datos_obj.get("justificacion_clave", "")
                
                # Mapear justificaciones de distractores
                justifs_list = datos_obj.get("justificaciones_distractores", [])
                justifs_map = {j.get('opcion'): j.get('justificacion') for j in justifs_list}
                
                st.session_state.editable_just_a = justifs_map.get("A", "Justificación para A no generada.")
                st.session_state.editable_just_b = justifs_map.get("B", "Justificación para B no generada.")
                st.session_state.editable_just_c = justifs_map.get("C", "Justificación para C no generada.")
                st.session_state.editable_just_d = justifs_map.get("D", "Justificación para D no generada.")
                
                # Bandera para mostrar el editor
                st.session_state.show_editor = True
                
            except json.JSONDecodeError:
                st.error("Error: La respuesta de la IA no fue un JSON válido.")
                st.text(resultado_generado_texto) # Mostrar el texto crudo para depurar
                st.session_state.show_editor = False
    else:
        st.warning("Por favor, sube una imagen primero.")

# --- NUEVA SECCIÓN: Editor de Ítems ---
# Esta sección solo aparece si show_editor es True
if 'show_editor' in st.session_state and st.session_state.show_editor:
    st.divider()
    st.header("3. Edita el Ítem Generado")
    
    # Campo para el Enunciado
    st.text_area("Enunciado (Pregunta Espejo)", key="editable_pregunta", height=150)
    
    # Columnas para Opciones
    st.subheader("Opciones")
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        st.text_input("Opción A", key="editable_opcion_a")
        st.text_input("Opción B", key="editable_opcion_b")
    with opt_col2:
        st.text_input("Opción C", key="editable_opcion_c")
        st.text_input("Opción D", key="editable_opcion_d")
        
    # Campo para la Clave
    st.text_input("Clave (Respuesta Correcta)", key="editable_clave")

    # Columnas para Justificaciones
    st.subheader("Justificaciones")
    just_col1, just_col2 = st.columns(2)
    with just_col1:
        st.text_area("Justificación Clave", key="editable_just_clave", height=100)
        st.text_area("Justificación A", key="editable_just_a", height=100)
        st.text_area("Justificación B", key="editable_just_b", height=100)
    with just_col2:
        st.text_area("Justificación C", key="editable_just_c", height=100)
        st.text_area("Justificación D", key="editable_just_d", height=100)

    # --- SECCIÓN DE DESCARGA (AHORA DEPENDE DE LOS DATOS EDITADOS) ---
    st.divider()
    st.header("4. Descargar Resultados")
    
    # --- LÓGICA DE RE-ENSAMBLE ---
    # Re-construir el diccionario 'datos' a partir del session_state
    # Esto asegura que los datos descargados sean los datos editados
    datos_editados = {
        "pregunta_espejo": st.session_state.editable_pregunta,
        "opciones": {
            "A": st.session_state.editable_opcion_a,
            "B": st.session_state.editable_opcion_b,
            "C": st.session_state.editable_opcion_c,
            "D": st.session_state.editable_opcion_d,
        },
        "clave": st.session_state.editable_clave,
        "justificacion_clave": st.session_state.editable_just_clave,
        "justificaciones_distractores": [
            {"opcion": "A", "justificacion": st.session_state.editable_just_a},
            {"opcion": "B", "justificacion": st.session_state.editable_just_b},
            {"opcion": "C", "justificacion": st.session_state.editable_just_c},
            {"opcion": "D", "justificacion": st.session_state.editable_just_d},
        ]
        # Nota: "descripcion_imagen_original" no se hizo editable,
        # pero podría añadirse si es necesario.
    }
    
    col_word, col_excel = st.columns(2)
    
    with col_word:
        # Pasar los datos editados a la función de creación
        archivo_word = crear_word(datos_editados)
        st.download_button(
            label="Descargar en Word (.docx)",
            data=archivo_word,
            file_name="item_espejo_editado.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        
    with col_excel:
        # Pasar los datos editados a la función de creación
        archivo_excel = crear_excel(datos_editados)
        st.download_button(
            label="Descargar en Excel (.xlsx)",
            data=archivo_excel,
            file_name="item_espejo_editado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
