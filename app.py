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

# --- Función Placeholder para llamar a Vertex AI (ACTUALIZADA) ---
def generar_item_espejo(imagen_cargada, taxonomia, contexto_adicional):
    """
    Llama a Vertex AI (Gemini) para analizar la imagen y el texto
    y generar el nuevo ítem y las justificaciones.
    """
    
    # 1. Inicializar el modelo multimodal
    model = GenerativeModel("gemini-2.5-flash-lite") 

    # 2. Cargar la imagen y convertirla para la API
    img_pil = Image.open(imagen_cargada)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    vertex_img = VertexImage.from_bytes(img_bytes)

    # 3. Diseño del Prompt (ACTUALIZADO CON NUEVAS REGLAS)
    prompt_texto = f"""
    Eres un experto en psicometría y diseño de ítems educativos.
    Tu tarea es analizar una pregunta de selección múltiple (presentada como imagen)
    y generar una "pregunta espejo" basada en el concepto de 'shell cognitivo' de Shavelson.

    **Shell Cognitivo (Pregunta Original):**
    Analiza la estructura lógica, el tipo de habilidad cognitiva (la "Tarea Cognitiva")
    y el formato de la pregunta en la imagen adjunta.

    **Taxonomía Requerida:**
    La nueva pregunta debe alinearse con esta taxonomía: {taxonomia}

    **Contexto Adicional del Usuario:**
    {contexto_adicional}

    --- INSTRUCCIONES DETALLADAS DE GENERACIÓN ---

    **1. Generar Pregunta Espejo (Enunciado):**
    * Crea una nueva pregunta que mantenga la misma estructura cognitiva (el 'shell') que la pregunta original, pero utiliza un contenido temático diferente.
    * Asegúrate de que la dificultad y la habilidad medida (la Tarea Cognitiva) sean equivalentes.
    * **CRÍTICO:** Escribe **únicamente el enunciado** o 'stem' de la pregunta. NO incluyas las opciones (A, B, C, D) en este campo.
    * Formula una pregunta clara, directa, sin ambigüedades ni tecnicismos innecesarios.
    * **¡INSTRUCCIÓN CRÍTICA DE ESTILO!** Evita terminantemente formular preguntas que pidan al estudiante comparar o jerarquizar opciones. **NO USES** frases como "¿cuál es la opción más...", "¿cuál es el mejor...", "¿cuál describe principalmente...?".
    * En su lugar, formula preguntas directas como: "**¿Cuál es la causa de...?**", "**¿Qué conclusión se deriva de...?**".
    * Si utilizas negaciones, resáltalas en MAYÚSCULAS Y NEGRITA (por ejemplo: **NO ES**, **EXCEPTO**).

    **2. Generar Opciones de Respuesta:**
    * Escribe exactamente cuatro opciones (A, B, C y D).
    * **Opción Correcta**: Debe ser la única conclusión válida tras ejecutar correctamente la Tarea Cognitiva (el 'shell').
    * **Distractores (Incorrectos)**: Deben ser plausibles y diseñados a partir de errores típicos en la ejecución de la Tarea Cognitiva (Ej: un distractor podría ser el resultado de aplicar un proceso cognitivo inferior, como simplemente recordar un dato, en lugar de analizarlo).
    * Las respuestas deben tener una estructura gramatical y longitud similares.
    * No utilices fórmulas vagas como “ninguna de las anteriores” o “todas las anteriores”.

    **3. Descripción de Imagen Original:**
    * Si la pregunta original usaba una imagen, genera una descripción textual detallada de esa imagen. Si no hay imagen, indica "N/A".

    **4. Justificaciones (Formato Estricto):**
    * Para la NUEVA pregunta espejo que generaste:
    * **Justificación de la Clave:** Explica detalladamente el razonamiento o proceso cognitivo que lleva a la respuesta correcta. NO justifiques por descarte.
    * **Justificaciones de Distractores:** Para CADA opción (incluida la correcta, para el mapeo), sigue este formato:
        * Si la opción es la clave: "Esta es la respuesta correcta porque..." (repites la justificación de la clave).
        * Si la opción es un distractor: "El estudiante podría escoger esta opción porque… Sin embargo, esto es incorrecto porque…"

    --- FORMATO DE SALIDA OBLIGATORIO (JSON) ---
    Responde ÚNICAMENTE con un objeto JSON válido con la siguiente estructura (esta estructura es fija para que la aplicación funcione):
    {{
      "pregunta_espejo": "Texto completo del enunciado/stem de la nueva pregunta...",
      "opciones": {{
        "A": "Texto de la opción A",
        "B": "Texto de la opción B",
        "C": "Texto de la opción C",
        "D": "Texto de la opción D"
      }},
      "clave": "A",
      "descripcion_imagen_original": "Descripción de la imagen en la pregunta de entrada...",
      "justificacion_clave": "Razón por la que la clave es correcta (sigue el formato estricto)...",
      "justificaciones_distractores": [
        {{ "opcion": "A", "justificacion": "Justificación para A (sigue el formato estricto)..." }},
        {{ "opcion": "B", "justificacion": "Justificación para B (sigue el formato estricto)..." }},
        {{ "opcion": "C", "justificacion": "Justificación para C (sigue el formato estricto)..." }},
        {{ "opcion": "D", "justificacion": "Justificación para D (sigue el formato estricto)..." }}
      ]
    }}
    """

    # 4. Realizar la llamada multimodal
    st.info("Generando ítem... esto puede tardar un momento.")
    
    try:
        response = model.generate_content([vertex_img, prompt_texto])
        
        # Es crucial limpiar el 'markdown' que a veces añade el modelo
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
# --- NUEVA SECCIÓN: Editor de Ítems (a 1 Columna) ---
# Esta sección solo aparece si show_editor es True
if 'show_editor' in st.session_state and st.session_state.show_editor:
    st.divider()
    st.header("3. Edita el Ítem Generado")
    
    # --- Todos los campos se apilarán verticalmente ---
    
    st.text_area("Enunciado (Pregunta Espejo)", key="editable_pregunta", height=150)
    
    st.subheader("Opciones")
    st.text_input("Opción A", key="editable_opcion_a")
    st.text_input("Opción B", key="editable_opcion_b")
    st.text_input("Opción C", key="editable_opcion_c")
    st.text_input("Opción D", key="editable_opcion_d")
        
    st.subheader("Clave")
    st.text_input("Clave (Respuesta Correcta)", key="editable_clave")

    st.subheader("Justificaciones")
    st.text_area("Justificación Clave", key="editable_just_clave", height=100)
    st.text_area("Justificación A", key="editable_just_a", height=100)
    st.text_area("Justificación B", key="editable_just_b", height=100)
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
