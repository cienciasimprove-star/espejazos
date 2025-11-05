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

# --- FUNCIÓN DE IA (CORREGIDA) ---
# Ahora acepta 'Componente1' y 'Componente2'
def generar_item_espejo(imagen_cargada, taxonomia_dict, contexto_adicional):
    """
    Llama a Vertex AI (Gemini) para analizar la imagen y el texto
    y generar el nuevo ítem y las justificaciones.
    """
    
    # 1. Inicializar el modelo multimodal
    model = GenerativeModel("gemini-1.5-flash-001") 

    # 2. Cargar la imagen y convertirla para la API
    img_pil = Image.open(imagen_cargada)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    vertex_img = VertexImage.from_bytes(img_bytes)

    # 3. Construir el string de taxonomía para el prompt
    #    (usando los nombres de columna correctos)
    taxonomia_texto = f"""
        * Grado: {taxonomia_dict.get('Grado', 'N/A')}
        * Área: {taxonomia_dict.get('Área', 'N/A')}
        * Componente (Estructura): {taxonomia_dict.get('Componente1', 'N/A')}
        * Componente (Temática): {taxonomia_dict.get('Componente2', 'N/A')}
        * Ref. Temática: {taxonomia_dict.get('Ref. Temática', 'N/A')}
        * Competencia: {taxonomia_dict.get('Competencia', 'N/A')}
        * Afirmación: {taxonomia_dict.get('Afirmación', 'N/A')}
        * Evidencia: {taxonomia_dict.get('Evidencia', 'N/A')}
    """

    # 3. Diseño del Prompt
    prompt_texto = f"""
    Eres un experto en psicometría y diseño de ítems educativos.
    Tu tarea es analizar una pregunta de selección múltiple (presentada como imagen)
    y generar una "pregunta espejo" basada en el concepto de 'shell cognitivo' de Shavelson.

    **Shell Cognitivo (Pregunta Original):**
    Analiza la estructura lógica, el tipo de habilidad cognitiva (la "Tarea Cognitiva")
    y el formato de la pregunta en la imagen adjunta.

    **Taxonomía Requerida:**
    La nueva pregunta debe alinearse con esta taxonomía detallada:
    {taxonomia_texto}

    **Contexto Adicional del Usuario:**
    {contexto_adicional}

    --- INSTRUCCIONES DETALLADAS DE GENERACIÓN ---

    **1. Generar Pregunta Espejo (Enunciado):**
    * Crea una nueva pregunta que mantenga la misma estructura cognitiva (el 'shell') que la pregunta original, pero utiliza un contenido temático diferente.
    * **CRÍTICO:** Escribe **únicamente el enunciado** o 'stem' de la pregunta. NO incluyas las opciones (A, B, C, D) en este campo.
    * Formula preguntas directas como: "**¿Cuál es la causa de...?**", "**¿Qué conclusión se deriva de...?**".
    * Evita preguntas de jerarquía (ej. "**¿cuál es la opción más...**").

    **2. Generar Opciones de Respuesta:**
    * Escribe exactamente cuatro opciones (A, B, C y D).
    * **Opción Correcta**: Debe ser la única conclusión válida tras ejecutar correctamente la Tarea Cognitiva.
    * **Distractores (Incorrectos)**: Deben ser plausibles y diseñados a partir de errores típicos en la ejecución de la Tarea Cognitiva.

    **3. Descripción de Imagen Original:**
    * Si la pregunta original usaba una imagen, genera una descripción textual detallada de esa imagen. Si no hay imagen, indica "N/A".

    **4. Justificaciones (Formato Estricto):**
    * **Justificación de la Clave:** Explica el razonamiento que lleva a la respuesta correcta.
    * **Justificaciones de Distractores:** Para CADA opción, sigue este formato:
        * Si es la clave: "Esta es la respuesta correcta porque..."
        * Si es un distractor: "El estudiante podría escoger esta opción porque… Sin embargo, esto es incorrecto porque…"

    --- FORMATO DE SALIDA OBLIGATORIO (JSON) ---
    Responde ÚNICAMENTE con un objeto JSON válido con la siguiente estructura:
    {{
      "pregunta_espejo": "Texto completo del enunciado/stem...",
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
        {{ "opcion": "A", "justificacion": "Justificación para A..." }},
        {{ "opcion": "B", "justificacion": "Justificación para B..." }},
        {{ "opcion": "C", "justificacion": "Justificación para C..." }},
        {{ "opcion": "D", "justificacion": "Justificación para D..." }}
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
# --- (Sin cambios) ---

def crear_excel(datos_generados):
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
        if just.get('opcion') != datos_generados.get('clave'):
            document.add_paragraph(f"**Justificación {just.get('opcion')}:** {just.get('justificacion')}")
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()

# --- Interfaz de Streamlit ---

st.set_page_config(layout="wide")
st.title("🤖 Generador de Ítemes Espejo (Basado en Shells Cognitivos)")

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

# --- COLUMNA 2 (CORREGIDA CON LÓGICA BIFURCADA) ---
with col2:
    st.header("2. Configurar Generación")
    
    # --- 1. Carga del Excel ---
    excel_file = st.file_uploader("Cargar Excel de Taxonomía (un solo .xlsx)", type=['xlsx'])
    
    # Variables para almacenar las selecciones
    grado_sel = None
    area_sel = None
    comp1_sel = None  # <-- Componente de Hoja 1
    comp2_sel = None  # <-- Componente de Hoja 2
    ref_sel = None
    competen_sel = None
    afirm_sel = None
    evid_sel = None
    
    # --- 2. Lógica de Filtros en Cascada (CORREGIDA) ---
    if excel_file is not None:
        try:
            # Cargar hojas en el estado de la sesión
            if 'df1' not in st.session_state or 'df2' not in st.session_state:
                data = pd.read_excel(excel_file, sheet_name=None)
                sheet_names = list(data.keys())
                if len(sheet_names) < 2:
                    st.error("Error: El archivo Excel debe tener al menos dos hojas.")
                    excel_file = None
                else:
                    st.session_state.df1 = data[sheet_names[0]]
                    st.session_state.df2 = data[sheet_names[1]]
                    st.success(f"Éxito: Cargadas hojas '{sheet_names[0]}' y '{sheet_names[1]}'.")
            
            # Solo continuar si los dataframes están cargados
            if 'df1' in st.session_state:
                df1 = st.session_state.df1
                df2 = st.session_state.df2

                # --- Filtro 1: Grado ---
                grados = df1['Grado'].unique()
                grado_sel = st.selectbox("Grado", options=grados)

                # --- Filtro 2: Area (CON TILDE) ---
                df_grado = df1[df1['Grado'] == grado_sel]
                areas = df_grado['Área'].unique()
                area_sel = st.selectbox("Área", options=areas)

                # --- BIFURCACIÓN DE LÓGICA ---
                
                # --- Cascada 1: (Hoja 1 - Estructura) ---
                st.subheader("Taxonomía (Hoja 1)")
                df_area_h1 = df_grado[df_grado['Área'] == area_sel]
                
                componentes1 = df_area_h1['Componente'].unique()
                comp1_sel = st.selectbox("Componente (Estructura)", options=componentes1) # <-- COMPONENTE 1

                df_comp1 = df_area_h1[df_area_h1['Componente'] == comp1_sel]
                competencias = df_comp1['Competencia'].unique()
                competen_sel = st.selectbox("Competencia", options=competencias)

                df_competencia = df_comp1[df_comp1['Competencia'] == competen_sel]
                
                # Lógica de Ciencias Naturales (usa comp1_sel)
                if area_sel == 'Ciencias Naturales': 
                    df_afirmacion_base = df_competencia[df_competencia['Componente'] == comp1_sel]
                else:
                    df_afirmacion_base = df_competencia
                    
                afirmaciones = df_afirmacion_base['Afirmación'].unique()
                afirm_sel = st.selectbox("Afirmación", options=afirmaciones)

                df_afirmacion = df_afirmacion_base[df_afirmacion_base['Afirmación'] == afirm_sel]
                evidencias = df_afirmacion['Evidencia'].unique()
                evid_sel = st.selectbox("Evidencia", options=evidencias)

                # --- Cascada 2: (Hoja 2 - Temática) ---
                st.subheader("Taxonomía (Hoja 2)")
                df_area_h2 = df2[
                    (df2['Grado'] == grado_sel) & 
                    (df2['Área'] == area_sel)
                ]
                
                componentes2 = df_area_h2['Componente'].unique()
                comp2_sel = st.selectbox("Componente (Temática)", options=componentes2) # <-- COMPONENTE 2

                df_comp2 = df_area_h2[df_area_h2['Componente'] == comp2_sel]
                
                if not df_comp2.empty:
                    refs = df_comp2['Ref. Temática'].unique()
                else:
                    refs = ["N/A"]
                ref_sel = st.selectbox("Ref. Temática", options=refs)

        except KeyError as e:
            st.error(f"Error de Columna: No se encontró la columna {e}. Revisa que los nombres en el Excel coincidan exactamente (incluyendo tildes y mayúsculas).")
            # Imprimir columnas para ayudar a depurar
            if 'df1' in st.session_state:
                 st.error(f"Columnas Hoja 1: {list(st.session_state.df1.columns)}")
            if 'df2' in st.session_state:
                 st.error(f"Columnas Hoja 2: {list(st.session_state.df2.columns)}")
            excel_file = None
        except Exception as e:
            st.error(f"Error inesperado al procesar el Excel. Detalle: {e}")
            excel_file = None
    
    # --- 3. Info Adicional (como estaba) ---
    info_adicional = st.text_area(
        "Información adicional (ej. tema específico, contexto)",
        height=150,
        placeholder="Ej: 'Usar el tema de fotosíntesis', 'Enfocar en estudiantes de grado 10'"
    )

# --- Botón de Generación (CORREGIDO) ---
st.divider()
if st.button("🚀 Generar Ítem Espejo", use_container_width=True, type="primary"):
    
    # --- Validaciones ---
    if imagen_subida is None:
        st.warning("Por favor, sube una imagen primero.")
    elif excel_file is None:
        st.warning("Por favor, carga el archivo Excel de taxonomía.")
    elif evid_sel is None or ref_sel is None: # Si los últimos filtros no están seteados
        st.warning("Error en los filtros de taxonomía. Revisa el Excel y las selecciones.")
    
    else:
        # --- Empaquetar la taxonomía (CORREGIDO) ---
        taxonomia_seleccionada = {
            "Grado": grado_sel,
            "Área": area_sel,
            "Componente1": comp1_sel,  # <-- Componente de Hoja 1
            "Componente2": comp2_sel,  # <-- Componente de Hoja 2
            "Ref. Temática": ref_sel,
            "Competencia": competen_sel,
            "Afirmación": afirm_sel,
            "Evidencia": evid_sel
        }
        
        # --- Llamar a la función de IA con los nuevos parámetros ---
        resultado_generado_texto = generar_item_espejo(
            imagen_subida, 
            taxonomia_seleccionada, # Pasa el diccionario
            info_adicional
        )
        
        if resultado_generado_texto:
            st.success("¡Ítem generado con éxito! Puedes editarlo abajo.")
            try:
                # --- LÓGICA DE INICIALIZACIÓN (Sin cambios) ---
                datos_obj = json.loads(resultado_generado_texto)
                st.session_state['resultado_json_obj'] = datos_obj
                st.session_state.editable_pregunta = datos_obj.get("pregunta_espejo", "")
                opciones = datos_obj.get("opciones", {})
                st.session_state.editable_opcion_a = opciones.get("A", "")
                st.session_state.editable_opcion_b = opciones.get("B", "")
                st.session_state.editable_opcion_c = opciones.get("C", "")
                st.session_state.editable_opcion_d = opciones.get("D", "")
                st.session_state.editable_clave = datos_obj.get("clave", "")
                st.session_state.editable_just_clave = datos_obj.get("justificacion_clave", "")
                justifs_list = datos_obj.get("justificaciones_distractores", [])
                justifs_map = {j.get('opcion'): j.get('justificacion') for j in justifs_list}
                st.session_state.editable_just_a = justifs_map.get("A", "Justificación para A no generada.")
                st.session_state.editable_just_b = justifs_map.get("B", "Justificación para B no generada.")
                st.session_state.editable_just_c = justifs_map.get("C", "Justificación para C no generada.")
                st.session_state.editable_just_d = justifs_map.get("D", "Justificación para D no generada.")
                st.session_state.show_editor = True
                
            except json.JSONDecodeError:
                st.error("Error: La respuesta de la IA no fue un JSON válido.")
                st.text(resultado_generado_texto) # Mostrar el texto crudo para depurar
                st.session_state.show_editor = False

# --- Editor de Ítems y Descarga (Sin cambios) ---
if 'show_editor' in st.session_state and st.session_state.show_editor:
    st.divider()
    st.header("3. Edita el Ítem Generado")
    
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

    # --- SECCIÓN DE DESCARGA ---
    st.divider()
    st.header("4. Descargar Resultados")
    
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
        archivo_word = crear_word(datos_editados)
        st.download_button(
            label="Descargar en Word (.docx)",
            data=archivo_word,
            file_name="item_espejo_editado.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        
    with col_excel:
        archivo_excel = crear_excel(datos_editados)
        st.download_button(
            label="Descargar en Excel (.xlsx)",
            data=archivo_excel,
            file_name="item_espejo_editado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
