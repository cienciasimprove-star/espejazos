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
import random # Necesario para la clave aleatoria

# --- Configuración de Google Cloud (hacer al inicio) ---
# Descomenta esta línea y configúrala con tu proyecto y región
# vertexai.init(project="TU_PROYECTO_GCP", location="TU_REGION")

# --- 1. FUNCIÓN DEL GENERADOR (ACTUALIZADA) ---
def generar_item_llm(imagen_cargada, taxonomia_dict, contexto_adicional, feedback_auditor=""):
    """
    GENERADOR: Genera el ítem Y los datos para un nuevo gráfico/tabla si es necesario.
    """
    
    model = GenerativeModel("gemini-2.5-flash-lite") 
    img_pil = Image.open(imagen_cargada)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    vertex_img = VertexImage.from_bytes(img_bytes)

    taxonomia_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])
    clave_aleatoria = random.choice(['A', 'B', 'C', 'D'])

    seccion_feedback = ""
    if feedback_auditor:
        seccion_feedback = f"""
        --- RETROALIMENTACIÓN DE AUDITORÍA (Error a corregir) ---
        El intento anterior fue rechazado. DEBES corregir los siguientes errores:
        {feedback_auditor}
        --- VUELVE A GENERAR EL ÍTEM CORRIGIENDO ESTO ---
        """

    # 4. Diseño del Prompt (Generador)
    prompt_texto = f"""
    Eres un psicómetra experto en "Shells Cognitivos". Tu tarea es crear un ítem espejo basado en la imagen adjunta, alineado con la taxonomía y el contexto.
    DEBES devolver un JSON válido.

    {seccion_feedback}

    **Shell Cognitivo (Pregunta Original):**
    Analiza la estructura lógica y la "Tarea Cognitiva" de la pregunta en la IMAGEN ADJUNTA. Si la pregunta original usa una tabla o gráfico, tu ítem espejo también debería usar uno de un tipo similar pero con contenido nuevo.

    **Taxonomía Requerida (Tu Guía):**
    {taxonomia_texto}
    
    **Contexto Adicional del Usuario (Tema del ítem nuevo):**
    {contexto_adicional}

    --- ANÁLISIS COGNITIVO OBLIGATORIO (Tu paso 1) ---
    Basado en la taxonomía (Evidencia, Afirmación, Competencia), define la Tarea Cognitiva exacta que el ítem espejo debe evaluar.
    
    --- CONSTRUCCIÓN DEL ÍTEM (Tu paso 2) ---
    Basado en tu análisis, construye el ítem.
    - ENUNCIADO: Debe ser claro y **NO** usar jerarquías ("más", "mejor", "principalmente").
    - OPCIONES: 4 opciones (A, B, C, D).
    - CLAVE: La respuesta correcta DEBE ser la opción **{clave_aleatoria}**.
    - DISTRACTORES: Plausibles, basados en errores comunes de la Tarea Cognitiva.
    - JUSTIFICACIONES:
        - Clave: "Esta es la respuesta correcta porque..."
        - Distractores: "El estudiante podría escoger esta opción porque… Sin embargo, esto es incorrecto porque…"

    --- INSTRUCCIONES DE SALIDA PARA GRÁFICO (¡NUEVO!) ---
    Si el ítem espejo que creaste REQUIERE una tabla, gráfico o diagrama para funcionar, sigue estas reglas:
    
    GRAFICO_NECESARIO: [Escribe "SÍ" o "NO"]
    DESCRIPCION_GRAFICO_NUEVO: [Si es "NO", escribe [] (un array vacío). Si es "SÍ", proporciona una LISTA DE OBJETOS JSON VÁLIDOS que describan el gráfico, siguiendo esta estructura:]
    
    Ejemplo de formato para DESCRIPCION_GRAFICO_NUEVO si GRAFICO_NECESARIO es "SÍ":
    [
      {{
        "ubicacion": "enunciado",
        "tipo_elemento": "tabla",
        "datos": {{
          "columnas": ["Producto", "Precio 2023", "Precio 2024"],
          "filas": [
            ["Manzanas", 1.00, 1.20],
            ["Bananas", 0.50, 0.55]
          ]
        }},
        "configuracion": {{ "titulo": "Precios de Frutas" }},
        "descripcion": "Una tabla que compara los precios de frutas entre 2023 y 2024."
      }}
    ]

    --- FORMATO DE SALIDA OBLIGATORIO (JSON VÁLIDO) ---
    Responde ÚNICAMENTE con el objeto JSON. No incluyas ```json.
    {{
      "pregunta_espejo": "Texto completo del enunciado/stem...",
      "opciones": {{
        "A": "Texto de la opción A",
        "B": "Texto de la opción B",
        "C": "Texto de la opción C",
        "D": "Texto de la opción D"
      }},
      "clave": "{clave_aleatoria}",
      "descripcion_imagen_original": "Descripción de la imagen que el usuario subió...",
      "justificacion_clave": "Razón por la que la clave es correcta...",
      "justificaciones_distractores": [
        {{ "opcion": "A", "justificacion": "Justificación para A..." }},
        {{ "opcion": "B", "justificacion": "Justificación para B..." }},
        {{ "opcion": "C", "justificacion": "Justificación para C..." }},
        {{ "opcion": "D", "justificacion": "Justificación para D..." }}
      ],
      "grafico_necesario": "SÍ" o "NO",
      "descripcion_grafico_nuevo": [ ... (el JSON del gráfico o un array vacío []) ... ]
    }}
    """

    config_generacion = GenerationConfig(
        response_mime_type="application/json"
    )

    try:
        response = model.generate_content(
            [vertex_img, prompt_texto], 
            generation_config=config_generacion
        )
        return response.text 
    except Exception as e:
        st.error(f"Error al contactar Vertex AI (Generador): {e}")
        return None

# --- 2. FUNCIÓN DEL AUDITOR (ACTUALIZADA) ---
def auditar_item_llm(item_json_texto, taxonomia_dict):
    """
    AUDITOR: Audita el ítem Y la coherencia del nuevo gráfico generado.
    """
    
    model = GenerativeModel("gemini-2.5-flash-lite")
    taxonomia_texto = "\n".join([f"* {k}: {v}" for k, v in taxonomia_dict.items()])

    prompt_auditor = f"""
    Eres un auditor psicométrico experto y riguroso. Tu tarea es auditar el siguiente ítem (en JSON)
    contra la taxonomía y las reglas de estilo.
    
    **Taxonomía de Referencia (Obligatoria):**
    {taxonomia_texto}

    **Ítem Generado (JSON a Auditar):**
    {item_json_texto}

    --- CRITERIOS DE AUDITORÍA (Evalúa uno por uno) ---
    1.  **Alineación con Taxonomía:** ¿El ítem (pregunta, opciones, clave) evalúa CLARAMENTE la Evidencia, Afirmación y Competencia de la taxonomía?
    2.  **Estilo del Enunciado (No Jerarquización):** ¿El enunciado usa palabras prohibidas como "más", "mejor", "principalmente"? (RECHAZO automático).
    3.  **Calidad de Distractores:** ¿Las justificaciones de los distractores explican el *error* (ej. "El estudiante podría...")?
    4.  **Clave y Opciones:** ¿Hay 4 opciones? ¿La clave coincide con una opción?
    5.  **Coherencia del Gráfico (¡NUEVO!):** Si "grafico_necesario" es "SÍ", ¿el contenido de "descripcion_grafico_nuevo" es un JSON válido y es *realmente necesario* y *coherente* con la pregunta? Si es "NO", ¿es correcto que no lo tenga?

    --- FORMATO DE SALIDA OBLIGATORIO (JSON VÁLIDO) ---
    Devuelve tu auditoría como un único objeto JSON. No uses ```json.
    {{
      "criterios": [
        {{ "criterio": "1. Alineación con Taxonomía", "estado": "✅ CUMPLE" o "❌ NO CUMPLE", "comentario": "Justificación breve." }},
        {{ "criterio": "2. Estilo (No Jerarquización)", "estado": "✅ CUMPLE" o "❌ NO CUMPLE", "comentario": "Justificación breve." }},
        {{ "criterio": "3. Calidad de Distractores", "estado": "✅ CUMPLE" o "❌ NO CUMPLE", "comentario": "Justificación breve." }},
        {{ "criterio": "4. Clave y Opciones", "estado": "✅ CUMPLE" o "❌ NO CUMPLE", "comentario": "Justificación breve." }},
        {{ "criterio": "5. Coherencia del Gráfico", "estado": "✅ CUMPLE" o "❌ NO CUMPLE", "comentario": "Justificación breve." }}
      ],
      "dictamen_final": "✅ CUMPLE" o "❌ RECHAZADO",
      "observaciones_finales": "Si es RECHAZADO, explica aquí CLARAMENTE qué debe corregir el generador. (Ej: 'El enunciado usa la palabra 'principalmente'. O 'El gráfico es SÍ pero la pregunta no lo usa.')"
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
        return response.text
    except Exception as e:
        st.error(f"Error al contactar Vertex AI (Auditor): {e}")
        return None

# --- 3. FUNCIONES DE EXPORTACIÓN (ACTUALIZADAS) ---

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
    
    # Añadir info del gráfico
    data_rows.append({"Componente": "Gráfico Necesario", "Contenido": datos_generados.get("grafico_necesario", "NO")})
    # Convertir el JSON del gráfico a string para el Excel
    grafico_json_str = json.dumps(datos_generados.get("descripcion_grafico_nuevo", []), indent=2)
    data_rows.append({"Componente": "Datos del Gráfico (JSON)", "Contenido": grafico_json_str})

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
    
    # Añadir info del gráfico (si existe)
    if datos_generados.get("grafico_necesario") == "SÍ":
        document.add_heading('Datos del Gráfico (JSON)', level=3)
        grafico_json_str = json.dumps(datos_generados.get("descripcion_grafico_nuevo", []), indent=2)
        document.add_paragraph(grafico_json_str)

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

# --- 4. INTERFAZ DE STREAMLIT (UI) ---

st.set_page_config(layout="wide")
st.title("🤖 Generador de Ítemes (con Auditoría de IA)")

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

# --- COLUMNA 2 (Lógica de Filtros Bifurcada) ---
with col2:
    st.header("2. Configurar Generación")
    
    excel_file = st.file_uploader("Cargar Excel de Taxonomía (un solo .xlsx)", type=['xlsx'])
    
    grado_sel, area_sel, comp1_sel, comp2_sel, ref_sel, competen_sel, afirm_sel, evid_sel = (None,) * 8
    
    if excel_file is not None:
        try:
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
            
            if 'df1' in st.session_state:
                df1 = st.session_state.df1
                df2 = st.session_state.df2

                # --- Filtros Comunes ---
                grados = df1['Grado'].unique()
                grado_sel = st.selectbox("Grado", options=grados)
                
                df_grado_h1 = df1[df1['Grado'] == grado_sel]
                areas = df_grado_h1['Área'].unique() # Con tilde
                area_sel = st.selectbox("Área", options=areas) # Con tilde

                # --- Cascada 1: (Hoja 1 - Estructura) ---
                st.subheader("Taxonomía (Hoja 1 - Estructura)")
                df_area_h1 = df_grado_h1[df_grado_h1['Área'] == area_sel]
                componentes1 = df_area_h1['Componente1'].unique()
                comp1_sel = st.selectbox("Componente (Estructura)", options=componentes1) 

                df_comp1 = df_area_h1[df_area_h1['Componente1'] == comp1_sel]
                competencias = df_comp1['Competencia'].unique()
                competen_sel = st.selectbox("Competencia", options=competencias)

                df_competencia = df_comp1[df_comp1['Competencia'] == competen_sel]
                
                if area_sel == 'Ciencias Naturales': 
                    df_afirmacion_base = df_competencia[df_competencia['Componente1'] == comp1_sel]
                else:
                    df_afirmacion_base = df_competencia
                    
                afirmaciones = df_afirmacion_base['Afirmación'].unique()
                afirm_sel = st.selectbox("Afirmación", options=afirmaciones)

                df_afirmacion = df_afirmacion_base[df_afirmacion_base['Afirmación'] == afirm_sel]
                evidencias = df_afirmacion['Evidencia'].unique()
                evid_sel = st.selectbox("Evidencia", options=evidencias)

                # --- Cascada 2: (Hoja 2 - Temática) ---
                st.subheader("Taxonomía (Hoja 2 - Temática)")
                df_area_h2 = df2[
                    (df2['Grado'] == grado_sel) & 
                    (df2['Área'] == area_sel) # Con tilde
                ]
                componentes2 = df_area_h2['Componente2'].unique()
                comp2_sel = st.selectbox("Componente (Temática)", options=componentes2)

                df_comp2 = df_area_h2[df_area_h2['Componente2'] == comp2_sel]
                
                refs = df_comp2['Ref. Temática'].unique() if not df_comp2.empty else ["N/A"] # Con tilde y espacio
                ref_sel = st.selectbox("Ref. Temática", options=refs) # Con tilde y espacio

        except KeyError as e:
            st.error(f"Error de Columna: No se encontró la columna {e}. Revisa las tildes/mayúsculas.")
            if 'df1' in st.session_state: st.error(f"Columnas H1: {list(st.session_state.df1.columns)}")
            if 'df2' in st.session_state: st.error(f"Columnas H2: {list(st.session_state.df2.columns)}")
            excel_file = None
        except Exception as e:
            st.error(f"Error inesperado al procesar el Excel: {e}")
            excel_file = None
    
    info_adicional = st.text_area(
        "Contexto Adicional (Tema para el ítem)",
        height=150,
        placeholder="Ej: 'Usar el tema de la fotosíntesis', 'Basarse en la Revolución Francesa'"
    )

# --- 5. LÓGICA DEL BOTÓN (Bucle Generador-Auditor) ---
st.divider()
if st.button("🚀 Generar Ítem Espejo (con Auditoría)", use_container_width=True, type="primary"):
    
    if imagen_subida is None:
        st.warning("Por favor, sube una imagen primero.")
    elif excel_file is None:
        st.warning("Por favor, carga el archivo Excel de taxonomía.")
    elif evid_sel is None or ref_sel is None:
        st.warning("Completa toda la selección de taxonomía.")
    else:
        taxonomia_seleccionada = {
            "Grado": grado_sel,
            "Área": area_sel,
            "Componente1_Estructura": comp1_sel,
            "Componente2_Tematica": comp2_sel,
            "Ref. Temática": ref_sel,
            "Competencia": competen_sel,
            "Afirmación": afirm_sel,
            "Evidencia": evid_sel
        }
        
        max_intentos = 3
        intento_actual = 0
        feedback_auditor = ""
        item_final_json = None

        with st.status("Iniciando proceso...", expanded=True) as status:
            while intento_actual < max_intentos:
                intento_actual += 1
                
                status.update(label=f"Intento {intento_actual}/{max_intentos}: Generando ítem...")
                item_json_str = generar_item_llm(
                    imagen_subida, 
                    taxonomia_seleccionada,
                    info_adicional,
                    feedback_auditor 
                )
                
                if item_json_str is None:
                    status.update(label=f"Error en la generación (Intento {intento_actual}).", state="error")
                    continue 

                status.update(label=f"Intento {intento_actual}/{max_intentos}: Auditando ítem...")
                audit_json_str = auditar_item_llm(item_json_str, taxonomia_seleccionada)

                if audit_json_str is None:
                    status.update(label=f"Error en la auditoría (Intento {intento_actual}).", state="error")
                    continue 

                try:
                    audit_data = json.loads(audit_json_str)
                    
                    if audit_data.get("dictamen_final") == "✅ CUMPLE":
                        status.update(label="¡Auditoría Aprobada!", state="complete")
                        item_final_json = item_json_str
                        break 
                    else:
                        feedback_auditor = audit_data.get("observaciones_finales", "Rechazado sin observaciones.")
                        status.update(label=f"Intento {intento_actual} Rechazado. Preparando re-intento...")
                        st.expander(f"Detalles del Rechazo (Intento {intento_actual})").json(audit_data)
                
                except json.JSONDecodeError:
                    status.update(label="Error al leer respuesta del auditor.", state="error")
                    feedback_auditor = "La respuesta del auditor no fue un JSON válido."

            if item_final_json is None:
                status.update(label=f"No se pudo generar un ítem de alta calidad después de {max_intentos} intentos.", state="error")
                st.error(f"Último feedback del auditor: {feedback_auditor}")
            
        if item_final_json:
            st.success("¡Ítem generado y auditado con éxito! Puedes editarlo abajo.")
            try:
                datos_obj = json.loads(item_final_json)
                st.session_state['resultado_json_obj'] = datos_obj
                
                # --- LÓGICA DE INICIALIZACIÓN (ACTUALIZADA) ---
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
                st.session_state.editable_just_a = justifs_map.get("A", "N/A")
                st.session_state.editable_just_b = justifs_map.get("B", "N/A")
                st.session_state.editable_just_c = justifs_map.get("C", "N/A")
                st.session_state.editable_just_d = justifs_map.get("D", "N/A")
                
                # --- INICIALIZACIÓN DEL GRÁFICO (NUEVO) ---
                st.session_state.editable_grafico_nec = datos_obj.get("grafico_necesario", "NO")
                # Convertir la lista de objetos JSON a un string JSON formateado para el text_area
                grafico_data = datos_obj.get("descripcion_grafico_nuevo", [])
                st.session_state.editable_grafico_json = json.dumps(grafico_data, indent=2)
                
                st.session_state.show_editor = True
                
            except json.JSONDecodeError:
                st.error(f"Error al parsear el JSON final: {item_final_json}")
                st.session_state.show_editor = False

# --- 6. EDITOR DE ÍTEMS Y DESCARGA (ACTUALIZADO) ---
if 'show_editor' in st.session_state and st.session_state.show_editor:
    st.divider()
    st.header("3. Edita el Ítem Generado")
    
    st.text_area("Enunciado (Pregunta Espejo)", key="editable_pregunta", height=150)
    
    # --- CAMPO DE EDICIÓN DEL GRÁFICO (NUEVO) ---
    st.subheader("Gráfico / Tabla del Ítem Espejo")
    st.selectbox(
        "¿Este ítem necesita un gráfico/tabla?", 
        options=["NO", "SÍ"], 
        key="editable_grafico_nec"
    )
    st.text_area(
        "Datos del Gráfico (Editar como JSON)", 
        key="editable_grafico_json", 
        height=200
    )
    
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
    
    # --- LÓGICA DE RE-ENSAMBLE (ACTUALIZADA) ---
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
        ],
        "grafico_necesario": st.session_state.editable_grafico_nec,
    }
    
    # Intentar parsear el JSON del gráfico, si falla, guardar como texto
    try:
        datos_editados["descripcion_grafico_nuevo"] = json.loads(st.session_state.editable_grafico_json)
    except json.JSONDecodeError:
        st.error("El JSON del gráfico tiene un error de formato, se guardará como texto.")
        datos_editados["descripcion_grafico_nuevo"] = st.session_state.editable_grafico_json
    
    
    col_word, col_excel = st.columns(2)
    
    with col_word:
        archivo_word = crear_word(datos_editados)
        st.download_button(
            label="Descargar en Word (.docx)",
            data=archivo_word,
            file_name="item_espejo_auditado.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        
    with col_excel:
        archivo_excel = crear_excel(datos_editados)
        st.download_button(
            label="Descargar en Excel (.xlsx)",
            data=archivo_excel,
            file_name="item_espejo_auditado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
