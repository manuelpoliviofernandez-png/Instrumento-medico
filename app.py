import os
import html
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO


# ==========================================================
# CONFIGURACIÓN GENERAL Y CARGA DEL MODELO
# ==========================================================

DIRECTORIO_BASE = Path(__file__).resolve().parent
RUTA_MODELO = DIRECTORIO_BASE / "best.pt"

if not RUTA_MODELO.is_file():
    raise FileNotFoundError(
        f"No se encontró el modelo YOLO en: {RUTA_MODELO}. "
        "Sube el archivo best.pt en la misma carpeta que app.py."
    )

# El modelo se carga una sola vez al iniciar la aplicación.
modelo = YOLO(str(RUTA_MODELO))

if modelo.task != "detect":
    raise ValueError(
        f"El modelo cargado es de tipo '{modelo.task}', no de detección."
    )

print(f"Modelo cargado correctamente: {RUTA_MODELO.name}")
print(f"Tipo de tarea: {modelo.task}")
print(f"Clases: {modelo.names}")
print(f"GPU disponible: {torch.cuda.is_available()}")

COLUMNAS = [
    "N.º",
    "Instrumento",
    "Confianza",
    "x1",
    "y1",
    "x2",
    "y2"
]


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================

def escapar(valor):
    return html.escape(str(valor))


def tabla_vacia():
    return pd.DataFrame(columns=COLUMNAS)


def estado_inicial():
    return """
    <div class="empty-result">

        <div class="empty-icon">
            SV
        </div>

        <div>
            <div class="empty-title">
                Esperando análisis
            </div>

            <div class="empty-description">
                Carga una imagen y presiona
                <b>Analizar imagen</b>.
            </div>
        </div>

    </div>
    """


def resumen_sin_detecciones(confianza):
    return f"""
    <div class="result-card result-empty">

        <div class="result-title">
            Instrumentos reconocidos
        </div>

        <div class="result-description">
            No se encontraron detecciones con una confianza
            mínima de {float(confianza) * 100:.0f}%.
        </div>

        <div class="tag-container">
            <span class="tag tag-empty">
                Sin detecciones
            </span>
        </div>

    </div>

    <div class="metrics-grid">

        <div class="metric-card">

            <div class="metric-label">
                INSTRUMENTOS<br>DETECTADOS
            </div>

            <div class="metric-value">
                0
            </div>

            <div class="metric-detail">
                Ninguna caja aceptada
            </div>

        </div>

        <div class="metric-card">

            <div class="metric-label">
                MAYOR<br>PROBABILIDAD
            </div>

            <div class="metric-value">
                —
            </div>

            <div class="metric-detail">
                Sin resultados
            </div>

        </div>

        <div class="metric-card">

            <div class="metric-label">
                ESTADO
            </div>

            <div class="metric-value status-small">
                No detectado
            </div>

            <div class="metric-detail">
                Ajusta la imagen o el umbral
            </div>

        </div>

    </div>
    """


def crear_resumen(detecciones, conteo):
    total = len(detecciones)

    if total == 0:
        return resumen_sin_detecciones(0)

    mejor = max(
        detecciones,
        key=lambda elemento: elemento["confianza_valor"]
    )

    etiquetas = "".join(
        f"""
        <span class="tag">
            {cantidad} × {escapar(nombre)}
        </span>
        """
        for nombre, cantidad in conteo.items()
    )

    if total == 1:
        texto_total = "Se detectó 1 instrumento."
    else:
        texto_total = f"Se detectaron {total} instrumentos."

    return f"""
    <div class="result-card">

        <div class="result-title">
            Instrumentos reconocidos
        </div>

        <div class="result-description">
            {texto_total}
        </div>

        <div class="tag-container">
            {etiquetas}
        </div>

    </div>

    <div class="metrics-grid">

        <div class="metric-card">

            <div class="metric-label">
                INSTRUMENTOS<br>DETECTADOS
            </div>

            <div class="metric-value">
                {total}
            </div>

            <div class="metric-detail">
                {len(conteo)} clase(s) diferente(s)
            </div>

        </div>

        <div class="metric-card">

            <div class="metric-label">
                MAYOR<br>PROBABILIDAD
            </div>

            <div class="metric-value">
                {mejor["confianza_valor"] * 100:.1f}%
            </div>

            <div class="metric-detail">
                {escapar(mejor["Instrumento"])}
            </div>

        </div>

        <div class="metric-card">

            <div class="metric-label">
                ESTADO
            </div>

            <div class="metric-value status-small">
                Detectado
            </div>

            <div class="metric-detail">
                Resultado positivo
            </div>

        </div>

    </div>
    """


def normalizar_imagen(imagen):
    if imagen is None:
        raise ValueError("No se recibió ninguna imagen.")

    if isinstance(imagen, Image.Image):
        imagen_pil = imagen

    else:
        arreglo = np.asarray(imagen)

        if arreglo.ndim == 2:
            arreglo = cv2.cvtColor(
                arreglo,
                cv2.COLOR_GRAY2RGB
            )

        elif arreglo.shape[-1] == 4:
            arreglo = cv2.cvtColor(
                arreglo,
                cv2.COLOR_RGBA2RGB
            )

        imagen_pil = Image.fromarray(
            arreglo.astype(np.uint8)
        )

    imagen_pil = ImageOps.exif_transpose(
        imagen_pil
    ).convert("RGB")

    return imagen_pil


# ==========================================================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN
# ==========================================================

def analizar_imagen(
    imagen,
    confianza_minima,
    umbral_iou,
    resolucion
):
    if imagen is None:
        return (
            None,
            estado_inicial(),
            tabla_vacia(),
            "Primero debes cargar una imagen."
        )

    try:
        imagen_pil = normalizar_imagen(imagen)
        imagen_rgb = np.asarray(imagen_pil)

        dispositivo = 0 if torch.cuda.is_available() else "cpu"

        resultados = modelo.predict(
            source=imagen_rgb,
            conf=float(confianza_minima),
            iou=float(umbral_iou),
            imgsz=int(resolucion),

            # Desactivado para evitar cajas duplicadas
            augment=False,

            max_det=50,
            agnostic_nms=False,
            device=dispositivo,
            verbose=False
        )

        resultado = resultados[0]

        imagen_anotada_bgr = resultado.plot(
            labels=True,
            conf=True,
            line_width=2
        )

        imagen_anotada_rgb = cv2.cvtColor(
            imagen_anotada_bgr,
            cv2.COLOR_BGR2RGB
        )

        imagen_salida = Image.fromarray(
            imagen_anotada_rgb
        )

        detecciones = []
        conteo = Counter()

        if (
            resultado.boxes is not None
            and len(resultado.boxes) > 0
        ):
            clases = (
                resultado.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confianzas = (
                resultado.boxes.conf
                .cpu()
                .numpy()
            )

            coordenadas = (
                resultado.boxes.xyxy
                .cpu()
                .numpy()
            )

            for numero, (
                clase_id,
                confianza,
                caja
            ) in enumerate(
                zip(
                    clases,
                    confianzas,
                    coordenadas
                ),
                start=1
            ):
                nombre = modelo.names[int(clase_id)]

                x1, y1, x2, y2 = caja

                conteo[nombre] += 1

                detecciones.append({
                    "N.º": numero,
                    "Instrumento": nombre,
                    "Confianza": (
                        f"{confianza * 100:.1f}%"
                    ),
                    "x1": round(float(x1)),
                    "y1": round(float(y1)),
                    "x2": round(float(x2)),
                    "y2": round(float(y2)),
                    "confianza_valor": float(confianza)
                })

        if detecciones:
            resumen = crear_resumen(
                detecciones,
                conteo
            )

            tabla = pd.DataFrame([
                {
                    columna: deteccion[columna]
                    for columna in COLUMNAS
                }
                for deteccion in detecciones
            ])

            desglose = "; ".join(
                f"{cantidad} × {nombre}"
                for nombre, cantidad in conteo.items()
            )

            informacion = (
                f"Confianza mínima: "
                f"{float(confianza_minima):.2f}. "
                f"IoU: {float(umbral_iou):.2f}. "
                f"Resolución: {int(resolucion)} píxeles. "
                f"TTA desactivado. "
                f"Conteo obtenido: {desglose}."
            )

        else:
            resumen = resumen_sin_detecciones(
                confianza_minima
            )

            tabla = tabla_vacia()

            informacion = (
                "El modelo no produjo cajas que superaran "
                f"la confianza mínima de "
                f"{float(confianza_minima):.2f}. "
                "Prueba con 0.10 o 0.15, una imagen más "
                "cercana o una resolución de 960."
            )

        return (
            imagen_salida,
            resumen,
            tabla,
            informacion
        )

    except Exception as error:
        return (
            imagen,
            f"""
            <div class="error-box">
                <b>Error durante la predicción:</b><br>
                {escapar(error)}
            </div>
            """,
            tabla_vacia(),
            "No fue posible completar el análisis."
        )


def limpiar():
    return (
        None,
        None,
        estado_inicial(),
        tabla_vacia(),
        ""
    )


# ==========================================================
# CSS
# ==========================================================

CSS = """
:root {
    --azul: #0b2d50;
    --azul-medio: #176986;
    --verde: #13bbaa;
    --verde-oscuro: #078f82;
    --verde-claro: #e9fff7;
    --fondo: #f1f6f9;
    --borde: #d2dfe7;
    --texto: #071f3a;
    --secundario: #607b8e;
}

body {
    background: var(--fondo) !important;
}

.gradio-container {
    max-width: 1260px !important;
    margin: auto !important;
    padding: 12px 18px 30px !important;
    background: var(--fondo) !important;
}

.surgi-header {
    background: linear-gradient(
        110deg,
        #0a294a 0%,
        #176783 57%,
        #118878 100%
    );

    border-radius: 17px;
    padding: 22px 25px;
    color: white;
    margin-bottom: 16px;
    box-shadow: 0 7px 0 rgba(21, 71, 96, 0.07);
}

.header-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

.logo {
    width: 47px;
    height: 47px;
    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 13px;
    border: 1px solid rgba(255,255,255,.25);
    background: rgba(255,255,255,.12);

    font-weight: 900;
}

.header-title {
    font-size: 27px;
    font-weight: 900;
}

.header-subtitle {
    margin-top: 4px;
    font-size: 13px;
    color: rgba(255,255,255,.9);
}

.badge {
    display: inline-block;
    margin-top: 15px;
    padding: 5px 10px;

    border: 1px solid rgba(255,255,255,.28);
    border-radius: 9px;
    background: rgba(255,255,255,.1);

    font-size: 10px;
}

.panel {
    background: white;
    border: 1px solid var(--borde);
    border-radius: 15px;
    padding: 18px;
    min-height: 710px;
}

.step {
    color: var(--verde-oscuro);
    font-size: 11px;
    font-weight: 900;
}

.section-title {
    margin-top: 4px;
    color: var(--texto);
    font-size: 19px;
    font-weight: 900;
}

.section-description {
    margin: 5px 0 17px;
    color: var(--secundario);
    font-size: 12px;
}

.result-card {
    background: var(--verde-claro);
    border: 1px solid #64e1ad;
    border-radius: 14px;
    padding: 18px;
    margin: 14px 0 18px;
}

.result-empty {
    background: #f8fafb;
    border-color: var(--borde);
}

.result-title {
    color: var(--texto);
    font-size: 15px;
    font-weight: 900;
}

.result-description {
    color: #355266;
    font-size: 12px;
    margin-top: 5px;
}

.tag-container {
    margin-top: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}

.tag {
    border: 1px solid #28ce91;
    background: white;
    color: #08755f;

    padding: 5px 10px;
    border-radius: 999px;

    font-size: 11px;
    font-weight: 800;
}

.tag-empty {
    border-color: #b9c8d1;
    color: #687f8e;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 11px;
    margin-bottom: 18px;
}

.metric-card {
    border: 1px solid var(--borde);
    background: white;
    border-radius: 12px;
    padding: 13px;
    min-height: 100px;
}

.metric-label {
    color: #527990;
    font-size: 10px;
    font-weight: 900;
}

.metric-value {
    margin-top: 7px;
    color: var(--texto);
    font-size: 22px;
    font-weight: 900;
}

.status-small {
    font-size: 18px;
}

.metric-detail {
    margin-top: 3px;
    color: #668196;
    font-size: 9px;
}

.empty-result {
    border: 1px dashed #b8cad4;
    border-radius: 13px;
    padding: 20px;
    background: #f7fafc;

    display: flex;
    align-items: center;
    gap: 12px;
}

.empty-icon {
    width: 38px;
    height: 38px;

    border-radius: 10px;
    background: #dfeaf0;

    display: flex;
    align-items: center;
    justify-content: center;

    color: #37647d;
    font-weight: 900;
}

.empty-title {
    font-weight: 900;
    color: var(--texto);
}

.empty-description {
    font-size: 11px;
    color: var(--secundario);
    margin-top: 3px;
}

.recommendation {
    margin-top: 12px;

    border: 1px solid var(--borde);
    border-radius: 12px;
    background: #f7fafc;

    padding: 13px;
    color: #4c687a;
    font-size: 11px;
}

.warning {
    margin-top: 16px;

    border: 1px solid #efb771;
    border-radius: 12px;
    background: #fff8ef;

    padding: 14px;
    color: #9d511c;
    font-size: 11px;
}

.error-box {
    padding: 16px;

    border: 1px solid #e3a2a2;
    border-radius: 12px;
    background: #fff1f1;

    color: #9b3030;
}

#analyze-button {
    background: var(--verde) !important;
    border: none !important;
    color: white !important;
    font-weight: 900 !important;
}

#analyze-button:hover {
    background: var(--verde-oscuro) !important;
}

#clear-button {
    background: white !important;
    border: 1px solid var(--borde) !important;
    color: var(--texto) !important;
    font-weight: 800 !important;
}

@media (max-width: 800px) {
    .metrics-grid {
        grid-template-columns: 1fr;
    }

    .panel {
        min-height: auto;
    }
}
"""


# ==========================================================
# INTERFAZ COMPLETA
# ==========================================================

with gr.Blocks(
    title="SurgiVision AI",
    css=CSS
) as pagina:

    gr.HTML(
        """
        <div class="surgi-header">

            <div class="header-row">

                <div class="logo">
                    SV
                </div>

                <div>
                    <div class="header-title">
                        SurgiVision AI
                    </div>

                    <div class="header-subtitle">
                        Detección y conteo inteligente de
                        instrumentos quirúrgicos
                    </div>
                </div>

            </div>

            <div class="badge">
                PROTOTIPO ACADÉMICO · DETECCIÓN DE OBJETOS YOLO
            </div>

        </div>
        """
    )

    with gr.Row(equal_height=True):

        # ==================================================
        # COLUMNA IZQUIERDA
        # ==================================================

        with gr.Column(
            scale=5,
            elem_classes=["panel"]
        ):

            gr.HTML(
                """
                <div class="step">
                    PASO 1
                </div>

                <div class="section-title">
                    Cargar imagen
                </div>

                <div class="section-description">
                    Utiliza una fotografía clara, centrada
                    y con iluminación uniforme.
                </div>
                """
            )

            imagen_entrada = gr.Image(
                type="pil",
                sources=[
                    "upload",
                    "webcam",
                    "clipboard"
                ],
                label="Vista previa de la imagen"
            )

            with gr.Accordion(
                "Configuración de predicción",
                open=False
            ):

                confianza = gr.Slider(
                    minimum=0.10,
                    maximum=0.80,
                    value=0.25,
                    step=0.05,
                    label="Confianza mínima"
                )

                iou = gr.Slider(
                    minimum=0.20,
                    maximum=0.80,
                    value=0.35,
                    step=0.05,
                    label="Umbral IoU"
                )

                resolucion = gr.Dropdown(
                    choices=[
                        640,
                        768,
                        960,
                        1280
                    ],
                    value=960,
                    label="Resolución de análisis"
                )

            with gr.Row():

                boton_analizar = gr.Button(
                    "Analizar imagen",
                    variant="primary",
                    elem_id="analyze-button"
                )

                boton_limpiar = gr.Button(
                    "Limpiar",
                    elem_id="clear-button"
                )

            gr.HTML(
                """
                <div class="recommendation">

                    <b>Recomendación:</b>
                    utiliza imágenes claras, instrumentos
                    completos y evita superposiciones excesivas.
                    Para comenzar usa confianza 0.25,
                    IoU 0.35 y resolución 960.

                </div>
                """
            )

        # ==================================================
        # COLUMNA DERECHA
        # ==================================================

        with gr.Column(
            scale=7,
            elem_classes=["panel"]
        ):

            gr.HTML(
                """
                <div class="step">
                    PASO 2
                </div>

                <div class="section-title">
                    Resultado del análisis
                </div>

                <div class="section-description">
                    Cada caja corresponde a un instrumento
                    localizado por el modelo.
                </div>
                """
            )

            imagen_resultado = gr.Image(
                label="Detecciones",
                interactive=False
            )

            resumen_resultado = gr.HTML(
                estado_inicial()
            )

            tabla_resultados = gr.Dataframe(
                headers=COLUMNAS,

                datatype=[
                    "number",
                    "str",
                    "str",
                    "number",
                    "number",
                    "number",
                    "number"
                ],

                label="Resultados por instrumento",
                interactive=False,
                wrap=True
            )

    with gr.Accordion(
        "Información técnica del análisis",
        open=False
    ):

        informacion_tecnica = gr.Textbox(
            show_label=False,
            interactive=False,
            lines=4
        )

    gr.HTML(
        """
        <div class="warning">

            <b>Aviso:</b>
            SurgiVision AI es una prueba de concepto académica.
            No constituye un dispositivo médico, no reemplaza
            la comprobación manual protocolizada y no debe
            emplearse para decisiones clínicas.

        </div>
        """
    )

    boton_analizar.click(
        fn=analizar_imagen,

        inputs=[
            imagen_entrada,
            confianza,
            iou,
            resolucion
        ],

        outputs=[
            imagen_resultado,
            resumen_resultado,
            tabla_resultados,
            informacion_tecnica
        ]
    )

    boton_limpiar.click(
        fn=limpiar,
        inputs=[],

        outputs=[
            imagen_entrada,
            imagen_resultado,
            resumen_resultado,
            tabla_resultados,
            informacion_tecnica
        ]
    )


# ==========================================================
# EJECUTAR LA PÁGINA
# ==========================================================

if __name__ == "__main__":
    # Limita la concurrencia para evitar varias inferencias simultáneas.
    pagina.queue(default_concurrency_limit=1, max_size=10)

    pagina.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        share=False,
        debug=False,
        show_error=True
    )
