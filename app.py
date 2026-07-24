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
# CARGA DEL MODELO
# ==========================================================

DIRECTORIO_BASE = Path(__file__).resolve().parent
RUTA_MODELO = DIRECTORIO_BASE / "best.pt"

if not RUTA_MODELO.is_file():
    raise FileNotFoundError(
        f"No se encontró el archivo {RUTA_MODELO.name}. "
        "Sube best.pt en la misma carpeta que app.py."
    )

modelo = YOLO(str(RUTA_MODELO))

if modelo.task != "detect":
    raise ValueError(
        f"El archivo cargado corresponde a una tarea '{modelo.task}', "
        "pero la aplicación necesita un modelo de detección."
    )

print(f"Modelo cargado correctamente: {RUTA_MODELO.name}")
print(f"Clases disponibles: {modelo.names}")
print(f"GPU disponible: {torch.cuda.is_available()}")


# Valores internos recomendados para mantener la interfaz sencilla.
CONFIANZA_MINIMA = 0.25
UMBRAL_IOU = 0.35
RESOLUCION = 960
COLUMNAS = [
    "N.º",
    "Instrumento identificado",
    "Nivel de reconocimiento"
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
        <div class="empty-icon">SV</div>
        <div>
            <div class="empty-title">Listo para comenzar</div>
            <div class="empty-description">
                Cargue o tome una fotografía y presione
                <b>Identificar instrumentos</b>.
            </div>
        </div>
    </div>
    """


def resumen_sin_detecciones():
    return """
    <div class="result-card result-empty">
        <div class="result-title">
            No fue posible identificar instrumentos
        </div>
        <div class="result-description">
            Tome una nueva fotografía con mayor iluminación,
            acerque los instrumentos a la cámara y evite que estén
            demasiado superpuestos.
        </div>
        <div class="tag-container">
            <span class="tag tag-empty">Sin instrumentos identificados</span>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">CANTIDAD IDENTIFICADA</div>
            <div class="metric-value">0</div>
            <div class="metric-detail">Revise la fotografía e inténtelo nuevamente</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">RESULTADO DEL ANÁLISIS</div>
            <div class="metric-value status-small">Sin identificación</div>
            <div class="metric-detail">No se encontraron coincidencias suficientes</div>
        </div>
    </div>
    """


def crear_resumen(detecciones, conteo):
    total = len(detecciones)
    mejor = max(detecciones, key=lambda elemento: elemento["confianza_valor"])

    etiquetas = "".join(
        f'<span class="tag">{cantidad} × {escapar(nombre)}</span>'
        for nombre, cantidad in conteo.items()
    )

    texto_total = (
        "Se identificó 1 instrumento."
        if total == 1
        else f"Se identificaron {total} instrumentos."
    )

    return f"""
    <div class="result-card">
        <div class="result-title">Instrumentos identificados</div>
        <div class="result-description">
            {texto_total} Revise la imagen marcada y confirme que
            el nombre y la cantidad coincidan con el instrumental observado.
        </div>
        <div class="tag-container">{etiquetas}</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">CANTIDAD IDENTIFICADA</div>
            <div class="metric-value">{total}</div>
            <div class="metric-detail">
                {len(conteo)} tipo(s) de instrumento
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">RECONOCIMIENTO MÁS SEGURO</div>
            <div class="metric-value">{mejor["confianza_valor"] * 100:.1f}%</div>
            <div class="metric-detail">
                {escapar(mejor["Instrumento identificado"])}
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
            arreglo = cv2.cvtColor(arreglo, cv2.COLOR_GRAY2RGB)
        elif arreglo.shape[-1] == 4:
            arreglo = cv2.cvtColor(arreglo, cv2.COLOR_RGBA2RGB)

        imagen_pil = Image.fromarray(arreglo.astype(np.uint8))

    return ImageOps.exif_transpose(imagen_pil).convert("RGB")


# ==========================================================
# IDENTIFICACIÓN DE INSTRUMENTOS
# ==========================================================

def analizar_imagen(imagen):
    if imagen is None:
        return (
            None,
            estado_inicial(),
            tabla_vacia(),
            "Primero debe cargar o tomar una fotografía."
        )

    try:
        imagen_pil = normalizar_imagen(imagen)
        imagen_rgb = np.asarray(imagen_pil)
        dispositivo = 0 if torch.cuda.is_available() else "cpu"

        resultados = modelo.predict(
            source=imagen_rgb,
            conf=CONFIANZA_MINIMA,
            iou=UMBRAL_IOU,
            imgsz=RESOLUCION,
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
        imagen_salida = Image.fromarray(imagen_anotada_rgb)

        detecciones = []
        conteo = Counter()

        if resultado.boxes is not None and len(resultado.boxes) > 0:
            clases = resultado.boxes.cls.cpu().numpy().astype(int)
            confianzas = resultado.boxes.conf.cpu().numpy()

            for numero, (clase_id, confianza) in enumerate(
                zip(clases, confianzas),
                start=1
            ):
                nombre = modelo.names[int(clase_id)]
                conteo[nombre] += 1

                detecciones.append({
                    "N.º": numero,
                    "Instrumento identificado": nombre,
                    "Nivel de reconocimiento": f"{confianza * 100:.1f}%",
                    "confianza_valor": float(confianza)
                })

        if detecciones:
            resumen = crear_resumen(detecciones, conteo)

            tabla = pd.DataFrame([
                {
                    columna: deteccion[columna]
                    for columna in COLUMNAS
                }
                for deteccion in detecciones
            ])

            detalle = (
                "Análisis completado. Confirme visualmente los nombres y "
                "cantidades antes de registrar, organizar o preparar el instrumental."
            )
        else:
            resumen = resumen_sin_detecciones()
            tabla = tabla_vacia()
            detalle = (
                "No se identificaron instrumentos. Pruebe con una fotografía "
                "más cercana, bien iluminada y con menos superposición."
            )

        return imagen_salida, resumen, tabla, detalle

    except Exception as error:
        return (
            imagen,
            f"""
            <div class="error-box">
                <b>No fue posible completar el análisis.</b><br>
                {escapar(error)}
            </div>
            """,
            tabla_vacia(),
            "Revise la imagen e inténtelo nuevamente."
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
# ESTILO VISUAL
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
    background: linear-gradient(110deg, #0a294a 0%, #176783 57%, #118878 100%);
    border-radius: 17px;
    padding: 24px 26px;
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
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    border: 1px solid rgba(255,255,255,.25);
    background: rgba(255,255,255,.12);
    font-weight: 900;
}

.header-title {
    font-size: 28px;
    font-weight: 900;
}

.header-subtitle {
    margin-top: 5px;
    font-size: 14px;
    color: rgba(255,255,255,.92);
}

.audience {
    display: inline-block;
    margin-top: 15px;
    padding: 6px 11px;
    border: 1px solid rgba(255,255,255,.28);
    border-radius: 9px;
    background: rgba(255,255,255,.1);
    font-size: 11px;
}

.panel {
    background: white;
    border: 1px solid var(--borde);
    border-radius: 15px;
    padding: 18px;
    min-height: 690px;
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
    line-height: 1.5;
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
    line-height: 1.5;
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
    grid-template-columns: repeat(2, 1fr);
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
    font-size: 10px;
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
    line-height: 1.55;
}

.warning {
    margin-top: 16px;
    border: 1px solid #efb771;
    border-radius: 12px;
    background: #fff8ef;
    padding: 14px;
    color: #8a4a1b;
    font-size: 11px;
    line-height: 1.5;
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
# INTERFAZ
# ==========================================================

with gr.Blocks(title="SurgiVision AI", css=CSS) as pagina:

    gr.HTML(
        """
        <div class="surgi-header">
            <div class="header-row">
                <div class="logo">SV</div>
                <div>
                    <div class="header-title">SurgiVision AI</div>
                    <div class="header-subtitle">
                        Identificación y conteo de instrumentos quirúrgicos mediante imágenes
                    </div>
                </div>
            </div>

            <div class="audience">
                Herramienta de apoyo para enfermería, internos y médicos
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=True):

        with gr.Column(scale=5, elem_classes=["panel"]):
            gr.HTML(
                """
                <div class="step">PASO 1</div>
                <div class="section-title">Cargar o tomar una fotografía</div>
                <div class="section-description">
                    Utilice una imagen en la que los instrumentos se observen completos,
                    separados y con buena iluminación.
                </div>
                """
            )

            imagen_entrada = gr.Image(
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                label="Fotografía del instrumental"
            )

            with gr.Row():
                boton_analizar = gr.Button(
                    "Identificar instrumentos",
                    variant="primary",
                    elem_id="analyze-button"
                )

                boton_limpiar = gr.Button(
                    "Nuevo análisis",
                    elem_id="clear-button"
                )

            gr.HTML(
                """
                <div class="recommendation">
                    <b>Para obtener mejores resultados:</b><br>
                    • coloque los instrumentos sobre una superficie uniforme;<br>
                    • evite reflejos, sombras fuertes y superposiciones;<br>
                    • mantenga todos los instrumentos dentro de la fotografía.
                </div>
                """
            )

        with gr.Column(scale=7, elem_classes=["panel"]):
            gr.HTML(
                """
                <div class="step">PASO 2</div>
                <div class="section-title">Revisar el resultado</div>
                <div class="section-description">
                    La aplicación marcará cada instrumento identificado y mostrará
                    su nombre, cantidad y nivel de reconocimiento.
                </div>
                """
            )

            imagen_resultado = gr.Image(
                label="Instrumentos identificados",
                interactive=False
            )

            resumen_resultado = gr.HTML(estado_inicial())

            tabla_resultados = gr.Dataframe(
                headers=COLUMNAS,
                datatype=["number", "str", "str"],
                label="Resumen de instrumentos",
                interactive=False,
                wrap=True
            )

    with gr.Accordion("Orientación para revisar el resultado", open=False):
        informacion = gr.Textbox(
            value=(
                "Los resultados deben compararse con la fotografía original. "
                "Antes de registrar, organizar o preparar el instrumental, "
                "confirme visualmente el nombre y la cantidad."
            ),
            show_label=False,
            interactive=False,
            lines=3
        )

    gr.HTML(
        """
        <div class="warning">
            <b>Importante:</b> SurgiVision AI es una herramienta de apoyo para
            la identificación y el conteo de instrumentos quirúrgicos.
            El personal responsable debe verificar visualmente los resultados.
            La aplicación no sustituye los protocolos institucionales de control,
            registro y seguridad.
        </div>
        """
    )

    boton_analizar.click(
        fn=analizar_imagen,
        inputs=[imagen_entrada],
        outputs=[
            imagen_resultado,
            resumen_resultado,
            tabla_resultados,
            informacion
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
            informacion
        ]
    )


# ==========================================================
# EJECUCIÓN EN RENDER
# ==========================================================

if __name__ == "__main__":
    pagina.queue(default_concurrency_limit=1, max_size=10)

    pagina.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "10000")),
        share=False,
        debug=False,
        show_error=True
    )
