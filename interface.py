import io
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import mlflow.pyfunc
from parsers.jdx import parse_jdx
from config import AUTH_URI

def list_models():
    client = mlflow.tracking.MlflowClient(tracking_uri=AUTH_URI)
    return [m.name for m in client.search_registered_models()]

def list_versions(model_name):
    client = mlflow.tracking.MlflowClient(tracking_uri=AUTH_URI)
    filter_str = f"name = '{model_name}'"
    mvs = client.search_model_versions(filter_str)
    versions = sorted({mv.version for mv in mvs}, key=lambda v: int(v))
    return versions

def predict(file, model_name, model_version, threshold=0.5, nitro=False):
    data = parse_jdx(file.name)
    x, y = data['x'], data['y']

    model_uri = f"models:/{model_name}/{model_version}"
    mlflow.set_tracking_uri(AUTH_URI)
    model = mlflow.pyfunc.load_model(model_uri)
    arr = np.asarray(y, dtype=float)
    spectra = arr.reshape(1, 1024)
    params = {"threshold": threshold, "nitro": bool(nitro)}
    result = model.predict(data=spectra, params=params)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Wavenumber (cm^-1)')
    plt.ylabel('Absorbance')
    plt.title('IR Spectrum')
    plt.gca().invert_xaxis()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    fg_list = result[0]["positive_targets"]
    probs = result[0]["positive_probabilities"]
    
    output_table = [[fg, round(float(p), 4)] for fg, p in zip(fg_list, probs)]
    
    return img, output_table

with gr.Blocks() as interface:
    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input & Settings")
                spectrum = gr.File(label="Upload .jdx File", file_types=['.jdx'])
                with gr.Accordion("Advanced Settings", open=False):
                    model_name = gr.Dropdown(choices=list_models(), value="ircnn_ftir_fcg", label="Model")
                    model_version = gr.Dropdown(choices=list_versions("ircnn_ftir_fcg"), value="14", label="Version")
                    threshold= gr.Number(value=0.5, label="Threshold", minimum=0, maximum=1, step=0.01)
                    nitro = gr.Checkbox(value=False, label="Nitro")
                predict_btn = gr.Button("Predict")

            with gr.Column(scale=2):
                gr.Markdown("### Prediction Output")
                spectrum_image = gr.Image(label="Spectrum")
                fcn_groups_output = gr.Dataframe(label="Functional Groups", headers=["Group", "Probability"], datatype=["str", "number"])

        predict_btn.click(
            fn=predict,
            inputs=[spectrum, model_name, model_version, threshold, nitro],
            outputs=[spectrum_image, fcn_groups_output]
        )

if __name__ == "__main__":
    interface.launch(share=False)
