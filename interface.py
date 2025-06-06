import os
import gradio as gr
import plotly.graph_objects as go
import mlflow
import mlflow.pyfunc
import pandas as pd

from parsers.jdx import parse_jdx
from config import AUTH_URI

CACHE_DIR = os.getenv("MLFLOW_MODEL_CACHE", "model_cache")

def list_models() -> list:
    client = mlflow.tracking.MlflowClient(tracking_uri=AUTH_URI)
    return [m.name for m in client.search_registered_models()]

def list_versions(model_name: str):
    client = mlflow.tracking.MlflowClient(tracking_uri=AUTH_URI)
    filter_str = f"name = '{model_name}'"
    mvs = client.search_model_versions(filter_str)
    versions = sorted({mv.version for mv in mvs}, key = lambda v: int(v))
    return gr.update(choices = versions, value = versions[0] if versions else None)

def get_local_model_path(model_name: str, model_version: str) -> str:
    local_path = os.path.join(CACHE_DIR, model_name, model_version)
    if not os.path.isdir(local_path):
        os.makedirs(local_path, exist_ok=True)
        model_uri = f"models:/{model_name}/{model_version}"
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=local_path
        )
    return local_path

def predict(file: gr.File, model_name: str, model_version: str, threshold: float = 0.5, fn_groups: list = None):
    mlflow.set_tracking_uri(AUTH_URI)
    local_model_path = get_local_model_path(model_name, model_version)
    model = mlflow.pyfunc.load_model(local_model_path)

    data = parse_jdx(file.name)
    x, y = data['x'], data['y']
    df = pd.DataFrame([{'spectrum_x': x, 'spectrum_y': y}])

    params = {"threshold": threshold}
    if fn_groups:
        for fg in fn_groups:
            params[fg] = True

    result = model.predict(data = df, params = params)
    attention = result[0]["attention"]
    fg_list = result[0]["positive_targets"]
    probs = result[0]["positive_probabilities"]
    output_table = sorted([[fg, round(float(p), 4)] for fg, p in zip(fg_list, probs)], key = lambda x:x[1], reverse = True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = x, y = y,
        mode = 'lines',
        name = 'IR Spectrum',
        hovertemplate = 'Wavenumber: %{x} cm⁻¹<br>Absorbance: %{y}<extra></extra>'
    ))
    fig.update_layout(
        title = 'IR Spectrum',
        xaxis = dict(title='Wavenumber (cm⁻¹)', autorange='reversed'),
        yaxis = dict(title='Absorbance'),
        hovermode = 'closest'
    )

    return gr.update(choices = fg_list, value = None), fig, attention, x, y, output_table, fg_list

def highlight_group(selected_group: str, all_groups: list, attention: list, x: list, y: list):
    if selected_group is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = x, y = y,
            mode = 'lines',
            name = 'IR Spectrum',
            hovertemplate = 'Wavenumber: %{x} cm⁻¹<br>Absorbance: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title = 'IR Spectrum',
            xaxis = dict(title='Wavenumber (cm⁻¹)', autorange = 'reversed'),
            yaxis = dict(title='Absorbance'),
            hovermode = 'closest'
        )
        return fig
    idx = all_groups.index(selected_group)
    group_attention = attention[idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name='IR Spectrum',
        hovertemplate='Wavenumber: %{x} cm⁻¹<br>Absorbance: %{y}<extra></extra>'
    ))

    y_min, y_max = min(y), max(y)
    for (start, end, importance) in group_attention:
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=y_min,
            y1=y_max,
            fillcolor=f"rgba(125, 0, 0, {importance / 2})",
            line=dict(width=0),
            layer="below"
        )

    fig.update_layout(
        title=f'IR Spectrum — Highlighting: {selected_group}',
        xaxis=dict(title='Wavenumber (cm⁻¹)', autorange='reversed'),
        yaxis=dict(title='Absorbance'),
        hovermode='closest'
    )
    return fig

with gr.Blocks(title="IR Spectrum Functional Group Predictor") as interface:
    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input & Settings")
                spectrum = gr.File(label="Upload .jdx File", file_types=['.jdx'])
                with gr.Accordion("Advanced Settings", open=False):
                    model_list = list_models()
                    model_name = gr.Dropdown(choices = model_list, value = model_list[0] if model_list else None, label = "Model")
                    model_version = gr.Dropdown(choices = [], label = "Version")
                    threshold= gr.Slider(value = 0.5, label = "Threshold", minimum = 0, maximum = 1, step = 0.01)
                    fn_groups = gr.CheckboxGroup(
                        choices = ["alkane","methyl","alkene","alkyne","alcohols","amines","nitriles","aromatics","alkyl halides","esters","ketones","aldehydes","carboxylic acids","ether","acyl halides","amides","nitro"],
                        label = "Functional Groups"
                    )
                predict_btn = gr.Button("Predict")

            with gr.Column(scale=2):
                gr.Markdown("### Prediction Output")
                group_dropdown = gr.Dropdown(choices = [], value = None, allow_custom_value = True, label = "Select Functional Group", interactive = True)
                spectrum_image = gr.Plot(label = "Spectrum")
                fcn_groups_output = gr.Dataframe(label = "Functional Groups", headers = ["Group", "Probability"], datatype = ["str", "number"])

        all_groups_state = gr.State()
        attention_state = gr.State()
        x_state = gr.State()
        y_state = gr.State()

        model_name.change(fn = list_versions, inputs = model_name, outputs = model_version)
    
        predict_btn.click(
            fn = predict,
            inputs = [spectrum, model_name, model_version, threshold, fn_groups],
            outputs = [group_dropdown, spectrum_image, attention_state, x_state, y_state, fcn_groups_output, all_groups_state]
        )

        group_dropdown.change(
            fn = highlight_group,
            inputs = [group_dropdown, all_groups_state, attention_state, x_state, y_state],
            outputs = [spectrum_image]
        )
    
    with gr.Tab("Compare Models"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input & Settings")
                spectrum_cmp = gr.File(label="Upload .jdx File", file_types=['.jdx'])
                model_list = list_models()

                gr.Markdown("### Model A Settings")
                model_a_name = gr.Dropdown(choices=model_list, value=model_list[0] if model_list else None, label="Model A")
                model_a_version = gr.Dropdown(choices=[], label="Version A")
                threshold_a = gr.Slider(value=0.5, label="Threshold A", minimum=0, maximum=1, step=0.01)
                nitro_a = gr.Checkbox(value=False, label="Nitro A")

                gr.Markdown("### Model B Settings")
                model_b_name = gr.Dropdown(choices=model_list, value=model_list[0] if model_list else None, label="Model B")
                model_b_version = gr.Dropdown(choices=[], label="Version B")
                threshold_b = gr.Slider(value=0.5, label="Threshold B", minimum=0, maximum=1, step=0.01)
                nitro_b = gr.Checkbox(value=False, label="Nitro B")

                compare_btn = gr.Button("Compare")

            with gr.Column(scale=2):
                gr.Markdown("### Comparison Output")
                spectrum_image_cmp = gr.Plot(label = "Spectrum Comparison")
                fcn_groups_cmp = gr.Dataframe(label = "Functional Groups Comparison", headers = ["Group A", "Prob A", "Group B", "Prob B"], datatype=["str","number","str","number"])

        model_a_name.change(fn = list_versions, inputs = model_a_name, outputs = model_a_version)
        model_b_name.change(fn = list_versions, inputs = model_b_name, outputs = model_b_version)

        def compare_models(file, a_name, a_version, a_thr, b_name, b_version, b_thr):
            fig_a, table_a = predict(file, a_name, a_version, a_thr)
            _, table_b = predict(file, b_name, b_version, b_thr)

            combined = [[r[0], r[1], s[0] if idx<len(s) else None, s[1] if idx<len(s) else None]
                        for idx, (r, s) in enumerate(zip(table_a, table_b))]
            return fig_a, combined

        compare_btn.click(
            fn = compare_models,
            inputs = [spectrum_cmp, model_a_name, model_a_version, threshold_a, nitro_a,
                    model_b_name, model_b_version, threshold_b, nitro_b],
            outputs = [spectrum_image_cmp, fcn_groups_cmp]
        )

if __name__ == "__main__":
    interface.launch(share=False)