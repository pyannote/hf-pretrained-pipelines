from huggingface_hub import HfApi
import matplotlib.pyplot as plt
import streamlit as st
from pyannote.audio import Pipeline
from pyannote.audio import Audio
from pyannote.core import notebook, Segment
import io
import base64

import streamlit.components.v1 as components


from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

import base64
import numpy as np
import scipy.io.wavfile
from typing import Text
def to_base64(waveform: np.ndarray, sample_rate: int = 16000) -> Text:
    """Convert waveform to base64 data"""
    with io.BytesIO() as content:
        scipy.io.wavfile.write(content, sample_rate, waveform)
        content.seek(0)
        b64 = base64.b64encode(content.read()).decode()
        b64 = f"data:audio/x-wav;base64,{b64}"
    return b64

def normalize(waveform: np.ndarray) -> np.ndarray:
    """Normalize waveform for better display in Prodigy UI"""
    return waveform / (np.max(np.abs(waveform)) + 1e-8)

PYANNOTE_LOGO = "https://avatars.githubusercontent.com/u/7559051?s=400&v=4"
EXCERPT = 30.0

st.set_page_config(
    page_title="pyannote.audio pretrained pipelines",
    page_icon=PYANNOTE_LOGO)


st.sidebar.image(PYANNOTE_LOGO)

st.markdown(
    f"""
# 🎹 Pretrained pipelines

Upload an audio file and the first {EXCERPT:g} seconds will be processed automatically.
"""
)

PIPELINES = [p.modelId for p in HfApi().list_models(filter="pyannote-audio-pipeline") if p.modelId.startswith("pyannote/")]

audio = Audio(sample_rate=16000, mono=True)

selected_pipeline = st.selectbox("", PIPELINES, index=0)

with st.spinner('Loading pipeline...'):
    pipeline = Pipeline.from_pretrained(selected_pipeline)

uploaded_file = st.file_uploader("")
if uploaded_file is not None:

    try:
        duration = audio.get_duration(uploaded_file)
    except RuntimeError as e:
        st.error(e)
        st.stop()
    waveform, sample_rate = audio.crop(uploaded_file, Segment(0, min(duration, EXCERPT)))
    file = {"waveform": waveform, "sample_rate": sample_rate, "uri": uploaded_file.name}

    with st.spinner('Running pipeline...'):
        output = pipeline(file)


    html_template = """
    <script src="https://unpkg.com/wavesurfer.js"></script>
    <script src="https://unpkg.com/wavesurfer.js/dist/plugin/wavesurfer.regions.min.js"></script>
    <div id="waveform"></div>    
    <script type="text/javascript">
        
        var wavesurfer = WaveSurfer.create({
            container: '#waveform',
            plugins: [
                WaveSurfer.regions.create({})
            ]
        });

        wavesurfer.load('BASE64');
        wavesurfer.on('ready', function () {
            wavesurfer.play();
        });

        REGIONS
    </script>
    """

    colors = ["#ffd700", "#00ffff", "#ff00ff", "#00ff00", "#9932cc", "#00bfff", "#ff7f50", "#66cdaa"]
    label2color = {label: color for label, color in zip(output.labels(), colors)}

    BASE64 = to_base64(normalize(waveform.numpy().T))
    REGIONS = "".join([
        f"wavesurfer.addRegion({{start: {segment.start:g}, end: {segment.end:g}, color: '{label2color[label]}'}});" 
        for segment, track, label in output.itertracks(yield_label=True)])
    html = html_template.replace('BASE64', BASE64).replace('REGIONS', REGIONS)
    components.html(html, height=200)

    with io.StringIO() as fp:
        output.write_rttm(fp)
        content = fp.getvalue()

        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a download="{output.uri}.rttm" href="data:file/text;base64,{b64}">Download as RTTM</a>'
        st.markdown(href, unsafe_allow_html=True)


st.sidebar.markdown(
    """
-------------------

To use these pipelines on more and longer files on your own (GPU, hence much faster) servers, check the [documentation](https://github.com/pyannote/pyannote-audio).  

For [technical questions](https://github.com/pyannote/pyannote-audio/discussions) and [bug reports](https://github.com/pyannote/pyannote-audio/issues), please check [pyannote.audio](https://github.com/pyannote/pyannote-audio) Github repository.

For commercial enquiries and scientific consulting, please contact [me](mailto:herve@niderb.fr).
"""
)
