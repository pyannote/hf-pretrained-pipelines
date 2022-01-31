# MIT License
#
# Copyright (c) 2022- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import io
import base64
import numpy as np
import scipy.io.wavfile
from typing import Text
from huggingface_hub import HfApi
import streamlit as st
from pyannote.audio import Pipeline
from pyannote.audio import Audio
from pyannote.core import Segment

import streamlit.components.v1 as components


def to_base64(waveform: np.ndarray, sample_rate: int = 16000) -> Text:
    """Convert waveform to base64 data"""
    waveform /= np.max(np.abs(waveform)) + 1e-8
    with io.BytesIO() as content:
        scipy.io.wavfile.write(content, sample_rate, waveform)
        content.seek(0)
        b64 = base64.b64encode(content.read()).decode()
        b64 = f"data:audio/x-wav;base64,{b64}"
    return b64


PYANNOTE_LOGO = "https://avatars.githubusercontent.com/u/7559051?s=400&v=4"
EXCERPT = 30.0

st.set_page_config(
    page_title="pyannote.audio pretrained pipelines", page_icon=PYANNOTE_LOGO
)


st.sidebar.image(PYANNOTE_LOGO)

st.markdown(
    f"""
# 🎹 Pretrained pipelines

Upload an audio file and the first {EXCERPT:g} seconds will be processed automatically.
"""
)

PIPELINES = [
    p.modelId
    for p in HfApi().list_models(filter="pyannote-audio-pipeline")
    if p.modelId.startswith("pyannote/")
]

audio = Audio(sample_rate=16000, mono=True)

selected_pipeline = st.selectbox("", PIPELINES, index=0)

with st.spinner("Loading pipeline..."):
    pipeline = Pipeline.from_pretrained(selected_pipeline)

uploaded_file = st.file_uploader("")
if uploaded_file is not None:

    try:
        duration = audio.get_duration(uploaded_file)
    except RuntimeError as e:
        st.error(e)
        st.stop()
    waveform, sample_rate = audio.crop(
        uploaded_file, Segment(0, min(duration, EXCERPT))
    )
    file = {"waveform": waveform, "sample_rate": sample_rate, "uri": uploaded_file.name}

    with st.spinner("Running pipeline..."):
        output = pipeline(file)

    with open('assets/template.html') as html:
        html_template = html.read()

    colors = [
        "#ffd70033",
        "#00ffff33",
        "#ff00ff33",
        "#00ff0033",
        "#9932cc33",
        "#00bfff33",
        "#ff7f5033",
        "#66cdaa33",
    ]
    label2color = {label: color for label, color in zip(output.labels(), colors)}

    BASE64 = to_base64(waveform.numpy().T)
    REGIONS = "".join(
        [
            f"var re = wavesurfer.addRegion({{start: {segment.start:g}, end: {segment.end:g}, color: '{label2color[label]}', resize : false, drag : false}}); addRegionLabel(re,'{label}');"
            for segment, _, label in output.itertracks(yield_label=True)
        ]
    )
    html = html_template.replace("BASE64", BASE64).replace("REGIONS", REGIONS)
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
