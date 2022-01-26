from huggingface_hub import HfApi
import matplotlib.pyplot as plt
import streamlit as st
from pyannote.audio import Pipeline
from pyannote.audio import Audio
from pyannote.core import notebook, Segment
import io
import base64

from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

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

    with _lock:

        notebook.reset()
        notebook.crop = Segment(0, min(duration, EXCERPT))

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_figwidth(12)
        fig.set_figheight(2.0)
        notebook.plot_annotation(output, ax=ax, time=True, legend=True)

        plt.tight_layout()
        st.pyplot(fig=fig, clear_figure=True)
        plt.close(fig)

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
