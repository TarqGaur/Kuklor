# import os
# os.environ["HF_HOME"] = "G:/01 Huggingface"
# os.environ["TRANSFORMERS_CACHE"] = "G:/01 Huggingface"
# os.environ["HF_HUB_CACHE"] = "G:/01 Huggingface"

from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL3-1B'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=16384, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
response = pipe(('describe this image', image))
print(response.text)
