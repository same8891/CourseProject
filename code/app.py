import torch
import torch.nn.functional as F
import streamlit as st
from scipy.special import softmax
import re
import pandas as pd
from simpletransformers.classification import ClassificationModel
from transformers import BertTokenizer, BertModel


st.title("Google Review Rating Analysis")
cuda_available = torch.cuda.is_available()
device = torch.device("cpu") if not cuda_available else torch.device("cuda:0")

with st.expander("About", expanded=True):
    st.write(
        "1.    This is a tool for sentiment analysis and automatic rating on reviews.\n"
        "2.    To use the tool, please enter the review in text in the input box below, and click \'Analyze\' button.\n"
    )

st.markdown("")
st.markdown("**Input your text here:**")
with st.form(key="my_form"):
    model_type = st.radio(
        'Model type to choose',
        ("roberta", "bert")
    )
    # ckpt_name = "code/" + model_type
    ckpt_name = model_type
    max_words = 256
    text = st.text_area(
        f"Input your text below (max {max_words} words)",
        height=max_words,
    )
    if len(re.findall(r"\w+", text)) > max_words:
        st.warning("Your content exceeds the text length limit.")
    submit_button = st.form_submit_button(label="Analyze")


if not submit_button:
    st.stop()

model = ClassificationModel(
    model_type, ckpt_name, use_cuda=cuda_available,
    args={"use_multiprocessing_for_evaluation": False,
          "use_multiprocessing": False}
)
prediction, raw_outputs = model.predict([text])
prediction = prediction[0]
raw_outputs = raw_outputs[0]
dist = softmax(raw_outputs, 0)
prediction = ["negative", "positive"][prediction]

st.markdown("**Results:**")
st.markdown("")
st.markdown(f"**Predicted label**: *{prediction}*")
st.markdown("**Estimated distribution**:")

dist_data = pd.DataFrame(
    dist,
    columns=["probability"])
st.bar_chart(dist_data)

all_encoded, sentences = torch.load("encoded.pt", map_location=device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
tokenized = tokenizer(text, return_tensors="pt")
text_encoded = model(**tokenized).pooler_output
similarity = F.cosine_similarity(text_encoded, all_encoded)
sort = similarity.sort(descending=True)
indices = sort.indices


with st.expander("Most similar reviews:", expanded=True):
    for i, _id in enumerate(indices[:5]):
        st.write(f"(score: {similarity[_id]:.04f}) {i}:  {sentences[_id]}")
