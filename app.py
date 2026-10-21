"""Lab 02 - Smart Keyboard V2: Interactive N-Gram Playground."""
import streamlit as st
import random
import math
import pandas as pd
import altair as alt
import os
import subprocess
import sys
from src.pipeline import NGramModel, EOS, UNK, load_documents, backoff_top_k

# --- Page Config ---
st.set_page_config(page_title="N-Gram Playground", layout="wide")

# ==========================================
# 1. LIGHT / DARK MODE TOGGLE & CSS
# ==========================================
st.sidebar.markdown("---")
dark_mode = st.sidebar.toggle("Dark Mode", value=True)

if dark_mode:
    bg_color = "#0e1117"
    text_color = "#fafafa"
    sidebar_color = "#262730"
    input_bg = "#262730"
    border_color = "#151313"
    hover_bg = "#3e3e4a"
else:
    bg_color = "#ffffff"
    text_color = "#31333f"
    sidebar_color = "#f0f2f6"
    input_bg = "#ffffff"
    border_color = "#3e3a3a"
    hover_bg = "#e0e0e0"

# EXTREME CSS OVERRIDE FOR ALL COMPONENTS
st.markdown(f"""
<style>
    /* 1. Global App, Body, & Top Header */
    body, .stApp {{ background-color: {bg_color} !important; }}
    header[data-testid="stHeader"] {{ background-color: {bg_color} !important; }}
    header[data-testid="stHeader"] svg {{ fill: {text_color} !important; }}
    
    /* 2. Sidebar */
    [data-testid="stSidebar"] {{ background-color: {sidebar_color} !important; }}
    
    /* 3. General Text */
    h1, h2, h3, h4, h5, h6, p, span, label, div {{ color: {text_color} !important; }}
    
    /* 4. Inputs and Text Areas */
    .stTextInput input, .stTextArea textarea {{ 
        background-color: {input_bg} !important; 
        color: {text_color} !important; 
        border: 1px solid {border_color} !important;
    }}
    
    /* 5. SELECTBOX - The "Kill It With Fire" Approach */
    /* Target the wrapper, the inner divs, the hidden input, and the text div */
    div[data-testid="stSelectbox"] * {{
        background-color: transparent !important;
        color: {text_color} !important;
    }}
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 4px !important;
    }}
    div[data-testid="stSelectbox"] div[data-baseweb="select"] input {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
    }}
    div[data-testid="stSelectbox"] svg {{
        fill: {text_color} !important;
    }}
    /* Remove the focus red border */
    div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {{
        border-color: {border_color} !important;
        box-shadow: none !important;
    }}
    
    /* 6. DROPDOWN MENU (The Portal) */
    div[data-baseweb="popover"] > div {{
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    ul[role="listbox"] {{ background-color: {input_bg} !important; }}
    li[role="option"], div[role="option"] {{ 
        background-color: {input_bg} !important; 
        color: {text_color} !important; 
    }}
    li[role="option"]:hover, div[role="option"]:hover, li[aria-selected="true"] {{ 
        background-color: {hover_bg} !important; 
        color: {text_color} !important;
    }}
    
    /* 7. Tabs */
    button[data-baseweb="tab"] {{ color: {text_color} !important; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: #FF4B4B !important; }}
    
    /* 8. Buttons */
    .stButton>button {{ 
        background-color: {input_bg} !important; 
        color: {text_color} !important; 
        border: 1px solid {border_color} !important; 
    }}
    
    /* 9. Slider */
    div[data-testid="stSlider"] div[role="slider"] {{ background-color: #FF4B4B !important; }}
</style>
""", unsafe_allow_html=True)

st.title("Smart Keyboard V2 - Interactive Playground")
st.caption("Powered by an n-gram model trained on WikiText-2. Experiment with smoothing and context!")

# --- Load Data & Cache ---
@st.cache_resource
def load_data():
    csv_path = "data/corpus_hf.csv"
    # If the file doesn't exist (like on a fresh deploy), download it
    if not os.path.exists(csv_path):
        with st.spinner("First run: Downloading and preparing WikiText-2 dataset. This may take a few minutes..."):
            subprocess.run([sys.executable, "data/load_hf_data.py"], check=True)
    
    # Now proceed to load as normal
    docs = load_documents(path=csv_path)
    sentences = [s for d in docs.values() for s in d]
    return sentences

sentences = load_data()

@st.cache_resource
def build_model(n, k):
    return NGramModel(sentences, n, k)

@st.cache_resource
def get_vocab_frequencies():
    uni = build_model(1, 0.1)
    freqs = {}
    for context, counter in uni.ctx_counts.items():
        for word, count in counter.items():
            freqs[word] = freqs.get(word, 0) + count
    return freqs

vocab_freqs = get_vocab_frequencies()

# --- Sidebar Model Config ---
st.sidebar.header("Model Configuration")

# If the CSS STILL fails for the dropdown, you can uncomment these lines to use Radio buttons instead:
# n_order = st.sidebar.radio("Select N-gram Order:", [1, 2, 3], index=2, horizontal=True)

n_order = st.sidebar.selectbox("Select N-gram Order:", [1, 2, 3], index=2)
k_smoothing = st.sidebar.slider("Smoothing constant (k):", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

if k_smoothing == 0.0:
    st.sidebar.warning("Warning: k=0 means no smoothing. Unseen words will cause infinite perplexity!")

model = build_model(n_order, k_smoothing)

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Autocomplete", "Compare Models", "Text Generation", "Perplexity & Vocab"])

# ==========================================
# TAB 1: Autocomplete & Visualization
# ==========================================
with tab1:
    st.header("Next-Word Suggestions")
    
    if "text_input" not in st.session_state:
        st.session_state.text_input = "the united states"

    text = st.text_input("Type a prompt:", value=st.session_state.text_input)
    st.session_state.text_input = text

    words = text.lower().replace(".", " . ").split()

    if text:
        if n_order == 3:
            tri, bi, uni = build_model(3, k_smoothing), build_model(2, k_smoothing), build_model(1, k_smoothing)
            suggestions, order_used = backoff_top_k(tri, bi, uni, words, 4)
        else:
            suggestions = model.top_k(words, 4)
            order_used = f"{n_order}-gram"

        st.subheader(f"Top Suggestions (using {order_used} model)")
        
        cols = st.columns(4)
        for col, (word, prob) in zip(cols, suggestions):
            label = "(end)" if word == EOS else word
            if col.button(f"{label} ({prob:.2%})", use_container_width=True):
                st.session_state.text_input += " " + word
                st.rerun()

        st.markdown("---")
        st.subheader("Probability Distribution (Top 10 words)")
        ctx = model.context(words)
        dist = {w: model.prob(ctx, w) for w in model.vocab if w != UNK}
        top_10 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:10]
        
        df = pd.DataFrame(top_10, columns=["Word", "Probability"])
        
        chart = alt.Chart(df).mark_bar(color="#4C78A8").encode(
            x=alt.X("Word", sort=None, axis=alt.Axis(labelColor=text_color, titleColor=text_color)),
            y=alt.Y("Probability", axis=alt.Axis(labelColor=text_color, titleColor=text_color))
        ).properties(
            background=bg_color,
            height=300
        )
        
        st.altair_chart(chart, use_container_width=True)

# ==========================================
# TAB 2: Model Comparison
# ==========================================
with tab2:
    st.header("Unigram vs Bigram vs Trigram")
    st.write("See how providing more context changes the model's predictions.")
    
    compare_text = st.text_input("Enter a context to compare:", "the city of")
    compare_words = compare_text.lower().split()
    
    col1, col2, col3 = st.columns(3)
    
    models = [
        ("Unigram (1-gram)", build_model(1, k_smoothing), col1),
        ("Bigram (2-gram)", build_model(2, k_smoothing), col2),
        ("Trigram (3-gram)", build_model(3, k_smoothing), col3)
    ]
    
    for name, m, col in models:
        with col:
            st.subheader(name)
            if m.n > 1 and not m.seen_context(compare_words):
                st.warning("Context unseen in training!")
            
            top_3 = m.top_k(compare_words, 3)
            for word, prob in top_3:
                st.write(f"**{word}**: {prob:.3%}")

# ==========================================
# TAB 3: Text Generation
# ==========================================
with tab3:
    st.header("Markov Chain Text Generation")
    st.write("The model repeatedly predicts the next word and feeds it back into itself.")
    
    gen_prompt = st.text_input("Start the sentence:", "in the beginning")
    max_words = st.slider("Maximum words to generate:", 5, 50, 20)
    
    if st.button("Generate Text", type="primary"):
        current_words = gen_prompt.lower().split()
        
        with st.spinner("Generating..."):
            for _ in range(max_words):
                ctx = model.context(current_words)
                dist = {w: model.prob(ctx, w) for w in model.vocab if w not in [UNK, EOS]}
                
                words_list = list(dist.keys())
                probs_list = list(dist.values())
                next_word = random.choices(words_list, weights=probs_list, k=1)[0]
                current_words.append(next_word)
                
            st.success("Generation Complete!")
            st.markdown(f"### Result:\n> {' '.join(current_words)}")
            
            st.download_button(
                label="Download Generated Text",
                data=" ".join(current_words),
                file_name="generated_text.txt",
                mime="text/plain"
            )

# ==========================================
# TAB 4: Perplexity & Vocab Explorer
# ==========================================
with tab4:
    st.header("Evaluate a Sentence (Perplexity)")
    st.write("Perplexity measures how 'surprised' the model is by your sentence. Lower is better.")
    
    ppl_text = st.text_area("Type a sentence:", "the united states of america is a country")
    
    if st.button("Calculate Perplexity"):
        tokens = ppl_text.lower().split()
        ppl, zero_share = model.perplexity([tokens])
        
        col_a, col_b = st.columns(2)
        if math.isinf(ppl):
            col_a.error(f"Perplexity: Infinity")
            col_b.metric("Zero-probability tokens", f"{zero_share:.1%}", delta_color="inverse")
            st.warning("Because k=0, the model assigned exactly 0 probability to an unseen sequence, making the perplexity infinite. Increase k in the sidebar!")
        else:
            col_a.metric("Sentence Perplexity", f"{ppl:.2f}")
            col_b.metric("Zero-probability tokens", f"{zero_share:.1%}")

    st.markdown("---")
    st.header("Vocabulary Explorer")
    search_word = st.text_input("Search for a word (e.g., 'the', 'cat', 'university'):", "university")
    
    if search_word:
        word = search_word.lower().strip()
        if word in model.vocab:
            count = vocab_freqs.get(word, 0)
            st.success(f"'{word}' is in the vocabulary.")
            st.info(f"It appears {count:,} times in the training data.")
        else:
            st.error(f"'{word}' is NOT in the vocabulary. It will be treated as <unk> (Unknown).")