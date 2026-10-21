# data/load_hf_data.py
import csv
import re
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download the tokenizer models
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def prepare_wikitext2():
    print("Loading WikiText-2 from Hugging Face (this may take a minute)...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    all_sentences = []
    for example in dataset:
        text = example['text'].strip()
        
        # Skip empty lines and Wikipedia section headers (e.g., "= = = History = = =")
        if not text or text.startswith('='):
            continue

        # Split paragraphs into individual sentences
        sentences = sent_tokenize(text)
        for sent in sentences:
            # Clean specific WikiText formatting artifacts
            sent = sent.replace(' @-@ ', '-').replace(' @.@ ', '.').replace(' @,@ ', ',')
            
            # Tokenize and lowercase
            tokens = word_tokenize(sent.lower())
            
            # Keep ONLY tokens that contain at least one letter (a-z). 
            # This strips out backticks, random symbols, and pure punctuation.
            tokens = [t for t in tokens if re.search(r'[a-z]', t)]
            
            # Keep sentences that have a reasonable length (e.g., at least 4 words)
            if len(tokens) >= 4:
                all_sentences.append(tokens)

    print(f"Tokenization complete. Total clean sentences: {len(all_sentences)}")
    
    # Group into "documents" of 100 sentences to keep your train/test split logic
    DOC_SIZE = 100
    
    with open('data/corpus_hf.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['doc_id', 'sentence'])
        for i, tokens in enumerate(all_sentences):
            doc_id = i // DOC_SIZE
            writer.writerow([doc_id, ' '.join(tokens)])

    print("Saved to data/corpus_hf.csv successfully!")

if __name__ == "__main__":
    prepare_wikitext2()