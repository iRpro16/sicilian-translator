from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

# Import tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("iRpro16/sicilian_translator")
model = AutoModelForSeq2SeqLM.from_pretrained("iRpro16/sicilian_translator")

# disallowed word list
disallowed_word_list = ["<pad>", "</s>", "<unk>", "__"]

# preprocess and remove pad and eos words
def generate_sentence(phrase: str, disallowed_word_list: list[str]):
    # encode
    inputs = tokenizer.encode(phrase, return_tensors="pt")

    # outputs of model
    outputs = model.generate(inputs, max_length=52).squeeze().tolist()

    # token to id
    token_to_id  = tokenizer.convert_ids_to_tokens(outputs)

    allowed_words_list = []

    for word in token_to_id:
        if word not in disallowed_word_list:
            allowed_words_list.append(word)

    sentence = " ".join(allowed_words_list)
    return sentence

# Streamlit app
st.title("Sicilian Translator")

# input sentence
input_sentence = st.text_input("Enter sentence to translate...")

if st.button('Translate'):

    # Preprocess
    preprocessed_sentence = generate_sentence(input_sentence, disallowed_word_list)

    # Remove underscores
    sentence = preprocessed_sentence.replace("▁","")

    # Show result
    st.header(sentence)

