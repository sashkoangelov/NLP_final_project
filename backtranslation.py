import json
import sentencepiece as spm
from transformers import MarianMTModel, MarianTokenizer
from contextlib import redirect_stdout, redirect_stderr
import io

def load_squad_dataset(path: str):
    with open(path, "r") as f:
        squad_dict = json.load(f)
    return squad_dict["data"]

def save_squad_dataset(squad_dict, path: str):
    with open(path, "w") as f:
        json.dump({"data": squad_dict}, f)

def apply_back_translation(squad_dict, src_lang: str, tgt_lang: str):
    for article in squad_dict:
        for paragraph in article["paragraphs"]:
            paragraph["context"] = back_translate(paragraph["context"], src_lang, tgt_lang)
            for qa in paragraph["qas"]:
                qa["question"] = back_translate(qa["question"], src_lang, tgt_lang)
                for ans in qa["answers"]:
                    ans["text"] = back_translate(ans["text"], src_lang, tgt_lang)
    return squad_dict

def merge_squad_datasets(squad_dict1, squad_dict2):
    return squad_dict1 + squad_dict2

def back_translate(text: str, src_lang: str, tgt_lang: str):
    # Use the backtranslation API
    model1_name = "Helsinki-NLP/opus-mt-en-" + tgt_lang
    tokenizer1 = MarianTokenizer.from_pretrained(model1_name)
    model1 = MarianMTModel.from_pretrained(model1_name)

    model2_name = "Helsinki-NLP/opus-mt-" + tgt_lang + "-en"
    tokenizer2 = MarianTokenizer.from_pretrained(model2_name)
    model2 = MarianMTModel.from_pretrained(model2_name)

    # Add the source and target language code to the text
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        translated = model1.generate(**tokenizer1(">>" + src_lang + "<< " + text, return_tensors="pt", padding=True))

    # decode the translated text
    text_array = [tokenizer1.decode(t, skip_special_tokens=True) for t in translated]

    # combine array into single string
    translated_text = ' '.join(text_array)

    # back translate the text
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        translated = model2.generate(**tokenizer2(">>" + tgt_lang + "<< " + translated_text, return_tensors="pt", padding=True))

    # decode the translated text
    text_array = [tokenizer2.decode(t, skip_special_tokens=True) for t in translated]
    # combine array into single string
    text = ' '.join(text_array)
    return text

if __name__ == "__main__": 
    # Load the SQuAD dataset
    squad_dataset = load_squad_dataset("train-v2.0.json")



    # Apply back translation to the dataset using Google Translate into 5 different languages
    for lang in ["fr", "de", "es", "zh", "ja"]:
        test_string = back_translate("This is a test string to see if this works as well as it should", "en", lang)
        print(lang, ":  ", test_string)	
        #squad_dataset = apply_back_translation(squad_dataset, "en", lang)
        #save_squad_dataset(squad_dataset, f"train-v2.0-with-back-translation-{lang}.json")