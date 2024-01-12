import math
import os
from dotenv import load_dotenv
from nltk_utils import *

USE_LEMMATIZER = True

def readFile(filepath: str) -> str:
    text = ''
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        # clear the text 
        for line in lines:
            text += line.strip() + '\n'

    return text

def count_occurrences(occurrences_dict: dict, text: list) -> None:
    """count word occurrence in a given text."""
    for word in text:
        if word in occurrences_dict:
            occurrences_dict[word] += 1
        else:
            occurrences_dict[word] = 1

def perform_pre_processing(doc: str) -> list:
    """Before creating the BoW is good to process your document"""
    # tokenize doc
    text = get_tokens(doc)
    # perform the stopwords elimination
    text = delete_stopwords(text)
    
    if(USE_LEMMATIZER):
        # perform lemmatization
        text = lemmatize_article(text)
    else:
        # perform stemming
        text = perform_stemming(text)

    return text

def calculate_doc_frequency(occurrences: dict) -> dict:
    frequencies = {}
    # sum all occurrences
    total_occurrences = sum(occurrences.values())
    for (word, occurrence) in occurrences.items():
        frequencies[word] = occurrence / total_occurrences
    
    return frequencies

def get_context_window_size() -> int:
    load_dotenv()
    return int(os.environ.get("CONTEXT_WINDOW_SIZE"))

def check_prompt_size(prompt: str) -> bool:
    """returns true if prompt is lesser than the context window size."""
    context_window_size = get_context_window_size()

    tokens = get_tokens(prompt)
    return len(tokens) < context_window_size
    
def cosine_similarity(bow1: dict, bow2: dict) -> int:
    # Calculate the dot product
    dot_product = sum(bow1.get(word, 0) * bow2.get(word, 0) for word in set(bow1) | set(bow2))

    # Calculate the magnitude (Euclidean norm) of each vector
    mag_bow1 = math.sqrt(sum(val**2 for val in bow1.values()))
    mag_bow2 = math.sqrt(sum(val**2 for val in bow2.values()))

    # Calculate the cosine similarity
    similarity = dot_product / (mag_bow1 * mag_bow2)

    return similarity

def get_bow(document: str) -> dict:
    occurrences_dict = {}
    # process the text
    processed_doc = perform_pre_processing(document)
    # count occurrences in all corpora
    count_occurrences(occurrences_dict, processed_doc)

    occurrences_dict = dict(sorted(occurrences_dict.items(), key=lambda item: item[1], reverse=True))
    bow = dict(sorted(calculate_doc_frequency(occurrences_dict).items(), key=lambda item: item[1], reverse=True))

    return bow

def slice_prompt(prompt: str, context_window_size: int) -> list:
    slices = []

    # recursive function, can't slice more than 64 tokens
    if(context_window_size <= 32):
        return [prompt]
    
    sentences = get_sentences(prompt)
    new_slice = ""

    # Reverse the list to use pop() properly
    sentences.reverse()
    
    while sentences:
        sentence = sentences.pop()
        length_slice = len(get_tokens(new_slice))
        length_sentence = len(get_tokens(sentence))
        # we need to stay below the context_window_size
        if((length_slice + length_sentence) >= context_window_size):
            # check adjacent similarity
            if(slices):
                last_slice = slices[len(slices) -1]
                bow_last_slice = get_bow(last_slice)
                bow_new_slice = get_bow(new_slice)
                similarity = cosine_similarity(bow_last_slice, bow_new_slice)
                if(similarity >= 0.8):
                    new_slices = slice_prompt(new_slice, math.ceil(context_window_size/2))
                    for el in new_slices:
                        slices.append(el)
                else:
                    if(new_slice):
                        slices.append(new_slice)
                    new_slice = sentence 
            else:
                if(new_slice):
                    slices.append(new_slice)
                new_slice = sentence
        else:
            new_slice += sentence
        
        if(len(sentences) == 20):
            continue

    # Add the last part of the prompt
    if(new_slice and new_slice not in slices):
        slices.append(new_slice)

    return slices