# Prompt slicer
An algorithm to generate slicing of excessive context window for LLMs.

## Dependencies
- [NLTK](https://www.nltk.org/)
- [Loadenv](https://pypi.org/project/loadenv/)
- [Replicate](https://pypi.org/project/replicate/)

## How to use
In order to use the script you need to add a Replicate API key to the [.env](.env) file.  
Launch `./start.sh` to start the script.

## Project description
Given the prompt, we generate an answer using Llama2 powered by [Replicate](https://replicate.com/).

In particular, if the prompt is below the standard size of the context window is then passed "as it is" to the LLM. Otherwise, the prompt will be sliced.

It use the classical pipeline to generate BoW of the sliced prompt:
1. perform ***tokenization***;
2. perform the ***stopwords*** elimination; 
3. perform either ***lemmatization*** or ***stemming*** using the Porter Stemmer algorithm;
4. calculate frequencies.

Then, having the BoW, it will check for cousine similarity with the previous slice (i.e. the adjacent one).
If the similarity is below 0.8 it will be selected as a slice. Otherwise it will try to generate a new slice with context window halved.
