from utils import *
from make_request import request

PROMPT_PATH = "prompt.txt"

def main():
    responses = []
    prompt = readFile(PROMPT_PATH)
    # if the content is below the standard size of the context window is then passed "as it is" to the LLM
    if(check_prompt_size(prompt)):
        responses.append(request(prompt))
    else:
        prompts = slice_prompt(prompt, get_context_window_size())

        for slice_of_prompt in prompts:
            responses.append(request(slice_of_prompt))
    
    for response in responses:
        print(response)

if(__name__ == "__main__"):
    main()