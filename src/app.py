from utils import *
from make_request import request

PROMPT_PATH = "prompt.txt"

def main():
    responses = []
    prompt = readFile(PROMPT_PATH)
    print(len(get_tokens(prompt)))
    # if the content is below the standard size of the context window is then passed "as it is" to the LLM
    if(check_prompt_size(prompt)):
        responses.append(request(prompt))
    else:
        prompts = slice_prompt(prompt, get_context_window_size())
        print(len(prompts))
        # print(len(get_tokens(prompts[0])))

        for slice_of_prompt in prompts:
            responses.append(request(slice_of_prompt))
            # print(slice_of_prompt)
            # print("\n==========\n")
    
    for response in responses:
        print(response)

if(__name__ == "__main__"):
    main()