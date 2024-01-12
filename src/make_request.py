import os
from dotenv import load_dotenv
import replicate as rep

def request(prompt: str) -> str:
    # init replicate
    load_dotenv()
    REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY")
    replicate = rep.Client(api_token=REPLICATE_API_KEY)

    # request
    response = ""
    for event in replicate.stream("meta/llama-2-70b-chat", input={"prompt": prompt}):
        response += str(event)
    
    return response