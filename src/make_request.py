import os
from dotenv import load_dotenv
import replicate

# init replicate
load_dotenv()
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY")
replicate = replicate.Client(api_token=REPLICATE_API_KEY)

# request
prompt = "Could you tell me the story of a child who dreamed of becoming a motorsports champion?"

for event in replicate.stream("meta/llama-2-70b-chat", input={"prompt": prompt}):
    print(str(event), end="")

prompt = "continue it"

for event in replicate.stream("meta/llama-2-70b-chat", input={"prompt": prompt}):
    print(str(event), end="")