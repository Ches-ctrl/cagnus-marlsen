from inference_sdk import InferenceHTTPClient
import json
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

result = CLIENT.infer("assets/chess_board_1.jpg", model_id="chess-piece-detection-5ipnt/3")
print(json.dumps(result, indent=4))
