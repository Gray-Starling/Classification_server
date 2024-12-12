from dotenv import load_dotenv
import os

load_dotenv()

SERVER_PORT = os.getenv('SERVER_PORT') if os.getenv('SERVER_PORT') else 4040
NEWS_DATA_API = os.getenv('NEWS_DATA_API') if os.getenv('NEWS_DATA_API') else "http://127.0.0.1:4040/data"