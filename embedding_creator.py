import os
from dotenv import load_dotenv
load_dotenv()


var1 = os.getenv('EMBEDDINGS_DEPLOYMENT')

print(var1)