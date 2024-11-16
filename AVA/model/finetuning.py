import pandas as pd
import google.generativeai as genai
import time 
import os 
from dotenv import load_dotenv
load_dotenv()

# Load the dataset
df = pd.read_parquet("hf://datasets/heliosbrahma/mental_health_chatbot_dataset/data/train-00000-of-00001-01391a60ef5c00d9.parquet")
print(df.head())

# Configure the Google API key
genai.configure(api_key=os.getenv("API_KEY")) # Alternatively, hardcode your API key if needed

# Prepare training data
training_data = []
for index, row in df.iterrows():
    human_text = row['text'].split('<HUMAN>')[1].split('<ASSISTANT>')[0].strip()
    assistant_text = row['text'].split('<ASSISTANT>')[1].strip()
    training_data.append({"text_input": human_text, "output": assistant_text})

# Initialize the base model
base_model = "models/gemini-1.5-flash-001-tuning"

# Create tuned model with operation tracking
# Use the model name as a string
operation = genai.create_tuned_model(
    display_name="mental-health-chatbot",
    source_model=base_model, 
    epoch_count=2,
    batch_size=4,
    learning_rate=0.001,
    training_data=training_data,
)


# Poll for operation status with a longer wait time
max_wait_time = 3600  # maximum wait time in seconds (1 hour)
poll_interval = 30    # polling interval in seconds (adjustable)

start_time = time.time()
for status in operation.wait_bar():
    if time.time() - start_time > max_wait_time:
        print("Max wait time reached, exiting.")
        break
    time.sleep(poll_interval)  # longer polling interval to reduce frequency of status checks


start_time = time.time()
while not operation.done():
    if time.time() - start_time > max_wait_time:
        print("Max wait time reached, exiting.")
        break
    print("Waiting for operation to complete...")
    time.sleep(poll_interval)  # Wait before polling again

if operation.done():
    result = operation.result()
    print(result)
else:
    print("Operation did not complete within the maximum wait time.")

