import json
import random
from copy import deepcopy
import os

TARGET_SIZE = 200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FILE = os.path.join(BASE_DIR, "bfsi_alpaca_base.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "bfsi_alpaca_expanded.json")


PARAPHRASE_PREFIXES = [
    "Please explain",
    "Kindly clarify",
    "I would like to know",
    "Can you tell me",
    "Help me understand",
    "Provide details about",
    "Give information about",
    "May I know",
    "I need information on",
    "Could you explain"
]

def paraphrase_instruction(text):
    prefix = random.choice(PARAPHRASE_PREFIXES)
    return f"{prefix} {text.lower()}"

with open(BASE_FILE, "r") as f:
    base_data = json.load(f)

expanded_data = []

while len(expanded_data) < TARGET_SIZE:
    for item in base_data:
        if len(expanded_data) >= TARGET_SIZE:
            break
        
        new_item = deepcopy(item)
        new_item["instruction"] = paraphrase_instruction(item["instruction"])
        expanded_data.append(new_item)

# Combining original + expanded
final_dataset = base_data + expanded_data
final_dataset = final_dataset[:TARGET_SIZE]

with open(OUTPUT_FILE, "w") as f:
    json.dump(final_dataset, f, indent=2)

print(f"Dataset expanded to {len(final_dataset)} samples.")
