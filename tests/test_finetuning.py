import pytest
import pandas as pd
import google.generativeai as genai
from unittest.mock import MagicMock

# Mock the generative model
genai.GenerativeModel = MagicMock()

def test_transform_data():
    # Sample data
    data = {
        'text': [
            '<HUMAN> 1 <ASSISTANT> 2',
            '<HUMAN> 3 <ASSISTANT> 4',
            '<HUMAN> -3 <ASSISTANT> -2',
            '<HUMAN> twenty two <ASSISTANT> twenty three'
        ]
    }
    df = pd.DataFrame(data)
    
    # Expected output
    expected_output = [
        {"text_input": "1", "output": "2"},
        {"text_input": "3", "output": "4"},
        {"text_input": "-3", "output": "-2"},
        {"text_input": "twenty two", "output": "twenty three"}
    ]
    
    # Transform the data
    training_data = []
    for index, row in df.iterrows():
        human_text = row['text'].split('<HUMAN>')[1].split('<ASSISTANT>')[0].strip()
        assistant_text = row['text'].split('<ASSISTANT>')[1].strip()
        training_data.append({"text_input": human_text, "output": assistant_text})
    
    assert training_data == expected_output

def test_generate_content():
    # Mock result
    mock_result = MagicMock()
    mock_result.text = "A panic attack is a sudden episode of intense fear."
    genai.GenerativeModel.return_value.generate_content.return_value = mock_result
    
    # Create model and generate content
    model = genai.GenerativeModel(model_name="test_model")
    result = model.generate_content("What is a panic attack?")
    
    assert result.text == "A panic attack is a sudden episode of intense fear."

if __name__ == "__main__":
    pytest.main()