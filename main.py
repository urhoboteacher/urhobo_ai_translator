from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Import tokenizer and model classes

# Load the translation model using Hugging Face Transformers (not pickle for this kind of task)
model_dir = 'models'  # Path where the model is saved
tokenizer = AutoTokenizer.from_pretrained(model_dir)  # Load tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)  # Load model

# Initialize Flask app
app = Flask(__name__)

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API route for translation
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json  # Get JSON data from the request
    input_text = data.get('text')  # Extract input text from the JSON

    if not input_text:
        return jsonify({"error": "No text provided"}), 400  # Return error if no text

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt')

    # Generate translation
    outputs = model.generate(**inputs)

    # Decode the generated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Return translated text as JSON response
    return jsonify({"translated_text": translated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)