from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained summarization model from Hugging Face
summarizer = pipeline('summarization')

# Define a route for the text summarizer API
@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    # Extract text from the request
    text = data['text']
    
    # Summarize the text
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)

    # Return the summary as a response
    return jsonify({"summary": summary[0]['summary_text']})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
