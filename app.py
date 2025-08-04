from flask import Flask, render_template, request, jsonify
from doc_vectorizer import DocVectorizer
import os

app = Flask(__name__)
vectorizer = DocVectorizer()
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.pdf', '.docx', '.doc', '.txt']:
        return jsonify({"error": "Unsupported file type"}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    try:
        doc_id = vectorizer.add_document(filepath)
        return jsonify({"message": "File uploaded and processed", "document_id": doc_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    show_pii = data.get('show_pii', False)
    if not query:
        return jsonify({"error": "No search query provided"}), 400
    try:
        results = vectorizer.search(query, n_results=5, redact_pii=not show_pii)
        # The search function now returns a list with a single dictionary.
        if results:
            result = results[0]
            response = {
                "results": [{
                    "score": float(result.get('score', 0)),
                    "content": result.get('document', 'No answer found.'),
                    "metadata": result.get('metadata', {}),
                    "context": result.get('context', 'No context retrieved.') # Pass context to UI
                }]
            }
        else:
            response = {"results": []}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_database():
    """Clear all documents from the vector database."""
    try:
        vectorizer.clear_database()
        return jsonify({"message": "Database cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
