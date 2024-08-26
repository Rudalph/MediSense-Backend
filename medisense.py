from flask import Flask, request, jsonify
from io import BytesIO
import tempfile
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = "AIzaSyC5agUKvQR7gBuutdV0FSo0tpz2MRn8uL4"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Read file data into memory
            file_data = file.read()

            # Create a temporary file
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_pdf.write(file_data)

            # Close the file to ensure it's written to disk
            temp_pdf.close()

            # Pass the file path to PyPDFLoader
            loader = PyPDFLoader(temp_pdf.name)
            pages = loader.load_and_split()

            llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=GOOGLE_API_KEY)
            chain = load_summarize_chain(llm, chain_type="stuff")

            result = chain.run(pages)

            return jsonify({'summary': result})
        except Exception as e:
            return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
