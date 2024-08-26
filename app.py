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
        
        

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

 
hf_api = "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"
neo4j_url = "neo4j+s://babf3722.databases.neo4j.io"
neo4j_user = "neo4j"
neo4j_password ="6qkA7VWxWKgfO5tJ7Lm2yKew2hVd3X7GBp_5F-93aNI"

gemini_api = "AIzaSyC5agUKvQR7gBuutdV0FSo0tpz2MRn8uL4"

graph = Neo4jGraph(neo4j_url, neo4j_user, neo4j_password)
graph.refresh_schema()

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api, temperature=0)
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)
        
@app.route('/brand', methods=['POST'])
def get_brand_details():
    # Get input value from frontend
    input_value = request.json.get('question')
    print(input_value)

    # Construct Cypher query to get generic name for the given brand name
    # cypher_query = f"MATCH (b:Brand {{name: '{input_value}'}})-[:HAS_GENERIC]->(g:Generic) RETURN g.name"
    cypher_query = "MATCH (b:Brand {name: $input_value})-[:HAS_GENERIC_NAME]->(g:GenericName) RETURN g.name AS genericName"
    result = graph.query(cypher_query, params={"input_value": input_value})
    print(result)
    if result:
        generic_name = result[0]['genericName']
        print(generic_name)
        
        # Construct Cypher query to get brand details based on generic name
        cypher_query_details = """
        MATCH (b:Brand)-[:HAS_GENERIC_NAME]->(g:GenericName {name: $generic_name})
        OPTIONAL MATCH (b)-[:PRICED_AT]->(p:Price)
        OPTIONAL MATCH (b)-[:HAS_STRENGTH]->(s:Strength)
        OPTIONAL MATCH (b)-[:PACKAGED_AS]->(pkg:Package)
        OPTIONAL MATCH (b)-[:MANUFACTURED_BY]->(c:Company)
        RETURN b.name AS brandName, 
        g.name AS genericName, 
        p.amount AS price, 
        s.value AS strength, 
        pkg.type AS packageName, 
        c.name AS companyName
        """
        
        result_details = graph.query(cypher_query_details,params={"generic_name": generic_name})
        print(result_details)
        
        return jsonify(result_details)
    else:
        return jsonify({'error': 'Brand not found'}), 404

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
