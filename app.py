from flask import Flask, request, jsonify
from io import BytesIO
import tempfile
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

#shruti patil new key : AIzaSyCebs219oM_IaRqkm09N0yJ9LhGK7nJ8gg

#shruti patil old key : AIzaSyAT8SR1HP4HDjZYDAwfur6qW1ppOEGkpSM

#Rudalph Gonsalves new api key: AIzaSyDNr-WITS3OgCnROMjVQk0jUblTPsCxVXs

GOOGLE_API_KEY = "AIzaSyDNr-WITS3OgCnROMjVQk0jUblTPsCxVXs"


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

# gemini_api = "AIzaSyC5agUKvQR7gBuutdV0FSo0tpz2MRn8uL4"

graph = Neo4jGraph(neo4j_url, neo4j_user, neo4j_password)
graph.refresh_schema()

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0)
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True)
        
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
    
    

import re
import google.generativeai as genai

model = genai.GenerativeModel('gemini-pro')
genai.configure(api_key=GOOGLE_API_KEY)
@app.route('/genai', methods=['POST'])
def generate_recommendations():
    try:
        # Get the question from the request data
        question = request.json.get('question')
        
        # Generate response based on the question
        response = model.generate_content(question)
        
        # Format the response
        # recommendations = textwrap.indent(response.text, '> ')
        
        
        recommendations = re.sub(r'[>*]+', '', response.text).strip()
        print(recommendations)
        
        return jsonify({'recommendations': recommendations}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

import os    
# GOOGLE_API_KEY = "AIzaSyCm-ow0oiLoVb2BrnGzrj6klQtYpjIsfk0"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

@app.route('/genai-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Save the file temporarily
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)

    # Process the image with Google Generative AI
    try:
        myfile = genai.upload_file(file_path)
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        result = model.generate_content(
            [myfile, "\n\n", """In reponse I dont want any text.
             I just want a JSON response in following template
             {
                 "medecine_1":"Medecine_name",
                 "medecine_2":"Medecine_name"...
             }"""]
        )

        response_text = result.text

        # Use regex to extract the first medicine
        match = re.search(r'"medecine_1"\s*:\s*"([^"]+)"', response_text)

        if match:
            medicine_1 = match.group(1)
        else:
            medicine_1 = "No medicine name found"

        # Return the result as a JSON response
        return jsonify({"medicine_1": medicine_1})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up the temporary file
        os.remove(file_path)






from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from flask_cors import CORS
import os
import google.generativeai as genai

model = genai.GenerativeModel('gemini-1.5-pro-latest')

# # GOOGLE_API_KEY"] = "AIzaSyAThmkRyMBm177cgKljHkVrAd3BLPpL2nk"

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.5)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

vectorstore_disk = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 5})

template = """
You are a helpful AI assistant.
Answer only based on the context provided. 
If the question provided to you is out of context just say I don't know.
context: {context}
input: {input}
answer:
"""

prompt = PromptTemplate.from_template(template)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


@app.route('/bot', methods=['POST'])
def ask_question():
    data = request.json
    question = data['question']
    # print(question)
    response = retrieval_chain.invoke({"input": question})
    answer1 = response["answer"]
    # print("answer1: ",answer1)
    
    generated_response = model.generate_content(f"""
The question provided is: '{question}'.
The generated answer is: '{answer1}'.

If this answer is relevant and correctly addresses the question, return only '1'.
If the answer is incorrect, incomplete, or irrelevant, generate a new and correct answer that directly answers the question, but only if the question is related to healthcare.

If the question is not related to healthcare, return: 'I don't know the answer'.
Do not include any additional text beyond what is requested.
""" 
)

    # print(generated_response)
    answer2 = generated_response._result.candidates[0].content.parts[0].text
    # print("answer2: ",answer2)
    
    final_answer=""
    
    if answer2.strip()=="1":
        final_answer=answer1
    else:
        final_answer=answer2
    
    print(final_answer)
    return jsonify({"answer": final_answer})



import requests
from apscheduler.schedulers.background import BackgroundScheduler
   
# Dummy route
@app.route('/keep_alive', methods=['GET'])
def keep_alive():
    return "Instance is alive!", 200

# Function to send dummy request
def send_dummy_request():
    try:
        # Replace 'http://your-domain.com/keep_alive' with your deployed API URL
        response = requests.get('https://medisense-backend.onrender.com/keep_alive')
        print(f"Keep-alive request sent: {response.status_code}")
    except Exception as e:
        print(f"Failed to send keep-alive request: {e}")

# Scheduler to run the dummy request every 10 minutes
scheduler = BackgroundScheduler()
scheduler.add_job(send_dummy_request, 'interval', minutes=1)
scheduler.start()



if __name__ == '__main__':
    app.run(debug=True, port=5000)

