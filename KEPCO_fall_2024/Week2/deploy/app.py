# path gguf : /workspace/nmquy/LLM_QA/model/gguf/Meta-Llama-3.1-8B-Instruct-bf16.gguf
#  /workspace/nmquy/RLHF/model/GGUF/test/ggml-model-Q4_K_M.gguf

from flask import Flask, request, jsonify
from llama_cpp import Llama

# Create a Flask object
app = Flask("Llama server")
model = None
n_gpu_layers = -1
n_batch = 512
model_path = "./Meta-Llama-3.1-8B-Instruct-bf16.gguf"
model = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers,n_batch=n_batch, verbose = True)

@app.route('/llama', methods=['POST'])
def generate_response():
    global model
    
    try:
        data = request.get_json()

        # Check if the required fields are present in the JSON data
        if 'system_message' in data and 'user_message' in data and 'max_tokens' in data:
            system_message = data['system_message']
            user_message = data['user_message']
            max_tokens = int(data['max_tokens'])

            # Prompt creation
            prompt = f"""<s>[INST] <<SYS>>
            {system_message}
            <</SYS>>
            {user_message} [/INST]"""
            # Run the model
            output = model(prompt, max_tokens=max_tokens, echo=True)
            
            return jsonify(output)

        else:
            return jsonify({"error": "Missing required parameters"}), 400

    except Exception as e:
        return jsonify({"Error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)