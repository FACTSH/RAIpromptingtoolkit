import os
import json
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq  # <-- 1. Import Groq

# --- 1. Configuration ---
app = Flask(__name__)
# Configure CORS for your *live* Firebase app URL
CORS(app, resources={
    r"/*": {"origins": "https://responsible-prompting-tool.web.app"}
})

# --- 🔴 NEW: Groq Configuration ---
try:
    # 2. Initialize the client
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY") # <-- Set this in Render
    )
    # 3. Use a fast Llama 3 model on Groq
    GROQ_MODEL = "llama3-8b-8192"
    print(f"✅ Groq backend configured to use: {GROQ_MODEL}")
except Exception as e:
    print(f"!!! ERROR: GROQ_API_KEY not set or client failed: {e}")

# --- 2. Core Toolkit Functions (Rewritten for Groq) ---

def call_groq(prompt, format_json=False):
    """
    Reusable helper function to call the Groq API.
    Groq's API is OpenAI-compatible.
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        if format_json:
            # Ask for JSON output
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}

        chat_completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            response_format=response_format,
            temperature=0.7 # You can adjust this
        )

        raw_response_text = chat_completion.choices[0].message.content

        if format_json:
            return json.loads(raw_response_text)
        else:
            return raw_response_text

    except Exception as e:
        print(f"!!! GROQ FAILED: {e}") 
        return {"error": f"Error during Groq call: {e}"}

def generate_text(prompt):
    result = call_groq(prompt, format_json=False)
    if isinstance(result, dict) and "error" in result:
        return result["error"]
    return result

def analyze_full_report(prompt, output):
    analysis_prompt = f"""
    You are a prompt analysis tool. Analyze the following pair.
    PROMPT: {prompt}
    OUTPUT: {output}
    Task 1: Word-by-Word Heatmap
    Assign an "impact_score" (integer 1-5, 1=low, 5=high) to *every single word* in the PROMPT.

    Task 2: Key Connections
    Identify the *key concepts* or *instructional words* from the PROMPT.
    For each key word, list the output phrases it influenced.

    Return *only* a valid JSON object in this format:
    {{
      "heatmap_data": [...],
      "connections": [...]
    }}
    """
    result = call_groq(analysis_prompt, format_json=True)
    if "error" in result:
        return {"heatmap_data": [], "connections": [], "error": result["error"]}
    return result

def run_ablation_study(original_prompt, words_to_remove):
    new_prompt = original_prompt
    for word in words_to_remove:
        new_prompt = re.sub(r'\b' + re.escape(word) + r'\b', '', new_prompt, flags=re.IGNORECASE)
    new_prompt = re.sub(r'\s+', ' ', new_prompt).strip()
    return generate_text(new_prompt), new_prompt

def run_counterfactual_study(original_prompt, replacement_map):
    new_prompt = original_prompt
    for old_word, new_word in replacement_map.items():
        new_prompt = re.sub(r'\b' + re.escape(old_word) + r'\b', new_word, new_prompt, flags=re.IGNORECASE)
    return generate_text(new_prompt), new_prompt

def analyze_experiment_results(original_output, new_prompt, new_output):
    if original_output == new_output:
        return {
            "new_analysis": {"heatmap_data": [], "connections": []},
            "change_data": {"semantic_change_score": 1, "change_summary": "No change."}
        }
    analysis_prompt = f"""
    You are an advanced prompt analysis tool. You will perform two tasks in one go.
    Task 1: Analyze New Output...
    Task 2: Quantify Change...
    Return *only* a valid JSON object:
    {{
      "new_analysis": {{...}},
      "change_data": {{...}}
    }}
    """
    result = call_groq(analysis_prompt, format_json=True)
    if "error" in result:
        return {
            "new_analysis": {"error": result["error"]},
            "change_data": {"semantic_change_score": -1, "summary": result["error"]}
        }
    return result

# --- 3. API Endpoints (UNMODIFIED) ---
# All your @app.route endpoints for /generate, /analyze, etc.,
# stay exactly the same as in your original file.

@app.route('/')
def index():
    return "This is the Groq-powered backend server."

@app.route('/generate', methods=['POST'])
def http_generate():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    output = generate_text(prompt)
    return jsonify({"output": output})

@app.route('/analyze', methods=['POST'])
def http_analyze():
    data = request.json
    prompt = data.get('prompt')
    output = data.get('output')
    if not prompt or not output:
        return jsonify({"error": "Prompt and output are required"}), 400
    analysis_data = analyze_full_report(prompt, output)
    return jsonify(analysis_data)

@app.route('/run_experiment_step1', methods=['POST'])
def http_run_experiment_step1():
    data = request.json
    experiment_type = data.get('type')
    original_prompt = data.get('original_prompt')
    if experiment_type == 'ablation':
        words_to_remove = data.get('changes', '').split(',')
        words_to_remove = [w.strip() for w in words_to_remove if w.strip()]
        new_output, new_prompt = run_ablation_study(original_prompt, words_to_remove)
    elif experiment_type == 'counterfactual':
        try:
            replacement_map = json.loads(data.get('changes', '{}'))
        except Exception:
            return jsonify({"error": "Invalid JSON for replacements"}), 400
        new_output, new_prompt = run_counterfactual_study(original_prompt, replacement_map)
    else:
        return jsonify({"error": "Invalid experiment type"}), 400
    return jsonify({"new_prompt": new_prompt, "new_output": new_output})

@app.route('/run_experiment_step2', methods=['POST'])
def http_run_experiment_step2():
    data = request.json
    original_output = data.get('original_output')
    new_prompt = data.get('new_prompt')
    new_output = data.get('new_output')
    if not all([original_output, new_prompt, new_output]):
        return jsonify({"error": "Missing data for analysis"}), 400
    analysis_data = analyze_experiment_results(original_output, new_prompt, new_output)
    return jsonify(analysis_data)

# --- 4. Run the Server ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)