import os
import json
import re

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# -------- 1. GEMINI CONFIG --------

GENERATION_MODEL = None
ANALYSIS_MODEL = None
MODEL_INIT_ERROR = None

try:
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

    # CHANGE THESE IF YOUR PROJECT USES DIFFERENT MODEL IDS
    # Common safe defaults:
    #   'gemini-1.5-flash' and 'gemini-1.5-pro'
    GENERATION_MODEL = genai.GenerativeModel("gemini-2.0-flash")
    ANALYSIS_MODEL = genai.GenerativeModel("gemini-2.5-flash")

    print("✅ Gemini models configured.")
except KeyError:
    MODEL_INIT_ERROR = "GEMINI_API_KEY environment variable not set."
    print(f"!!! ERROR: {MODEL_INIT_ERROR}")
except Exception as e:
    MODEL_INIT_ERROR = f"Error initializing Gemini models: {e}"
    print(f"!!! ERROR: {MODEL_INIT_ERROR}")


# -------- 2. CORE TOOL FUNCTIONS --------

def generate_text(prompt: str) -> str:
    """Generate an output for the given prompt."""
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if GENERATION_MODEL is None:
        raise RuntimeError("Generation model is not initialized.")
    response = GENERATION_MODEL.generate_content(prompt)
    return response.text


def analyze_full_report(prompt: str, output: str) -> dict:
    """
    Returns:
      {
        "heatmap_data": [
          {"word": "Write", "impact_score": 4},
          ...
        ],
        "connections": [
          {
            "prompt_word": "formal",
            "impact_score": 5,
            "influenced_output_words": ["professional", "polite tone", ...]
          },
          ...
        ]
      }
    """
    if MODEL_INIT_ERROR:
        return {
            "heatmap_data": [],
            "connections": [],
            "error": MODEL_INIT_ERROR,
        }

    analysis_prompt = f"""
    You are a prompt analysis tool. Analyze the following pair.

    PROMPT: {prompt}
    OUTPUT: {output}

    Task 1: Word-by-Word Heatmap
    - Go through the PROMPT word by word (including punctuation as tokens).
    - For every token, assign an "impact_score" (integer 1–5):
      1 = minimal influence on the final output
      3 = moderate influence
      5 = very high influence / critical to structure or meaning

    Task 2: Key Connections
    - Identify key or instructional words from the PROMPT (e.g., 'formal', 'short', 'in 3 bullet points').
    - For each key word, list the output words or short phrases it most strongly influenced.

    Return ONLY a valid JSON object:
    {{
      "heatmap_data": [
        {{"word": "word1", "impact_score": 1}},
        {{"word": "word2", "impact_score": 5}}
      ],
      "connections": [
        {{
          "prompt_word": "formal",
          "impact_score": 5,
          "influenced_output_words": ["professional tone", "polite closing"]
        }}
      ]
    }}
    """

    try:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json"
        )
        response = ANALYSIS_MODEL.generate_content(
            analysis_prompt,
            generation_config=generation_config,
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"!!! ANALYSIS FAILED: {e}")
        return {
            "heatmap_data": [],
            "connections": [],
            "error": f"Error during analysis: {e}",
        }


def run_ablation_study(original_prompt: str, words_to_remove):
    """
    Ablation:
    - Remove the specified words from the original prompt.
    - Re-generate the output on this 'ablated' prompt.
    """
    new_prompt = original_prompt
    for word in words_to_remove:
        if not word:
            continue
        new_prompt = re.sub(
            r"\b" + re.escape(word) + r"\b",
            "",
            new_prompt,
            flags=re.IGNORECASE,
        )

    new_prompt = re.sub(r"\s+", " ", new_prompt).strip()
    new_output = generate_text(new_prompt)
    return new_output, new_prompt


def run_counterfactual_study(original_prompt: str, replacement_map: dict):
    """
    Counterfactual:
    - Replace words in the prompt according to replacement_map, e.g.:
      {"formal": "informal", "sick": "personal"}
    - Re-generate output on this counterfactual prompt.
    """
    new_prompt = original_prompt
    for old_word, new_word in replacement_map.items():
        new_prompt = re.sub(
            r"\b" + re.escape(old_word) + r"\b",
            new_word,
            new_prompt,
            flags=re.IGNORECASE,
        )

    new_output = generate_text(new_prompt)
    return new_output, new_prompt


def quantify_change(original_output: str, new_output: str) -> dict:
    """
    Compare original vs new outputs and compute a semantic_change_score (1–10).

    The analysis model is asked to judge:
      1 = almost no change
      5 = medium change (tone, style, minor details)
      10 = meaning/content is fundamentally different
    """
    if original_output == new_output:
        return {
            "semantic_change_score": 1,
            "change_summary": "No change.",
        }

    if MODEL_INIT_ERROR:
        return {
            "semantic_change_score": -1,
            "change_summary": MODEL_INIT_ERROR,
        }

    analysis_prompt = f"""
    You are an output analysis tool. Analyze the semantic difference between these two texts.

    Original Output:
    {original_output}

    ---
    New Output:
    {new_output}

    Your Task:
    1. Assign a "semantic_change_score" (integer 1–10):
       - 1 = no meaningful difference
       - 5 = moderate change (tone, style, minor details changed, but core meaning same)
       - 10 = major change in meaning, facts, or intent
    2. Write a brief "change_summary" (one short sentence) explaining the most important difference.

    Return ONLY a JSON object:
    {{
      "semantic_change_score": <int>,
      "change_summary": "<short summary>"
    }}
    """

    try:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json"
        )
        response = ANALYSIS_MODEL.generate_content(
            analysis_prompt,
            generation_config=generation_config,
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "semantic_change_score": -1,
            "change_summary": f"Error during change analysis: {e}",
        }


# -------- 3. API ENDPOINTS (/api/...) --------

@app.route("/api/generate", methods=["POST"])
def http_generate():
    data = request.json or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        output = generate_text(prompt)
        return jsonify({"output": output})
    except Exception as e:
        app.logger.exception("Error during generation")
        return jsonify({"error": f"Error during generation: {str(e)}"}), 500


@app.route("/api/analyze", methods=["POST"])
def http_analyze():
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")
    if not prompt or not output:
        return jsonify({"error": 'Both "prompt" and "output" are required'}), 400

    analysis_data = analyze_full_report(prompt, output)
    return jsonify(analysis_data)


@app.route("/api/run_experiment", methods=["POST"])
def http_run_experiment():
    """
    Request body:
    {
      "type": "ablation" | "counterfactual",
      "original_prompt": "...",
      "original_output": "...",
      "changes": "formal, polite"
         OR
      "changes": "{\"formal\": \"informal\", \"sick\": \"personal\"}"
    }
    """
    data = request.json or {}
    experiment_type = data.get("type")
    original_prompt = data.get("original_prompt")
    original_output = data.get("original_output")

    if not original_prompt or not original_output:
        return jsonify({
            "error": "original_prompt and original_output are required"
        }), 400

    if experiment_type == "ablation":
        words_to_remove = (data.get("changes", "") or "").split(",")
        words_to_remove = [w.strip() for w in words_to_remove if w.strip()]
        try:
            new_output, new_prompt = run_ablation_study(
                original_prompt,
                words_to_remove,
            )
        except Exception as e:
            app.logger.exception("Error during ablation study")
            return jsonify({"error": f"Ablation failed: {str(e)}"}), 500

    elif experiment_type == "counterfactual":
        raw_changes = data.get("changes", "{}") or "{}"
        try:
            replacement_map = json.loads(raw_changes)
            if not isinstance(replacement_map, dict):
                raise ValueError("changes must be a JSON object")
        except Exception:
            return jsonify({"error": "Invalid JSON for replacements in 'changes'"}), 400

        try:
            new_output, new_prompt = run_counterfactual_study(
                original_prompt,
                replacement_map,
            )
        except Exception as e:
            app.logger.exception("Error during counterfactual study")
            return jsonify({"error": f"Counterfactual failed: {str(e)}"}), 500

    else:
        return jsonify({"error": "Invalid experiment type"}), 400

    new_analysis_data = analyze_full_report(new_prompt, new_output)
    change_data = quantify_change(original_output, new_output)

    return jsonify({
        "new_prompt": new_prompt,
        "new_output": new_output,
        "new_analysis": new_analysis_data,
        "change_data": change_data,
    })


# Local dev only; Vercel ignores this.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)
