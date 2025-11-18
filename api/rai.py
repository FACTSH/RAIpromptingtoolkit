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


def init_models():
    """
    Initialize GEMINI models once, rather than on every request.
    """
    global GENERATION_MODEL, ANALYSIS_MODEL, MODEL_INIT_ERROR
    try:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

        genai.configure(api_key=api_key)

        GENERATION_MODEL = genai.GenerativeModel("gemini-1.5-flash")
        ANALYSIS_MODEL = genai.GenerativeModel("gemini-1.5-flash")
        MODEL_INIT_ERROR = None
    except Exception as e:
        MODEL_INIT_ERROR = str(e)


init_models()


# -------- 2. UTILS --------

def clean_prompt(text: str) -> str:
    """
    Minimal cleaning, in case you want to standardize whitespace, etc.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# -------- 3. CORE GENERATION (100-WORD OUTPUT LIMIT) --------

def generate_text(prompt: str) -> str:
    """
    Generate an output for the given prompt, with a hard limit of 100 words.
    The model is explicitly instructed to stay under 100 words, and the result
    is post-processed to enforce the cap.
    """
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if GENERATION_MODEL is None:
        raise RuntimeError("Generation model is not initialized.")
    
    # Add explicit instruction so the model aims to stay concise
    constrained_prompt = prompt.strip() + "\n\nRespond in 100 words or fewer."

    response = GENERATION_MODEL.generate_content(constrained_prompt)
    full_output = response.text or ""
    
    # Post-process to enforce 100-word cap
    words = full_output.split()
    if len(words) > 100:
        return " ".join(words[:100]) + "..."
    return full_output


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
            "influence": [
              {
                "output_phrase": "I hope this email finds you well",
                "relationship_type": "style_match",
                "direction": "positive",
                "confidence": 0.9
              },
              ...
            ]
          },
          ...
        ]
      }
    """
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if ANALYSIS_MODEL is None:
        raise RuntimeError("Analysis model is not initialized.")

    system_instructions = """
You are an analysis engine for prompt-output pairs.

Given:
1) A user prompt.
2) The model's generated output.

You produce:
- A 'heatmap' of the prompt words, indicating how strongly they influenced
  the final output.
- A set of 'connections' describing how each significant prompt word or phrase
  affected specific output phrases, including relationship type, direction,
  and a confidence score.

Return strictly valid JSON. Fields:
{
  "heatmap_data": [
    {
      "word": "string",
      "impact_score": integer from 1 (weak) to 5 (very strong)
    }
  ],
  "connections": [
    {
      "prompt_word": "string",
      "impact_score": integer 1-5,
      "influence": [
        {
          "output_phrase": "string",
          "relationship_type": "style_match" | "topic_match" | "sentiment_shift" | "constraint_enforcement" | "other",
          "direction": "positive" | "negative" | "neutral",
          "confidence": float between 0 and 1
        }
      ]
    }
  ]
}
Only return JSON.
"""

    prompt_text = f"""
User Prompt:
{prompt}

Model Output:
{output}
"""

    response = ANALYSIS_MODEL.generate_content(
        [
            {"role": "system", "parts": [system_instructions]},
            {"role": "user", "parts": [prompt_text]},
        ]
    )

    raw_text = response.text.strip()

    try:
        data = json.loads(raw_text)
    except Exception:
        data = {
            "heatmap_data": [],
            "connections": [],
            "raw_analysis_text": raw_text,
        }

    if "heatmap_data" not in data or not isinstance(data["heatmap_data"], list):
        data["heatmap_data"] = []
    if "connections" not in data or not isinstance(data["connections"], list):
        data["connections"] = []

    return data


# -------- 4. FLASK ROUTES --------

@app.route("/health", methods=["GET"])
def health():
    if MODEL_INIT_ERROR:
        return jsonify({"status": "error", "detail": MODEL_INIT_ERROR}), 500
    return jsonify({"status": "ok"}), 200


@app.route("/api/generate", methods=["POST"])
def http_generate():
    data = request.json or {}
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    # Input length limit: 20 words as a proxy for 20 tokens
    if len(prompt.split()) > 20:
        return jsonify({"error": "Input prompt is limited to 20 tokens (words)."}), 400

    try:
        output = generate_text(prompt)
        return jsonify({"output": output})
    except Exception as e:
        app.logger.exception("Error during generation")
        if MODEL_INIT_ERROR:
            return jsonify({"error": f"Model initialization error: {MODEL_INIT_ERROR}"}), 500
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def http_analyze():
    data = request.json or {}
    prompt = data.get("prompt")
    output = data.get("output")
    if not prompt or not output:
        return jsonify({"error": 'Both "prompt" and "output" are required'}), 400

    try:
        analysis_data = analyze_full_report(prompt, output)
        return jsonify(analysis_data)
    except Exception as e:
        app.logger.exception("Error during analysis")
        if MODEL_INIT_ERROR:
            return jsonify({"error": f"Model initialization error: {MODEL_INIT_ERROR}"}), 500
        return jsonify({"error": str(e)}), 500


@app.route("/api/run_experiment", methods=["POST"])
def http_run_experiment():
    """
    Request body:
    {
      "type": "ablation" | "counterfactual",
      "original_prompt": "string",
      "variations": [
        "string variation 1",
        "string variation 2",
        ...
      ]
    }

    For 'ablation':
      - variations might remove or alter certain words
      - we compare the model outputs and highlight changes in content/style

    For 'counterfactual':
      - variations might flip some conditions in the prompt
      - we compare the outputs to see how they differ logically
    """
    data = request.json or {}
    experiment_type = data.get("type")
    original_prompt = data.get("original_prompt")
    variations = data.get("variations", [])

    if not experiment_type or experiment_type not in ["ablation", "counterfactual"]:
        return jsonify({"error": 'Invalid or missing "type". Must be "ablation" or "counterfactual".'}), 400
    if not original_prompt:
        return jsonify({"error": 'Missing "original_prompt".'}), 400
    if not isinstance(variations, list) or not variations:
        return jsonify({"error": 'Missing or invalid "variations". Must be a non-empty list of prompts.'}), 400

    # Input length limit for original and variations
    if len(original_prompt.split()) > 20:
        return jsonify({"error": "Original prompt is limited to 20 tokens (words)."}), 400
    for v in variations:
        if len(str(v).split()) > 20:
            return jsonify({"error": "Each variation prompt is limited to 20 tokens (words)."}), 400

    try:
        original_output = generate_text(original_prompt)

        variation_outputs = []
        for v in variations:
            out = generate_text(v)
            variation_outputs.append({"prompt": v, "output": out})

        original_analysis = analyze_full_report(original_prompt, original_output)

        experiments = []
        for var in variation_outputs:
            var_prompt = var["prompt"]
            var_output = var["output"]
            var_analysis = analyze_full_report(var_prompt, var_output)

            experiments.append({
                "variation_prompt": var_prompt,
                "variation_output": var_output,
                "variation_analysis": var_analysis
            })

        result = {
            "type": experiment_type,
            "original": {
                "prompt": original_prompt,
                "output": original_output,
                "analysis": original_analysis
            },
            "experiments": experiments
        }
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error running experiment")
        if MODEL_INIT_ERROR:
            return jsonify({"error": f"Model initialization error: {MODEL_INIT_ERROR}"}), 500
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)