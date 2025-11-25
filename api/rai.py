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
    Initialize Gemini models once at startup.
    """
    global GENERATION_MODEL, ANALYSIS_MODEL, MODEL_INIT_ERROR
    try:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

        genai.configure(api_key=api_key)

        generation_model_name = os.environ.get(
            "GENERATION_MODEL_NAME", "gemini-2.0-flash"
        )
        analysis_model_name = os.environ.get(
            "ANALYSIS_MODEL_NAME", "gemini-2.5-flash"
        )

        GENERATION_MODEL = genai.GenerativeModel(generation_model_name)
        ANALYSIS_MODEL = genai.GenerativeModel(analysis_model_name)
        MODEL_INIT_ERROR = None
    except Exception as e:
        # Store the error so endpoints can return a helpful message
        MODEL_INIT_ERROR = str(e)


init_models()  # run once on import


# -------- 2. HELPERS --------

def clean_prompt(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_json_object(text: str):
    """
    Try to extract a JSON object from a raw model response.
    Returns (parsed_dict, error_message_or_none).
    """
    if not text:
        return None, "Empty analysis response from model."

    text = text.strip()
    # First try direct JSON
    try:
        return json.loads(text), None
    except Exception:
        pass

    # Try to find the first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None, "No JSON object found in analysis response."

    candidate = m.group(0)
    try:
        return json.loads(candidate), None
    except Exception as e:
        return None, f"Failed to parse JSON from analysis response: {e}"


# -------- 3. CORE GENERATION (NO LIMITS) --------

def generate_text(prompt: str) -> str:
    """
    Generate an output for the given prompt without word limits.
    """
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if GENERATION_MODEL is None:
        raise RuntimeError("Generation model is not initialized.")

    # Removed the artificial constraint instructions
    constrained_prompt = prompt.strip()

    response = GENERATION_MODEL.generate_content(constrained_prompt)
    full_output = response.text or ""

    # Removed truncation logic
    return full_output


# -------- 4. ANALYSIS (HEATMAP + CONNECTIONS) --------

def analyze_full_report(prompt: str, output: str) -> dict:
    """
    Use the analysis model to return structured JSON.
    """
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if ANALYSIS_MODEL is None:
        raise RuntimeError("Analysis model is not initialized.")

    system_instructions = """
You are an analysis model that must output ONLY a single JSON object.

You receive:
- A user prompt.
- The model's output for that prompt.

You must:
1. Estimate which prompt words most strongly influenced the output.
2. Represent your reasoning as a JSON object with this exact structure:

{
  "heatmap_data": [
    {
      "word": "string (a single word from the prompt)",
      "impact_score": number (1-5, integer or float; higher = more influential)
    },
    ...
  ],
  "connections": [
    {
      "prompt_word": "string (one influential word from the prompt)",
      "impact_score": number (1-5, consistent with heatmap scale)",
      "influenced_output_words": [
        "short phrase or word from the output",
        ...
      ]
    },
    ...
  ]
}

Guidelines:
- Focus on content, tone, and constraints (e.g., "formal", "polite", "short").
- Omit words that clearly have negligible impact (score 1) unless necessary.
- "influenced_output_words" should list 0-5 SHORT tokens/phrases from the model output.
- DO NOT include any explanations outside the JSON.
- DO NOT wrap the JSON in backticks.
- DO NOT add extra top-level keys beyond those requested (you may add a "raw_notes" field inside the object ONLY if needed).
"""

    prompt_text = f"""
User Prompt:
{prompt}

Model Output:
{output}
"""

    analysis_prompt = system_instructions + "\n\n" + prompt_text

    response = ANALYSIS_MODEL.generate_content(analysis_prompt)
    raw_text = (response.text or "").strip()

    data, parse_error = extract_json_object(raw_text)
    if parse_error:
        # Return error plus raw text so the frontend can show something helpful
        return {
            "error": parse_error,
            "heatmap_data": [],
            "connections": [],
            "raw_analysis_text": raw_text,
        }

    if not isinstance(data, dict):
        return {
            "error": "Analysis response was not a JSON object.",
            "heatmap_data": [],
            "connections": [],
            "raw_analysis_text": raw_text,
        }

    # Ensure required fields
    if "heatmap_data" not in data or not isinstance(data["heatmap_data"], list):
        data["heatmap_data"] = []
    if "connections" not in data or not isinstance(data["connections"], list):
        data["connections"] = []

    # Attach raw text for debugging if not already present
    if "raw_analysis_text" not in data:
        data["raw_analysis_text"] = raw_text

    return data


# -------- 5. EXPERIMENTS (ABLATION & COUNTERFACTUAL) --------

def ablate_prompt(original_prompt: str, changes_text: str) -> str:
    """
    Remove comma-separated terms from the prompt (case-insensitive).
    """
    if not changes_text:
        return original_prompt

    terms = [t.strip() for t in changes_text.split(",") if t.strip()]
    new_prompt = original_prompt
    for term in terms:
        # Remove as whole word (roughly)
        pattern = r"\b" + re.escape(term) + r"\b"
        new_prompt = re.sub(pattern, "", new_prompt, flags=re.IGNORECASE)

    # Collapse whitespace
    new_prompt = re.sub(r"\s+", " ", new_prompt).strip()
    return new_prompt


def parse_counterfactual_pairs(text: str) -> dict:
    """
    Parse lines of 'word, replacement' into a mapping.
    """
    mapping = {}
    if not text:
        return mapping

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        idx = line.find(",")
        if idx == -1:
            continue
        key = line[:idx].strip()
        value = line[idx + 1 :].strip()
        if key and value:
            mapping[key] = value

    return mapping


def apply_counterfactual(original_prompt: str, mapping: dict) -> str:
    """
    Replace words/phrases based on a mapping.
    """
    new_prompt = original_prompt
    if not isinstance(mapping, dict):
        return new_prompt

    # Replace longer keys first to avoid partial overlaps
    items = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for src, dst in items:
        src = src.strip()
        dst = (dst or "").strip()
        if not src:
            continue
        pattern = r"\b" + re.escape(src) + r"\b"
        new_prompt = re.sub(pattern, dst, new_prompt, flags=re.IGNORECASE)

    new_prompt = re.sub(r"\s+", " ", new_prompt).strip()
    return new_prompt


def evaluate_change(original_prompt: str,
                    original_output: str,
                    new_prompt: str,
                    new_output: str) -> dict:
    """
    Ask the model to produce a semantic change score between 1 and 10
    plus a short summary.
    """
    if MODEL_INIT_ERROR:
        raise RuntimeError(MODEL_INIT_ERROR)
    if ANALYSIS_MODEL is None:
        raise RuntimeError("Analysis model is not initialized.")

    instructions = """
You are a model that compares two prompt–output pairs.

You must return ONLY a JSON object with:

{
  "semantic_change_score": number (1-10),
  "change_summary": "short natural language explanation (1-3 sentences)"
}

Where:
- 1 = almost no change in meaning, tone, or structure.
- 10 = very large change in meaning, tone, or structure.

Do NOT include any text outside the JSON.
"""

    comparison_text = f"""
Original Prompt:
{original_prompt}

Original Output:
{original_output}

New Prompt:
{new_prompt}

New Output:
{new_output}
"""

    full_prompt = instructions + "\n\n" + comparison_text
    response = ANALYSIS_MODEL.generate_content(full_prompt)
    raw_text = (response.text or "").strip()

    data, parse_error = extract_json_object(raw_text)
    if parse_error or not isinstance(data, dict):
        return {
            "semantic_change_score": 1,
            "change_summary": "Could not reliably compute change score; treating as minimal change.",
            "raw_change_text": raw_text,
        }

    # Defaults if missing
    score = data.get("semantic_change_score", 1)
    try:
        score = float(score)
    except Exception:
        score = 1.0

    summary = data.get("change_summary") or "No summary provided."

    return {
        "semantic_change_score": score,
        "change_summary": summary,
        "raw_change_text": raw_text,
    }


# -------- 6. FLASK ROUTES --------

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

    # LIMIT REMOVED: No 20-word check here anymore

    try:
        clean = clean_prompt(prompt)
        output = generate_text(clean)
        return jsonify({"output": output})
    except Exception as e:
        app.logger.exception("Error during generation")
        if MODEL_INIT_ERROR:
            return jsonify(
                {"error": f"Model initialization error: {MODEL_INIT_ERROR}"}
            ), 500
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
            return jsonify(
                {"error": f"Model initialization error: {MODEL_INIT_ERROR}"}
            ), 500
        return jsonify({"error": str(e)}), 500


@app.route("/api/run_experiment", methods=["POST"])
def http_run_experiment():
    data = request.json or {}
    experiment_type = data.get("type")
    original_prompt = data.get("original_prompt")
    original_output = data.get("original_output")
    changes = data.get("changes")

    if experiment_type not in ["ablation", "counterfactual"]:
        return jsonify(
            {"error": 'Invalid or missing "type". Must be "ablation" or "counterfactual".'}
        ), 400

    if not original_prompt or not original_output:
        return jsonify(
            {"error": '"original_prompt" and "original_output" are required.'}
        ), 400

    try:
        if experiment_type == "ablation":
            if not isinstance(changes, str) or not changes.strip():
                return jsonify(
                    {"error": 'For "ablation", "changes" must be a non-empty string like "formal, polite".'}
                ), 400
            new_prompt = ablate_prompt(original_prompt, changes)
        else:  # counterfactual
            if not isinstance(changes, str) or not changes.strip():
                return jsonify(
                    {
                        "error": 'For "counterfactual", "changes" must be a non-empty multiline string like:\nformal, informal\npolite, rude'
                    }
                ), 400
            mapping = parse_counterfactual_pairs(changes)
            if not mapping:
                return jsonify(
                    {
                        "error": 'Could not parse any "word, replacement" pairs. Use lines like: formal, informal'
                    }
                ), 400
            new_prompt = apply_counterfactual(original_prompt, mapping)

        new_output = generate_text(new_prompt)
        new_analysis = analyze_full_report(new_prompt, new_output)
        change_data = evaluate_change(
            original_prompt, original_output, new_prompt, new_output
        )

        result = {
            "new_prompt": new_prompt,
            "new_output": new_output,
            "new_analysis": new_analysis,
            "change_data": change_data,
        }
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Error running experiment")
        if MODEL_INIT_ERROR:
            return jsonify(
                {"error": f"Model initialization error: {MODEL_INIT_ERROR}"}
            ), 500
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)