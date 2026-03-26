import os
import json
import logging
import joblib
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

model_path = os.getenv("MODEL_PATH", "./data/model/model.joblib")
meta_path  = os.getenv("META_PATH",  "./data/model/meta.json")
host = "0.0.0.0"
port = int(os.getenv("PORT", 8080))

pipeline = joblib.load(model_path)
with open(meta_path) as f:
    meta = json.load(f)

LABELS: list[str] = meta["labels"]

#back-end
app = FastAPI(title="Language Detector API",
              description="Identify the language of a text snippet.",
              version="1.0.0",)

class PredictRequest(BaseModel):
    text: str

    model_config = {"json_schema_extra": {"example": {"text": "Bonjour tout le monde"}}}


class PredictResponse(BaseModel):
    language: str
    confidence: float
    top_3: list[dict]


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "num_languages": len(LABELS)}


@app.get("/languages", tags=["meta"])
def languages():
    return {"languages": LABELS}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty.")

    proba  = pipeline.predict_proba([text])[0]
    top_idx = proba.argsort()[::-1][:3]

    top_3 = [{"language": pipeline.classes_[i], "confidence": round(float(proba[i]), 4)} for i in top_idx]

    return PredictResponse(language=top_3[0]["language"],
                           confidence=top_3[0]["confidence"],
                           top_3=top_3,)


#front-end
def gradio_predict(text: str) -> str:
    if not text.strip():
        return "⚠️ Please enter some text."
 
    proba   = pipeline.predict_proba([text])[0]
    top_idx = proba.argsort()[::-1][:3]

    colours = ["#C44BFF", "#26d0a8", "#FF9F43"]
    flags = {
    "Estonian":   "🇪🇪",
    "Swedish":    "🇸🇪",
    "Thai":       "🇹🇭",
    "Tamil":      "🇮🇳",
    "Dutch":      "🇳🇱",
    "Japanese":   "🇯🇵",
    "Turkish":    "🇹🇷",
    "Latin":      "🏛️",
    "Urdu":       "🇵🇰",
    "Indonesian": "🇮🇩",
    "Portugese":  "🇵🇹",
    "French":     "🇫🇷",
    "Chinese":    "🇨🇳",
    "Korean":     "🇰🇷",
    "Hindi":      "🇮🇳",
    "Spanish":    "🇪🇸",
    "Pushto":     "🇦🇫",
    "Persian":    "🇮🇷",
    "Romanian":   "🇷🇴",
    "Russian":    "🇷🇺",
    "English":    "🇬🇧",
    "Arabic":     "🇸🇦",}

    medals = ["🥇", "🥈", "🥉"]
 
    lines = ["### 🔍 Top predictions\n"]
    for rank, i in enumerate(top_idx):
        lang  = pipeline.classes_[i]
        score = proba[i]
        flag  = flags.get(lang, "🌍")
        #bar   = "█" * int(score * 20)
        pct   = f"{score:.1%}"
        lines.append(f"{medals[rank]} **{flag} {lang}** &nbsp; "
    f"<span style='color:{colours[rank]}'>`{pct}`</span> &nbsp; ")
    #f"<span style='color:{colours[rank]}'>`{bar}`</span>")
 
    return "\n\n".join(lines)

with gr.Blocks(title="🌐 Language Detector", theme=gr.themes.Soft()) as demo:
 
    gr.HTML("""
        <div id="hero-title">
            <h1>🌐 Language Detector</h1>
            <p>Type or paste any text — the model will identify its language instantly ✨</p>
        </div>
    """)
 
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Your text",
                placeholder="Type something here…",
                lines=4,
                elem_id="text-input",
            )
            submit_btn = gr.Button(
                "✨ Detect Language",
                variant="primary",
                elem_id="detect-btn",
            )
 
        with gr.Column(scale=1):
            output = gr.Markdown(
                value="*Enter text on the left to see predictions.*",
                elem_id="output-box",
            )
 
    submit_btn.click(fn=gradio_predict, inputs=text_input, outputs=output)
    text_input.submit(fn=gradio_predict, inputs=text_input, outputs=output)
 
    gr.Examples(
        examples=[
            ["Bonjour, comment allez-vous?"],
            ["Hello, how are you today?"],
            ["Hola, ¿cómo estás?"],
            ["Guten Morgen, wie geht es Ihnen?"],
            ["مرحبا، كيف حالك؟"],
            ["こんにちは、お元気ですか？"],
        ],
        inputs=text_input,
    )

    
app = gr.mount_gradio_app(app, demo, path="/web")

if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, log_level="info")