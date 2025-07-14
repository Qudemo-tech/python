from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import process_video_endpoint, ask_question_endpoint, health_check, ProcessVideoRequest, AskQuestionRequest

app = FastAPI(title="QuDemo Video Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://qu-demo.vercel.app",
        "https://qudemo.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints from main.py
app.add_api_route("/process-video/{company_name}", process_video_endpoint, methods=["POST"])
app.add_api_route("/ask-question", ask_question_endpoint, methods=["POST"])
app.add_api_route("/health", health_check, methods=["GET"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 