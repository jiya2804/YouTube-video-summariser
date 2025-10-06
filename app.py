from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import googleapiclient.discovery
import re
import os
import time
from random import uniform

app = Flask(__name__)
CORS(app)

# ========== CONFIG ==========
# Add your YouTube Data API key in Render dashboard or .env
YT_API_KEY = os.environ.get("YT_API_KEY", "")
summarizer = None


# ========== SUMMARIZER ==========
def get_summarizer():
    """Load the summarization model once"""
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer


# ========== YOUTUBE HELPERS ==========
def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)",
        r"youtube\.com\/embed\/([^&\n?#]+)",
        r"youtube\.com\/v\/([^&\n?#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_transcript_from_api(video_id):
    """Try using YouTube Data API v3 first"""
    try:
        if not YT_API_KEY:
            return None, "YouTube API key not configured."

        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YT_API_KEY)
        captions_response = youtube.captions().list(part="snippet", videoId=video_id).execute()

        if not captions_response["items"]:
            return None, "No captions available through YouTube Data API."

        caption_id = captions_response["items"][0]["id"]
        caption_download = youtube.captions().download(id=caption_id).execute()
        return caption_download, None

    except Exception as e:
        return None, str(e)


def get_transcript(video_id):
    """Get transcript with retry + fallback"""
    try:
        # First try official API
        transcript_text, api_error = get_transcript_from_api(video_id)
        if transcript_text:
            return transcript_text, None

        # Fallback to youtube-transcript-api
        time.sleep(uniform(1.2, 2.5))
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([item["text"] for item in transcript_list])
        return transcript, None

    except Exception as e:
        err = str(e)
        if "Too Many Requests" in err or "429" in err:
            return None, "⚠️ YouTube is temporarily blocking requests. Please try again later."
        elif "TranscriptsDisabled" in err:
            return None, "🚫 This video has transcripts disabled."
        elif "NoTranscriptFound" in err:
            return None, "❌ No transcript found for this video."
        else:
            return None, f"Unexpected error: {err}"


# ========== TEXT PROCESSING ==========
def chunk_text(text, max_length=1024):
    """Split large text into smaller chunks"""
    words = text.split()
    chunks, current_chunk, current_length = [], [], 0

    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def summarize_text(text):
    """Summarize transcript"""
    model = get_summarizer()
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        if len(chunk.split()) < 50:
            summaries.append(chunk)
            continue
        summary = model(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]["summary_text"])
    return " ".join(summaries)


# ========== ROUTES ==========
@app.route("/")
def home():
    return jsonify(
        {
            "message": "🧠 YouTube Video Summarizer API",
            "endpoints": {
                "/summarize": "POST - Summarize a YouTube video",
                "/health": "GET - Health check",
            },
        }
    )


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "No URL provided"}), 400

        url = data["url"]
        video_id = extract_video_id(url)
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        transcript, error_message = get_transcript(video_id)
        if not transcript:
            return jsonify({"error": error_message}), 429

        summary = summarize_text(transcript)
        return jsonify(
            {
                "video_id": video_id,
                "summary": summary,
                "transcript_length": len(transcript.split()),
                "summary_length": len(summary.split()),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
