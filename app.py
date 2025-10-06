from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re
import os
import time
from random import uniform

app = Flask(__name__)
CORS(app)

# Initialize summarization pipeline
summarizer = None

def get_summarizer():
    """Load the summarization model only once"""
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_transcript(video_id):
    """Fetch transcript safely with delay and proxy support"""
    try:
        # Add random delay to avoid rate limits
        time.sleep(uniform(1.2, 2.5))

        # Optional: use a proxy (uncomment if needed)
        # proxies = {"https": "http://your-proxy:port"}
        # transcript_list = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies)

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([item['text'] for item in transcript_list])
        return transcript, None

    except Exception as e:
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "429" in error_msg:
            return None, "⚠️ YouTube is temporarily blocking requests. Please try again in a few minutes."
        elif "TranscriptsDisabled" in error_msg:
            return None, "🚫 This video has transcripts disabled."
        elif "NoTranscriptFound" in error_msg:
            return None, "❌ No transcript found for this video."
        else:
            return None, f"Unexpected error while fetching transcript: {error_msg}"


def chunk_text(text, max_length=1024):
    """Split large text into smaller chunks for summarization"""
    words = text.split()
    chunks, current_chunk, current_length = [], [], 0

    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def summarize_text(text):
    """Summarize the full transcript"""
    model = get_summarizer()
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        if len(chunk.split()) < 50:
            summaries.append(chunk)
            continue
        summary = model(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return ' '.join(summaries)


@app.route('/')
def home():
    return jsonify({
        "message": "🧠 YouTube Video Summarizer API",
        "endpoints": {
            "/summarize": "POST - Summarize a YouTube video",
            "/health": "GET - Check API status"
        }
    })


@app.route('/health')
def health():
    return jsonify({"status": "healthy"})


@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "No URL provided"}), 400

        url = data['url']
        video_id = extract_video_id(url)
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        # Get transcript safely
        transcript, error_message = get_transcript(video_id)
        if not transcript:
            return jsonify({"error": error_message}), 429

        # Summarize
        summary = summarize_text(transcript)

        return jsonify({
            "video_id": video_id,
            "summary": summary,
            "transcript_length": len(transcript.split()),
            "summary_length": len(summary.split())
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
