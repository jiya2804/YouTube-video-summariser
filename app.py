from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import re
import os

app = Flask(__name__)
CORS(app)

# Initialize summarization pipeline
summarizer = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
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
    """Get transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([item['text'] for item in transcript_list])
        return transcript
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

def chunk_text(text, max_length=1024):
    """Split text into chunks for summarization"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
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
    """Summarize text using transformers"""
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
        "message": "YouTube Video Summarizer API",
        "endpoints": {
            "/summarize": "POST - Summarize a YouTube video",
            "/health": "GET - Health check"
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
        
        # Get transcript
        transcript = get_transcript(video_id)
        
        if not transcript:
            return jsonify({"error": "Could not fetch transcript"}), 400
        
        # Summarize
        summary = summarize_text(transcript)
        
        return jsonify({
            "video_id": video_id,
            "summary": summary,
            "transcript_length": len(transcript.split()),
            "summary_length": len(summary.split())
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
