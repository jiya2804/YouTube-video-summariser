from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

# Load a smaller summarization model
# This is lighter than the default distilbart-cnn-12-6
summariser = pipeline(
    "summarization",
    model="facebook/bart-base"  # smaller model
)

@app.get('/summary')
def summary_api():
    url = request.args.get('url', '')
    if not url:
        return {"error": "URL parameter missing"}, 400

    try:
        video_id = url.split('=')[1]
    except IndexError:
        return {"error": "Invalid YouTube URL"}, 400

    try:
        transcript = get_transcript(video_id)
        summary = get_summary(transcript)
        return jsonify({"summary": summary}), 200
    except Exception as e:
        return {"error": str(e)}, 500

def get_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([d['text'] for d in transcript_list])
    return transcript

def get_summary(transcript):
    summary = ''
    chunk_size = 500  # smaller chunks to save memory
    for i in range(0, (len(transcript) // chunk_size) + 1):
        chunk = transcript[i*chunk_size:(i+1)*chunk_size]
        summary_text = summariser(chunk)[0]['summary_text']
        summary += summary_text + ' '
    return summary.strip()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
