from flask import Flask, request, render_template, jsonify, session
import os
import pandas as pd
import numpy as np
from parser import parse_whatsapp_chat
from searcher import build_index, search, model

app = Flask(__name__)
app.secret_key = 'chattrace-secret-2024'

# Store data in memory during session
chat_data = {}

UPLOAD_FOLDER = '/tmp/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.txt'):
        return jsonify({'error': 'Please upload a .txt file'}), 400

    # Save file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Parse the chat
        df = parse_whatsapp_chat(filepath)

        if df.empty:
            return jsonify({'error': 'No messages found in file'}), 400

        # Build AI index
        embeddings = build_index(df)

        # Store in memory
        session_id = file.filename
        chat_data[session_id] = {
            'df': df,
            'embeddings': embeddings,
            'filename': file.filename
        }

        return jsonify({
            'success': True,
            'session_id': session_id,
            'total_messages': len(df),
            'participants': df['sender'].unique().tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search_chat():
    data = request.json
    query = data.get('query', '').strip()
    session_id = data.get('session_id', '')

    if not query:
        return jsonify({'error': 'Please enter a search query'}), 400

    if session_id not in chat_data:
        return jsonify({'error': 'Please upload a chat file first'}), 400

    try:
        df = chat_data[session_id]['df']
        embeddings = chat_data[session_id]['embeddings']

        results = search(query, df, embeddings, top_k=5)

        # Clean results for JSON
        clean_results = []
        for r in results:
            clean_results.append({
                'similarity_score': round(r['similarity_score'] * 100, 1),
                'matched_message': r['matched_message'],
                'sender': r['sender'],
                'date': r['date'],
                'time': r['time'],
                'context': r['context']
            })

        return jsonify({
            'success': True,
            'query': query,
            'results': clean_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['POST'])
def get_stats():
    data = request.json
    session_id = data.get('session_id', '')

    if session_id not in chat_data:
        return jsonify({'error': 'Please upload a chat file first'}), 400

    try:
        from stats import generate_stats
        df = chat_data[session_id]['df']
        stats = generate_stats(df)
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)