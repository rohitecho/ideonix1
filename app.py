from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from pypdf import PdfReader

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

def get_subject_context_path(subject):
    safe_subject = secure_filename(subject) if subject else "default"
    return os.path.join(app.config['UPLOAD_FOLDER'], f"{safe_subject}_context") # Changed to directory

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    subject = data.get('subject')
    output_language = data.get('output_language', 'English') # Get Lang
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Get Context
    context_path = get_subject_context_path(subject)
    context_text = ""
    if os.path.exists(context_path):
        files = sorted([f for f in os.listdir(context_path) if os.path.isfile(os.path.join(context_path, f))], 
                       key=lambda x: os.path.getmtime(os.path.join(context_path, x)), reverse=True)[:3]
        for f in files:
            with open(os.path.join(context_path, f), 'r', encoding='utf-8', errors='ignore') as cf:
                context_text += f"\n--- {f} ---\n{cf.read()}\n"

    system_prompt = (f"You are a helpful AI tutor for the subject: {subject}. "
                     f"Use the following document context to answer questions: {context_text}. "
                     f"IMPORTANT: Answer strictly in the {output_language} language.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return jsonify({'response': response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    task = request.form.get('task', 'summary')
    target_lang = request.form.get('target_lang', 'C++')
    subject = request.form.get('subject', 'default')
    output_language = request.form.get('output_language', 'English')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        # Hack to prevent Flask reloader from restarting on .py file save in debug mode
        if filename.endswith('.py'):
            filename += '.txt'
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Extract text from PDF
            text = ""
            if filename.lower().endswith('.pdf'):
                reader = PdfReader(filepath)
                for page in reader.pages:
                    text += page.extract_text() or ""
            else:
                # Basic text file support
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

            # Truncate text
            text = text[:4000] 
            
            # SAVE CONTEXT TO FILE (now saves to a new file in the subject's context directory)
            # SAVE CONTEXT TO FILE (now saves to a new file in the subject's context directory)
            context_dir = get_subject_context_path(subject)
            if not os.path.exists(context_dir):
                os.makedirs(context_dir)
                
            context_filename = f"{secure_filename(os.path.splitext(filename)[0])}_{os.urandom(4).hex()}.txt"
            context_path = os.path.join(context_dir, context_filename)
            with open(context_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # Get Subject Context
            # Read relevant files (Limit to latest 3 for brevity in this demo)
            context_text = ""
            if os.path.exists(context_dir):
                files = sorted([f for f in os.listdir(context_dir) if os.path.isfile(os.path.join(context_dir, f))], 
                               key=lambda x: os.path.getmtime(os.path.join(context_dir, x)), reverse=True)[:3]
                for f in files:
                    with open(os.path.join(context_dir, f), 'r', encoding='utf-8', errors='ignore') as cf:
                        context_text += f"\n--- {f} ---\n{cf.read()}\n"

            # Construct Prompt
            base_instruction = f"Context from previous documents:\n{context_text}\n\nCurrent Document:\n"
            
            # Language Instruction
            lang_instruction = f"IMPORTANT: You MUST provide the ENTIRE response in {output_language} language." if output_language != "English" else ""

            prompt = ""
            if task == 'summary':
                prompt = f"{base_instruction}Analyze and summarize.\n\nText:\n{text}\n\n{lang_instruction}"
            elif task == 'keypoints': # Kept original keypoints task
                prompt = f"{base_instruction}Extract key points from the following text:\n\n{text}\n\n{lang_instruction}"
            elif task == 'quiz_generator': # Kept original quiz_generator task
                prompt = f"{base_instruction}Generate 3 quiz questions based on the following text:\n\n{text}\n\n{lang_instruction}"
            elif task == 'rap_song':
                prompt = f"{base_instruction}Write a rap song.\n\nText:\n{text}\n\n{lang_instruction}"
            elif task == 'feynman':
                prompt = f"{base_instruction}Explain like I'm 5.\n\nText:\n{text}\n\n{lang_instruction}"
            elif task == 'mind_map':
                prompt = (f"Generate a Mermaid.js graph. Return ONLY the Mermaid code (starting with 'graph TD'). "
                          f"Do NOT translate the Mermaid keywords (graph, TD, -->), but translate the node labels to {output_language}.\n\nText:\n{text}")
            elif task == 'cornell':
                prompt = (f"{base_instruction}Format into Cornell Notes (HTML). "
                          f"Translate content to {output_language}.\n\nText:\n{text}")
            elif task == 'code_translate':
                prompt = (f"Translate the following code ENTIRELY into {target_lang}. "
                          f"Return the FULL TRANSLATED CODE inside a markdown code block first. "
                          f"Then provide a brief explanation in {output_language}.\n\nCode:\n{text}")
            elif task == 'pseudocode':
                prompt = f"Convert to Pseudocode. Comments in {output_language}.\n\n{text}"
            elif task == 'optimization':
                prompt = (f"Optimize the following code for better performance (Time/Space Complexity). "
                          f"Return the FULL OPTIMIZED CODE inside a markdown code block first. "
                          f"Then explain the optimizations in {output_language}.\n\nCode:\n{text}")
            elif task == 'flashcards':
                prompt = f"Generate 5 flashcards. Return JSON array with 'question' and 'answer'. Translate content to {output_language}.\n\nText:\n{text}"
            else:
                prompt = f"{base_instruction}Analyze.\n\nText:\n{text}\n\n{lang_instruction}"

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = completion.choices[0].message.content
            return jsonify({"content": content})

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(filepath):
                 os.remove(filepath) # Clean up original upload

if __name__ == '__main__':
    app.run(debug=True, port=5001)