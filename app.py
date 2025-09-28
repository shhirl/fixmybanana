from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory
import os
import base64
from werkzeug.utils import secure_filename
import requests
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_handstand_posture(image_path):
    """
    Analyze handstand posture using OpenAI's vision model
    """
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        return {
            'analysis': 'Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.',
            'form_quality': 'error'
        }
    
    # Test API key works
    try:
        test_headers = {"Authorization": f"Bearer {api_key}"}
        test_response = requests.get("https://api.openai.com/v1/models", headers=test_headers)
        if test_response.status_code != 200:
            return {
                'analysis': f'API key validation failed: {test_response.status_code}',
                'form_quality': 'error'
            }
    except Exception as e:
        return {
            'analysis': f'API connection test failed: {str(e)}',
            'form_quality': 'error'
        }
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Try models that support vision
        models_to_try = ["gpt-4o", "gpt-4-turbo", "gpt-4-turbo-2024-04-09"]
        
        for model_name in models_to_try:
            payload = {
                    "model": model_name,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a strict vision classifier. "
                                "Goal: From a SIDE-ON photo of a handstand, output exactly one label: "
                                "\"good form\" or \"banana back\". "
                                "Definitions: "
                                "• banana back = clear lumbar/spinal arch; ribs flare forward; hips in front of shoulders; "
                                "  legs/feet drift behind the body, making a C/banana shape. "
                                "• good form = wrists–shoulders–hips–ankles vertically stacked; neutral spine; ribs tucked; "
                                "  no visible midsection curve. "
                                "Rules: Output ONLY one of these strings with no punctuation or explanation."
                            )
                        },

                        # --- Few-shot text-only examples (no images needed) ---
                        {
                            "role": "user",
                            "content": (
                                "Side-on handstand description: hips are ahead of the shoulder line, "
                                "lower back is arched, chest/ribs flaring, legs trailing behind."
                            )
                        },
                        {"role": "assistant", "content": "banana back"},

                        {
                            "role": "user",
                            "content": (
                                "Side-on handstand description: wrists, shoulders, hips, ankles form one vertical line; "
                                "spine looks neutral; ribs tucked; toes stacked over hips."
                            )
                        },
                        {"role": "assistant", "content": "good form"},
                        # --- End few-shot ---

                        # Now ask the model to classify the actual uploaded image
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Classify this SIDE-ON handstand image as exactly one label: "
                                        "\"good form\" or \"banana back\"."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 5
                }

            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content'].strip().lower()
                if "banana" in analysis_text:
                    analysis_text = "banana back"
                elif "good" in analysis_text:
                    analysis_text = "good form"
                else:
                    analysis_text = analysis_text.splitlines()[0].strip()
                
                # Determine form quality based on response
                if "good form" in analysis_text:
                    form_quality = "good"
                elif "banana back" in analysis_text:
                    form_quality = "bad"
                else:
                    form_quality = "unclear"
                    
                return {
                    'analysis': analysis_text,
                    'form_quality': form_quality
                }
            # If this model fails, try the next one
        # If we get here, all models failed
        return {
            'analysis': 'All vision models failed. Please try again.',
            'form_quality': 'error'
        }
        
    except requests.exceptions.RequestException as e:
        return {
            'analysis': f'API request error: {str(e)}',
            'form_quality': 'error'
        }
    except Exception as e:
        return {
            'analysis': f'Error analyzing image: {str(e)}',
            'form_quality': 'error'
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the handstand
        analysis_result = analyze_handstand_posture(filepath)
        
        # Clean up uploaded file (optional)
        # os.remove(filepath)
        
        return render_template('result.html', 
                             analysis=analysis_result['analysis'],
                             form_quality=analysis_result['form_quality'],
                             uploaded_image=filename
                             )
    
    return redirect(url_for('index'))

# Add route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 1010))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)