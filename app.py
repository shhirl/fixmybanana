from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory, abort
import os
import base64
import time
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import requests
import json
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour"],
    storage_uri="memory://",
)

@app.errorhandler(429)
def ratelimit_handler(e):
    return render_template('429.html', limit_description=str(e.description)), 429

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def friendly_ai_error(last_error):
    """
    Turn the last OpenAI failure into a message safe to show users.
    The real status code and response body go to the logs (Railway → Deploy logs).
    """
    status = last_error[1] if last_error else None
    if status == 429:
        return "Our AI has hit its daily budget. Please try again tomorrow."
    if status in (401, 403):
        return "Our AI connection is misconfigured. Please try again later."
    if status == 404:
        return "Our AI model is temporarily unavailable. Please try again later."
    return "Our AI analysis is temporarily unavailable. Please try again in a few minutes."

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def purge_old_uploads(max_age_hours=24):
    """Delete uploaded photos older than max_age_hours, backing the privacy promise on the homepage."""
    cutoff = time.time() - max_age_hours * 3600
    for name in os.listdir(app.config['UPLOAD_FOLDER']):
        path = os.path.join(app.config['UPLOAD_FOLDER'], name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass

def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_banana_back_feedback(base64_image, api_key, model_name):
    """
    Get detailed feedback explaining why a handstand is classified as banana back
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model_name,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a handstand coach providing detailed feedback. "
                        "Analyze the handstand image and explain specifically why it shows 'banana back' form. "
                        "Focus on: spinal alignment, hip position, shoulder position, and overall body line. "
                        "Provide constructive, specific feedback in 2-3 sentences."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This handstand has been classified as 'banana back'. "
                                "Please analyze the image and explain specifically why this is banana back form. "
                                "What do you see in terms of spinal alignment, hip position, and overall body line?"
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
            "max_tokens": 200
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            app.logger.error("OpenAI feedback call failed: model=%s status=%s body=%s",
                             model_name, response.status_code, response.text[:300])
            return "Unable to generate detailed feedback at this time."
            
    except Exception as e:
        app.logger.exception("OpenAI feedback call raised")
        return f"Error generating feedback: {str(e)}"

def analyze_handstand_posture(image_path):
    """
    Analyze handstand posture using OpenAI's vision model
    """
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        return {
            'analysis': 'Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.',
            'form_quality': 'error',
            'detailed_feedback': None
        }
    
    # Test API key works
    try:
        test_headers = {"Authorization": f"Bearer {api_key}"}
        test_response = requests.get("https://api.openai.com/v1/models", headers=test_headers)
        if test_response.status_code != 200:
            app.logger.error("OpenAI key check failed: status=%s body=%s",
                             test_response.status_code, test_response.text[:300])
            return {
                'analysis': f'API key validation failed: {test_response.status_code}',
                'form_quality': 'error',
                'detailed_feedback': None
            }
    except Exception as e:
        return {
            'analysis': f'API connection test failed: {str(e)}',
            'form_quality': 'error',
            'detailed_feedback': None
        }
    
    try:
        # Encode image to base64
        base64_image = encode_image_to_base64(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Try models that support vision
        # Original v0 list first (so v0 behaviour is unchanged while those models exist),
        # then current models as fallbacks. gpt-4-turbo* shut down 2026-10-23 per
        # developers.openai.com/api/docs/deprecations; gpt-5.6-terra (lighter) and
        # gpt-5.6-sol (flagship) are the documented replacements.
        models_to_try = ["gpt-4o", "gpt-4-turbo", "gpt-4-turbo-2024-04-09",
                         "gpt-5.6-terra", "gpt-5.6-sol"]
        
        last_error = None
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
                    return {
                        'analysis': analysis_text,
                        'form_quality': form_quality,
                        'detailed_feedback': None
                    }
                elif "banana back" in analysis_text:
                    form_quality = "bad"
                    
                    # Get detailed feedback for banana back
                    detailed_feedback = get_banana_back_feedback(base64_image, api_key, model_name)
                    
                    return {
                        'analysis': analysis_text,
                        'form_quality': form_quality,
                        'detailed_feedback': detailed_feedback
                    }
                else:
                    form_quality = "unclear"
                    return {
                        'analysis': analysis_text,
                        'form_quality': form_quality,
                        'detailed_feedback': None
                    }
            else:
                last_error = (model_name, response.status_code, response.text[:300])
                app.logger.error("OpenAI classify failed: model=%s status=%s body=%s", *last_error)
            # If this model fails, try the next one
        # If we get here, all models failed
        app.logger.error("All vision models failed; last error: %s", last_error)
        return {
            'analysis': friendly_ai_error(last_error),
            'form_quality': 'error',
            'detailed_feedback': None
        }
        
    except requests.exceptions.RequestException as e:
        app.logger.exception("OpenAI request error")
        return {
            'analysis': f'API request error: {str(e)}',
            'form_quality': 'error',
            'detailed_feedback': None
        }
    except Exception as e:
        app.logger.exception("Error analyzing image")
        return {
            'analysis': f'Error analyzing image: {str(e)}',
            'form_quality': 'error',
            'detailed_feedback': None
        }

def load_samples():
    """The 'try a sample' photos and their recorded results (eval/samples.json, written by eval/make_samples.py)."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval/samples.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []

@app.route('/')
def index():
    feedback_submitted = request.args.get('submitted') == '1'
    return render_template('index.html', feedback_submitted=feedback_submitted, samples=load_samples())

@app.route('/sample/<sample_id>')
def sample_result(sample_id):
    """Result page for a sample photo: a recorded run from the eval, no live model call and no rate-limit slot."""
    sample = next((s for s in load_samples() if s['id'] == sample_id), None)
    if not sample:
        abort(404)
    return render_template('result.html', analysis=sample['analysis'], form_quality=sample['form_quality'],
                           detailed_feedback=sample.get('detailed_feedback'), uploaded_image=None, sample=sample)

@app.route('/sample-photo/<sample_id>')
def sample_photo(sample_id):
    sample = next((s for s in load_samples() if s['id'] == sample_id), None)
    if not sample:
        abort(404)
    testset = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval', 'testset')
    return send_from_directory(testset, sample['photo'])

@app.route('/how-its-built')
def how_its_built():
    """Static lab-notebook page. All numbers come from files the eval scripts write (eval/results/summary.json,
    eval/results/v0-taxonomy.json, eval/v0/raw_response.json, eval/labels.csv); missing files degrade gracefully."""
    import csv
    from collections import Counter
    base = os.path.dirname(os.path.abspath(__file__))

    def load_json(rel):
        try:
            with open(os.path.join(base, rel)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    summary = load_json('eval/results/summary.json') or {}
    counts = None
    try:
        with open(os.path.join(base, 'eval/labels.csv'), newline='') as f:
            rows = list(csv.DictReader(f))
        hs = [r for r in rows if r['is_handstand'] == 'yes']
        counts = dict(n=len(rows), hs=len(hs), ctl=len(rows) - len(hs),
                      view=Counter(r['view'] for r in hs), support=Counter(r['support'] for r in hs),
                      quality=Counter(r['quality'] for r in hs),
                      commons=sum('Commons' in r['source'] and 'derived' not in r['source'] for r in rows),
                      derived=sum('derived' in r['source'] for r in rows),
                      synthetic=sum('synthetic' in r['source'] for r in rows))
    except (OSError, KeyError):
        pass
    prompt_md = None
    try:
        with open(os.path.join(base, 'eval/v0/prompt.md')) as f:
            prompt_md = f.read()
    except OSError:
        pass
    return render_template('how_its_built.html', v0=summary.get('v0'), tax=load_json('eval/results/v0-taxonomy.json'),
                           raw=load_json('eval/v0/raw_response.json'), prompt_md=prompt_md, c=counts)

@app.route('/upload', methods=['POST'])
@limiter.limit("5 per day")
@limiter.limit("50 per day", key_func=lambda: "global_upload")
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        purge_old_uploads()
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
                             detailed_feedback=analysis_result.get('detailed_feedback'),
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