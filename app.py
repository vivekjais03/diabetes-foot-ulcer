from flask import Flask, render_template, request, jsonify, send_file, session
import os
import base64
import io
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Import our enhanced model
from notebooks.enhanced_model import EnhancedFootUlcerModel

app = Flask(__name__, static_url_path='', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.urandom(24)

# Initialize the model (prefer EfficientNet if available)
MODEL_PATHS = [
    "models/foot_ulcer_efficientnet.h5",
    "models/foot_ulcer_efficientnet_best.h5",
    "models/foot_ulcer_model.h5",
]

model = None
for mp in MODEL_PATHS:
    if os.path.exists(mp):
        model = EnhancedFootUlcerModel(mp)
        print(f"✅ Model loaded successfully from {mp}!")
        break
if model is None:
    print("❌ No model found! Expected one of:")
    for mp in MODEL_PATHS:
        print(f" - {mp}")

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Save uploaded file
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Analyze image
            confidence_threshold = float(request.form.get('confidence_threshold', 0.8))
            print(f"🔍 Analyzing image with threshold: {confidence_threshold}")

            result = model.analyze_image_with_xai(filepath, confidence_threshold)
            print(f"📊 Analysis result: {result['class']}, Confidence: {result['confidence']}")
            print(f"🔍 XAI available: SHAP={result.get('shap_explanation') is not None}, LIME={result.get('lime_explanation') is not None}")
            print(f"🎯 Final result: {result['class']} with confidence: {result['confidence']:.3f}")

            # Get remedies
            remedies = model.get_remedies(result)
            print(f"💊 Remedies generated for severity: {remedies['severity']}")

            # Convert overlay image to base64 for display
            overlay_b64 = None
            if result.get('overlay') is not None:
                try:
                    # Convert BGR to RGB
                    overlay_rgb = cv2.cvtColor(result['overlay'], cv2.COLOR_BGR2RGB)
                    overlay_pil = Image.fromarray(overlay_rgb)

                    # Convert to base64
                    buffer = io.BytesIO()
                    overlay_pil.save(buffer, format='JPEG')
                    overlay_b64 = base64.b64encode(buffer.getvalue()).decode()
                except Exception as e:
                    print(f"Warning: Could not process overlay image: {e}")
                    overlay_b64 = None

            # Convert original image to base64
            original_img = result['original_img']
            buffer = io.BytesIO()
            original_img.save(buffer, format='JPEG')
            original_b64 = base64.b64encode(buffer.getvalue()).decode()

            # Ensure all values are JSON serializable
            response_data = {
                'success': True,
                'filename': str(filename),
                'class': str(result['class']),
                'meets_threshold': bool(result['meets_threshold']),
                'original_image': str(original_b64) if original_b64 else None,
                'overlay_image': str(overlay_b64) if overlay_b64 else None,
                'remedies': {
                    'severity': str(remedies['severity']),
                    'remedies': [str(r) for r in remedies['remedies']]
                },
                'xai_explanation': result.get('textual_explanation', ''),
                'shap_available': result.get('shap_explanation') is not None,
                'lime_available': result.get('lime_explanation') is not None,
                'timestamp': str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            }

            print(f"✅ Sending response with class: {response_data['class']}")
            return jsonify(response_data)

        except Exception as e:
            import traceback
            print(f"❌ Error during analysis: {str(e)}")
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    return jsonify({'error': 'File upload failed'}), 400

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.json
        # Log only essential data, not the large base64 images
        log_data = {
            'filename': data.get('filename'),
            'class': data.get('class'),
            'timestamp': data.get('timestamp')
        }
        print(f"🔍 Generating PDF with data: {log_data}")
        
        # Create PDF
        filename = f"foot_ulcer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
            textColor=colors.darkblue
        )
        story.append(Paragraph("Foot Ulcer Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Date and Time
        story.append(Paragraph(f"<b>Date & Time:</b> {data['timestamp']}", styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Prediction Results
        story.append(Paragraph("<b>Analysis Results:</b>", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Determine color based on prediction
        if "Abnormal" in data['class']:
            result_color = colors.red
            severity_text = f"Severity: {data['remedies']['severity']}"
        else:
            result_color = colors.green
            severity_text = "Status: Healthy"
        
        result_style = ParagraphStyle(
            'ResultStyle',
            parent=styles['Normal'],
            fontSize=14,
            textColor=result_color,
            spaceAfter=10
        )
        
        story.append(Paragraph(f"<b>Prediction:</b> {data['class']}", result_style))
        story.append(Paragraph(f"<b>{severity_text}</b>", result_style))
        story.append(Spacer(1, 20))
        
        # Remedies and Care Tips
        story.append(Paragraph("<b>Care Recommendations:</b>", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        for remedy in data['remedies']['remedies']:
            story.append(Paragraph(f"• {remedy}", styles['Normal']))
            story.append(Spacer(1, 5))
        
        story.append(Spacer(1, 20))
        
        # Add XAI Textual Explanation if available
        if data.get('textual_explanation'):
            story.append(Paragraph("<b>Explainable AI (XAI) Explanation:</b>", styles['Heading2']))
            story.append(Spacer(1, 10))
            # Split textual explanation into paragraphs by line breaks if any
            for para in data['textual_explanation'].split('\n'):
                story.append(Paragraph(para.strip(), styles['Normal']))
                story.append(Spacer(1, 5))
            story.append(Spacer(1, 20))
        
        # Important Notes
        story.append(Paragraph("<b>Important Notes:</b>", styles['Heading3']))
        story.append(Spacer(1, 10))
        story.append(Paragraph("• This analysis is for screening purposes only", styles['Normal']))
        story.append(Paragraph("• Always consult healthcare professionals for medical decisions", styles['Normal']))
        story.append(Paragraph("• Keep this report for your medical records", styles['Normal']))
        
        # Add images to PDF if available
        story.append(Spacer(1, 20))
        if data.get('original_image'):
            original_img_bytes = base64.b64decode(data['original_image'])
            original_img_io = io.BytesIO(original_img_bytes)
            story.append(Paragraph("<b>Original Image:</b>", styles['Heading3']))
            story.append(RLImage(original_img_io, width=3*inch, height=3*inch))
            story.append(Spacer(1, 10))
        if data.get('overlay_image'):
            overlay_img_bytes = base64.b64decode(data['overlay_image'])
            overlay_img_io = io.BytesIO(overlay_img_bytes)
            story.append(Paragraph("<b>Overlay Image:</b>", styles['Heading3']))
            story.append(RLImage(overlay_img_io, width=3*inch, height=3*inch))
            story.append(Spacer(1, 10))
        
        # Add SHAP explanation image if available
        if data.get('shap_explanation_image'):
            try:
                shap_img_bytes = base64.b64decode(data['shap_explanation_image'])
                shap_img_io = io.BytesIO(shap_img_bytes)
                story.append(Paragraph("<b>SHAP Explanation:</b>", styles['Heading3']))
                story.append(RLImage(shap_img_io, width=3*inch, height=3*inch))
                story.append(Spacer(1, 10))
            except Exception as e:
                print(f"⚠️ Could not add SHAP image to PDF: {e}")
        
        # Add LIME explanation image if available
        if data.get('lime_explanation_image'):
            try:
                lime_img_bytes = base64.b64decode(data['lime_explanation_image'])
                lime_img_io = io.BytesIO(lime_img_bytes)
                story.append(Paragraph("<b>LIME Explanation:</b>", styles['Heading3']))
                story.append(RLImage(lime_img_io, width=3*inch, height=3*inch))
                story.append(Spacer(1, 10))
            except Exception as e:
                print(f"⚠️ Could not add LIME image to PDF: {e}")
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ PDF generated successfully: {filename}")
        
        return jsonify({
            'success': True,
            'pdf_filename': filename,
            'download_url': f'/download_pdf/{filename}'
        })
    except Exception as e:
        import traceback
        print(f"❌ Error during PDF generation: {str(e)}")
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    else:
        return "PDF not found", 404

@app.route('/xai_analysis', methods=['POST'])
def xai_analysis():
    """Endpoint for XAI-only analysis without full image processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Save uploaded file temporarily
        filename = f"xai_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Analyze image with XAI features only
            confidence_threshold = float(request.form.get('confidence_threshold', 0.8))
            print(f"🔍 Performing XAI analysis with threshold: {confidence_threshold}")

            result = model.analyze_image_with_xai(filepath, confidence_threshold)

            # Clean up the temporary file
            os.remove(filepath)

            # Prepare XAI-specific response
            xai_response = {
                'success': True,
                'filename': str(file.filename),
                'class': str(result['class']),
                'confidence': float(result['confidence']),
                'meets_threshold': bool(result['meets_threshold']),
                'xai_explanation': result.get('textual_explanation', ''),
                'shap_available': result.get('shap_explanation') is not None,
                'lime_available': result.get('lime_explanation') is not None,
                'xai_features': {
                    'shap_available': result.get('shap_explanation') is not None,
                    'lime_available': result.get('lime_explanation') is not None,
                    'textual_explanation': result.get('textual_explanation', ''),
                    'uncertainty': result.get('uncertainty', 0.0)
                },
                'timestamp': str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            }

            # Add SHAP data if available
            if result.get('shap_explanation'):
                xai_response['xai_features']['shap_values'] = result['shap_explanation'].get('values', [])
                xai_response['xai_features']['shap_base_values'] = result['shap_explanation'].get('base_values', [])
                xai_response['xai_features']['feature_names'] = result['shap_explanation'].get('feature_names', [])

            # Add LIME data if available
            if result.get('lime_explanation'):
                xai_response['xai_features']['lime_explanation'] = result['lime_explanation']

            print(f"✅ XAI analysis completed: SHAP={xai_response['xai_features']['shap_available']}, LIME={xai_response['xai_features']['lime_available']}")
            return jsonify(xai_response)

        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            import traceback
            print(f"❌ Error during XAI analysis: {str(e)}")
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return jsonify({'error': f'XAI analysis failed: {str(e)}'}), 500

    return jsonify({'error': 'File upload failed'}), 400

@app.route('/chat_image_analysis', methods=['POST'])
def chat_image_analysis():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save uploaded file temporarily
        filename = f"chat_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Analyze image
            confidence_threshold = float(request.form.get('confidence_threshold', 0.8))
            result = model.analyze_image(filepath, confidence_threshold)
            remedies = model.get_remedies(result)
            
            # Clean up the temporary file
            os.remove(filepath)

            response_data = {
                'success': True,
                'filename': str(file.filename),
                'class': str(result['class']),
                'confidence': float(result['confidence']),
                'remedies': {
                    'severity': str(remedies['severity']),
                    'remedies': [str(r) for r in remedies['remedies']]
                }
            }
            
            # Store key results in session for chatbot context
            session['last_analysis'] = {
                'class': str(result['class']),
                'confidence': float(result['confidence']),
                'severity': str(remedies['severity']),
                'remedies': [str(r) for r in remedies['remedies']]
            }

            return jsonify(response_data)
            
        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            import traceback
            print(f"❌ Error during chat image analysis: {str(e)}")
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'error': 'File upload failed'}), 400

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """AI Medical Assistant Chatbot with Multi-Language Support"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').lower().strip()
        user_language = data.get('language', 'en')  # Get user's language preference
        
        # Get context from session
        last_analysis = session.get('last_analysis')

        # Multi-language medical knowledge base
        medical_responses = {
            'en': get_english_responses(),
            'hi': get_hindi_responses(),
            'es': get_spanish_responses(),
            'fr': get_french_responses()
        }
        
        # Get responses for user's language
        responses = medical_responses.get(user_language, medical_responses['en'])
        
        # Generate intelligent response based on user input
        response = generate_ai_response(user_message, responses, user_language, last_analysis)
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        return jsonify({'response': 'I apologize, but I encountered an error. Please try rephrasing your question or contact your healthcare provider for immediate assistance.'})

def get_english_responses():
    """English medical responses"""
    return {
        'foot care': """Proper foot care is essential for preventing ulcers. Here are key practices:

• Wash feet daily with mild soap and warm water
• Dry thoroughly, especially between toes
• Moisturize with foot cream (avoid between toes)
• Check feet daily for cuts, blisters, or changes
• Wear well-fitting, supportive shoes
• Avoid walking barefoot

Would you like specific advice on any of these areas?""",
        'clean': """Here's how to properly clean foot ulcers:

• Use sterile saline solution or mild soap
• Gently clean with clean gauze or cloth
• Rinse thoroughly with clean water
• Pat dry (don't rub)
• Apply prescribed dressing
• Wash hands before and after

⚠️ Never use hydrogen peroxide or alcohol on ulcers as they can damage tissue and slow healing.""",
        'infection': """Watch for these signs of infection in foot ulcers:

🚨 **RED FLAGS** (Seek immediate medical care):
• Increased pain or swelling
• Redness spreading from ulcer
• Foul-smelling discharge
• Fever or chills
• Warmth around the area

⚠️ **EARLY WARNING SIGNS**:
• Increased drainage
• Color changes in tissue
• Delayed healing
• Unusual odor

Early detection is crucial for successful treatment.""",
        'emergency': """🚨 **IMMEDIATE MEDICAL CARE** if you experience:
SEEK 
• Severe pain that doesn't improve
• Fever above 100.4°F (38°C)
• Redness spreading from ulcer
• Foul-smelling discharge
• Black or dark tissue
• Difficulty breathing
• Chest pain

📞 **CALL 911 OR GO TO ER FOR**:
• Uncontrolled bleeding
• Severe infection signs
• Signs of sepsis

Early intervention saves lives and limbs!""",
        'diabetes': """Diabetes significantly affects ulcer healing:

🔬 **WHY DIABETES MATTERS**:
• Poor blood circulation
• Nerve damage (neuropathy)
• Reduced immune function
• Slower healing

📋 **DIABETES MANAGEMENT**:
• Control blood sugar levels
• Regular foot exams
• Immediate attention to any foot problems
• Work closely with healthcare team

Diabetic foot ulcers require specialized care and close monitoring.""",
        'prevent': """Prevent foot ulcers with these strategies:

👟 **FOOTWEAR**:
• Well-fitting, supportive shoes
• Avoid high heels and tight shoes
• Check shoes for foreign objects
• Replace worn-out shoes

🦶 **FOOT CARE**:
• Daily inspections
• Proper hygiene
• Moisturize dry skin
• Trim nails carefully
• Exercise feet regularly

📋 **HEALTH MANAGEMENT**:
• Control blood sugar (if diabetic)
• Maintain healthy weight
• Don't smoke
• Regular medical checkups""",
        'medical terminology': """Here are some common medical terms related to foot ulcers:

• **Debridement:** The removal of dead, damaged, or infected tissue to improve the healing potential of the remaining healthy tissue.
• **Granulation tissue:** New connective tissue and microscopic blood vessels that form on the surfaces of a wound during the healing process.
• **Exudate:** Fluid, such as pus or clear fluid, that leaks out of blood vessels into nearby tissues.
• **Slough:** A layer or mass of dead tissue separated from surrounding living tissue, often yellow or white.
• **Necrosis:** The death of most or all of the cells in an organ or tissue due to disease, injury, or failure of the blood supply."""
    }

def get_hindi_responses():
    """Hindi medical responses"""
    return {
        'foot care': "अल्सर को रोकने के लिए उचित पैर की देखभाल आवश्यक है। यहाँ मुख्य प्रथाएं हैं:\n\n• हल्के साबुन और गर्म पानी से रोज पैर धोएं\n• पूरी तरह से सुखाएं, विशेष रूप से उंगलियों के बीच\n• पैर क्रीम से मॉइस्चराइज करें (उंगलियों के बीच नहीं)\n• कट, छाले या परिवर्तन के लिए रोज पैर की जांच करें\n• अच्छी फिटिंग, सहायक जूते पहनें\n• नंगे पैर न चलें\n\nक्या आप इन क्षेत्रों में से किसी के बारे में विशिष्ट सलाह चाहते हैं?",
        'clean': "यहाँ फुट अल्सर को सही तरीके से साफ करने का तरीका है:\n\n• बाँझ नमकीन घोल या हल्का साबुन उपयोग करें\n• साफ गौज या कपड़े से धीरे से साफ करें\n• साफ पानी से अच्छी तरह धोएं\n• थपथपाकर सुखाएं (रगड़ें नहीं)\n• निर्धारित ड्रेसिंग लगाएं\n• पहले और बाद में हाथ धोएं\n\n⚠️ कभी भी हाइड्रोजन पेरोक्साइड या अल्कोहल का उपयोग न करें क्योंकि ये ऊतक को नुकसान पहुंचा सकते हैं और उपचार को धीमा कर सकते हैं।",
        'infection': "फुट अल्सर में संक्रमण के इन संकेतों पर ध्यान दें:\n\n🚨 लाल झंडे (तुरंत चिकित्सा देखभाल लें):\n• बढ़ता दर्द या सूजन\n• अल्सर से फैलता लालपन\n• बदबूदार निर्वहन\n• बुखार या ठंड लगना\n• क्षेत्र के आसपास गर्मी\n\n⚠️ प्रारंभिक चेतावनी संकेत:\n• बढ़ता निर्वहन\n• ऊतक में रंग परिवर्तन\n• देरी से उपचार\n• असामान्य गंध\n\nसफल उपचार के लिए शीघ्र पता लगाना महत्वपूर्ण है।",
        'emergency': "🚨 यदि आप इनका अनुभव करते हैं तो तुरंत चिकित्सा देखभाल लें:\n\n• गंभीर दर्द जो सुधरता नहीं है\n• 100.4°F (38°C) से ऊपर बुखार\n• अल्सर से फैलता लालपन\n• बदबूदार निर्वहन\n• काला या गहरा ऊतक\n• सांस लेने में कठिनाई\n• छाती में दर्द\n\n📞 इनके लिए 911 पर कॉल करें या ER में जाएं:\n• अनियंत्रित रक्तस्राव\n• गंभीर संक्रमण संकेत\n• सेप्सिस के संकेत\n\nशीघ्र हस्तक्षेप जीवन और अंगों को बचाता है!",
        'diabetes': "मधुमेह अल्सर उपचार को महत्वपूर्ण रूप से प्रभावित करता है:\n\n🔬 मधुमेह क्यों मायने रखता है:\n• खराब रक्त परिसंचरण\n• तंत्रिका क्षति (न्यूरोपैथी)\n• कम प्रतिरक्षा कार्य\n• धीमा उपचार\n\n📋 मधुमेह प्रबंधन:\n• रक्त शर्करा स्तर नियंत्रित करें\n• नियमित पैर परीक्षा\n• किसी भी पैर की समस्या पर तुरंत ध्यान\n• स्वास्थ्य देखभाल टीम के साथ काम करें\n\nमधुमेही पैर अल्सर को विशेष देखभाल और निकट निगरानी की आवश्यकता होती है।",
        'prevent': "इन रणनीतियों के साथ फुट अल्सर को रोकें:\n\n👟 फुटवियर:\n• अच्छी फिटिंग, सहायक जूते\n• ऊँची एड़ी और तंग जूते से बचें\n• जूतों में विदेशी वस्तुओं की जांच करें\n• पुराने जूते बदलें\n\n🦶 पैर की देखभाल:\n• दैनिक निरीक्षण\n• उचित स्वच्छता\n• सूखी त्वचा को मॉइस्चराइज करें\n• नाखूनों को सावधानी से काटें\n• नियमित रूप से पैर का व्यायाम करें\n\n📋 स्वास्थ्य प्रबंधन:\n• रक्त शर्करा नियंत्रित करें (यदि मधुमेही)\n• स्वस्थ वजन बनाए रखें\n• धूम्रपान न करें\n• नियमित चिकित्सा जांच",
        'medical terminology': "यहाँ फुट अल्सर से संबंधित कुछ सामान्य चिकित्सा शब्द दिए गए हैं:\n\n• **देब्रिडमेंट:** शेष स्वस्थ ऊतक की उपचार क्षमता में सुधार के लिए मृत, क्षतिग्रस्त, या संक्रमित ऊतक को हटाना।\n• **ग्रैनुलेशन ऊतक:** उपचार प्रक्रिया के दौरान घाव की सतहों पर बनने वाले नए संयोजी ऊतक और सूक्ष्म रक्त वाहिकाएं।\n• **एक्सयूडेट:** द्रव, जैसे मवाद या स्पष्ट द्रव, जो रक्त वाहिकाओं से आस-पास के ऊतकों में रिसता है।\n• **स्लफ:** आसपास के जीवित ऊतक से अलग मृत ऊतक की एक परत या द्रव्यमान, अक्सर पीला या सफेद।\n• **नेक्रोसिस:** बीमारी, चोट, या रक्त की आपूर्ति की विफलता के कारण किसी अंग या ऊतक में अधिकांश या सभी कोशिकाओं की मृत्यु।"
    }

def get_spanish_responses():
    """Spanish medical responses"""
    return {
        'foot care': "El cuidado adecuado de los pies es esencial para prevenir úlceras. Aquí están las prácticas clave:\n\n• Lave los pies diariamente con jabón suave y agua tibia\n• Seque completamente, especialmente entre los dedos\n• Hidratez con crema para pies (evite entre los dedos)\n• Revise los pies diariamente por cortes, ampollas o cambios\n• Use zapatos bien ajustados y de apoyo\n• Evite caminar descalzo\n\n¿Le gustaría consejos específicos sobre alguna de estas áreas?",
        'infection': "Esté atento a estos signos de infección en las úlceras del pie:\n\n🚨 BANDERAS ROJAS (Busque atención médica inmediata):\n• Dolor o hinchazón aumentados\n• Enrojecimiento que se extiende desde la úlcera\n• Descarga con mal olor\n• Fiebre o escalofríos\n• Calor alrededor del área\n\n⚠️ SIGNOS DE ADVERTENCIA TEMPRANA:\n• Drenaje aumentado\n• Cambios de color en el tejido\n• Cicatrisación retardada\n• Olor inusual\n\nLa detección temprana es crucial para el tratamiento exitoso.",
        'emergency': "🚨 BUSQUE ATENCIÓN MÉDICA INMEDIATA si experimenta:\n\n• Dolor severo que no mejora\n• Fiebre por encima de 100.4°F (38°C)\n• Enrojecimiento que se extiende desde la úlcera\n• Descarga con mal olor\n• Tejido negro o oscuro\n• Dificultad para respirar\n• Dolor en el pecho\n\n📞 LLAME AL 911 O VAYA A LA SALA DE EMERGENCIAS PARA:\n• Sangrado incontrolado\n• Signos de infección severa\n• Signos de sepsis\n\n¡La intervención temprana salva vidas y extremidades!",
        'medical terminology': "Aquí hay algunos términos médicos comunes relacionados con las úlceras del pie:\n\n• **Desbridamiento:** La eliminación de tejido muerto, dañado o infectado para mejorar el potencial de curación del tejido sano restante.\n• **Tejido de granulación:** Nuevo tejido conectivo y vasos sanguíneos microscópicos que se forman en las superficies de una herida durante el proceso de curación.\n• **Exudado:** Líquido, como pus o líquido claro, que se filtra de los vasos sanguíneos a los tejidos cercanos.\n• **Esfacelo:** Una capa o masa de tejido muerto separado del tejido vivo circundante, a menudo de color amarillo o blanco.\n• **Necrosis:** La muerte de la mayoría o la totalidad de las células de un órgano o tejido debido a una enfermedad, lesión o insuficiencia del suministro sanguíneo."
    }

def get_french_responses():
    """French medical responses"""
    return {
        'foot care': "Les soins appropriés des pieds sont essentiels pour prévenir les ulcères. Voici les pratiques clés :\n\n• Lavez les pieds quotidiennement avec du savon doux et de l'eau tiède\n• Séchez complètement, surtout entre les orteils\n• Hydratez avec de la crème pour pieds (évitez entre les orteils)\n• Vérifiez les pieds quotidiennement pour les coupures, ampoules ou changements\n• Portez des chaussures bien ajustées et de soutien\n• Évitez de marcher pieds nus\n\nSouhaitez-vous des conseils spécifiques sur l'un de ces domaines ?",
        'infection': "Surveillez ces signes d'infection dans les ulcères du pied :\n\n🚨 DRAPEAUX ROUGES (Consultez immédiatement un médecin) :\n• Douleur ou gonflement augmentés\n• Rougeur qui s'étend de l'ulcère\n• Écoulement malodorant\n• Fièvre ou frissons\n• Chaleur autour de la zone\n\n⚠️ SIGNAUX D'ALARME PRÉCOCES :\n• Écoulement augmenté\n• Changements de couleur dans les tissus\n• Cicatrisation retardée\n• Odeur inhabituelle\n\nLa détection précoce est cruciale pour un traitement réussi.",
        'emergency': "🚨 CONSULTEZ IMMÉDIATEMENT UN MÉDECIN si vous ressentez :\n\n• Douleur sévère qui ne s'améliore pas\n• Fièvre au-dessus de 100.4°F (38°C)\n• Rougeur qui s'étend de l'ulcère\n• Écoulement malodorant\n• Tissu noir ou sombre\n• Difficulté à respirer\n• Douleur thoracique\n\n📞 APPELEZ LE 911 OU ALLEZ AUX URGENCES POUR :\n• Saignement incontrôlé\n• Signes d'infection sévère\n• Signes de septicémie\n\nL'intervention précoce sauve des vies et des membres !",
        'medical terminology': "Voici quelques termes médicaux courants liés aux ulcères du pied :\n\n• **Débridement :** L'élimination des tissus morts, endommagés ou infectés pour améliorer le potentiel de guérison des tissus sains restants.\n• **Tissu de granulation :** Nouveau tissu conjonctif et vaisseaux sanguins microscopiques qui se forment à la surface d'une plaie pendant le processus de guérison.\n• **Exsudat :** Liquide, tel que du pus ou un liquide clair, qui s'échappe des vaisseaux sanguins vers les tissus voisins.\n• **Escarre :** Une couche ou une masse de tissu mort séparée des tissus vivants environnants, souvent de couleur jaune ou blanche.\n• **Nécrose :** La mort de la plupart ou de la totalité des cellules d'un organe ou d'un tissu en raison d'une maladie, d'une blessure ou d'une défaillance de l'apport sanguin."
    }

def get_result_interpretation(analysis_context, language='en'):
    """Provides a detailed, user-friendly interpretation of the analysis results."""
    
    # Default to English if language not supported
    interpretations = {
        'en': {
            'header': "Here's an explanation of your analysis results:",
            'class_title': "Prediction",
            'confidence_title': "Confidence Score",
            'severity_title': "Assessed Severity",
            'abnormal_desc': "The model detected signs consistent with a foot ulcer.",
            'normal_desc': "The model did not detect signs of a foot ulcer. The skin appears healthy.",
            'confidence_desc': "This score ({confidence:.2%}) indicates how confident the model is in its prediction. Higher is more certain.",
            'severity_desc': "This assesses the potential severity. '{severity}' suggests the ulcer's current state. This helps in understanding the required care.",
            'next_steps_title': "What to do next?",
            'next_steps_abnormal': "Please consult a healthcare professional to get a formal diagnosis and treatment plan. You can ask me for 'recommendations' for general care tips.",
            'next_steps_normal': "Continue with good foot care practices to maintain healthy skin. You can ask me about 'foot care' or 'prevention'.",
            'disclaimer': "<b>Disclaimer:</b> This is an automated analysis and not a medical diagnosis. Always consult a doctor for health concerns."
        },
        # Add other languages here if needed
    }
    
    lang_interp = interpretations.get(language, interpretations['en'])
    
    class_desc = lang_interp['abnormal_desc'] if 'Abnormal' in analysis_context['class'] else lang_interp['normal_desc']
    next_steps = lang_interp['next_steps_abnormal'] if 'Abnormal' in analysis_context['class'] else lang_interp['next_steps_normal']
    
    response = f"<b>{lang_interp['header']}</b><br><br>"
    response += f"• <b>{lang_interp['class_title']}:</b> {analysis_context['class']}<br><i>{class_desc}</i><br><br>"
    response += f"• <b>{lang_interp['confidence_title']}:</b> {analysis_context['confidence']:.2%}<br><i>{lang_interp['confidence_desc'].format(confidence=analysis_context['confidence'])}</i><br><br>"
    response += f"• <b>{lang_interp['severity_title']}:</b> {analysis_context['severity']}<br><i>{lang_interp['severity_desc'].format(severity=analysis_context['severity'])}</i><br><br>"
    response += f"• <b>{lang_interp['next_steps_title']}:</b><br><i>{next_steps}</i><br><br>"
    response += f"{lang_interp['disclaimer']}"
    
    return response

def generate_ai_response(user_message, medical_responses, language='en', analysis_context=None):
    """Generate intelligent AI response based on user input, language, and analysis context"""
    
    # Handle conversational closings
    if any(word in user_message for word in ['ok', 'okay']):
        casual_responses = {
            'en': "👍",
            'hi': "👍",
            'es': "👍",
            'fr': "👍"
        }
        return casual_responses.get(language, "👍")

    # Farewell "bye" response
    if 'bye' in user_message:
        bye_responses = {
            'en': "Goodbye! Take care of your feet. 👋",
            'hi': "अलविदा! अपने पैरों का ध्यान रखें। 👋",
            'es': "¡Adiós! Cuide sus pies. 👋",
            'fr': "Au revoir ! Prenez soin de vos pieds. 👋"
        }
        return bye_responses.get(language, bye_responses['en'])

    # "thanks"/"thank you" response
    if any(word in user_message for word in ['thanks', 'thank you']):
        closing_responses = {
            'en': "You're welcome! If you want to ask me anything, I'm here.",
            'hi': "आपका स्वागत है! यदि आपके कोई और प्रश्न हैं, तो बेझिझक पूछें। मैं यहाँ मदद करने के लिए हूँ।",
            'es': "¡De nada! Si tiene más preguntas, no dude en preguntar. Estoy aquí para ayudar.",
            'fr': "De rien ! Si vous avez d'autres questions, n'hésitez pas à les poser. Je suis là pour vous aider."
        }
        return closing_responses.get(language, closing_responses['en'])

    # Handle questions related to the last analysis if context exists
    if analysis_context:
        if any(word in user_message for word in ['result', 'analysis', 'what does it mean', 'explain', 'my results', 'understanding']):
            return get_result_interpretation(analysis_context, language)
        if any(word in user_message for word in ['remedies', 'recommendations', 'what should i do', 'care']):
            remedies_list = "<br>• ".join(analysis_context['remedies'])
            return f"Based on the analysis, here are the recommendations:<br>• {remedies_list}"
        if any(word in user_message for word in ['severity', 'how bad is it']):
            return f"The severity level was assessed as '{analysis_context['severity']}' based on the analysis."
        if any(word in user_message for word in ['class', 'prediction', 'what is it']):
            return f"The prediction was '{analysis_context['class']}' with a confidence of {analysis_context['confidence']:.2%}."

    # Check for exact matches first
    for key, response in medical_responses.items():
        if key in user_message:
            return response
    
    # Check for related terms and provide appropriate responses
    if any(word in user_message for word in ['help', 'what', 'how', 'why', 'when', 'where']):
        if any(word in user_message for word in ['clean', 'wash', 'hygiene']):
            return medical_responses.get('clean', medical_responses.get('foot care', 'I can help you with foot care questions.'))
        elif any(word in user_message for word in ['infection', 'infected', 'bacteria']):
            return medical_responses.get('infection', 'I can help you understand infection signs.')
        elif any(word in user_message for word in ['prevent', 'avoid', 'stop']):
            return medical_responses.get('prevent', 'I can help you with prevention strategies.')
        elif any(word in user_message for word in ['diabetes', 'diabetic', 'blood sugar']):
            return medical_responses.get('diabetes', 'I can help you understand diabetes and foot care.')
        elif any(word in user_message for word in ['emergency', 'urgent', 'immediate']):
            return medical_responses.get('emergency', 'I can help you understand emergency situations.')
        else:
            return medical_responses.get('foot care', 'I can help you with general foot care.')
    
    # Default response based on language
    default_responses = {
        'en': "I'm here to help with foot ulcer care and prevention. You can ask me about cleaning, infection signs, prevention, diabetes, or emergency situations. What would you like to know?",
        'hi': "मैं फुट अल्सर देखभाल और रोकथाम में मदद करने के लिए यहाँ हूँ। आप मुझसे सफाई, संक्रमण के संकेत, रोकथाम, मधुमेह, या आपातकालीन स्थितियों के बारे में पूछ सकते हैं। आप क्या जानना चाहते हैं?",
        'es': "Estoy aquí para ayudar con el cuidado y prevención de úlceras del pie. Puede preguntarme sobre limpieza, signos de infección, prevención, diabetes o situaciones de emergencia. ¿Qué le gustaría saber?",
        'fr': "Je suis ici pour vous aider avec les soins et la prévention des ulcères du pied. Vous pouvez me demander sur le nettoyage, les signes d'infection, la prévention, le diabète ou les situations d'urgence. Que souhaitez-vous savoir ?"
    }
    
    return default_responses.get(language, default_responses['en'])

if __name__ == '__main__':
    if model is None:
        print("❌ Cannot start app without model!")
        print("Please ensure models/foot_ulcer_model.h5 exists")
    else:
        print("🚀 Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
