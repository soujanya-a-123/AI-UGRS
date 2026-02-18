from datetime import datetime, timedelta
import datetime as dt  # Add this

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, session, jsonify
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib 
from email.message import EmailMessage
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
from flask import send_file
from datetime import datetime, timedelta, timezone
import io
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
import random
import pickle
import numpy as np
import tensorflow as tf
from functools import wraps

# Initialize stemmer
stemmer = LancasterStemmer()

# ---------------------------
# Flask & DB configuration
# ---------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///grievances.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "poisonousplants2024@gmail.com"
app.config["MAIL_PASSWORD"] = "wtfghdcknihmbaog"
app.config["MAIL_DEFAULT_SENDER"] = "poisonousplants2024@gmail.com"

db = SQLAlchemy(app)

# Global variables for chatbot
chat_model = None
chat_words = None
chat_labels = None
chat_data = None

def init_chatbot():
    """Initialize chatbot model and components"""
    global chat_model, chat_words, chat_labels, chat_data
    
    try:
        print("Initializing chatbot...")
        
        # Load intents
        with open("sample.json", encoding="utf-8") as f:
            chat_data = json.load(f)
        print("✓ Loaded intents data")
        
        # Load processed data
        with open("assets/input_data.pickle", "rb") as f:
            chat_words, chat_labels, chat_training, chat_output = pickle.load(f)
        print("✓ Loaded processed data")
        
        # Load model
        chat_model = tf.keras.models.load_model("assets/chatbot_model.keras")
        print("✓ Loaded chatbot model")
        
        print("Chatbot initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        # Create a simple fallback dataset
        create_fallback_chatbot()
        return False

def create_fallback_chatbot():
    """Create a simple fallback chatbot if ML model fails"""
    global chat_data, chat_words, chat_labels
    
    print("Creating fallback chatbot...")
    
    # Simple intents for fallback
    chat_data = {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["hi", "hello", "hey", "good morning", "good afternoon"],
                "responses": ["Hello! I'm your AI grievance assistant. How can I help you today?"]
            },
            {
                "tag": "fallback",
                "patterns": [""],
                "responses": ["I can help with grievance submissions, tracking complaints, department information, and general queries about the AI-UGRS system."]
            }
        ]
    }

def simple_bag_of_words(s, words):
    """Simple bag of words implementation"""
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s.lower())
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return np.array(bag)

def get_chat_response(inp):
    """Get chatbot response using ML model with proper error handling"""
    if not inp or len(inp.strip()) < 2:
        return "Please type a longer message (at least 2 characters)."
    
    # Clean input
    inp = inp.strip()
    
    # Ensure we have chat_data
    if not chat_data:
        create_fallback_chatbot()
    
    try:
        # Try ML model first if available
        if chat_model is not None and chat_words is not None and chat_labels is not None:
            # Create bag of words
            bow = simple_bag_of_words(inp, chat_words)
            
            # Predict using the model
            results = chat_model.predict(np.array([bow]), verbose=0)[0]
            results_index = np.argmax(results)
            tag = chat_labels[results_index]
            
            print(f"Model prediction - Tag: {tag}, Confidence: {results[results_index]:.4f}")
            
            # Use confidence threshold (lower for better coverage)
            if results[results_index] > 0.5:
                for tg in chat_data["intents"]:
                    if tg["tag"] == tag:
                        response = random.choice(tg["responses"])
                        print(f"ML Model response selected: {response}")
                        return response
        
        # Fall back to pattern matching if ML model is not confident or not available
        print("Falling back to pattern matching...")
        return get_simple_response(inp)
        
    except Exception as e:
        print(f"Chatbot ML error: {e}")
        # Final fallback to pattern matching
        return get_simple_response(inp)

def get_simple_response(inp):
    """Simple pattern-based response matching as fallback"""
    inp_lower = inp.lower()
    
    # Check each intent's patterns
    for intent in chat_data["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in inp_lower:
                response = random.choice(intent["responses"])
                print(f"Pattern match response: {response}")
                return response
    
    # If no pattern matches, use the noanswer intent
    noanswer_intent = next((tg for tg in chat_data["intents"] if tg["tag"] == "noanswer"), None)
    if noanswer_intent:
        response = random.choice(noanswer_intent["responses"])
        print(f"No answer fallback: {response}")
        return response
    
    # Ultimate fallback
    return "I'm here to help with the grievance redressal system. You can ask me about submitting complaints, tracking status, or general information about AI-UGRS."

def send_acknowledgement_email(citizen, complaint, ticket_id, department_label, priority):
    """Send complaint acknowledgement email to citizen."""
    
    # Email subject
    email_subject = f"Grievance Acknowledgement - Ticket ID: {ticket_id}"
    
    # Email body (your existing body content)
    email_body = f"""
Dear {citizen.name},

Thank you for submitting your grievance through the AI-Driven Unified Grievance Redressal System. 
We have successfully received your complaint and here are the details:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 COMPLAINT DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Ticket ID       : {ticket_id}
• Title           : {complaint.title}
• Description     : {complaint.description}
• Submitted Date  : {complaint.created_at.strftime('%d-%m-%Y %H:%M')}
• Priority        : {priority}
• Assigned Department : {department_label}
• Current Status  : Pending

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 LOCATION DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Address  : {complaint.address or 'Not provided'}
• District : {complaint.district or 'Not provided'}
• Ward/Zone: {complaint.ward or 'Not provided'}
• Location : {complaint.location or 'Not provided'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📞 NEXT STEPS & EXPECTATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Your complaint has been logged and will be reviewed by the concerned department.
2. Expected resolution timeframe: 7 days from submission.
3. You can track your complaint status using your Ticket ID: {ticket_id}
4. For urgent matters, please contact the respective department directly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 TRACK YOUR COMPLAINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To check the status of your complaint, please visit:
- Grievance Tracking Portal: [System URL]/track
- Use your Ticket ID: {ticket_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We appreciate you bringing this matter to our attention and will work towards 
a prompt resolution.

For any queries, please contact our support team or respond to this email.

Best regards,

AI-Driven Unified Grievance Redressal System
Urban Local Body - Citizen Services
[Your City/Municipality Name]
Support Email: support@ugrs.gov.in
Helpline: [Your Helpline Number]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 IMPORTANT NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• This is an automated acknowledgement. Please do not reply to this email.
• Keep this Ticket ID safe for future reference.
• You will receive updates when your complaint status changes.
• For emergency situations, please contact emergency services directly.
"""

    try:
        send_email(citizen.email, email_body, email_subject)
        return True
    except Exception as e:
        app.logger.error(f"Email sending failed for {citizen.email}: {e}")
        raise e

def generate_acknowledgement_pdf(complaint, citizen):
    """Generate PDF acknowledgement for a complaint."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=1,  # Center aligned
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=12,
    )
    
    normal_style = styles["Normal"]
    
    # Build the story (content)
    story = []
    
    # Title
    story.append(Paragraph("GRIEVANCE ACKNOWLEDGEMENT RECEIPT", title_style))
    story.append(Spacer(1, 20))
    
    # Complaint details
    story.append(Paragraph("Complaint Details:", heading_style))
    
    # Create table for complaint details
    ticket_id = format_ticket_id(complaint.id)
    complaint_data = [
        ["Ticket ID:", ticket_id],
        ["Title:", complaint.title],
        ["Description:", complaint.description],
        ["Submitted On:", complaint.created_at.strftime("%d-%m-%Y %H:%M")],
        ["Priority:", complaint.priority],
        ["Status:", complaint.status],
        ["Department:", complaint.department.name if complaint.department else "Not Assigned"],
    ]
    
    if complaint.address:
        complaint_data.append(["Address:", complaint.address])
    if complaint.district:
        complaint_data.append(["District:", complaint.district])
    if complaint.ward:
        complaint_data.append(["Ward/Zone:", complaint.ward])
    if complaint.location:
        complaint_data.append(["Location:", complaint.location])
    
    complaint_table = Table(complaint_data, colWidths=[1.5*inch, 4*inch])
    complaint_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    
    story.append(complaint_table)
    story.append(Spacer(1, 20))
    
    # Citizen details
    story.append(Paragraph("Citizen Details:", heading_style))
    
    citizen_data = [
        ["Name:", citizen.name],
        ["Email:", citizen.email],
        ["Complaint ID:", str(complaint.id)],
    ]
    
    citizen_table = Table(citizen_data, colWidths=[1.5*inch, 4*inch])
    citizen_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    
    story.append(citizen_table)
    story.append(Spacer(1, 20))
    
    # Notes section
    notes = [
        "This is an automated acknowledgement of your grievance submission.",
        "Please keep this receipt for future reference.",
        f"Use Ticket ID: {ticket_id} to track your complaint status.",
        "Expected resolution timeframe: 7 days from submission.",
        "For urgent matters, contact the respective department directly."
    ]
    
    story.append(Paragraph("Important Notes:", heading_style))
    for note in notes:
        story.append(Paragraph(f"• {note}", normal_style))
        story.append(Spacer(1, 6))
    
    # Footer
    story.append(Spacer(1, 30))
    story.append(Paragraph("AI-Driven Unified Grievance Redressal System", 
                          ParagraphStyle('Footer', parent=normal_style, alignment=1)))
    story.append(Paragraph("Urban Local Body - Citizen Services", 
                          ParagraphStyle('Footer', parent=normal_style, alignment=1)))
    story.append(Paragraph(f"Generated on: {datetime.now(timezone.utc).strftime('%d-%m-%Y %H:%M UTC')}", 
                          ParagraphStyle('Footer', parent=normal_style, alignment=1)))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------------------------
# Database Models
# ---------------------------
def send_email(receiver_email, message, subject):
    """Send email with error handling and logging."""
    try:
        msg = EmailMessage()
        msg.set_content(message)
        msg['Subject'] = subject
        msg['From'] = "poisonousplants2024@gmail.com"
        msg['To'] = receiver_email
        
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login("poisonousplants2024@gmail.com", "wtfghdcknihmbaog")
        s.send_message(msg)
        s.quit()
        
        app.logger.info(f"Email sent successfully to {receiver_email}")
        return True
    except smtplib.SMTPException as e:
        app.logger.error(f"SMTP error sending email to {receiver_email}: {e}")
        raise e
    except Exception as e:
        app.logger.error(f"Unexpected error sending email to {receiver_email}: {e}")
        raise e

class Citizen(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    complaints = db.relationship("Complaint", backref="citizen", lazy=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Department(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)

    complaints = db.relationship("Complaint", backref="department", lazy=True)
    officials = db.relationship("Official", backref="department_ref", lazy=True)


class Official(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'department' or 'admin'
    department_id = db.Column(db.Integer, db.ForeignKey("department.id"))

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # Ticket ID
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    address = db.Column(db.String(255))
    district = db.Column(db.String(120))
    ward = db.Column(db.String(120))
    location = db.Column(db.String(120))        # e.g. area / GPS text
    citizen_id = db.Column(db.Integer, db.ForeignKey("citizen.id"))
    department_id = db.Column(db.Integer, db.ForeignKey("department.id"))

    # NEW: priority predicted via NLP
    priority = db.Column(db.String(20), default="Medium")  # Low / Medium / Urgent

    status = db.Column(db.String(50), default="Pending")  # Pending / In-Progress / Resolved / Escalated
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deadline = db.Column(db.DateTime)
    is_escalated = db.Column(db.Boolean, default=False)
    resolution = db.Column(db.Text)
    resolved_at = db.Column(db.DateTime)


# ---------------------------
# Helper functions
# ---------------------------

def login_required_citizen(f):
    """Simple decorator for citizen-only routes."""
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("citizen_id"):
            flash("Please login as citizen first.", "warning")
            return redirect(url_for("citizen_login"))
        return f(*args, **kwargs)

    return wrapper


def login_required_official(role=None):
    """Decorator for official routes. role can be 'department' or 'admin'."""
    from functools import wraps

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not session.get("official_id"):
                flash("Please login as official first.", "warning")
                return redirect(url_for("official_login"))
            if role and session.get("official_role") != role:
                flash("You are not authorized to view that page.", "danger")
                return redirect(url_for("index"))
            return f(*args, **kwargs)

        return wrapper

    return decorator


def escalate_overdue_complaints():
    """Mark complaints as escalated if deadline passed and not resolved."""
    now = datetime.now(timezone.utc)
    overdue = Complaint.query.filter(
        Complaint.deadline.isnot(None),
        Complaint.deadline < now,
        Complaint.status != "Resolved",
        Complaint.is_escalated == False,
    ).all()
    for c in overdue:
        c.is_escalated = True
        c.status = "Escalated"
    if overdue:
        db.session.commit()


def format_ticket_id(complaint_id: int) -> str:
    """Format ticket like AI-UGRS-0001."""
    return f"AI-UGRS-{complaint_id:04d}"


# ---------------------------
# Simple NLP routing & priority
# ---------------------------

def classify_department_and_priority(description: str):
    """
    Very simple NLP classifier using keyword matching.
    Later you can replace this with a proper ML model.

    Returns: (department_object_or_None, priority_string)
    """
    text = (description or "").lower()

    # Keywords for department detection
    dept_keywords = {
        "Water Supply": [
            "water", "pipe", "tap", "sewage", "sewer", "drain",
            "drainage", "leak", "leaking", "overflow", "manhole"
        ],
        "Roads & Transport": [
            "road", "pothole", "traffic", "signal", "bus",
            "street light", "streetlight", "footpath", "bridge"
        ],
        "Sanitation": [
            "garbage", "trash", "waste", "cleaning", "sweep",
            "dirty", "litter", "toilet", "sanitation", "dustbin"
        ],
    }

    # Score each department based on matched keywords
    scores = {}
    for dept_name, keywords in dept_keywords.items():
        scores[dept_name] = sum(1 for kw in keywords if kw in text)

    chosen_dept = None
    if scores:
        best_dept = max(scores, key=scores.get)
        if scores[best_dept] > 0:
            chosen_dept = Department.query.filter_by(name=best_dept).first()

    # Priority classification
    urgent_keywords = [
        "accident", "injury", "injured", "flood", "fire", "collapsed",
        "danger", "dangerous", "no water", "no electricity", "blocked",
        "blockage", "burst", "major leak", "overflow", "snake", "short circuit"
    ]
    low_keywords = [
        "suggestion", "request", "feedback", "query", "inquiry",
        "information", "clarification", "minor"
    ]

    priority = "Medium"
    for kw in urgent_keywords:
        if kw in text:
            priority = "Urgent"
            break
    else:
        for kw in low_keywords:
            if kw in text:
                priority = "Low"
                break

    return chosen_dept, priority


# ---------------------------
# Routes – Public / Portal
# ---------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------
# Citizen Auth + Dashboard
# ---------------------------

@app.route("/citizen/register", methods=["GET", "POST"])
def citizen_register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(url_for("citizen_register"))
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("citizen_register"))

        existing = Citizen.query.filter_by(email=email).first()
        if existing:
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("citizen_login"))

        citizen = Citizen(name=name, email=email)
        citizen.set_password(password)
        db.session.add(citizen)
        db.session.commit()

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("citizen_login"))

    return render_template("citizen_register.html")


@app.route("/citizen/login", methods=["GET", "POST"])
def citizen_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        citizen = Citizen.query.filter_by(email=email).first()
        if not citizen or not citizen.check_password(password):
            flash("Invalid email or password.", "danger")
            return redirect(url_for("citizen_login"))

        session.clear()
        session["citizen_id"] = citizen.id
        session["citizen_name"] = citizen.name

        flash("Logged in successfully.", "success")
        return redirect(url_for("citizen_dashboard"))

    return render_template("citizen_login.html")


@app.route("/citizen/logout")
def citizen_logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/citizen/dashboard")
@login_required_citizen
def citizen_dashboard():
    citizen_id = session["citizen_id"]
    complaints = Complaint.query.filter_by(citizen_id=citizen_id).order_by(Complaint.created_at.desc()).all()
    total = len(complaints)
    resolved = len([c for c in complaints if c.status == "Resolved"])
    pending = len([c for c in complaints if c.status != "Resolved"])
    return render_template(
        "citizen_dashboard.html",
        total=total,
        resolved=resolved,
        pending=pending,
    )


@app.route("/citizen/submit", methods=["GET", "POST"])
@login_required_citizen
def submit_grievance():
    departments = Department.query.order_by(Department.name).all()

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        address = request.form.get("address", "").strip()
        district = request.form.get("district", "").strip()
        ward = request.form.get("ward", "").strip()
        location = request.form.get("location", "").strip()
        department_id = request.form.get("department_id") or None

        if not title or not description:
            flash("Title and description are required.", "danger")
            return redirect(url_for("submit_grievance"))

        # --- NLP routing & priority ---
        predicted_dept, priority = classify_department_and_priority(description)

        # if citizen selected department manually, that overrides NLP routing
        if department_id:
            final_dept_id = int(department_id)
            chosen_label = next((d.name for d in departments if d.id == final_dept_id), "Selected by citizen")
        else:
            final_dept_id = predicted_dept.id if predicted_dept else None
            chosen_label = predicted_dept.name if predicted_dept else "Not decided (admin will route)"

        complaint = Complaint(
            title=title,
            description=description,
            address=address,
            district=district,
            ward=ward,
            location=location,
            citizen_id=session["citizen_id"],
            department_id=final_dept_id,
            status="Pending",
            priority=priority,
            deadline=datetime.now(timezone.utc) + timedelta(days=7),
        )
        db.session.add(complaint)
        db.session.commit()

        ticket = format_ticket_id(complaint.id)
        
        # --- Send acknowledgement email to citizen ---
        citizen = Citizen.query.get(session["citizen_id"])
        try:
            send_acknowledgement_email(citizen, complaint, ticket, chosen_label, priority)
            email_status = "Acknowledgement email has been sent to your registered email address."
        except Exception as e:
            app.logger.error(f"Failed to send acknowledgement email for complaint {complaint.id}: {e}")
            email_status = "Note: Could not send acknowledgement email. Please check your email address or contact support."

        flash(
            f"Complaint submitted successfully! Ticket ID: {ticket}. "
            f"Auto department: {chosen_label}, Priority: {priority}. "
            f"{email_status}",
            "success",
        )
        return redirect(url_for("citizen_dashboard"))

    return render_template("submit_grievance.html", departments=departments)


@app.route("/citizen/my-complaints")
@login_required_citizen
def my_complaints():
    complaints = Complaint.query.filter_by(citizen_id=session["citizen_id"]).order_by(
        Complaint.created_at.desc()
    ).all()
    return render_template("my_complaints.html", complaints=complaints, format_ticket_id=format_ticket_id)


@app.route("/track", methods=["GET", "POST"])
def track_complaint():
    complaint = None
    ticket_id = None
    if request.method == "POST":
        ticket_id = request.form.get("ticket_id", "").strip()
        # ticket format: AI-UGRS-0001 or just number
        if ticket_id.upper().startswith("AI-UGRS-"):
            try:
                cid = int(ticket_id.split("-")[-1])
            except ValueError:
                cid = None
        else:
            try:
                cid = int(ticket_id)
            except ValueError:
                cid = None

        if cid is None:
            flash("Invalid Ticket ID format.", "danger")
        else:
            complaint = Complaint.query.get(cid)
            if not complaint:
                flash("No complaint found with that Ticket ID.", "warning")

    return render_template(
        "track_complaint.html",
        complaint=complaint,
        ticket_id=ticket_id,
        format_ticket_id=format_ticket_id,
    )


# ---------------------------
# Officials – Login
# ---------------------------

@app.route("/official/login", methods=["GET", "POST"])
def official_login():
    if request.method == "POST":
        role = request.form.get("role")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        official = Official.query.filter_by(username=username, role=role).first()
        if not official or not official.check_password(password):
            flash("Invalid credentials.", "danger")
            return redirect(url_for("official_login"))

        session.clear()
        session["official_id"] = official.id
        session["official_role"] = official.role
        session["official_name"] = official.name

        if role == "department":
            return redirect(url_for("department_dashboard"))
        else:
            return redirect(url_for("admin_dashboard"))

    roles = [("department", "Department"), ("admin", "System Admin")]
    return render_template("official_login.html", roles=roles)


@app.route("/official/logout")
def official_logout():
    session.clear()
    flash("Official logged out.", "info")
    return redirect(url_for("index"))


# ---------------------------
# Department Portal
# ---------------------------

@app.route("/department/dashboard")
@login_required_official(role="department")
def department_dashboard():
    official = Official.query.get(session["official_id"])
    if not official.department_id:
        flash("No department linked with this official account.", "danger")
        return redirect(url_for("official_logout"))

    dept_id = official.department_id
    complaints = Complaint.query.filter_by(department_id=dept_id).all()
    total = len(complaints)
    in_progress = len([c for c in complaints if c.status == "In-Progress"])
    resolved = len([c for c in complaints if c.status == "Resolved"])

    return render_template(
        "dept_dashboard.html",
        department=official.department_ref,
        total=total,
        in_progress=in_progress,
        resolved=resolved,
        complaints=complaints  # Pass complaints for location analysis
    )

@app.route("/department/login", methods=["GET", "POST"])
def department_login():
    # List of departments for dropdown
    departments = Department.query.order_by(Department.name).all()

    if request.method == "POST":
        dept_id = request.form.get("department_id")
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not dept_id or not username or not password:
            flash("Please select department and enter username & password.", "danger")
            return redirect(url_for("department_login"))

        try:
            dept_id = int(dept_id)
        except ValueError:
            flash("Invalid department selected.", "danger")
            return redirect(url_for("department_login"))

        # Only officials with role='department' for that department can login here
        official = Official.query.filter_by(
            username=username,
            role="department",
            department_id=dept_id
        ).first()

        if not official or not official.check_password(password):
            flash("Invalid credentials for selected department.", "danger")
            return redirect(url_for("department_login"))

        # Login success
        session.clear()
        session["official_id"] = official.id
        session["official_role"] = official.role
        session["official_name"] = official.name

        flash("Logged in successfully as Department Admin.", "success")
        return redirect(url_for("department_dashboard"))

    return render_template("department_login.html", departments=departments)

@app.route("/department/grievances", methods=["GET", "POST"])
@login_required_official(role="department")
def department_grievances():
    official = Official.query.get(session["official_id"])
    dept_id = official.department_id
    status_filter = request.args.get("status", "all")

    query = Complaint.query.filter_by(department_id=dept_id)
    if status_filter != "all":
        query = query.filter_by(status=status_filter)
    complaints = query.order_by(Complaint.created_at.desc()).all()

    return render_template(
        "dept_grievances.html",
        department=official.department_ref,
        complaints=complaints,
        status_filter=status_filter,
        format_ticket_id=format_ticket_id,
    )


@app.route("/department/update/<int:complaint_id>", methods=["POST"])
@login_required_official(role="department")
def update_complaint_status(complaint_id):
    complaint = Complaint.query.get_or_404(complaint_id)
    new_status = request.form.get("status")
    resolution = request.form.get("resolution", "").strip()
    new_priority = request.form.get("priority", complaint.priority)

    if new_status not in ["Pending", "In-Progress", "Resolved", "Escalated"]:
        flash("Invalid status.", "danger")
        return redirect(url_for("department_grievances"))

    if new_priority not in ["Low", "Medium", "Urgent"]:
        new_priority = complaint.priority

    complaint.status = new_status
    complaint.priority = new_priority

    if resolution:
        complaint.resolution = resolution

    just_resolved_now = False
    if new_status == "Resolved" and complaint.resolved_at is None:
        complaint.resolved_at = datetime.now(timezone.utc)
        just_resolved_now = True   # <- remember that this transition happened

    db.session.commit()

    # ---- Send email to citizen AFTER successfully resolving ----
    if just_resolved_now and complaint.citizen and complaint.citizen.email:
        try:
            ticket_id = format_ticket_id(complaint.id)
            citizen_name = complaint.citizen.name
            body = f"""Dear {citizen_name},
            Your grievance has been resolved.Ticket ID : {ticket_id}Title     : {complaint.title}
Resolution details:
{complaint.resolution or "The department has marked your complaint as resolved."}

Thank you for using the grievance redressal system.

Regards,
Urban Grievance Redressal System
"""

            send_email(
            complaint.citizen.email,
            body,
            "Your grievance has been resolved")
        except Exception as e:
            # Optional: log error instead of crashing the request
            app.logger.error(f"Failed to send resolution email for complaint {complaint.id}: {e}")

    flash("Complaint updated successfully.", "success")
    return redirect(url_for("department_grievances"))

# ---------------------------
# System Admin Portal
# ---------------------------

@app.route("/admin/dashboard")
@login_required_official(role="admin")
def admin_dashboard():
    escalate_overdue_complaints()

    total = Complaint.query.count()
    resolved = Complaint.query.filter_by(status="Resolved").count()
    escalated = Complaint.query.filter_by(status="Escalated").count()
    pending = total - resolved

    # Get priority counts for the chart
    low_priority = Complaint.query.filter_by(priority="Low").count()
    medium_priority = Complaint.query.filter_by(priority="Medium").count()
    urgent_priority = Complaint.query.filter_by(priority="Urgent").count()

    # Get all complaints for the table (optional)
    complaints = Complaint.query.all()

    return render_template(
        "admin_dashboard.html",
        total=total,
        resolved=resolved,
        escalated=escalated,
        pending=pending,
        low_priority=low_priority,
        medium_priority=medium_priority,
        urgent_priority=urgent_priority,
        complaints=complaints  # Pass complaints for other potential uses
    )


@app.route("/admin/grievances")
@login_required_official(role="admin")
def admin_grievances():
    escalate_overdue_complaints()

    dept_id = request.args.get("department_id")
    status_filter = request.args.get("status", "all")

    departments = Department.query.order_by(Department.name).all()

    query = Complaint.query
    if dept_id and dept_id != "all":
        query = query.filter_by(department_id=int(dept_id))
    if status_filter != "all":
        query = query.filter_by(status=status_filter)

    complaints = query.order_by(Complaint.created_at.desc()).all()

    return render_template(
        "admin_grievances.html",
        complaints=complaints,
        departments=departments,
        dept_id=dept_id or "all",
        status_filter=status_filter,
        format_ticket_id=format_ticket_id,
    )


@app.route("/admin/escalated")
@login_required_official(role="admin")
def admin_escalated():
    escalate_overdue_complaints()

    complaints = Complaint.query.filter_by(status="Escalated").order_by(
        Complaint.created_at.desc()
    ).all()
    
    current_time = datetime.now(timezone.utc)
    
    # Calculate statistics
    total_escalated = len(complaints)
    
    # Unique departments
    departments_count = len(set(c.department for c in complaints if c.department))
    
    # Average days overdue
    total_days = 0
    count_overdue = 0
    complaints_with_days = []
    
    for c in complaints:
        days_overdue = 0
        if c.deadline and c.deadline < current_time:
            days_overdue = (current_time - c.deadline).days
            total_days += days_overdue
            count_overdue += 1
        
        # Add days overdue to each complaint for the template
        complaints_with_days.append({
            'complaint': c,
            'days_overdue': days_overdue
        })
    
    avg_days_overdue = round(total_days / count_overdue, 1) if count_overdue > 0 else 0
    
    return render_template(
        "admin_escalated.html",
        complaints=complaints_with_days,  # Pass complaints with pre-calculated days
        format_ticket_id=format_ticket_id,
        total_escalated=total_escalated,
        departments_count=departments_count,
        avg_days_overdue=avg_days_overdue,
    )

@app.route("/admin/overview")
@login_required_official(role="admin")
def admin_overview():
    escalate_overdue_complaints()

    departments = Department.query.order_by(Department.name).all()
    overview = []
    for d in departments:
        total = Complaint.query.filter_by(department_id=d.id).count()
        resolved = Complaint.query.filter_by(department_id=d.id, status="Resolved").count()
        escalated = Complaint.query.filter_by(department_id=d.id, status="Escalated").count()
        overview.append(
            {
                "department": d,
                "total": total,
                "resolved": resolved,
                "escalated": escalated,
            }
        )

    return render_template("admin_overview.html", overview=overview)


# ---------------------------
# DB initialization & seed
# ---------------------------

def seed_data():
    """Create some default departments and officials for demo."""
    # 1) Ensure departments exist
    if Department.query.count() == 0:
        d1 = Department(name="Water Supply")
        d2 = Department(name="Roads & Transport")
        d3 = Department(name="Sanitation")
        db.session.add_all([d1, d2, d3])
        db.session.commit()

    # Fetch departments by name (works even if they already existed)
    water_dept = Department.query.filter_by(name="Water Supply").first()
    roads_dept = Department.query.filter_by(name="Roads & Transport").first()
    sanitation_dept = Department.query.filter_by(name="Sanitation").first()

    # 2) System admin
    if not Official.query.filter_by(username="admin").first():
        admin = Official(
            name="System Admin",
            username="admin",
            role="admin",
            department_id=None,
        )
        admin.set_password("admin")
        db.session.add(admin)
        db.session.commit()

    # 3) Department admins – each mapped to its own department

    if water_dept and not Official.query.filter_by(username="dept1").first():
        official = Official(
            name="Water Supply Officer",
            username="dept1",
            role="department",
            department_id=water_dept.id,
        )
        official.set_password("dept1")
        db.session.add(official)
        db.session.commit()

    if roads_dept and not Official.query.filter_by(username="dept2").first():
        official = Official(
            name="Roads & Transport Officer",
            username="dept2",
            role="department",
            department_id=roads_dept.id,
        )
        official.set_password("dept2")
        db.session.add(official)
        db.session.commit()

    if sanitation_dept and not Official.query.filter_by(username="dept3").first():
        official = Official(
            name="Sanitation Officer",
            username="dept3",
            role="department",
            department_id=sanitation_dept.id,
        )
        official.set_password("dept3")
        db.session.add(official)
        db.session.commit()

@app.route("/citizen/complaint/<int:complaint_id>/download")
@login_required_citizen
def download_acknowledgement(complaint_id):
    """Download complaint acknowledgement as PDF."""
    complaint = Complaint.query.get_or_404(complaint_id)
    citizen = Citizen.query.get(session["citizen_id"])
    
    # Ensure the complaint belongs to the logged-in citizen
    if complaint.citizen_id != citizen.id:
        flash("You are not authorized to download this acknowledgement.", "danger")
        return redirect(url_for("my_complaints"))
    
    pdf_buffer = generate_acknowledgement_pdf(complaint, citizen)
    ticket_id = format_ticket_id(complaint.id)
    filename = f"complaint_acknowledgement_{ticket_id}.pdf"
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )

# ---------------------------
# Chatbot Routes
# ---------------------------

@app.route('/chatbot', methods=['POST'])
def chatbot():

    """Chatbot endpoint - accessible without login for public use"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'response': 'Invalid request format'}), 400
            
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'response': 'Please enter a message.'}), 400
        
        if len(user_message) < 2:
            return jsonify({'response': 'Please type a longer message (at least 2 characters).'}), 400
        
        print(f"Received message: {user_message}")
        
        # Get response from chatbot
        response = get_chat_response(user_message)
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Chatbot endpoint error: {e}")
        return jsonify({'response': "I'm experiencing technical difficulties. Please try again later."}), 500

@app.route('/chatbot/status')
def chatbot_status():
    """Debug endpoint to check chatbot status"""
    status = {
        'chat_data_loaded': bool(chat_data),
        'chat_model_loaded': bool(chat_model),
        'chat_words_loaded': bool(chat_words),
        'chat_labels_loaded': bool(chat_labels),
        'intents_count': len(chat_data['intents']) if chat_data else 0
    }
    return jsonify(status)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        seed_data()
        init_chatbot()
    app.run(host='0.0.0.0', port=5000,debug=True)