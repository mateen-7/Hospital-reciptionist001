"""
LangGraph Workflow for Hospital Receptionist
Manages conversation state and AI classification
"""

import os
import json
from typing import TypedDict, Optional, Literal
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# ─── State Schema ─────────────────────────────────────────────────────────────

class PatientState(TypedDict):
    session_id: str
    messages: list[dict]           # full conversation history
    patient_name: Optional[str]
    patient_age: Optional[str]
    patient_query: Optional[str]
    ward: Optional[str]
    step: str                      # current_step in collection flow
    is_complete: bool
    response: str                  # bot response to send back
    language: str                  # detected language: en / hi / te

# ─── LLM Setup ────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "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")

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key=ANTHROPIC_API_KEY,
    max_tokens=512
)

# ─── Emergency Keywords ────────────────────────────────────────────────────────

EMERGENCY_KEYWORDS = [
    # English
    "chest pain", "heart attack", "stroke", "unconscious", "bleeding", "accident",
    "emergency", "can't breathe", "cannot breathe", "severe pain", "overdose",
    "poisoning", "seizure", "falling", "broken bone", "fracture", "head injury",
    "trauma", "not breathing", "choking", "dying", "critical",
    # Hindi transliterated
    "seene mein dard", "dil ka daura", "behosh", "khoon", "saans nahi",
    "bahut dard", "haddi toot", "accident",
    # Telugu transliterated
    "gurram", "guunde noppi", "muttayi", "blood vastundi", "emergency",
]

MENTAL_HEALTH_KEYWORDS = [
    # English
    "depression", "anxiety", "suicidal", "mental", "stress", "panic",
    "hallucination", "voices", "trauma", "ptsd", "bipolar", "schizophrenia",
    "self harm", "self-harm", "cutting", "hopeless", "worthless",
    # Hindi transliterated
    "mansik", "udaas", "tension", "chinta", "darr",
    # Telugu transliterated
    "manasik", "tension", "bayam",
]

# ─── Utility Helpers ──────────────────────────────────────────────────────────

def detect_emergency(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in EMERGENCY_KEYWORDS)

def detect_mental_health(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in MENTAL_HEALTH_KEYWORDS)

def detect_language(text: str) -> str:
    """Basic language detection based on script/keywords"""
    # Telugu script range
    if any('\u0C00' <= c <= '\u0C7F' for c in text):
        return "te"
    # Hindi/Devanagari script range
    if any('\u0900' <= c <= '\u097F' for c in text):
        return "hi"
    return "en"

def greeting_for_lang(lang: str) -> str:
    greetings = {
        "en": "Hello! Welcome to the hospital. I'm your AI receptionist. May I know your name please?",
        "hi": "Namaste! Aspatal mein aapka swagat hai. Main aapka AI receptionist hoon. Kripya apna naam batayein?",
        "te": "Namaskaram! Asupathri ki swaagatam. Nenu mee AI receptionist ni. Meeru peru cheppagalara?"
    }
    return greetings.get(lang, greetings["en"])

def age_prompt_for_lang(lang: str, name: str) -> str:
    prompts = {
        "en": f"Thank you, {name}! Could you please tell me your age?",
        "hi": f"Shukriya, {name}! Kripya apni umar batayein?",
        "te": f"Dhanyavaadalu, {name}! Meeru vayassu cheppagalara?"
    }
    return prompts.get(lang, prompts["en"])

def query_prompt_for_lang(lang: str) -> str:
    prompts = {
        "en": "I understand. Could you please describe your symptoms or reason for visiting the hospital today?",
        "hi": "Samajh gaya. Kripya apni takleef ya hospital aane ka karan batayein?",
        "te": "Artham chesukunnanu. Meeru ee roju asupathri ki vachina kaaranam cheppagalara?"
    }
    return prompts.get(lang, prompts["en"])

# ─── Graph Nodes ──────────────────────────────────────────────────────────────

def start_node(state: PatientState) -> PatientState:
    """Initial greeting or process first message"""
    messages = state.get("messages", [])
    if not messages:
        lang = "en"
        return {**state, "step": "collect_name", "language": lang, "response": greeting_for_lang(lang)}

    last_msg = messages[-1]["content"]
    lang = detect_language(last_msg)

    # If we're at the very start and this is the first user message, treat it as name
    if state.get("step") == "greeting":
        return {**state, "step": "collect_name", "language": lang, "response": greeting_for_lang(lang)}

    return {**state, "language": lang}


def collect_name_node(state: PatientState) -> PatientState:
    """Extract patient name from user message"""
    messages = state.get("messages", [])
    if not messages:
        return {**state, "step": "collect_name", "response": greeting_for_lang(state.get("language", "en"))}

    last_msg = messages[-1]["content"].strip()
    lang = state.get("language", "en")

    # Use LLM to extract name cleanly
    try:
        result = llm.invoke([
            SystemMessage(content="Extract only the person's name from this message. Reply with ONLY the name, nothing else. If no name found, reply with 'UNKNOWN'."),
            HumanMessage(content=last_msg)
        ])
        name = result.content.strip()
        if name == "UNKNOWN" or len(name) < 2:
            return {
                **state,
                "step": "collect_name",
                "response": "I'm sorry, I didn't catch your name. Could you please tell me your name?"
            }
    except Exception:
        name = last_msg.split()[-1] if last_msg else "Patient"

    return {
        **state,
        "patient_name": name,
        "step": "collect_age",
        "response": age_prompt_for_lang(lang, name)
    }


def collect_age_node(state: PatientState) -> PatientState:
    """Extract and validate patient age"""
    messages = state.get("messages", [])
    last_msg = messages[-1]["content"].strip() if messages else ""
    lang = state.get("language", "en")
    name = state.get("patient_name", "")

    # Extract numeric age
    import re
    numbers = re.findall(r'\b\d+\b', last_msg)
    if not numbers:
        error_msgs = {
            "en": "Please provide a valid age (numbers only). For example: 25",
            "hi": "Kripya sahi umar batayein (sirf number mein). Jaise: 25",
            "te": "Dayachesi valid vayassu ivvandi (numbers matrame). Udaharana: 25"
        }
        return {**state, "step": "collect_age", "response": error_msgs.get(lang, error_msgs["en"])}

    age = int(numbers[0])
    if age < 0 or age > 120:
        return {
            **state,
            "step": "collect_age",
            "response": "Please provide a valid age between 0 and 120."
        }

    return {
        **state,
        "patient_age": str(age),
        "step": "collect_query",
        "response": query_prompt_for_lang(lang)
    }


def collect_query_node(state: PatientState) -> PatientState:
    """Collect patient symptoms/query"""
    messages = state.get("messages", [])
    last_msg = messages[-1]["content"].strip() if messages else ""
    lang = state.get("language", "en")

    if len(last_msg) < 5:
        return {
            **state,
            "step": "collect_query",
            "response": "Could you please describe your symptoms in a bit more detail?"
        }

    return {
        **state,
        "patient_query": last_msg,
        "step": "classify_ward",
    }


def classify_ward_node(state: PatientState) -> PatientState:
    """Classify which ward the patient should go to"""
    query = state.get("patient_query", "")
    lang = state.get("language", "en")
    name = state.get("patient_name", "Patient")

    # Priority: Emergency > Mental Health > General
    if detect_emergency(query):
        ward = "Emergency Ward"
    elif detect_mental_health(query):
        ward = "Mental Health Ward"
    else:
        # Use LLM for nuanced classification
        try:
            result = llm.invoke([
                SystemMessage(content="""You are a hospital triage AI. Based on the patient's symptoms, 
                classify them into one of these three wards ONLY:
                - Emergency Ward (life-threatening, urgent)
                - Mental Health Ward (psychological, psychiatric issues)
                - General Ward (routine, non-urgent)
                
                Reply with ONLY the ward name, nothing else."""),
                HumanMessage(content=f"Patient symptoms: {query}")
            ])
            ward = result.content.strip()
            if ward not in ["Emergency Ward", "Mental Health Ward", "General Ward"]:
                ward = "General Ward"
        except Exception:
            ward = "General Ward"

    # Craft final confirmation message
    ward_msgs = {
        "Emergency Ward": {
            "en": f"⚠️ {name}, this sounds urgent! I'm immediately routing you to the **Emergency Ward**. Please head to the Emergency department right away. Staff will attend to you immediately.",
            "hi": f"⚠️ {name}, yeh acil lagta hai! Hum aapko turant **Emergency Ward** bhej rahe hain. Kripya Emergency department mein jayein.",
            "te": f"⚠️ {name}, idi urgent ga kanipistundi! Meeru ni vantane **Emergency Ward** ki pathimpistunnanu. Dayachesi Emergency department ki vellandi."
        },
        "Mental Health Ward": {
            "en": f"💙 {name}, thank you for sharing this with me. I'm routing you to our **Mental Health Ward** where caring specialists will support you. You're not alone.",
            "hi": f"💙 {name}, yeh share karne ke liye shukriya. Hum aapko **Mental Health Ward** bhej rahe hain jahan visheshagya aapki madad karenge.",
            "te": f"💙 {name}, idi share chesinduku dhanyavaadalu. Meeru ni **Mental Health Ward** ki pathimpistunnanu, specialists meeru ni support chestaaru."
        },
        "General Ward": {
            "en": f"✅ {name}, I've registered your visit. You'll be seen at the **General Ward**. Please take a seat and a staff member will call your name shortly.",
            "hi": f"✅ {name}, aapki registration ho gayi. Aapko **General Ward** mein dekha jayega. Kripya baithein, staff jald hi aapko bulayega.",
            "te": f"✅ {name}, meeru registration complete ayyindi. Meeru ni **General Ward** lo chustaru. Dayachesi kadupu, staff meeru peru pillustaru."
        }
    }

    msg = ward_msgs.get(ward, ward_msgs["General Ward"]).get(lang, ward_msgs.get(ward, ward_msgs["General Ward"])["en"])

    return {
        **state,
        "ward": ward,
        "step": "complete",
        "is_complete": True,
        "response": msg
    }


def router_node(state: PatientState) -> str:
    """Route to the correct node based on current step"""
    step = state.get("step", "collect_name")
    step_map = {
        "greeting": "start",
        "collect_name": "collect_name",
        "collect_age": "collect_age",
        "collect_query": "collect_query",
        "classify_ward": "classify_ward",
        "complete": END,
    }
    return step_map.get(step, END)

# ─── Build the Graph ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PatientState)

    graph.add_node("start", start_node)
    graph.add_node("collect_name", collect_name_node)
    graph.add_node("collect_age", collect_age_node)
    graph.add_node("collect_query", collect_query_node)
    graph.add_node("classify_ward", classify_ward_node)

    graph.set_entry_point("start")

    graph.add_conditional_edges("start", router_node)
    graph.add_conditional_edges("collect_name", lambda s: "collect_age" if s.get("patient_name") else "collect_name")
    graph.add_conditional_edges("collect_age", lambda s: "collect_query" if s.get("patient_age") else "collect_age")
    graph.add_conditional_edges("collect_query", lambda s: "classify_ward" if s.get("patient_query") else "collect_query")
    graph.add_edge("classify_ward", END)

    return graph.compile()


# Singleton compiled graph
hospital_graph = build_graph()
