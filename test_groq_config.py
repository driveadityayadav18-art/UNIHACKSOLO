import os

from agents import build_structured_chat_model

# Ensure a mock key exists if not set
if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "test-only-placeholder"
os.environ["LLM_PROVIDER"] = "groq"
model = build_structured_chat_model()
print({"provider": "groq", "model_class": model.__class__.__name__, "model_name": getattr(model, "model_name", None)})
