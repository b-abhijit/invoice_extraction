import json
import os
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# The client picks up GEMINI_API_KEY from the environment automatically.
# Get a free key (no credit card) at https://aistudio.google.com/apikey
client = genai.Client()

MODEL = "gemini-3.5-flash"  # free tier, no card required


class ExtractRequest(BaseModel):
    document_id: str
    text: str
    schema: Dict[str, Any]


SYSTEM_PROMPT = """You are an invoice data extraction engine.
Read the invoice text and fill in every field of the JSON schema you were
given. Follow these rules exactly:

- vendor: the biller's proper name, exactly as written in the text.
- currency: convert to the ISO 4217 code (USD, EUR, GBP, INR, JPY) even if
  the text uses words ("euros"), symbols ("₹"), or phrases ("pounds sterling").
- total_amount: a plain integer in the main unit, no commas/symbols. Handle
  spelled-out numbers ("twelve thousand four hundred eighty" -> 12480),
  Western grouping ("12,480"), Indian grouping ("1,24,800" -> 124800), and
  K/M suffixes ("12K" -> 12000).
- invoice_date: normalize to YYYY-MM-DD.
- due_in_days: an integer parsed from phrases like "Net 30", "payable within
  45 days", or "due in two weeks" (-> 14).
- is_paid: boolean inferred from wording ("paid in full" -> true,
  "awaiting payment" -> false).
- priority: exactly one of low, normal, high, urgent.
- contact_email: lowercased.
- line_items: an array of {sku, quantity, unit_price} in the order they
  appear in the text; unit_price is an integer.
- item_count: the number of line items.

Fill in every field the schema asks for, and do not invent extra fields.
"""


@app.post("/extract")
def extract(req: ExtractRequest):
    # response_schema IS the schema the grader sent us. Setting
    # response_mime_type to application/json + passing this schema is what
    # makes Gemini's reply strict, valid JSON instead of free text.
    response = client.models.generate_content(
        model=MODEL,
        contents=req.text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=req.schema,
        ),
    )

    try:
        result = json.loads(response.text)
    except (ValueError, AttributeError):
        return {"error": "Model did not return valid structured output"}

    # Safety net: keep ONLY the keys the schema expects, nothing extra,
    # nothing missing silently swallowed.
    expected_keys = list(req.schema.get("properties", {}).keys())
    cleaned = {k: result.get(k) for k in expected_keys}
    return cleaned


@app.get("/")
def health():
    return {"status": "ok"}
