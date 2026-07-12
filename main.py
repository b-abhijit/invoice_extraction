import json
import os
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# AI Pipe proxies to real OpenAI models using the course-provided AIPIPE_TOKEN
# instead of a personal paid API key. Get yours at https://aipipe.org/login
client = OpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url="https://aipipe.org/openai/v1",
)

MODEL = "gpt-4.1-nano"


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

# OpenAI's structured-output "strict" mode requires every object node to
# have additionalProperties: false and list every one of its properties in
# "required" - including nested objects like the ones inside line_items.
# We enforce that here rather than trusting the incoming schema already has
# it everywhere, since strict mode rejects the whole request if it's missing
# even on one nested object.
def harden_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        schema = dict(schema)
        if schema.get("type") == "object" and "properties" in schema:
            schema["properties"] = {
                k: harden_schema(v) for k, v in schema["properties"].items()
            }
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"].keys())
        elif "items" in schema:
            schema["items"] = harden_schema(schema["items"])
        return schema
    return schema


@app.post("/extract")
def extract(req: ExtractRequest):
    safe_schema = harden_schema(req.schema)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "invoice_extraction",
                "schema": safe_schema,
                "strict": True,
            },
        },
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except (ValueError, AttributeError, IndexError):
        return {"error": "Model did not return valid structured output"}

    # Safety net: keep ONLY the keys the schema expects, nothing extra,
    # nothing missing silently swallowed.
    expected_keys = list(req.schema.get("properties", {}).keys())
    cleaned = {k: result.get(k) for k in expected_keys}
    return cleaned


@app.get("/")
def health():
    return {"status": "ok"}