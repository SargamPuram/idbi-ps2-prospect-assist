"""
Pre-LLM input sanitizer for the DeepSeek pitch-generation call in app/main.py
(POST /recommend/{customer_id} -> generate_recommendation).

The three values interpolated into the DeepSeek prompt (lead name, occupation,
recommended product) are sourced from the synthetic dataset via a customer_id
lookup, not typed by an end user at request time -- but they still cross the
same trust boundary any string does the moment it is spliced into an LLM
prompt: nothing stops the CSV/dataframe from someday being edited, re-generated
with different assumptions, or replaced with a real data feed that includes
attacker-controlled or merely malformed strings. This module applies the same
defense-in-depth principle a chat-facing app would use, scaled down to match
the actual risk of this endpoint.

This is a deterministic, dependency-free, pre-LLM check -- no network/model
call. It is heuristic, not exhaustive (see DISCLAIMER.md "Known Limitations"):
regex pattern matches plus a hard length cap catch known
prompt-injection-style phrasing and obviously-anomalous input, but a novel
phrasing could evade it. That is an accepted, documented limitation for a
hackathon prototype, not a claim of complete coverage.
"""

import re

# Values longer than this are rejected outright and replaced with the safe
# default. A real name/occupation/product string has no legitimate reason to
# be this long; unusually long strings are also a classic vector for stuffing
# extra "instructions" into a field that is expected to be a few words.
MAX_FIELD_LENGTH = 200

# Prompt-injection-style phrasing. Deliberately similar in spirit to (but
# written independently of, not copied from) common jailbreak/instruction-
# override archetypes: telling the model to disregard its current
# instructions, or to adopt a new persona/role.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions?", re.I),
    re.compile(r"forget\s+(all\s+|any\s+)?(previous|prior|your)\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"act\s+as\s+(a|an|the)\s+", re.I),
    re.compile(r"pretend\s+(to\s+be|you\s+are)\s+", re.I),
    re.compile(r"new\s+(instructions?|system\s+prompt|role|persona)\s*:", re.I),
    re.compile(r"\bsystem\s*prompt\b", re.I),
    re.compile(r"reveal\s+(your|the)\s+(prompt|instructions?|system)", re.I),
    re.compile(r"<\s*/?\s*(system|assistant|user)\s*>", re.I),  # fake chat-role delimiters
    re.compile(r"```"),  # code-fence break-out attempt
]


def sanitize_prompt_field(value, field_name: str, default: str) -> str:
    """
    Validate a single untrusted string before it is interpolated into an LLM
    prompt. Returns the original value if it passes all checks, otherwise
    returns `default`.

    Checks (in order):
      1. Must be a non-empty string (after stripping whitespace).
      2. Must not exceed MAX_FIELD_LENGTH characters.
      3. Must not match any known prompt-injection-style pattern.

    This is intentionally conservative: on any failure, the field is replaced
    wholesale rather than partially cleaned/escaped, so a rejected value can
    never partially leak into the prompt.
    """
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    if len(text) > MAX_FIELD_LENGTH:
        print(f"[safety] rejected {field_name}: exceeds {MAX_FIELD_LENGTH} chars (len={len(text)})")
        return default

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            print(f"[safety] rejected {field_name}: matched injection pattern {pattern.pattern!r}")
            return default

    return text


def sanitize_lead_fields(name, occupation, recommended_product) -> dict:
    """
    Convenience wrapper for the three fields interpolated into the DeepSeek
    prompt in generate_recommendation(). Returns a dict with sanitized
    values, each falling back to a safe, generic default on failure so the
    prompt always remains well-formed.
    """
    return {
        "name": sanitize_prompt_field(name, "name", default="Customer"),
        "occupation": sanitize_prompt_field(occupation, "occupation", default="professional"),
        "recommended_product": sanitize_prompt_field(
            recommended_product, "recommended_product", default="Personal Loan"
        ),
    }
