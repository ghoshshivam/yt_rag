import time

from src import config as cfg

SYSTEM_PROMPT = (
    "You are a study assistant for a DSA video playlist. Answer ONLY from the numbered "
    "transcript excerpts provided. The transcripts are auto-generated translations of Hindi "
    "lectures and may contain small errors (for example 'area' or 'error' often means 'array', "
    "and '2.2' or '2 pointer' means 'two pointer'); interpret them sensibly. "
    "Read ALL excerpts and use every relevant one, not just the first. When the lecturer compares "
    "approaches, include the time and space trade-off. If a number in an excerpt looks inconsistent "
    "(for example 1 lakh elements stored as '2 x 10^4'), do not repeat it as fact; state the "
    "correct value or say the excerpt is unclear. "
    "If the excerpts do not contain the answer, say so clearly instead of guessing. "
    "Cite sources like, [2] after the statements they support. Do not invent links or "
    "timestamps. Write plain text with simple bullet points and no LaTeX or math markup. "
    "Keep the answer clear and concise."
)

RETRYABLE_CODES = {429, 500, 502, 503, 504}
_client = None


def _get_client():
    global _client
    if _client is None:
        if not cfg.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        from google import genai
        _client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    return _client


def build_prompt(question: str, chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] ({c['video_title']}, {c['timestamp']})\n{c['text']}")
    context = "\n\n".join(blocks)
    return f"Transcript excerpts:\n\n{context}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant content found in the indexed playlist."

    from google.genai import errors, types

    prompt = build_prompt(question, chunks)
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2)
    models = [cfg.LLM_MODEL] + [m for m in cfg.LLM_FALLBACK_MODELS if m != cfg.LLM_MODEL]
    last_error = None

    for model_name in models:
        for attempt in range(cfg.LLM_RETRIES):
            try:
                resp = _get_client().models.generate_content(
                    model=model_name, contents=prompt, config=config
                )
                return resp.text
            except errors.APIError as e:
                last_error = e
                code = getattr(e, "code", None)
                if code in RETRYABLE_CODES:
                    wait = 2 * (2 ** attempt)          # 2s, 4s, 8s
                    print(f"[{model_name}] error {code}, retry {attempt + 1}/{cfg.LLM_RETRIES} in {wait}s")
                    time.sleep(wait)
                    continue
                if code == 404:                         # model name retired or wrong: try next model
                    print(f"[{model_name}] not found, trying next model")
                    break
                raise                                   # bad key, bad request, etc.

    raise RuntimeError(f"All Gemini models were unavailable: {last_error}")
