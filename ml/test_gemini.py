
import os

from google import genai


# --------------------------------------------------
# 1. Initialize Gemini client
# --------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# 2. Send prompt to Gemini
# --------------------------------------------------

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Explain vector search in simple terms."
)


# --------------------------------------------------
# 3. Print generated response
# --------------------------------------------------

print("\nGemini response:\n")
print(interaction.output_text)
