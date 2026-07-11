from openai import OpenAI
from dotenv import load_dotenv
import requests
import json
import os

# Load API key from .env file (project root, two levels up from this script)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------
# External API function / tool
# -------------------------------
def get_country_info(country_name):
    """
    This function calls a real REST API and returns country information.
    API used: https://restcountries.com
    """

    url = f"https://restcountries.com/v3.1/name/{country_name}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()[0]

        country_info = {
            "country": data.get("name", {}).get("common"),
            "capital": data.get("capital", ["Not available"])[0],
            "region": data.get("region"),
            "population": data.get("population"),
            "currency": list(data.get("currencies", {}).keys()),
            "languages": list(data.get("languages", {}).values())
        }

        return country_info

    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# Tool schema given to LLM
# -------------------------------
tools = [
    {
        "type": "function",
        "name": "get_country_info",
        "description": "Get information about a country using an external REST API.",
        "parameters": {
            "type": "object",
            "properties": {
                "country_name": {
                    "type": "string",
                    "description": "Name of the country, for example India, USA, Japan"
                }
            },
            "required": ["country_name"]
        }
    }
]


# -------------------------------
# Step 1: User prompt
# -------------------------------
user_prompt = "Tell me the capital, population, currency, and languages of India."


# -------------------------------
# Step 2: Send prompt to the model along with tool definitions
# -------------------------------
input_messages = [{"role": "user", "content": user_prompt}]

response = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools
)

# -------------------------------
# Step 3: Handle tool calls, if the model requested any
# -------------------------------
tool_calls = [item for item in response.output if item.type == "function_call"]

if tool_calls:
    input_messages += response.output

    for tool_call in tool_calls:
        args = json.loads(tool_call.arguments)
        result = get_country_info(args["country_name"])

        input_messages.append({
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": json.dumps(result)
        })

    final_response = client.responses.create(
        model="gpt-4.1",
        input=input_messages,
        tools=tools
    )

    print(final_response.output_text)
else:
    print(response.output_text)
