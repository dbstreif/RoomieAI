from openai import OpenAI, APIError, AuthenticationError, RateLimitError

BASE_URL = "https://openrouter.ai/api/v1"
with open("APIKEY.txt", encoding="utf-8") as file:
    API_KEY = file.readline().strip()



def openai_call(text: str):
    client = OpenAI(
      base_url= BASE_URL,
      api_key= API_KEY
    )

    messages = [
        {
            'role': 'user', 
            'content': text
        }
    ]

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=messages,
            max_tokens=50
        )

        return {
            "status_code": 200,
            "response": response.choices[0].message.content
        }

    except AuthenticationError as e:
        print(f"Auth Error: {str(e)}")
        return {
            "status_code": 401,
            "error": f"Auth Error: {str(e)}"
        }

    except RateLimitError as e:
        print(f"Rate Limit Error: {str(e)}")
        return {
            "status_code": 429,
            "error": f"Rate Limit Error: {str(e)}"
        }

    except APIError as e:
        print(f"API Error: {str(e)}")
        return {
            "status_code": 500,
            "error": f"API Error: {str(e)}"
        }

    except Exception as e:
        print(f"Unknown Exception Occurred: {str(e)}")
        return {
            "status_code": 500,
            "error": f"Internal Error: {str(e)}"
        }
