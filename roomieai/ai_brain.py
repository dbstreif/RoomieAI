from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from devtext import DEV1, DEV2

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
            'role': "developer",
            'content': [
            {
                'type': "text",
                'text': DEV1 
            }
        ]
        },
        {
            'role': "assistant",
            'content': DEV2
        },
        {
            'role': 'user', 
            'content': text
        }
    ]

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=messages,
            max_tokens=1000
        )

        reply = response.choices[0].message.content

        # Parse the flag and the actual response
        flag_start = reply.find("[FLAG:") + len("[FLAG:")
        flag_end = reply.find("]", flag_start)
        flag = reply[flag_start:flag_end].strip()
        parsed_reply = reply[flag_end + 1:].strip()

        return {
            "status_code": 200,
            "flag": flag,
            "response": parsed_reply
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
