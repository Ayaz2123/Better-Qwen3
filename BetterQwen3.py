"""
title: Better Qwen3
author: AaronFeng753
author_url: https://github.com/AaronFeng753
funding_url: https://github.com/AaronFeng753
version: 0.1
"""

from pydantic import BaseModel, Field
from typing import Optional
import re
import requests


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        max_turns: int = Field(
            default=99999,
            description="Maximum allowable conversation turns for a user.",
        )
        pass

    class UserValves(BaseModel):
        max_turns: int = Field(
            default=99999,
            description="Maximum allowable conversation turns for a user.",
        )
        pass

    def __init__(self):
        # Indicates custom file handling logic. This flag helps disengage default routines in favor of custom
        # implementations, informing the WebUI to defer file-related operations to designated methods within this class.
        # Alternatively, you can remove the files directly from the body in from the inlet hook
        # self.file_handler = True

        # Initialize 'valves' with specific configurations. Using 'Valves' instance helps encapsulate settings,
        # which ensures settings are managed cohesively and not confused with operational flags like 'file_handler'.
        self.valves = self.Valves()
        pass

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[callable] = None,
    ) -> dict:
        # Modify the request body or validate it before processing by the chat completion API.
        # This function is the pre-processor for the API where various checks on the input can be performed.
        # It can also modify the request before sending it to the API.

        messages = body.get("messages", [])
        modified_messages = []

        for message in messages:
            if message.get("role") == "assistant":
                message_content = message.get("content", "")
                pattern = r"<think[^>]*>.*?</think>"
                modified_content = re.sub(pattern, "", message_content, flags=re.DOTALL)
                modified_message = {"role": "assistant", "content": modified_content}
            else:
                modified_message = message
            modified_messages.append(modified_message)

        body["messages"] = modified_messages

        messages = body.get("messages", [])

        if not messages:
            return body

        # Get the latest user message

        latest_user_msg = next(
            (msg for msg in reversed(messages) if msg.get("role") == "user"), None
        )

        if not latest_user_msg:
            return body

        First_user_msg = next(
            (msg for msg in messages if msg.get("role") == "user"), None
        )

        if not First_user_msg:
            return body

        # ================================================================================================================
        #  Evaluate the difficulty of the latest user message OR the first user message
        # ================================================================================================================
        # Evaluate the difficulty of the latest user message:
        User_request_str = latest_user_msg.get("content", "")
        # Only evaluate the difficulty of the first user message:
        # User_request_str = First_user_msg.get("content", "")
        # ================================================================================================================

        # If the length of the user's request exceeds 1000 characters, only keep the first and last 500 characters to speed up processing.
        if len(User_request_str) > 1010:
            User_request_str = User_request_str[:500] + "\n\n" + User_request_str[-500:]

        # Construct API request
        # ================================================================================================================
        # Please configure the API URL and Model name according to your LLM backend.
        # ================================================================================================================
        api_url = "http://11.0.0.11:1111/v1/chat/completions"  # OpenAI API URL
        payload = {
            "model": "qwen3-32b-i4_xs",  # Model Name
            # ============================================================================================================
            "messages": [
                {
                    "role": "system",
                    "content": """/no_think
You are a specialized AI model acting as a Request Difficulty Assessor.""",
                },
                {
                    "role": "user",
                    "content": """You are a specialized AI model acting as a Request Difficulty Assessor.
Your SOLE and ONLY task is to evaluate the inherent difficulty of a user's request that is intended for another AI.
You will receive a user's request message.
Your objective is to determine if this request requires careful, deliberate thought from the downstream AI, or if it's straightforward.

Criteria for your decision:
1. If the user's request is complex, nuanced, requires multi-step reasoning, creative generation, in-depth analysis, or careful consideration by the AI to produce a high-quality response, you MUST respond with: `hard`
2. If the user's request is simple, factual, straightforward, or can likely be answered quickly and directly by the AI with minimal processing or deliberation, you MUST respond with: `easy`

IMPORTANT:
- Your response MUST be EXACTLY one of the two commands: `hard` and `easy`
- Do NOT add any other text, explanations, or pleasantries.
- Your assessment is about the processing difficulty for the *AI that will ultimately handle the user's request*.

---

### User's request:

<Users_request>\n"""
                    + User_request_str
                    + "\n</Users_request>",
                },
                {
                    "role": "assistant",
                    "content": "<think>\n\n</think>\n\n",
                },
            ],
            "temperature": 0.7,
        }

        try:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "🤖 Evaluating request difficulty",
                        "done": False,
                    },
                }
            )
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            api_reply = response.json()["choices"][0]["message"]["content"]
            pattern = r"<think[^>]*>.*?</think>"
            api_reply = re.sub(pattern, "", api_reply, flags=re.DOTALL)
            api_reply = api_reply.lower()

            if "hard" in api_reply:
                latest_user_msg["content"] += f"\n\n/think"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "🧠 /think",
                            "done": True,
                        },
                    }
                )
            elif "easy" in api_reply:
                latest_user_msg["content"] += f"\n\n/no_think"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "⚡ /no_think",
                            "done": True,
                        },
                    }
                )
            else:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "❌ Invalid reply from Assessor",
                            "done": True,
                        },
                    }
                )

            # Update the message list
            for i in reversed(range(len(messages))):
                if messages[i]["role"] == "user":
                    messages[i] = latest_user_msg
                    break

            body["messages"] = messages

        except Exception as e:
            # Error handling (optional add error identification)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "❌ API request failed",
                        "done": True,
                    },
                }
            )

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Modify or analyze the response body after processing by the API.
        # This function is the post-processor for the API, which can be used to modify the response
        # or perform additional checks and analytics.

        return body
