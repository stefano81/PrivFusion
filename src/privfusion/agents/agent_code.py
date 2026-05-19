import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_fixed

from privfusion.agents.llms import LLM
from privfusion.utils.utils import print_colored_dict

logger = logging.getLogger(__name__)


class AgentCode:
    def __init__(self, llm: LLM, system_prompt: str, kwargs: Any):
        self._llm = llm
        self._system_prompt = system_prompt
        self._kwargs = kwargs

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def code(self, code_context: dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": str(code_context)},
        ]

        result = self._llm.chat(messages=messages, **self._kwargs)
        print_colored_dict(result)
        try:
            function_str = result["response"]
        except Exception as e:
            logger.error("Exception converting code output to json.")
            raise ValueError("Exception converting code output to json.") from e

        # do some evaluation on the function to ensure it is executable
        exec(function_str)  # nosec[B102]

        return function_str
