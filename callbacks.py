from typing import Dict, Any, List, Optional
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class AgentsCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print("******prompts to LLM :******")
        print(f"{prompts[0]}")
        print("******prompts to LLM :******")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print("******Responce from LLM :******")
        print(f"{response.generations[0][0].text}")
        print("************")
