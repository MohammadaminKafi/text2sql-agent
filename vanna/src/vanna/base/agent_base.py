import json
import os
import re
import sqlite3
import traceback
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Any

import pandas as pd
import requests
import sqlparse

from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.llms.base import LLM

from ..exceptions import DependencyError, ImproperlyConfigured, ValidationError
from ..types import TrainingPlan, TrainingPlanItem
from ..utils import validate_config_path

from .base import VannaBase


class AVannaBase(VannaBase):
    def __init__(self, config=None):
        super().__init__(config)
        self._react_agent = None 

    def _build_react_agent(self):
        """
        Create a ReAct-style LangChain agent that can *only* call ``run_sql`` and
        that delegates all LLM calls to ``self.submit_prompt`` (i.e. the same
        path used by ``generate_sql``).  Every intermediate step is aggressively
        logged through ``self.log``.
        """

        class _LogHandler(BaseCallbackHandler):
            def __init__(self, vanna):         # remember enclosing instance
                self.vanna = vanna

            # chain-of-thought from the LLM
            def on_llm_new_token(self, token, **kw):
                self.vanna.log(title="LLM Token", message=token, echo=False)

            # agent state changes
            def on_agent_action(self, action, **kw):
                self.vanna.log(title="Agent Action", message=str(action))

            def on_agent_finish(self, finish, **kw):
                self.vanna.log(title="Agent Finish", message=str(finish))

            # tool activity
            def on_tool_start(self, serialized, input_str, **kw):
                self.vanna.log(title=f"Tool Start – {serialized['name']}",
                               message=input_str)

            def on_tool_end(self, output, **kw):
                self.vanna.log(title="Tool End", message=str(output))

        # ── 2.  Expose run_sql as the *only* tool available to the agent ──────
        def _sql_tool_func(query: str) -> str:
            """
            Wrapper around self.run_sql that returns a *string* (markdown table)
            so the ReAct agent can splice it back into the prompt.
            """
            try:
                df = self.run_sql(query)
                # Log & preserve the DataFrame for debugging
                self.log(title="Tool run_sql executed", message="Captured DF", save_df=True, df=df)
                # Minimal but readable – change to .to_json() if you prefer JSON
                return df.to_markdown()
            except Exception as e:
                return f"Error running SQL: {e}"

        sql_tool = Tool(
            name="run_sql",
            func=_sql_tool_func,
            description=(
                "Execute an SQL statement and return the result table as markdown text."
            ),
        )


        # ── 3.  Lightweight LangChain LLM that forwards to submit_prompt ──────
        class _VannaLLM(LLM):
            vanna: Any
            model_config = {"arbitrary_types_allowed": True}

            @property
            def _llm_type(self) -> str:
                return "vanna_submit_prompt"

            def _call(self, prompt: str, stop=None, run_manager=None, **kw):
                # 1️⃣ extract the natural-language question that LangChain embeds
                m = re.search(r"Question:\s*(.+)", prompt)
                question = m.group(1).strip() if m else prompt

                # 2️⃣ use Vanna’s built-in pipeline to make SQL
                try:
                    sql = self.vanna.generate_sql(question)
                except Exception as e:
                    sql = f"-- error generating SQL: {e}"

                # 3️⃣ return **ReAct-compatible** final answer
                return f"Final Answer: {sql}"

            @property
            def _identifying_params(self):
                return {}
   
        llm = _VannaLLM(vanna=self)

        # ── 4.  System instructions – keep it lean; rely on tools for schema ──
        system_prefix = (
            "You are an expert SQL assistant.\n"
            "Given a natural-language question, think step-by-step, inspect the "
            "database with the `run_sql` tool when needed, and finish by "
            "outputting ONLY the final SQL query (no markdown)."
        )

        # ── 5.  Assemble ReAct agent (description-based) ──────────────────────
        agent = initialize_agent(
        tools=[sql_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        callbacks=[_LogHandler(self)],
        )

        return agent


    def _ensure_react_agent(self):
        if not hasattr(self, "_react_agent"):
            self._react_agent = self._build_react_agent()
        return self._react_agent

    def ask_react_agent(
        self,
        question: Union[str, None] = None,
    ) -> Union[Tuple[Union[str, None], Union[pd.DataFrame, None]], None]:
        """
        Replacement for ``ask`` that delegates SQL generation to an internal
        ReAct agent and immediately executes it.

        Returns
        -------
        Tuple[str | None, pd.DataFrame | None] or ``None`` on failure.
        """
        # ---------------------------------------------------------------- input
        if question is None:
            question = input("Enter a question: ")
            if question == "" or question == None:
                question = "List all the territories"

        self.create_new_thread()
        self.log(message=question, title="Asked Question")

        # ---------------------------------------------------------------- guard
        if not getattr(self, "run_sql_is_set", False):
            self.log(
                title="Exit",
                message="Database not connected (`run_sql_is_set` is False).",
            )
            return None

        # ---------------------------------------------------------------- agent
        agent = self._ensure_react_agent()
        try:
            # modern API (v0.1+) — returns {'output': "..."}
            result = agent.invoke({"input": question})
            raw_sql = result.get("output", "")
            # strip the "Final Answer:" prefix that ReAct adds
            sql = raw_sql.replace("Final Answer:", "", 1).strip()
            self.log(title="Generated SQL", message=sql)
        except Exception as e:
            self.log(title="Exception in ReAct agent", message=str(e))
            return None, None

        # ---------------------------------------------------------------- query
        try:
            df = self.run_sql(sql)
            # Aggressive logging: save dataframe snapshot
            self.log(
                title="SQL Executed",
                message="Saved dataframe from agent SQL",
                save_df=True,
                df=df,
            )
            return sql, df
        except Exception as e:
            self.log(title="Exception in run_sql", message=str(e))
            return sql, None
