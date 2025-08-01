r"""

# Nomenclature

| Prefix | Definition | Examples |
| --- | --- | --- |
| `vn.get_` | Fetch some data | [`vn.get_related_ddl(...)`][vanna.base.base.VannaBase.get_related_ddl] |
| `vn.add_` | Adds something to the retrieval layer | [`vn.add_question_sql(...)`][vanna.base.base.VannaBase.add_question_sql] <br> [`vn.add_ddl(...)`][vanna.base.base.VannaBase.add_ddl] |
| `vn.generate_` | Generates something using AI based on the information in the model | [`vn.generate_sql(...)`][vanna.base.base.VannaBase.generate_sql] <br> [`vn.generate_explanation()`][vanna.base.base.VannaBase.generate_explanation] |
| `vn.run_` | Runs code (SQL) | [`vn.run_sql`][vanna.base.base.VannaBase.run_sql] |
| `vn.remove_` | Removes something from the retrieval layer | [`vn.remove_training_data`][vanna.base.base.VannaBase.remove_training_data] |
| `vn.connect_` | Connects to a database | [`vn.connect_to_snowflake(...)`][vanna.base.base.VannaBase.connect_to_snowflake] |
| `vn.update_` | Updates something | N/A -- unused |
| `vn.set_` | Sets something | N/A -- unused  |

# Open-Source and Extending

Vanna.AI is open-source and extensible. If you'd like to use Vanna without the servers, see an example [here](https://vanna.ai/docs/postgres-ollama-chromadb/).

The following is an example of where various functions are implemented in the codebase when using the default "local" version of Vanna. `vanna.base.VannaBase` is the base class which provides a `vanna.base.VannaBase.ask` and `vanna.base.VannaBase.train` function. Those rely on abstract methods which are implemented in the subclasses `vanna.openai_chat.OpenAI_Chat` and `vanna.chromadb_vector.ChromaDB_VectorStore`. `vanna.openai_chat.OpenAI_Chat` uses the OpenAI API to generate SQL and Plotly code. `vanna.chromadb_vector.ChromaDB_VectorStore` uses ChromaDB to store training data and generate embeddings.

If you want to use Vanna with other LLMs or databases, you can create your own subclass of `vanna.base.VannaBase` and implement the abstract methods.

```mermaid
flowchart
    subgraph VannaBase
        ask
        train
    end

    subgraph OpenAI_Chat
        get_sql_prompt
        submit_prompt
        generate_question
        generate_plotly_code
    end

    subgraph ChromaDB_VectorStore
        generate_embedding
        add_question_sql
        add_ddl
        add_documentation
        get_similar_question_sql
        get_related_ddl
        get_related_documentation
    end
```

"""

from no_commit_utils.credentials_utils import read_avalai_api_key, read_langsmith_api_key, read_metis_api_key

# Enviromental Variables
import os
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGSMITH_API_KEY"] = read_langsmith_api_key()
# os.environ["LANGSMITH_PROJECT"] = "react-sql"

import json
# import os
import re
import sqlite3
import traceback
import dataclasses
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from urllib.parse import urlparse
from datetime import datetime

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlparse

# Agent imports
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool
from pydantic import Field, BaseModel as PydanticBaseModelForTool

# Local imports
from ..exceptions import DependencyError, ImproperlyConfigured, ValidationError
from ..types import TrainingPlan, TrainingPlanItem
from ..utils import validate_config_path


class VannaBase(ABC):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config

        # DBMS
        self.run_sql_is_set = False
        self.static_documentation = ""
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 14000)

        # Log
        self.current_thread_name = None
        self.current_log_id = None
        self.save_log = self.config.get("save_log", True)
        self.verbose = self.config.get("verbose", False)
        self.log_dir = self.config.get("log_dir", "log")

        # Agent
        self.agent = None
        self.max_iters = 20
        self.agent_last_sql = None
        self.agent_last_df = None

    # ---------- Test Methods
    def test_llm_connection(self):
        self.create_new_thread(thread_type="test")

        try:
            test_response = self.submit_prompt([
                    self.system_message("Respond to user's message in one word"),
                    self.user_message("Hi!"),
                ])
            print(f"LLM responded to test connection with: {test_response}")
            if test_response.startswith('H'):
                return True
            else:
                return False
        except Exception as e:
            print(f"Exception occurred in connection to LLM: {e}")
            return False

    # ---------- Log Methods
    def create_new_thread(self, thread_type='chat'):
        if self.save_log:
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            self.current_thread_name = f"{thread_type}-{current_time}"
            self.current_log_id = 0

    def log_df(self, df: pd.DataFrame, path):
        df.to_csv(path, index=False)

    def log(self, message: str, title: str = "Info", echo: bool = False, save_df: bool = False, df: pd.DataFrame | None = None):
        if self.save_log:
            # Ensure thread name is set
            if not self.current_thread_name:
                raise ValueError("current_thread_name is not set.")
            
            # Ensure log ID is initialized
            if self.current_log_id is None:
                self.current_log_id = 0
        
            # Create directory if it doesn't exist
            log_dir = f"./{self.log_dir}/{self.current_thread_name}"
            os.makedirs(log_dir, exist_ok=True)
        
            # Define file path
            filename = f"{self.current_log_id:02d}-{title}.txt"
            filepath = os.path.join(log_dir, filename)
        
            # Save log message to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(message))
        
            if save_df:
                df_name = f"{self.current_log_id:02d}-dataframe.csv"
                df_path = os.path.join(log_dir, df_name)
                self.log_df(path=df_path, df=df)

            self.current_log_id += 1

        # Print to console if echo
        if echo and self.verbose:
            print(f"{title}: {message}")

    def log_json(self, data, title: str = "Info"):
        """Save the provided data as a JSON log file."""
        if self.save_log:
            if not self.current_thread_name:
                raise ValueError("current_thread_name is not set.")
            if self.current_log_id is None:
                self.current_log_id = 0

            log_dir = f"./{self.log_dir}/{self.current_thread_name}"
            os.makedirs(log_dir, exist_ok=True)

            filename = f"{self.current_log_id:02d}-{title}.json"
            filepath = os.path.join(log_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.current_log_id += 1
    
    def _response_language(self) -> str:
        if self.language is None:
            return ""

        return f"Respond in the {self.language} language."

    # ---------- SQL Helper Methods
    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        """
        Example:
        ```python
        vn.generate_sql("What are the top 10 customers by sales?")
        ```

        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:

        - [`get_similar_question_sql`][vanna.base.base.VannaBase.get_similar_question_sql]

        - [`get_related_ddl`][vanna.base.base.VannaBase.get_related_ddl]

        - [`get_related_documentation`][vanna.base.base.VannaBase.get_related_documentation]

        - [`get_sql_prompt`][vanna.base.base.VannaBase.get_sql_prompt]

        - [`submit_prompt`][vanna.base.base.VannaBase.submit_prompt]


        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)

        self.log(
            message="Retrieved question-sql pairs from VDB:\n\n" + "\n\n".join(question_sql_list),
            title="Retrieved Question-SQL Pair",
        )
        self.log(
            message="Retrieved DDL from VDB:\n\n" + "\n\n".join(ddl_list),
            title="Retrieved DDL",
        )
        self.log(
            message="Retrieved documents from VDB:\n\n" + "\n\n".join(doc_list),
            title="Retrieved Documents",
        )

        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list+[f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n" + df.to_markdown()],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message=prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"


        return self.extract_sql(llm_response)

    def extract_sql(self, llm_response: str) -> str:
        """
        Example:
        ```python
        vn.extract_sql("Here's the SQL query in a code block: ```sql\nSELECT * FROM customers\n```")
        ```

        Extracts the SQL query from the LLM response. This is useful in case the LLM response contains other information besides the SQL query.
        Override this function if your LLM responses need custom extraction logic.

        Args:
            llm_response (str): The LLM response.

        Returns:
            str: The extracted SQL query.
        """

        import re
        """
        Extracts the SQL query from the LLM response, handling various formats including:
        - WITH clause
        - SELECT statement
        - CREATE TABLE AS SELECT
        - Markdown code blocks
        """

        # Match CREATE TABLE ... AS SELECT
        sqls = re.findall(r"\bCREATE\s+TABLE\b.*?\bAS\b.*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # Match WITH clause (CTEs)
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # Match SELECT ... ;
        sqls = re.findall(r"\bSELECT\b .*?;", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # Match ```sql ... ``` blocks
        sqls = re.findall(r"```sql\s*\n(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1].strip()
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # Match any ``` ... ``` code blocks
        sqls = re.findall(r"```(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
        if sqls:
            sql = sqls[-1].strip()
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        return llm_response

    def is_sql_valid(self, sql: str) -> bool:
        """
        Example:
        ```python
        vn.is_sql_valid("SELECT * FROM customers")
        ```
        Checks if the SQL query is valid. This is usually used to check if we should run the SQL query or not.
        By default it checks if the SQL query is a SELECT statement. You can override this method to enable running other types of SQL queries.

        Args:
            sql (str): The SQL query to check.

        Returns:
            bool: True if the SQL query is valid, False otherwise.
        """

        parsed = sqlparse.parse(sql)

        for statement in parsed:
            if statement.get_type() == 'SELECT':
                return True

        return False

    # ---------- Generation Methods
    def should_generate_chart(self, df: pd.DataFrame) -> bool:
        """
        Example:
        ```python
        vn.should_generate_chart(df)
        ```

        Checks if a chart should be generated for the given DataFrame. By default, it checks if the DataFrame has more than one row and has numerical columns.
        You can override this method to customize the logic for generating charts.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Returns:
            bool: True if a chart should be generated, False otherwise.
        """

        if len(df) > 1 and df.select_dtypes(include=['number']).shape[1] > 0:
            return True

        return False

    def generate_rewritten_question(self, last_question: str, new_question: str, **kwargs) -> str:
        """
        **Example:**
        ```python
        rewritten_question = vn.generate_rewritten_question("Who are the top 5 customers by sales?", "Show me their email addresses")
        ```

        Generate a rewritten question by combining the last question and the new question if they are related. If the new question is self-contained and not related to the last question, return the new question.

        Args:
            last_question (str): The previous question that was asked.
            new_question (str): The new question to be combined with the last question.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The combined question if related, otherwise the new question.
        """
        if last_question is None:
            return new_question

        prompt = [
            self.system_message("Your goal is to combine a sequence of questions into a singular question if they are related. If the second question does not relate to the first question and is fully self-contained, return the second question. Return just the new combined question with no additional explanations. The question should theoretically be answerable with a single SQL statement."),
            self.user_message("First question: " + last_question + "\nSecond question: " + new_question),
        ]

        return self.submit_prompt(prompt=prompt, **kwargs)

    def generate_followup_questions(
        self, question: str, sql: str, df: pd.DataFrame, n_questions: int = 5, **kwargs
    ) -> list:
        """
        **Example:**
        ```python
        vn.generate_followup_questions("What are the top 10 customers by sales?", sql, df)
        ```

        Generate a list of followup questions that you can ask Vanna.AI.

        Args:
            question (str): The question that was asked.
            sql (str): The LLM-generated SQL query.
            df (pd.DataFrame): The results of the SQL query.
            n_questions (int): Number of follow-up questions to generate.

        Returns:
            list: A list of followup questions that you can ask Vanna.AI.
        """

        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe SQL query for this question was: {sql}\n\nThe following is a pandas DataFrame with the results of the query: \n{df.head(25).to_markdown()}\n\n"
            ),
            self.user_message(
                f"Generate a list of {n_questions} followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions. Remember that there should be an unambiguous SQL query that can be generated from the question. Prefer questions that are answerable outside of the context of this conversation. Prefer questions that are slight modifications of the SQL query that was generated that allow digging deeper into the data. Each question will be turned into a button that the user can click to generate a new SQL query so don't use 'example' type questions. Each question must have a one-to-one correspondence with an instantiated SQL query." +
                self._response_language()
            ),
        ]

        llm_response = self.submit_prompt(message_log, **kwargs)

        numbers_removed = re.sub(r"^\d+\.\s*", "", llm_response, flags=re.MULTILINE)
        return numbers_removed.split("\n")

    def generate_questions(self, **kwargs) -> List[str]:
        """
        **Example:**
        ```python
        vn.generate_questions()
        ```

        Generate a list of questions that you can ask Vanna.AI.
        """
        question_sql = self.get_similar_question_sql(question="", **kwargs)

        return [q["question"] for q in question_sql]

    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        """
        **Example:**
        ```python
        vn.generate_summary("What are the top 10 customers by sales?", df)
        ```

        Generate a summary of the results of a SQL query.

        Args:
            question (str): The question that was asked.
            df (pd.DataFrame): The results of the SQL query.

        Returns:
            str: The summary of the results of the SQL query.
        """

        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary." +
                self._response_language()
            ),
        ]

        summary = self.submit_prompt(message_log, **kwargs)

        return summary

    # ---------- Embedding Methods
    @abstractmethod
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        pass

    # ---------- Vector Database Abstract Methods
    @abstractmethod
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and their corresponding SQL statements for.

        Returns:
            list: A list of similar questions and their corresponding SQL statements.
        """
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """
        pass

    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        This method is used to add a question and its corresponding SQL query to the training data.

        Args:
            question (str): The question to add.
            sql (str): The SQL query to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        This method is used to add a DDL statement to the training data.

        Args:
            ddl (str): The DDL statement to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        This method is used to add documentation to the training data.

        Args:
            documentation (str): The documentation to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        """
        Example:
        ```python
        vn.get_training_data()
        ```

        This method is used to get all the training data from the retrieval layer.

        Returns:
            pd.DataFrame: The training data.
        """
        pass

    @abstractmethod
    def remove_training_data(self, id: str, **kwargs) -> bool:
        """
        Example:
        ```python
        vn.remove_training_data(id="123-ddl")
        ```

        This method is used to remove training data from the retrieval layer.

        Args:
            id (str): The ID of the training data to remove.

        Returns:
            bool: True if the training data was removed, False otherwise.
        """
        pass

    # ---------- LLM Abstract Methods
    @abstractmethod
    def system_message(self, message: str) -> any:
        pass

    @abstractmethod
    def user_message(self, message: str) -> any:
        pass

    @abstractmethod
    def assistant_message(self, message: str) -> any:
        pass

    # ---------- LLM Methods
    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) / 4

    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += "\n===Tables \n"

            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"

        return initial_prompt

    def add_documentation_to_prompt(
        self,
        initial_prompt: str,
        documentation_list: list[str],
        max_tokens: int = 14000,
    ) -> str:
        if len(documentation_list) > 0:
            initial_prompt += "\n===Additional Context \n\n"

            for documentation in documentation_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    def add_sql_to_prompt(
        self, initial_prompt: str, sql_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(sql_list) > 0:
            initial_prompt += "\n===Question-SQL Pairs\n\n"

            for question in sql_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(question["sql"])
                    < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"

        return initial_prompt

    def get_sql_prompt(
        self,
        initial_prompt : str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if initial_prompt is None:
            initial_prompt = f"You are a {self.dialect} expert. " + \
            "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
            "3. If the provided context is insufficient, please explain why it can't be generated. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
            f"6. Ensure that the output SQL is {self.dialect}-compliant and executable, and free of syntax errors. \n"
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log

    def get_followup_questions_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ) -> list:
        initial_prompt = f"The user initially asked the question: '{question}': \n\n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
                "Generate a list of followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the questions."
            )
        )

        return message_log

    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        Example:
        ```python
        vn.submit_prompt(
            [
                vn.system_message("The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."),
                vn.user_message("What are the top 10 customers by sales?"),
            ]
        )
        ```

        This method is used to submit a prompt to the LLM.

        Args:
            prompt (any): The prompt to submit to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def generate_question(self, sql: str, **kwargs) -> str:
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response

    def _extract_python_code(self, markdown_string: str) -> str:
        # Strip whitespace to avoid indentation errors in LLM-generated code
        markdown_string = markdown_string.strip()

        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("fig.show()", "")

        return plotly_code

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Can you generate the Python plotly code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    # ---------- Database Methods
    def connect_to_snowflake(
        self,
        account: str,
        username: str,
        password: str,
        database: str,
        role: Union[str, None] = None,
        warehouse: Union[str, None] = None,
        **kwargs
    ):
        try:
            snowflake = __import__("snowflake.connector")
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[snowflake]"
            )

        if username == "my-username":
            username_env = os.getenv("SNOWFLAKE_USERNAME")

            if username_env is not None:
                username = username_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake username.")

        if password == "mypassword":
            password_env = os.getenv("SNOWFLAKE_PASSWORD")

            if password_env is not None:
                password = password_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake password.")

        if account == "my-account":
            account_env = os.getenv("SNOWFLAKE_ACCOUNT")

            if account_env is not None:
                account = account_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake account.")

        if database == "my-database":
            database_env = os.getenv("SNOWFLAKE_DATABASE")

            if database_env is not None:
                database = database_env
            else:
                raise ImproperlyConfigured("Please set your Snowflake database.")

        conn = snowflake.connector.connect(
            user=username,
            password=password,
            account=account,
            database=database,
            client_session_keep_alive=True,
            **kwargs
        )

        def run_sql_snowflake(sql: str) -> pd.DataFrame:
            cs = conn.cursor()

            if role is not None:
                cs.execute(f"USE ROLE {role}")

            if warehouse is not None:
                cs.execute(f"USE WAREHOUSE {warehouse}")
            cs.execute(f"USE DATABASE {database}")

            cur = cs.execute(sql)

            results = cur.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(results, columns=[desc[0] for desc in cur.description])

            return df

        self.dialect = "Snowflake SQL"
        self.run_sql = run_sql_snowflake
        self.run_sql_is_set = True

    def connect_to_sqlite(self, url: str, check_same_thread: bool = False,  **kwargs):
        """
        Connect to a SQLite database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            url (str): The URL of the database to connect to.
            check_same_thread (str): Allow the connection may be accessed in multiple threads.
        Returns:
            None
        """

        # URL of the database to download

        # Path to save the downloaded database
        path = os.path.basename(urlparse(url).path)

        # Download the database if it doesn't exist
        if not os.path.exists(url):
            response = requests.get(url)
            response.raise_for_status()  # Check that the request was successful
            with open(path, "wb") as f:
                f.write(response.content)
            url = path

        # Connect to the database
        conn = sqlite3.connect(
            url,
            check_same_thread=check_same_thread,
            **kwargs
        )

        def run_sql_sqlite(sql: str):
            return pd.read_sql_query(sql, conn)

        self.dialect = "SQLite"
        self.run_sql = run_sql_sqlite
        self.run_sql_is_set = True

    def connect_to_postgres(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        """
        Connect to postgres using the psycopg2 connector. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_postgres(
            host="myhost",
            dbname="mydatabase",
            user="myuser",
            password="mypassword",
            port=5432
        )
        ```
        Args:
            host (str): The postgres host.
            dbname (str): The postgres database name.
            user (str): The postgres user.
            password (str): The postgres password.
            port (int): The postgres Port.
        """

        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[postgres]"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your postgres host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your postgres database")

        if not user:
            user = os.getenv("PG_USER")

        if not user:
            raise ImproperlyConfigured("Please set your postgres user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your postgres password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your postgres port")

        conn = None

        try:
            conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port,
                **kwargs
            )
        except psycopg2.Error as e:
            raise ValidationError(e)

        def connect_to_db():
            return psycopg2.connect(host=host, dbname=dbname,
                        user=user, password=password, port=port, **kwargs)


        def run_sql_postgres(sql: str) -> Union[pd.DataFrame, None]:
            conn = None
            try:
                conn = connect_to_db()  # Initial connection attempt
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.InterfaceError as e:
                # Attempt to reconnect and retry the operation
                if conn:
                    conn.close()  # Ensure any existing connection is closed
                conn = connect_to_db()
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.Error as e:
                if conn:
                    conn.rollback()
                    raise ValidationError(e)

            except Exception as e:
                        conn.rollback()
                        raise e

        self.dialect = "PostgreSQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_postgres

    def connect_to_mysql(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        try:
            import pymysql.cursors
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install PyMySQL"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your MySQL host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your MySQL database")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your MySQL user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your MySQL password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your MySQL port")

        conn = None

        try:
            conn = pymysql.connect(
                host=host,
                user=user,
                password=password,
                database=dbname,
                port=port,
                cursorclass=pymysql.cursors.DictCursor,
                **kwargs
            )
        except pymysql.Error as e:
            raise ValidationError(e)

        def run_sql_mysql(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    conn.ping(reconnect=True)
                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except pymysql.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_mysql

    def connect_to_clickhouse(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        try:
            import clickhouse_connect
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install clickhouse_connect"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your ClickHouse host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your ClickHouse database")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your ClickHouse user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your ClickHouse password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your ClickHouse port")

        conn = None

        try:
            conn = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=user,
                password=password,
                database=dbname,
                **kwargs
            )
            print(conn)
        except Exception as e:
            raise ValidationError(e)

        def run_sql_clickhouse(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    result = conn.query(sql)
                    results = result.result_rows

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(results, columns=result.column_names)
                    return df

                except Exception as e:
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_clickhouse

    def connect_to_oracle(
        self,
        user: str = None,
        password: str = None,
        dsn: str = None,
        **kwargs
    ):

        """
        Connect to an Oracle db using oracledb package. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_oracle(
        user="username",
        password="password",
        dsn="host:port/sid",
        )
        ```
        Args:
            USER (str): Oracle db user name.
            PASSWORD (str): Oracle db user password.
            DSN (str): Oracle db host ip - host:port/sid.
        """

        try:
            import oracledb
        except ImportError:

            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install oracledb"
            )

        if not dsn:
            dsn = os.getenv("DSN")

        if not dsn:
            raise ImproperlyConfigured("Please set your Oracle dsn which should include host:port/sid")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your Oracle db user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your Oracle db password")

        conn = None

        try:
            conn = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
                **kwargs
            )
        except oracledb.Error as e:
            raise ValidationError(e)

        def run_sql_oracle(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    sql = sql.rstrip()
                    if sql.endswith(';'): #fix for a known problem with Oracle db where an extra ; will cause an error.
                        sql = sql[:-1]

                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except oracledb.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_oracle

    def connect_to_bigquery(
        self,
        cred_file_path: str = None,
        project_id: str = None,
        **kwargs
    ):
        """
        Connect to gcs using the bigquery connector. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_bigquery(
            project_id="myprojectid",
            cred_file_path="path/to/credentials.json",
        )
        ```
        Args:
            project_id (str): The gcs project id.
            cred_file_path (str): The gcs credential file path
        """

        try:
            from google.api_core.exceptions import GoogleAPIError
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command:"
                " \npip install vanna[bigquery]"
            )

        if not project_id:
            project_id = os.getenv("PROJECT_ID")

        if not project_id:
            raise ImproperlyConfigured("Please set your Google Cloud Project ID.")

        import sys

        if "google.colab" in sys.modules:
            try:
                from google.colab import auth

                auth.authenticate_user()
            except Exception as e:
                raise ImproperlyConfigured(e)
        else:
            print("Not using Google Colab.")

        conn = None

        if not cred_file_path:
            try:
                conn = bigquery.Client(project=project_id)
            except:
                print("Could not found any google cloud implicit credentials")
        else:
            # Validate file path and pemissions
            validate_config_path(cred_file_path)

        if not conn:
            with open(cred_file_path, "r") as f:
                credentials = service_account.Credentials.from_service_account_info(
                    json.loads(f.read()),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

            try:
                conn = bigquery.Client(
                    project=project_id,
                    credentials=credentials,
                    **kwargs
                )
            except:
                raise ImproperlyConfigured(
                    "Could not connect to bigquery please correct credentials"
                )

        def run_sql_bigquery(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                job = conn.query(sql)
                df = job.result().to_dataframe()
                return df
            return None

        self.dialect = "BigQuery SQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_bigquery

    def connect_to_duckdb(self, url: str, init_sql: str = None, **kwargs):
        """
        Connect to a DuckDB database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            url (str): The URL of the database to connect to. Use :memory: to create an in-memory database. Use md: or motherduck: to use the MotherDuck database.
            init_sql (str, optional): SQL to run when connecting to the database. Defaults to None.

        Returns:
            None
        """
        try:
            import duckdb
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[duckdb]"
            )
        # URL of the database to download
        if url == ":memory:" or url == "":
            path = ":memory:"
        else:
            # Path to save the downloaded database
            print(os.path.exists(url))
            if os.path.exists(url):
                path = url
            elif url.startswith("md") or url.startswith("motherduck"):
                path = url
            else:
                path = os.path.basename(urlparse(url).path)
                # Download the database if it doesn't exist
                if not os.path.exists(path):
                    response = requests.get(url)
                    response.raise_for_status()  # Check that the request was successful
                    with open(path, "wb") as f:
                        f.write(response.content)

        # Connect to the database
        conn = duckdb.connect(path, **kwargs)
        if init_sql:
            conn.query(init_sql)

        def run_sql_duckdb(sql: str):
            return conn.query(sql).to_df()

        self.dialect = "DuckDB SQL"
        self.run_sql = run_sql_duckdb
        self.run_sql_is_set = True

    def connect_to_mssql(self, odbc_conn_str: str, **kwargs):
        """
        Connect to a Microsoft SQL Server database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            odbc_conn_str (str): The ODBC connection string.

        Returns:
            None
        """
        try:
            import pyodbc
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install pyodbc"
            )

        try:
            import sqlalchemy as sa
            from sqlalchemy.engine import URL
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install sqlalchemy"
            )

        connection_url = URL.create(
            "mssql+pyodbc", query={"odbc_connect": odbc_conn_str}
        )

        from sqlalchemy import create_engine

        engine = create_engine(connection_url, **kwargs)

        def run_sql_mssql(sql: str):
            # Execute the SQL statement and return the result as a pandas DataFrame
            with engine.begin() as conn:
                df = pd.read_sql_query(sa.text(sql), conn)
                conn.close()
                return df

            raise Exception("Couldn't run sql")

        self.dialect = "T-SQL / Microsoft SQL Server"
        self.run_sql = run_sql_mssql
        self.run_sql_is_set = True
    
    def connect_to_presto(
        self,
        host: str,
        catalog: str = 'hive',
        schema: str = 'default',
        user: str = None,
        password: str = None,
        port: int = None,
        combined_pem_path: str = None,
        protocol: str = 'https',
        requests_kwargs: dict = None,
        **kwargs
    ):
      """
        Connect to a Presto database using the specified parameters.

        Args:
            host (str): The host address of the Presto database.
            catalog (str): The catalog to use in the Presto environment.
            schema (str): The schema to use in the Presto environment.
            user (str): The username for authentication.
            password (str): The password for authentication.
            port (int): The port number for the Presto connection.
            combined_pem_path (str): The path to the combined pem file for SSL connection.
            protocol (str): The protocol to use for the connection (default is 'https').
            requests_kwargs (dict): Additional keyword arguments for requests.

        Raises:
            DependencyError: If required dependencies are not installed.
            ImproperlyConfigured: If essential configuration settings are missing.

        Returns:
            None
      """
      try:
        from pyhive import presto
      except ImportError:
        raise DependencyError(
          "You need to install required dependencies to execute this method,"
          " run command: \npip install pyhive"
        )

      if not host:
        host = os.getenv("PRESTO_HOST")

      if not host:
        raise ImproperlyConfigured("Please set your presto host")

      if not catalog:
        catalog = os.getenv("PRESTO_CATALOG")

      if not catalog:
        raise ImproperlyConfigured("Please set your presto catalog")

      if not user:
        user = os.getenv("PRESTO_USER")

      if not user:
        raise ImproperlyConfigured("Please set your presto user")

      if not password:
        password = os.getenv("PRESTO_PASSWORD")

      if not port:
        port = os.getenv("PRESTO_PORT")

      if not port:
        raise ImproperlyConfigured("Please set your presto port")

      conn = None

      try:
        if requests_kwargs is None and combined_pem_path is not None:
          # use the combined pem file to verify the SSL connection
          requests_kwargs = {
            'verify': combined_pem_path,  # 使用转换后得到的 PEM 文件进行 SSL 验证
          }
        conn = presto.Connection(host=host,
                                 username=user,
                                 password=password,
                                 catalog=catalog,
                                 schema=schema,
                                 port=port,
                                 protocol=protocol,
                                 requests_kwargs=requests_kwargs,
                                 **kwargs)
      except presto.Error as e:
        raise ValidationError(e)

      def run_sql_presto(sql: str) -> Union[pd.DataFrame, None]:
        if conn:
          try:
            sql = sql.rstrip()
            # fix for a known problem with presto db where an extra ; will cause an error.
            if sql.endswith(';'):
                sql = sql[:-1]
            cs = conn.cursor()
            cs.execute(sql)
            results = cs.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(
              results, columns=[desc[0] for desc in cs.description]
            )
            return df

          except presto.Error as e:
            print(e)
            raise ValidationError(e)

          except Exception as e:
            print(e)
            raise e

      self.run_sql_is_set = True
      self.run_sql = run_sql_presto

    def connect_to_hive(
        self,
        host: str = None,
        dbname: str = 'default',
        user: str = None,
        password: str = None,
        port: int = None,
        auth: str = 'CUSTOM',
        **kwargs
    ):
      """
        Connect to a Hive database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        Connect to a Hive database. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]

        Args:
            host (str): The host of the Hive database.
            dbname (str): The name of the database to connect to.
            user (str): The username to use for authentication.
            password (str): The password to use for authentication.
            port (int): The port to use for the connection.
            auth (str): The authentication method to use.

        Returns:
            None
      """

      try:
        from pyhive import hive
      except ImportError:
        raise DependencyError(
          "You need to install required dependencies to execute this method,"
          " run command: \npip install pyhive"
        )

      if not host:
        host = os.getenv("HIVE_HOST")

      if not host:
        raise ImproperlyConfigured("Please set your hive host")

      if not dbname:
        dbname = os.getenv("HIVE_DATABASE")

      if not dbname:
        raise ImproperlyConfigured("Please set your hive database")

      if not user:
        user = os.getenv("HIVE_USER")

      if not user:
        raise ImproperlyConfigured("Please set your hive user")

      if not password:
        password = os.getenv("HIVE_PASSWORD")

      if not port:
        port = os.getenv("HIVE_PORT")

      if not port:
        raise ImproperlyConfigured("Please set your hive port")

      conn = None

      try:
        conn = hive.Connection(host=host,
                               username=user,
                               password=password,
                               database=dbname,
                               port=port,
                               auth=auth)
      except hive.Error as e:
        raise ValidationError(e)

      def run_sql_hive(sql: str) -> Union[pd.DataFrame, None]:
        if conn:
          try:
            cs = conn.cursor()
            cs.execute(sql)
            results = cs.fetchall()

            # Create a pandas dataframe from the results
            df = pd.DataFrame(
              results, columns=[desc[0] for desc in cs.description]
            )
            return df

          except hive.Error as e:
            print(e)
            raise ValidationError(e)

          except Exception as e:
            print(e)
            raise e

      self.run_sql_is_set = True
      self.run_sql = run_sql_hive

    def run_sql(self, sql: str, **kwargs) -> pd.DataFrame:
        """
        Example:
        ```python
        vn.run_sql("SELECT * FROM my_table")
        ```

        Run a SQL query on the connected database.

        Args:
            sql (str): The SQL query to run.

        Returns:
            pd.DataFrame: The results of the SQL query.
        """
        raise Exception(
            "You need to connect to a database first by running vn.connect_to_snowflake(), vn.connect_to_postgres(), similar function, or manually set vn.run_sql"
        )

    # ---------- Main API Methods
    def ask(
        self,
        question: Union[str, None] = None,
        print_results: bool = True,
        auto_train: bool = False,
        visualize: bool = False,  # if False, will not generate plotly code
        allow_llm_to_see_data: bool = True,
    ) -> Union[
        Tuple[
            Union[str, None],
            Union[pd.DataFrame, None],
            Union[plotly.graph_objs.Figure, None],
        ],
        None,
    ]:
        """
        **Example:**
        ```python
        vn.ask("What are the top 10 customers by sales?")
        ```

        Ask Vanna.AI a question and get the SQL query that answers it.

        Args:
            question (str): The question to ask.
            print_results (bool): Whether to print the results of the SQL query.
            auto_train (bool): Whether to automatically train Vanna.AI on the question and SQL query.
            visualize (bool): Whether to generate plotly code and display the plotly figure.

        Returns:
            Tuple[str, pd.DataFrame, plotly.graph_objs.Figure]: The SQL query, the results of the SQL query, and the plotly figure.
        """

        self.create_new_thread()
        self.log(message=f"New thread initialized: {self.current_thread_name}")

        if question is None:
            question = input("Enter a question: ")

        self.log(echo=False, message=question, title="Asked Question")

        try:
            sql = self.generate_sql(question=question, allow_llm_to_see_data=allow_llm_to_see_data)
            self.log(echo=False, message=sql, title="Generated SQL")
        except Exception as e:
            self.log(message=f"The following exception occurred: {e}", title="Exception in generating SQL")
            return None, None, None

        if print_results:
            try:
                Code = __import__("IPython.display", fromList=["Code"]).Code
                display(Code(sql))
            except Exception as e:
                print(sql)

        if self.run_sql_is_set is False:
            print(
                "If you want to run the SQL query, connect to a database first."
            )

            self.log(echo=False, message="Exited beacuse no database engine is connected", title="Exit")

            if print_results:
                return None
            else:
                return sql, None, None

        try:
            df = self.run_sql(sql)
            self.log(echo=False, message="Saved dataframe from running the previous query", save_df=True, df=df)

            if print_results:
                try:
                    display = __import__(
                        "IPython.display", fromList=["display"]
                    ).display
                    display(df)
                except Exception as e:
                    print(df)

            if auto_train and len(df) > 0:
                self.add_question_sql(question=question, sql=sql)
            # Only generate plotly code if visualize is True
            if visualize:
                try:
                    plotly_code = self.generate_plotly_code(
                        question=question,
                        sql=sql,
                        df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
                    )
                    fig = self.get_plotly_figure(plotly_code=plotly_code, df=df)
                    if print_results:
                        try:
                            display = __import__(
                                "IPython.display", fromlist=["display"]
                            ).display
                            Image = __import__(
                                "IPython.display", fromlist=["Image"]
                            ).Image
                            img_bytes = fig.to_image(format="png", scale=2)
                            display(Image(img_bytes))
                        except Exception as e:
                            fig.show()
                except Exception as e:
                    # Print stack trace
                    traceback.print_exc()
                    print("Couldn't run plotly code: ", e)
                    if print_results:
                        return None
                    else:
                        return sql, df, None
            else:
                return sql, df, None

        except Exception as e:
            self.log(message=e, title="Exception in running SQL")
            if print_results:
                return None
            else:
                return sql, None, None
        return sql, df, fig

    def ask_agent(
        self,
        question: str,
        print_results: bool = True,
        return_sql: bool = True,
    ) -> Tuple[Union[str, None], Union[pd.DataFrame, None], Union[plotly.graph_objs.Figure, None]]:
        """Use the ReAct agent to answer a question.

        Parameters
        ----------
        question : str
            The natural language question.
        print_results : bool, optional
            If ``True`` the resulting dataframe will be printed, by default ``True``.
        return_sql : bool, optional
            If ``True`` include the SQL generated by the agent in the return tuple.

        Returns
        -------
        tuple
            ``(sql, dataframe, figure)`` similar to :py:meth:`ask`.
        """

        if self.agent is None:
            self.create_agent()

        self.create_new_thread(thread_type="react")

        # reset last agent state
        self.agent_last_sql = None
        self.agent_last_df = None

        # doc_list = self.get_related_documentation(question)

        # prompt_content = (
        #     "You are a T-SQL / Microsoft SQL Server expert. Please help to generate a SQL query to answer the question. "
        #     "You are provided with a tool for querying the AdventureWorks2022 database. The tool name is \"run_sql\".\n"
        #     f"You are provided with the description of related tables retrieved from RAG:\n{'\n\n'.join(doc_list)}\n"
        #     f"The question is: {question}"
        # )

        prompt_content = self._build_agent_prompt(question)

        self.log(message=f"Prompt:\n{prompt_content}", title="Prompt")

        prompt = [{"role": "user", "content": prompt_content}]

        try:
            result = self.agent.invoke({"messages": prompt})
            self._log_agent_chain(result)
        except Exception as e:
            self.log(message=e, title="Exception in agent")
            if print_results:
                return None
            else:
                return None, None, None

        sql = self.agent_last_sql if return_sql else None
        df = self.agent_last_df
        fig = None

        if print_results and df is not None:
            try:
                display = __import__("IPython.display", fromlist=["display"]).display
                display(df)
            except Exception:
                print(df)

        return sql, df, fig

    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
    ) -> str:
        """
        **Example:**
        ```python
        vn.train()
        ```

        Train Vanna.AI on a question and its corresponding SQL query.
        If you call it with no arguments, it will check if you connected to a database and it will attempt to train on the metadata of that database.
        If you call it with the sql argument, it's equivalent to [`vn.add_question_sql()`][vanna.base.base.VannaBase.add_question_sql].
        If you call it with the ddl argument, it's equivalent to [`vn.add_ddl()`][vanna.base.base.VannaBase.add_ddl].
        If you call it with the documentation argument, it's equivalent to [`vn.add_documentation()`][vanna.base.base.VannaBase.add_documentation].
        Additionally, you can pass a [`TrainingPlan`][vanna.types.TrainingPlan] object. Get a training plan with [`vn.get_training_plan_generic()`][vanna.base.base.VannaBase.get_training_plan_generic].

        Args:
            question (str): The question to train on.
            sql (str): The SQL query to train on.
            ddl (str):  The DDL statement.
            documentation (str): The documentation to train on.
            plan (TrainingPlan): The training plan to train on.
        """

        self.create_new_thread(thread_type="train")

        if question and not sql:
            self.log(message="Exiting training because a question is provided without the sql", title="Exit", echo=False)
            raise ValidationError("Please also provide a SQL query")

        if documentation:
            self.log(f"Training on document {documentation}", title="Train on Document")
            return self.add_documentation(documentation)

        if sql:
            if question is None:
                question = self.generate_question(sql)
                self.log(message=f"Generating question for the SQL:\n{sql}", title="Generate Question")
            self.log(message=f"Training on question-SQL pair:\n\n{question}\n\n{sql}", title="Train on Question")
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            self.log(message=f"Training on DDL:\n{ddl}", title="Train on DDL")
            return self.add_ddl(ddl)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)

            self.log(
                message="Training on a plan:\n\n" + "\n\n".join(
                    item.item_value for item in plan._plan
                ),
                title="Train on Plan",
                echo=False,
            )

    # ---------- Helper Methods
    def _get_databases(self) -> List[str]:
        try:
            print("Trying INFORMATION_SCHEMA.DATABASES")
            df_databases = self.run_sql("SELECT * FROM INFORMATION_SCHEMA.DATABASES")
        except Exception as e:
            print(e)
            try:
                print("Trying SHOW DATABASES")
                df_databases = self.run_sql("SHOW DATABASES")
            except Exception as e:
                print(e)
                return []

        return df_databases["DATABASE_NAME"].unique().tolist()

    def _get_information_schema_tables(self, database: str) -> pd.DataFrame:
        df_tables = self.run_sql(f"SELECT * FROM {database}.INFORMATION_SCHEMA.TABLES")

        return df_tables

    def get_training_plan_generic(self, df) -> TrainingPlan:
        """
        This method is used to generate a training plan from an information schema dataframe.

        Basically what it does is breaks up INFORMATION_SCHEMA.COLUMNS into groups of table/column descriptions that can be used to pass to the LLM.

        Args:
            df (pd.DataFrame): The dataframe to generate the training plan from.

        Returns:
            TrainingPlan: The training plan.
        """

        self.create_new_thread(thread_type="plan")
        self.log(message="Planning started with the given dataframe", save_df=True, df=df, echo=False)

        # For each of the following, we look at the df columns to see if there's a match:
        database_column = df.columns[
            df.columns.str.lower().str.contains("database") |
            df.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]

        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]

        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]

        columns = [database_column,
                    schema_column,
                    table_column]
        
        candidates = ["column_name",
                      "data_type",
                      "comment"]
        
        matches = df.columns.str.lower().str.contains("|".join(candidates), regex=True)
        columns += df.columns[matches].to_list()

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                df.query(f'{database_column} == "{database}"')[schema_column]
                .unique()
                .tolist()
            ):
                for table in (
                    df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}"'
                    )[table_column]
                    .unique()
                    .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}" and {table_column} == "{table}"'
                    )
                    doc = f"The following columns are in the {schema}.{table} table in the {database} database:\n"
                    doc += f"{self.extract_column_types(df_columns_filtered_to_table[columns])}\n"
                    doc += (
                        "Provided description for the table: "
                        + self.llm_describe_table(
                            doc=doc,
                            rows=self.get_table_top_rows(table_name=f"{schema}.{table}"),
                        )
                    )

                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

        self.log(
            message="The following tables found:\n" + "\n".join(plan.get_summary()),
            title="Plan Table",
            echo=False,
        )

        return plan

    def get_training_plan_snowflake(
        self,
        filter_databases: Union[List[str], None] = None,
        filter_schemas: Union[List[str], None] = None,
        include_information_schema: bool = False,
        use_historical_queries: bool = True,
    ) -> TrainingPlan:
        plan = TrainingPlan([])

        if self.run_sql_is_set is False:
            raise ImproperlyConfigured("Please connect to a database first.")

        if use_historical_queries:
            try:
                print("Trying query history")
                df_history = self.run_sql(
                    """ select * from table(information_schema.query_history(result_limit => 5000)) order by start_time"""
                )

                df_history_filtered = df_history.query("ROWS_PRODUCED > 1")
                if filter_databases is not None:
                    mask = (
                        df_history_filtered["QUERY_TEXT"]
                        .str.lower()
                        .apply(
                            lambda x: any(
                                s in x for s in [s.lower() for s in filter_databases]
                            )
                        )
                    )
                    df_history_filtered = df_history_filtered[mask]

                if filter_schemas is not None:
                    mask = (
                        df_history_filtered["QUERY_TEXT"]
                        .str.lower()
                        .apply(
                            lambda x: any(
                                s in x for s in [s.lower() for s in filter_schemas]
                            )
                        )
                    )
                    df_history_filtered = df_history_filtered[mask]

                if len(df_history_filtered) > 10:
                    df_history_filtered = df_history_filtered.sample(10)

                for query in df_history_filtered["QUERY_TEXT"].unique().tolist():
                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_SQL,
                            item_group="",
                            item_name=self.generate_question(query),
                            item_value=query,
                        )
                    )

            except Exception as e:
                print(e)

        databases = self._get_databases()

        for database in databases:
            if filter_databases is not None and database not in filter_databases:
                continue

            try:
                df_tables = self._get_information_schema_tables(database=database)

                print(f"Trying INFORMATION_SCHEMA.COLUMNS for {database}")
                df_columns = self.run_sql(
                    f"SELECT * FROM {database}.INFORMATION_SCHEMA.COLUMNS"
                )

                for schema in df_tables["TABLE_SCHEMA"].unique().tolist():
                    if filter_schemas is not None and schema not in filter_schemas:
                        continue

                    if (
                        not include_information_schema
                        and schema == "INFORMATION_SCHEMA"
                    ):
                        continue

                    df_columns_filtered_to_schema = df_columns.query(
                        f"TABLE_SCHEMA == '{schema}'"
                    )

                    try:
                        tables = (
                            df_columns_filtered_to_schema["TABLE_NAME"]
                            .unique()
                            .tolist()
                        )

                        for table in tables:
                            df_columns_filtered_to_table = (
                                df_columns_filtered_to_schema.query(
                                    f"TABLE_NAME == '{table}'"
                                )
                            )
                            doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                            doc += df_columns_filtered_to_table[
                                [
                                    "TABLE_CATALOG",
                                    "TABLE_SCHEMA",
                                    "TABLE_NAME",
                                    "COLUMN_NAME",
                                    "DATA_TYPE",
                                    "COMMENT",
                                ]
                            ].to_markdown()

                            plan._plan.append(
                                TrainingPlanItem(
                                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                                    item_group=f"{database}.{schema}",
                                    item_name=table,
                                    item_value=doc,
                                )
                            )

                    except Exception as e:
                        print(e)
                        pass
            except Exception as e:
                print(e)

        return plan

    def get_plotly_figure(
        self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """
        **Example:**
        ```python
        fig = vn.get_plotly_figure(
            plotly_code="fig = px.bar(df, x='name', y='salary')",
            df=df
        )
        fig.show()
        ```
        Get a Plotly figure from a dataframe and Plotly code.

        Args:
            df (pd.DataFrame): The dataframe to use.
            plotly_code (str): The Plotly code to use.

        Returns:
            plotly.graph_objs.Figure: The Plotly figure.
        """
        ldict = {"df": df, "px": px, "go": go}
        try:
            exec(plotly_code, globals(), ldict)

            fig = ldict.get("fig", None)
        except Exception as e:
            # Inspect data types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Decision-making for plot type
            if len(numeric_cols) >= 2:
                # Use the first two numeric columns for a scatter plot
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                # Use a bar plot if there's one numeric and one categorical column
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
                # Use a pie chart for categorical data with fewer unique values
                fig = px.pie(df, names=categorical_cols[0])
            else:
                # Default to a simple line plot if above conditions are not met
                fig = px.line(df)

        if fig is None:
            return None

        if dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig
    
    def extract_column_types(self, df : pd.DataFrame):
        """
        Extracts COLUMN_NAME and DATA_TYPE pairs from a pandas DataFrame
        and returns them as a formatted string.
        
        Example output:
        - CountryRegionCode -> nvarchar
        - CurrencyCode -> nchar
        - ModifiedDate -> datetime
        """
        required_cols = ["COLUMN_NAME", "DATA_TYPE"]
        
        # Normalize column names (to handle different casing or spacing)
        col_map = {col.lower(): col for col in df.columns}
        if not all(col.lower() in col_map for col in required_cols):
            raise ValueError("DataFrame must contain 'COLUMN_NAME' and 'DATA_TYPE' columns.")

        column_col = col_map["column_name"]
        datatype_col = col_map["data_type"]

        lines = [
            f"- {row[column_col]} ({row[datatype_col]})"
            for _, row in df[[column_col, datatype_col]].iterrows()
        ]
        return "\n".join(lines)

    def get_table_top_rows(self, table_name : str, n=5):
        try:
            query = f"SELECT * FROM {table_name}"
            return self.run_sql(query).head(n)
        except Exception as e:
            self.log(message=f"Error fetching top rows from {table_name}: {e}", title="Exception in Fetch Table Rows")
            return None

    def llm_describe_table(self, doc: str, rows: pd.DataFrame) -> str:
        # Convert the sample rows to markdown for better formatting
        if rows is not None:
            sample_data_str = rows.to_markdown(index=False)
        else:
            sample_data_str = "Rows of the table is not provided for this table"

        # Construct the prompt for the LLM
        system_message = (
            "You are a data analyst. Respond straightforwardly in a single, concise paragraph."
        )

        user_message = (
            "Below is the schema of a database table, showing column names and their data types:\n\n"
            f"{doc}\n\n"
            "Here are the first few rows of data from this table:\n\n"
            f"{sample_data_str}\n\n"
            "Write a brief (less than one paragraph), clear description of what this table likely represents, "
            "the kind of information it contains, and what each column might mean. Use professional and informative language."
        )

        prompt = [
                self.system_message(system_message),
                self.user_message(user_message)
            ]
        
        self.log(message=prompt, title="Prompt", echo=False)

        # Submit the prompt to the LLM and return its response
        try:
            return self.submit_prompt(prompt)
        except Exception as e:
            self.log(message=f"Error prompting LLM: {e}", title="Exception in Prompt")
            return ""
        
    # ---------- Agent Helper Methods
    def create_agent(
            self,
            model = "gpt-4o-mini",
            model_provider = "openai",
            api_base = "https://api.metisai.ir/openai/v1",
            api_key = read_metis_api_key(),
            agent_toolkit: list = ["run_sql", "ask_user", "query_rag"],
    ):
        self.create_new_thread(thread_type="agent-init")

        chat_model = init_chat_model(
            model=model,
            model_provider=model_provider, 
            openai_api_base=api_base,
            openai_api_key=api_key,
        )

        toolkit = []
        self.agent_toolkit = agent_toolkit
        if "run_sql" in agent_toolkit:
            query_tool = StructuredTool.from_function(
                func=self.agent_run_sql_query,
                name="run_sql",
                description = (
                            "Execute a raw SQL query against the connected database and return first 5 rows of the result "
                            "as a Pandas DataFrame formatted in Markdown. Useful for reading, analyzing, "
                            "or summarizing tabular data. The input should be a valid SQL query string. "
                        ),
                args_schema=QueryArgs
            )
            toolkit.append(query_tool)
        if "ask_user" in agent_toolkit:
            clarify_tool = StructuredTool.from_function(
                func=self.agent_ask_user_clarification,
                name="ask_user",
                description = (
                    "Asks user for more clarification and returns the user's response. "
                    "Should be asked in the same language as user has asked the question"
                ),
                args_schema=AskUserArgs
            )
            toolkit.append(clarify_tool)
        if "query_rag" in agent_toolkit:
            retrieve_similar_vectors = StructuredTool.from_function(
                func=self.agent_query_rag,
                name="retrieve_similar_vectors",
                description = (
                    "Query a RAG system containing definition and natural language description of database tables. "
                    "Should be used as natural language"
                ),
                args_schema=QueryRAGArgs
            )
            toolkit.append(retrieve_similar_vectors)
        if "shamsi_to_gregorian" in agent_toolkit:
            date_tool = StructuredTool.from_function(
                func=self.agent_shamsi_to_gregorian,
                name="shamsi_to_gregorian",
                description="Convert a Shamsi (Jalali) date string to Gregorian YYYY-MM-DD format.",
                args_schema=ShamsiDateArgs,
            )
            toolkit.append(date_tool)

        self.agent = create_react_agent(model=chat_model, tools=toolkit)

        self.log(
            message=f"Agent initialized with {model} model and {model_provider} provider with {toolkit} as its toolkit",
            title="Agent Initialization"
        )

        return

    def agent_run_sql_query(self, query : str) -> str:
        sql = self.extract_sql(query)
        self.agent_last_sql = sql
        try:
            df = self.run_sql(sql)
        except Exception:
            self.agent_last_df = None
            raise
        self.agent_last_df = df
        
        self.log(message=f"Run SQL query is called with prompt:\n{query}", title="Tool Call", save_df=True, df=df)

        return df.head(n=5).to_markdown()
    
    def agent_ask_user_clarification(self, question : str) -> str:

        prebuilt_responses = {
            0: "I don't know",
            1: "Yes",
            2: "No",
            3: "Doesn't matter"
        }
        user_response = input(f"Agent has asked you for more clarification: {question}\n0 -> I don't know, 1 -> Yes, 2 -> No, 3 -> Doesn't matter\n")
        if user_response == "0":
            user_response =  prebuilt_responses[0]
        elif user_response == "1":
            user_response = prebuilt_responses[1]
        elif user_response == "2":
            user_response = prebuilt_responses[2]
        elif user_response == "3":
            user_response = prebuilt_responses[3]
        elif user_response == None or user_response == "":
            user_response = "User did not respond"
        
        self.log(message=f"Agent ask user:\n{question}\n\nUser responded:\n{user_response}", title="Tool Call")

        return user_response

    def agent_query_rag(self, query : str, count : int) -> list:
        docs = "\n\n".join(self.get_related_documentation(query))

        self.log(message=f"Agent queried the RAG for {count} similar vectors with:\n{query}\n\nRAG output:\n{docs}", title="Tool Call")

        return docs

    def agent_shamsi_to_gregorian(self, date: str) -> str:
        """Convert a Shamsi (Jalali) date string to Gregorian ``YYYY-MM-DD``."""
        try:
            import jdatetime  # type: ignore
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: pip install jdatetime"
            )

        date_normalized = date.replace("-", "/")
        try:
            parts = [int(p) for p in date_normalized.split("/")]
            if len(parts) != 3:
                raise ValueError("Date must have year, month and day")
            jalali = jdatetime.date(parts[0], parts[1], parts[2])
            gregorian = jalali.togregorian()
            result = gregorian.strftime("%Y-%m-%d")
        except Exception as e:
            self.log(message=f"Error converting date '{date}': {e}", title="Tool Error")
            raise

        self.log(message=f"Converted {date} -> {result}", title="Tool Call")
        return result
    
    def agent_gets_db_functions(self):
        pass

    # ---------- Agent Utility Methods
    def _to_serializable(self, obj: any) -> any:
        """Attempt to convert various message objects to simple Python types."""
        if isinstance(obj, dict):
            return obj
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        for attr in ("model_dump", "dict", "to_dict"):
            if hasattr(obj, attr):
                try:
                    return getattr(obj, attr)()
                except Exception:
                    pass
        if hasattr(obj, "model_dump_json"):
            try:
                return json.loads(obj.model_dump_json())
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            try:
                return json.loads(json.dumps(obj.__dict__, default=str))
            except Exception:
                return {k: self._to_serializable(v) for k, v in obj.__dict__.items()}
        try:
            return json.loads(str(obj))
        except Exception:
            return str(obj)

    def _format_agent_messages(self, messages: list) -> str:
        parts = []
        for m in messages:
            data = self._to_serializable(m)
            if isinstance(data, dict) and "role" in data and "content" in data:
                role = data.get("role", "assistant")
                content = data.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(data))
        return "\n".join(parts)

    def _agent_messages_to_tree(self, messages: list):
        tree = []
        for m in messages:
            data = self._to_serializable(m)
            if not isinstance(data, dict):
                tree.append({"message": data})
                continue

            node = {"role": data.get("role")}
            if data.get("id"):
                node["id"] = data.get("id")
            if data.get("name"):
                node["name"] = data.get("name")
            if data.get("content"):
                node["content"] = data.get("content")
            if data.get("tool_call_id"):
                node["tool_call_id"] = data.get("tool_call_id")

            add_kwargs = data.get("additional_kwargs", {})
            tool_calls = add_kwargs.get("tool_calls")
            if tool_calls:
                calls = []
                for tc in tool_calls:
                    tc_data = self._to_serializable(tc)
                    call = {
                        "id": tc_data.get("id"),
                        "type": tc_data.get("type"),
                        "name": tc_data.get("function", {}).get("name"),
                        "arguments": tc_data.get("function", {}).get("arguments"),
                    }
                    calls.append(call)
                node["tool_calls"] = calls

            tree.append(node)
        return tree

    def _log_agent_chain(self, result):
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
        else:
            messages = result
        if messages:
            formatted = self._format_agent_messages(messages)
            self.log(message=formatted, title="Agent Chain")
            tree = self._agent_messages_to_tree(messages)
            self.log_json(tree, title="Agent Chain")

    def _build_agent_prompt(self, question: str) -> str:
        toolkit = getattr(self, "agent_toolkit", [])
        tool_lines = []
        if "run_sql" in toolkit:
            tool_lines.append("* `run_sql` to execute SQL queries on the database.")
        if "ask_user" in toolkit:
            tool_lines.append("* `ask_user` to request clarification from the user when needed.")
        if "query_rag" in toolkit:
            tool_lines.append("* `retrieve_similar_vectors` to search documentation about tables and columns.")
        if "shamsi_to_gregorian" in toolkit:
            tool_lines.append("* `shamsi_to_gregorian` to convert Persian (Shamsi) dates to Gregorian.")

        tools_section = "\n".join(tool_lines)
        if tools_section:
            tools_section = "Available tools:\n" + tools_section + "\n"

        prompt = (
            "You are a T-SQL / Microsoft SQL Server expert. "
            "If asking the user for more information, always ask in the same language as user has asked the question"
            "Use the tools below to reason your way to the best answer.\n"
            f"{tools_section}"
            "Guidelines:\n"
            "1. Translate the question to English if it's written in another language. Never ask user for clarification about the translation.\n"
            "2. Search the RAG for related tables and columns.\n"
            "3. Try to create a structured prompt that contains the exact filters, column names, groupings and orderings if required.\n"
            "4. If the current structured prompt is ambigious (there is mismatch between names) try providing user with options to choose.\n"
            "5. Plan a query and execute it using `run_sql`.\n"
            "6. Iterate until the results answer the question with confidence.\n"
            f"User question: {question}"
        )
        return prompt

class QueryArgs(PydanticBaseModelForTool):
    query: str = Field(..., description="SQL query to be executed")

class AskUserArgs(PydanticBaseModelForTool):
    question: str = Field(..., description="Question to ask user for more clarification")

class QueryRAGArgs(PydanticBaseModelForTool):
    query: str = Field(..., description="Query to find similarities from vector database")
    count: int = Field(..., description="Number of similar vectors to retreive from vector database")

class ShamsiDateArgs(PydanticBaseModelForTool):
    date: str = Field(..., description="Date in Shamsi (Jalali) calendar")
