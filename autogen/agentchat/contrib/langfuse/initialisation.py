from datetime import datetime, timezone
import functools
import secrets
import inspect
import os

import tiktoken

from langfuse import Langfuse
from langfuse.model import ModelUsage

from llm_pricing import oai_prices


class VariableNotFoundError(Exception):
    pass


class LangFuseInitializer:
    def __init__(self, langfuse_secret_key, langfuse_public_key, langfuse_host, langfuse_sample_rate="1", langfuse_debug="False"):
        self.langfuse_secret_key = langfuse_secret_key
        self.langfuse_public_key = langfuse_public_key
        self.langfuse_host = langfuse_host
        self.langfuse_sample_rate = langfuse_sample_rate
        self.langfuse_debug = langfuse_debug

        self.langfuse = None
        self.user_proxy_agent = None
        self.coordination_agent = None
        self.tracks_store = None

        self.initialize()

    def initialize(self):
        os.environ["LANGFUSE_SECRET_KEY"] = self.langfuse_secret_key
        os.environ["LANGFUSE_PUBLIC_KEY"] = self.langfuse_public_key
        os.environ["LANGFUSE_HOST"] = self.langfuse_host
        os.environ["LANGFUSE_SAMPLE_RATE"] = self.langfuse_sample_rate
        os.environ["LANGFUSE_DEBUG"] = self.langfuse_debug

        self.langfuse = Langfuse()

        assert self.langfuse.auth_check()

    def set_tracks(self, tracks_store: list):
        self._is_tracks_store_a_variable(tracks_store)

        self.tracks_store = tracks_store

    def observe(
            self,
            user_id: str = None,
            name: str = None,
            session_id: str = None,
            tags: list = None,
            metadata: dict = None,
            tracks_store: list = None
    ):
        if not tracks_store:
            raise RuntimeError("LangFuseInitializer.set_tracks(tracks_store: list) must be called before observations")

        if not user_id:
            user_id = self._generate_user_id()

        if not session_id:
            session_id = self._generate_session_id(user_id)

        start_time = self._get_timestamp()

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # update trace with function name and dynamic session/user_id
                trace = self.langfuse.trace(
                    name=name if name else func.__name__,
                    start_time=start_time,
                    session_id=session_id,
                    user_id=user_id,
                )

                # then execute the function
                chat_result, finale_answer, initial_message = func(*args, **kwargs)

                end_time = self._get_timestamp()

                # capture output data if set
                trace.update(
                    input=initial_message,
                    end_time=end_time,
                    output=finale_answer
                )

                if tags:
                    trace.update(tags=tags)

                if tags:
                    trace.update(metadata=metadata)

                # get the model name
                model = self._get_model(chat_result)

                for data in self.tracks_store:
                    if data["name"] != "user_proxy":

                        generation = trace.generation(
                            name=data["name"] if data["name"] != "speaker_selection_agent" else "coordination_agent",
                            model=model,
                            start_time=data["start_time"],
                            end_time=data["end_time"],
                            output=data["output"],
                            input=data["input"]
                        )

                        input_tokens = self._get_tokens_count(data["input"], model)
                        output_tokens = self._get_tokens_count(data["output"], model)

                        input_cost = self._get_cost(input_tokens, model, "input")
                        output_cost = self._get_cost(output_tokens, model, "output")

                        _langfuse_usage = ModelUsage(
                            input=input_tokens,
                            output=output_tokens,
                            total=input_tokens + output_tokens,
                            input_cost=input_cost,
                            output_cost=output_cost,
                            total_cost=input_cost + output_cost,
                            unit="TOKENS"
                        )

                        generation.update(
                            model=model,
                            usage=_langfuse_usage
                        )

                return chat_result

            return wrapper

        return decorator

    def _generate_user_id(self):
        timestamp = self._get_timestamp().strftime("%Y%m%d_%H%M%S")
        random_token = secrets.token_hex(4 * 4)

        user_id = f"{timestamp}_{random_token}"

        return user_id

    def _generate_session_id(self, user_id):
        timestamp = self._get_timestamp().strftime("%Y%m%d_%H%M%S")
        user_tag = user_id.split("-")[0]
        random_token = secrets.token_hex(4)

        session_id = f"{timestamp}_{user_tag}_{random_token}"

        return session_id

    @staticmethod
    def _get_tokens_count(text, model):
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(str(text))

        return len(tokens)

    @staticmethod
    def _get_cost(tokens_count, model, side):
        return oai_prices[model][side] * tokens_count / 1e6

    @staticmethod
    def _get_timestamp():
        return datetime.now(timezone.utc)

    @staticmethod
    def _is_tracks_store_a_variable(tracks_store):
        frame = inspect.currentframe()
        caller_locals = frame.f_back.f_locals
        variable_name = [name for name, value in caller_locals.items() if value is tracks_store]

        if variable_name:
            return True
        else:
            raise VariableNotFoundError("tracks_store must be a variable")

    @staticmethod
    def _get_model(chat_result):
        cost = chat_result.cost["usage_including_cached_inference"]
        model = [i for i in cost.keys() if "gpt" in i][0]

        return model
