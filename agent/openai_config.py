import os
from types import SimpleNamespace

import instructor
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv('/home/ubuntu/lilac/agent/.env')

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_MODEL_CHOICES = (
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o4-mini",
)


def openai_api_key_from_args(args):
    return getattr(args, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")


def openai_model_from_args(args):
    return getattr(args, "gen_model_name", None) or DEFAULT_OPENAI_MODEL


def ensure_openai_args(args):
    if not hasattr(args, "openai_api_key"):
        args.openai_api_key = os.getenv("OPENAI_API_KEY")
    if not hasattr(args, "openai_base_url"):
        args.openai_base_url = os.getenv("OPENAI_BASE_URL")
    if not hasattr(args, "gen_model_name"):
        args.gen_model_name = DEFAULT_OPENAI_MODEL

    return args


def build_openai_client(args):
    args = ensure_openai_args(args)
    kwargs = {}
    api_key = openai_api_key_from_args(args)
    base_url = getattr(args, "openai_base_url", None)
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return instructor.from_openai(OpenAI(**kwargs))


def args_namespace(**overrides):
    defaults = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        "gen_model_name": DEFAULT_OPENAI_MODEL,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
