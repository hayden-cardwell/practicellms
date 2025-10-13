from langchain_openai import ChatOpenAI
import os
import dotenv

dotenv.load_dotenv()


def get_lc_llm():

    temp_env = os.getenv("OPENAI_TEMPERATURE")
    temp = float(temp_env) if temp_env else 0.0

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        temperature=temp,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
