from shared.lc_llm import get_lc_llm
from langchain_core.prompts import PromptTemplate


def prompt_by_temp(llm, rendered_prompt, temperature: float):
    print(f"\nTemperature: {temperature}")
    print(llm.invoke(rendered_prompt, temperature=temperature).content)


def main():
    llm = get_lc_llm()

    prompt = PromptTemplate(
        template="As a {role}, write a short explanation about {topic}.",
        input_variables=["role", "topic"],
    )

    prompt = prompt.from_template(
        "As a {role}, write a short explanation about {topic}."
    )

    rendered_prompt = prompt.format(role="Writer", topic="What makes a good email?")

    print(f"\nPrompt: {rendered_prompt}")

    temps = [0.0, 0.3, 0.5, 0.7, 1.0]
    for temp in temps:
        prompt_by_temp(llm, rendered_prompt, temp)


if __name__ == "__main__":
    main()
