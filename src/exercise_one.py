from shared.lc_llm import get_lc_llm


def main():
    llm = get_lc_llm()

    user_name = input("What is your name? ")
    prompt = f"Hello, my name is {user_name}! Can you repeat my name back to me?"
    llm_response = llm.invoke(prompt)
    print(llm_response.content)


if __name__ == "__main__":
    main()
