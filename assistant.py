from transformers import pipeline, TextStreamer


def start_chat_session():

    pipe = pipeline(
        "text-generation",
        "./Qwen2-1.5B-Instruct",
        torch_dtype="auto",
        device_map="auto",
    )

    streamer = TextStreamer(
        pipe.tokenizer,
        skip_prompt=True,
        skip_special_token=True,
    )

    print("You: ", end="")
    query = input()

    messages = [
        {"role": "system", "content": "Here are some instructions: ..."},
        {"role": "system", "content": "Here are some relevant data: ..."},
        {"role": "user", "content": query},
    ]

    while True:
        print("\nAssistant: ", end="")
        output = pipe(messages, max_new_tokens=512, streamer=streamer)
        answer = output[0]["generated-text"][-1]["content"].strip()

        if query.lower() == "goodbye":
            break

        messages.append({"role": "assistant", "content": answer})

        print("\nYou: ", end="")
        query = input()
        messages.append({"role": "user", "content": query})


start_chat_session()
