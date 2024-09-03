from transformers import pipeline, TextStreamer

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def start_chat_session(query, relevant_content):

    pipe = pipeline(
        "text-generation",
        "./Qwen2-1.5B-Instruct",
        torch_dtype="auto",
        device_map="auto",
    )

    streamer = TextStreamer(
        pipe.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    instructions = (
        "You provide accurate, helpful, and consise responses based on DOCUMENT. "
        + "Reference the DOCUMENTS you found the information from. "
        + "If unsure, admit uncertainty. Do not make up information. "
        + "Maintain a neutal and professional tone."
    )

    context = (
        "DOCUMENT1:\n"
        + relevant_content[0]
        + "DOCUMENT2:\n"
        + relevant_content[0]
        + "DOCUMENT3:\n"
        + relevant_content[0]
    )

    init_messages = [
        {"role": "system", "content": instructions},
        {"role": "system", "content": context},
        {"role": "user", "content": query},
    ]

    while True:
        print("\nAssistant: ", end="")
        output = pipe(init_messages, max_new_tokens=512, streamer=streamer)
        answer = output[0]["generated_text"][-1]["content"].strip()

        if query.lower() == "goodbye":
            break

        init_messages.append({"role": "assistant", "content": answer})

        print("\nYou: ", end="")
        query = input()
        init_messages.append({"role": "user", "content": query})
