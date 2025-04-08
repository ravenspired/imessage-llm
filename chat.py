import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./fine_tuned_model"  
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("You are now chatting with your model. Type 'exit' to stop.\n")

chat_history = ""
inserted_content = 2

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Append to chat history
    chat_history += f"<|me|>: {user_input}\n<|friend|>:"
    # print(f" inserting into model: {chat_history}")

    # Tokenize and create attention mask
    input_ids = tokenizer.encode(chat_history, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)

    # Generate response
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=input_ids.shape[1] + 80,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )

    # Initialize variables
    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("\n")

    # print(raw_output[:inserted_content])
    output = raw_output[:inserted_content]
    output[-1] = output[-1] + "\n"
    print(f"Model: {output[-1].split(': ')[1]}")

    inserted_content += 2


    chat_history = "\n".join(output)
    # print("chat history is now:\n", chat_history)