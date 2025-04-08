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

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Append to chat history
    chat_history += f"<|user|>: {user_input}\n<|friend|>:"

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

    # Decode only the newly generated part
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract just the reply after "<|friend|>:"
    reply_start = full_output.rfind("<|friend|>:")
    if reply_start != -1:
        generated_reply = full_output[reply_start + len("<|friend|>:"):].split("<|user|>")[0].strip()
    else:
        generated_reply = "[no response]"

    print(f"Friend: {generated_reply}")

    # Update chat history with the reply
    chat_history += f" {generated_reply}\n"
