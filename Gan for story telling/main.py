from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_story(prompt, max_length=200, num_return_sequences=1):
    # Load pre-trained model and tokenizer
    model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl' for larger models
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,  # Prevents repeating phrases
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode generated text
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    
    return generated_texts

# Example usage
prompt = "Once upon a time in a land far, far away,"
generated_stories = generate_story(prompt, max_length=500, num_return_sequences=3)

for i, story in enumerate(generated_stories):
    print(f"Story {i + 1}:\n{story}\n")
