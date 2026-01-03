import tiktoken

enc = tiktoken.encoding_for_model('gpt-4')

# Read the prompt from file
with open('llm_engineering/applications/datasets/generation.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract prompt between triple quotes
start = content.find('prompt_template_str = """') + len('prompt_template_str = """')
end = content.find('"""', start)
prompt = content[start:end]

# Count
lines = prompt.splitlines()
tokens = enc.encode(prompt)

print(f"Prompt statistics:")
print(f"- Lines: {len(lines)}")
print(f"- Characters: {len(prompt)}")
print(f"- Tokens: {len(tokens)}")
print(f"\nCost analysis (GPT-4):")
print(f"- Input: $0.0000025/token")
print(f"- Output: $0.00001/token (512 tokens)")
print(f"- Cost per call: ${len(tokens) * 0.0000025 + 512 * 0.00001:.4f}")
print(f"- Cost per 100 samples: ${100 * (len(tokens) * 0.0000025 + 512 * 0.00001):.2f}")
print(f"\nToken breakdown:")
print(f"- User prompt (input): {len(tokens)} tokens")
print(f"- System prompt: ~50 tokens")
print(f"- Total input: ~{len(tokens) + 50} tokens")
