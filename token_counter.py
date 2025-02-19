#!/usr/bin/env python3
import sys
import os
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

def count_tokens(file_content):
    """Count tokens in the given file content using Mistral v3 tokenizer"""
    tokenizer = MistralTokenizer.v3()
    
    chat_request = ChatCompletionRequest(
        messages=[UserMessage(content=file_content)],
        model="test"
    )
    
    tokenized = tokenizer.encode_chat_completion(chat_request)
    return len(tokenized.tokens)

def process_file(filepath):
    """Process a single file and return its token count"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            token_count = count_tokens(content)
            return token_count
    except Exception as e:
        return f"Error processing file: {str(e)}"

def main():
    # Check if files were provided
    if len(sys.argv) < 2:
        print("Usage: python token_counter.py <file1.v> <file2.v> ...")
        sys.exit(1)

    # Process each file
    total_tokens = 0
    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        result = process_file(filepath)
        if isinstance(result, int):
            print(f"{filepath}: {result:,} tokens")
            total_tokens += result
        else:
            print(f"{filepath}: {result}")
    
    # Print summary if multiple files were processed
    if len(sys.argv) > 2:
        print(f"\nTotal tokens across all files: {total_tokens:,}")

if __name__ == "__main__":
    main()