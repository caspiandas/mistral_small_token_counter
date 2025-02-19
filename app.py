import streamlit as st
import pandas as pd
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import io

def count_tokens(file_content):
    """Count tokens in the given file content using Mistral v3 tokenizer"""
    tokenizer = MistralTokenizer.v3()
    
    # Create a chat completion request with the file content as a user message
    chat_request = ChatCompletionRequest(
        messages=[UserMessage(content=file_content)],
        model="test"  # Required parameter but not used for tokenization
    )
    
    # Encode the chat completion request
    tokenized = tokenizer.encode_chat_completion(chat_request)
    return len(tokenized.tokens)

def main():
    st.title("SystemVerilog File Token Counter")
    st.write("Upload SystemVerilog files (.v) to count tokens using Mistral v3 tokenizer")
    
    # File uploader that accepts multiple files
    uploaded_files = st.file_uploader(
        "Choose SystemVerilog file(s)", 
        accept_multiple_files=True,
        type=['v']
    )
    
    if uploaded_files:
        # Create a list to store results
        results = []
        
        # Process each uploaded file
        for file in uploaded_files:
            try:
                # Read file content
                content = file.read().decode('utf-8')
                
                # Count tokens
                token_count = count_tokens(content)
                
                # Add to results
                results.append({
                    'Filename': file.name,
                    'File Size (bytes)': len(content),
                    'Token Count': token_count
                })
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        # Create DataFrame and display results
        if results:
            df = pd.DataFrame(results)
            st.subheader("Token Count Results")
            st.dataframe(
                df,
                column_config={
                    "Filename": st.column_config.TextColumn("Filename"),
                    "File Size (bytes)": st.column_config.NumberColumn("File Size (bytes)", format="%d"),
                    "Token Count": st.column_config.NumberColumn("Token Count", format="%d")
                },
                hide_index=True
            )
            
            # Add download button for results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="token_counts.csv",
                mime="text/csv"
            )
            
            # Display total statistics
            st.subheader("Summary Statistics")
            st.write(f"Total Files Processed: {len(results)}")
            st.write(f"Total Tokens: {sum(r['Token Count'] for r in results):,}")
            st.write(f"Average Tokens per File: {sum(r['Token Count'] for r in results) / len(results):,.2f}")

if __name__ == "__main__":
    main()