import json
import re

def json_to_list(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)


def split_into_sentences(text):
    """
    Split text into sentences, handling special cases like abbreviations,
    decimal numbers, etc.
    """
    # Handle common abbreviations and special cases
    text = re.sub(r'([A-Z]\.)([A-Z]\.)', r'\1\n\2', text)  # Handle abbreviations like U.S.A.
    text = re.sub(r'(Dr\.|Mr\.|Mrs\.|Ms\.|Ph\.D\.|e\.g\.|i\.e\.|etc\.)', 
                 lambda m: m.group().replace('.', '[DOT]'), text)
    
    # Split on periods followed by a space and capital letter or periods at end of text
    text = re.sub(r'\.(\s+[A-Z]|$)', r'.\n\1', text)
    
    # Restore abbreviations
    text = text.replace('[DOT]', '.')
    
    # Remove empty lines and extra whitespace
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    return sentences

def process_json(json_data, keys_to_process=None, min_length=100):
    """
    Process JSON data by splitting long text values into sentences.
    
    Args:
        json_data: The JSON data to process
        keys_to_process: List of keys to process (None means all string values)
        min_length: Minimum text length to trigger splitting
    
    Returns:
        Processed JSON data with long text values split into arrays of sentences
    """
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if isinstance(value, str) and (keys_to_process is None or key in keys_to_process):
                if len(value) > min_length:
                    json_data[key] = split_into_sentences(value)
            elif isinstance(value, (dict, list)):
                process_json(value, keys_to_process, min_length)
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            if isinstance(item, (dict, list)):
                process_json(item, keys_to_process, min_length)
    
    return json_data

def process_json_to_str(json_data, result_list):

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if isinstance(value, str):
                result_list.append(value)
            elif isinstance(value, (dict, list)):
                process_json_to_str(value, result_list)
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            if isinstance(item, (dict, list)):
                process_json_to_str(item, result_list)
            else:
                result_list.append(item)
def main():
    # Keys to process (add or remove as needed)
    keys_to_process = ["Title", "Abstract", "Intro", "Method", "Results", "Conclusion", "References"]
    
    input_file = "data/full_text_samples.json"  
    output_file = "data/processed_output.json"
    output_file_raw = "data/processed_output_raw.txt"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = process_json(data, keys_to_process)
    
    # Write the processed data to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Output written to {output_file}")

    result_list = []
    process_json_to_str(processed_data, result_list)

    with open(output_file_raw, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result_list))

    print(f"Raw processing complete. Output written to {output_file_raw}")

if __name__ == "__main__":
    main()