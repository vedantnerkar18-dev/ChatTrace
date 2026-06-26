import pandas as pd
import re

def parse_whatsapp_chat(file_path):
    """
    Reads a WhatsApp exported .txt file and converts it
    into a clean pandas DataFrame
    """
    
    messages = []
    
    # WhatsApp message pattern: date, time, sender, message
    pattern = r'\[?(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?\s?-?\s?([^:]+):\s(.*)'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_message = None
    
    for line in lines:
        line = line.strip()
        match = re.match(pattern, line)
        
        if match:
            # Save previous message
            if current_message:
                messages.append(current_message)
            
            date, time, sender, message = match.groups()
            
            # Skip system messages
            if 'Messages and calls are end-to-end encrypted' in message:
                current_message = None
                continue
            if 'omitted' in message.lower():
                current_message = None
                continue
                
            current_message = {
                'date': date,
                'time': time,
                'sender': sender.strip(),
                'message': message.strip()
            }
        else:
            # This line is continuation of previous message
            if current_message and line:
                current_message['message'] += ' ' + line
    
    # Add last message
    if current_message:
        messages.append(current_message)
    
    df = pd.DataFrame(messages)
    return df


def test_parser():
    """Quick test to check parser is working"""
    sample = """12/06/2025, 10:30 - Priya: Hey did you try that pasta recipe?
12/06/2025, 10:31 - Sneha: Yes!! It was amazing, add extra garlic
12/06/2025, 10:32 - Priya: How much garlic exactly?
12/06/2025, 10:33 - Sneha: Like 6-7 cloves, trust me"""
    
    # Write sample to temp file
    with open('test_chat.txt', 'w', encoding='utf-8') as f:
        f.write(sample)
    
    df = parse_whatsapp_chat('test_chat.txt')
    print("✅ Parser working!")
    print(f"Total messages parsed: {len(df)}")
    print("\nSample output:")
    print(df.to_string())
    
if __name__ == "__main__":
    test_parser()