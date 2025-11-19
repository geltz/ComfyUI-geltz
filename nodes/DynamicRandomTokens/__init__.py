import re
import random
import nodes

# 1. Define the parsing logic
def process_dynamic_syntax(text):
    """
    Recursively replaces {a|b|c} with a random choice.
    Handles nesting like {color {red|blue}| texture}.
    """
    if not isinstance(text, str):
        return text

    pattern = re.compile(r'\{([^{}]+)\}')
    
    # Keep replacing until no braces remain
    while pattern.search(text):
        def replace(match):
            options = match.group(1).split('|')
            # Strip whitespace to be clean, though not strictly necessary
            options = [opt.strip() for opt in options] 
            return random.choice(options)
        
        text = pattern.sub(replace, text)
        
    return text

# 2. Capture the original method so we can still call it
original_encode = nodes.CLIPTextEncode.encode

# 3. Define our replacement method
def hijacked_encode(self, clip, text):
    # Process the text to resolve {a|b} logic
    new_text = process_dynamic_syntax(text)
    
    # Log to console so you know what was picked (optional, but helpful)
    if new_text != text:
        print(f"\033[96m[DynamicPrompt] Converted:\033[0m {text}")
        print(f"\033[96m[DynamicPrompt]      To ->:\033[0m {new_text}")

    # Pass the modified text to the original function
    return original_encode(self, clip, new_text)

# 4. Apply the patch (Monkey Patching)
print("\033[92m[Global Patch] Enabled {word|word} syntax support.\033[0m")
nodes.CLIPTextEncode.encode = hijacked_encode

# Standard ComfyUI node mappings (required file structure, even if empty)
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}