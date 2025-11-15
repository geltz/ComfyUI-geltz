import random


class PromptShufflerNode:
    """Shuffles comma-separated Prompts on each execution."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "shuffle"
    CATEGORY = "text"
    
    @classmethod
    def IS_CHANGED(cls, text, seed):
        return float(seed)
    
    def shuffle(self, text, seed):
        # Parse items
        items = [part.strip() for part in text.split(",") if part.strip()]
        
        if len(items) < 2:
            return (text,)  # Return original if < 2 items
        
        # Shuffle and format
        random.shuffle(items)
        output = ", ".join(items)
        
        # Ensure trailing comma
        output = output.rstrip(",").rstrip() + ","
        
        return (output,)


NODE_CLASS_MAPPINGS = {
    "PromptShuffler": PromptShufflerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptShuffler": "Prompt Shuffler"
}