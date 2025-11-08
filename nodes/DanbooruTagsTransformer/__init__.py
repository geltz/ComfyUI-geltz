import numpy as np
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class DanbooruTagsTransformerMoeV2:
    """Self-contained Danbooru Tags Transformer MoE V2 node"""
    
    CATEGORY = "Dart"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "copyright": ("STRING", {"multiline": False, "default": "original"}),
                "character": ("STRING", {"multiline": False, "default": ""}),
                "rating": (["sfw", "general", "sensitive", "nsfw", "questionable", "explicit"], {"default": "sfw"}),
                "aspect_ratio": (["ultra_wide", "wide", "square", "tall", "ultra_tall"], {"default": "square"}),
                "length": (["very_short", "short", "medium", "long", "very_long"], {"default": "long"}),
                "general": ("STRING", {"multiline": True, "default": "1girl"}),
                "identity": (["none", "lax", "strict"], {"default": "none"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 256}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 20, "min": 1, "max": 500}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "ban_tags": ("STRING", {"multiline": True, "default": ""}),
                "remove_tags": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "p1atdev/dart-v2-moe-sft"
        
    def load_model(self):
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
    
    def compose_prompt(self, copyright, character, rating, aspect_ratio, length, general, identity):
        return (
            f"<|bos|>"
            f"<copyright>{copyright}</copyright>"
            f"<character>{character}</character>"
            f"<|rating:{rating}|><|aspect_ratio:{aspect_ratio}|><|length:{length}|>"
            f"<general>{general}<|identity:{identity}|><|input_end|>"
        )
    
    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(seed)
    
    def generate(self, copyright, character, rating, aspect_ratio, length, general, 
                 identity, seed, max_new_tokens, temperature, top_k, top_p,
                 ban_tags="", remove_tags=""):
        
        self.load_model()
        self.set_seed(seed)
        
        prompt = self.compose_prompt(copyright, character, rating, aspect_ratio, 
                                     length, general, identity)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        bad_words_ids = None
        if ban_tags.strip():
            bad_words_ids = self.tokenizer.encode_plus(ban_tags).input_ids
            if len(bad_words_ids) > 0:
                bad_words_ids = [[token] for token in bad_words_ids]
            else:
                bad_words_ids = None
        
        with torch.no_grad():
            generation_config = self.model.generation_config
            generation_config.max_new_tokens = max_new_tokens
            generation_config.temperature = temperature
            generation_config.top_k = top_k
            generation_config.top_p = top_p
            
            outputs = self.model.generate(
                inputs,
                generation_config=generation_config,
                bad_words_ids=bad_words_ids,
                do_sample=True,
            )
        
        token_ids = outputs[0].tolist()
        
        # Remove tags if specified
        if remove_tags.strip():
            remove_tag_token_ids = self.tokenizer.encode_plus(remove_tags).input_ids
            token_ids = [token for token in token_ids if token not in remove_tag_token_ids]
        
        # Decode in Animagine order
        result = self.decode_by_animagine(token_ids)
        
        return (result,)
    
    def decode_by_animagine(self, token_ids):
        """Decode tokens in Animagine XL v3 order"""
        people_tags = ["1girl", "2girls", "3girls", "4girls", "5girls", "6+girls",
                      "1boy", "2boys", "3boys", "4boys", "5boys", "6+boys",
                      "1other", "2others", "3others", "4others", "5others", "6+others",
                      "no humans"]
        
        special_tags = ["<|bos|>", "<|eos|>", "<copyright>", "</copyright>",
                       "<character>", "</character>", "<general>", "</general>",
                       "<|input_end|>", "<|rating:sfw|>", "<|rating:general|>",
                       "<|rating:sensitive|>", "<|rating:nsfw|>", "<|rating:questionable|>",
                       "<|rating:explicit|>", "<|aspect_ratio:ultra_wide|>",
                       "<|aspect_ratio:wide|>", "<|aspect_ratio:square|>",
                       "<|aspect_ratio:tall|>", "<|aspect_ratio:ultra_tall|>",
                       "<|length:very_short|>", "<|length:short|>", "<|length:medium|>",
                       "<|length:long|>", "<|length:very_long|>", "<|identity:none|>",
                       "<|identity:lax|>", "<|identity:strict|>"]
        
        special_token_ids = self.tokenizer.convert_tokens_to_ids(special_tags)
        people_tag_ids = self.tokenizer.convert_tokens_to_ids(people_tags)
        
        copyright_eos = self.tokenizer.convert_tokens_to_ids("</copyright>")
        character_eos = self.tokenizer.convert_tokens_to_ids("</character>")
        general_eos = self.tokenizer.convert_tokens_to_ids("</general>")
        
        # Split into sections
        sections = []
        section = []
        for token_id in token_ids:
            if token_id in [copyright_eos, character_eos, general_eos]:
                sections.append(section)
                section = []
            elif token_id not in special_token_ids:
                section.append(token_id)
        
        copyright_part = sections[0] if len(sections) > 0 else []
        character_part = sections[1] if len(sections) > 1 else []
        general_part = sections[2] if len(sections) > 2 else []
        
        # Split people tags from general
        people_part = [t for t in general_part if t in people_tag_ids]
        other_general_part = [t for t in general_part if t not in people_tag_ids]
        
        # Animagine order: people, character, copyright, other general
        rearranged = people_part + character_part + copyright_part + other_general_part
        
        tokens = [self.tokenizer.decode([tid], skip_special_tokens=True) 
                 for tid in rearranged]
        
        return ", ".join([token for token in tokens if token])

NODE_CLASS_MAPPINGS = {
    "DanbooruTagsTransformerMoeV2": DanbooruTagsTransformerMoeV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DanbooruTagsTransformerMoeV2": "Danbooru Tags Transformer"
}