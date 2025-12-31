import torch

class TextFM:
    """
    Text Feature Manager:
    Converts symbolic strings (Passwords, PINs, Seeds) into fixed-length 
    integer sequences for the TextEncoder.
    
    Uses Character-Level tokenization (best for random strings/passwords).
    """
    def __init__(self, max_len=16, vocab_size=1000):
        self.max_len = max_len
        # Reserve 0 for Padding, 1 for Unknown
        self.vocab_size = vocab_size 
        
        # Simple ASCII mapping: 'a'->97, etc.
        # We shift them to ensure they fit in range and leave 0/1 empty.
        self.offset = 2 

    def process(self, text: str):
        """
        Input: "password123"
        Output: Tensor (1, max_len) e.g., [[114, 99, 117, ... 0, 0]]
        """
        # 1. Standardization
        # For security tokens, we might keep case sensitivity. 
        # For semantics, we might lower(). Let's keep case for passwords.
        clean_text = str(text).strip()
        
        # 2. Tokenize (Char -> Int)
        indices = []
        for char in clean_text:
            # Ord gives ASCII value. We clamp to vocab_size to prevent crashes.
            idx = ord(char) + self.offset
            if idx >= self.vocab_size:
                idx = 1 # Unknown token
            indices.append(idx)
            
        # 3. Pad / Truncate
        if len(indices) < self.max_len:
            # Pad with 0
            indices += [0] * (self.max_len - len(indices))
        else:
            # Truncate
            indices = indices[:self.max_len]
            
        # 4. To Tensor
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

# Default instance
processor = TextFM()