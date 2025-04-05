from typing import List
from .token_utils import get_token_counter

class SemanticTextSplitter:
    def __init__(self, chunk_size: int = 1048, model_name: str = "gpt-4o-mini"):
        """
        Initialize the text splitter with chunk size and model name parameters.
        
        Args:
            chunk_size (int): Maximum number of tokens per chunk
            model_name (str): Model name for token counting
        """
        self.chunk_size = chunk_size
        self.token_counter = get_token_counter(model_name)

    def split(self, text: str) -> List[str]:
        """
        Split text into chunks based on both token count and semantic boundaries.
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # add 4 times of chunk_size string to the start position
            end = start + self.chunk_size * 4  # assume each token is 4 characters
            if end > text_len:
                end = text_len
                
            # get the current text fragment
            current_chunk = text[start:end]
            
            # if the token count of the current fragment exceeds the limit, need to find the split point
            while self.token_counter(current_chunk) > self.chunk_size and start < end:
                # find semantic boundary in the current range
                boundaries = ['\n\n', '\n', '。', '.', '！', '!', '？', '?', '；', ';']
                semantic_end = end
                
                for boundary in boundaries:
                    boundary_pos = current_chunk.rfind(boundary)
                    if boundary_pos != -1:
                        semantic_end = start + boundary_pos + len(boundary)
                        break
                
                # if found semantic boundary, use it; otherwise, force truncation by character
                if semantic_end < end:
                    end = semantic_end
                else:
                    # 没找到合适的语义边界，往回数token直到满足大小限制
                    end = start + int(len(current_chunk) // 1.2)
                
                current_chunk = text[start:end]
            
            # 添加处理好的文本块
            chunk = current_chunk.strip()
            if chunk:
                chunks.append(chunk)
            
            # 移动到下一个起始位置
            start = end

        return chunks
