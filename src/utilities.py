import human_id

def generate_human_id(num_words=3):
    
    # The module requires a minimum of three words
    if num_words<3:
        num_words = 3
    
    return human_id.generate_id(word_count=num_words)