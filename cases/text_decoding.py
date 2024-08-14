from transformers import LlamaTokenizer

# Assuming you've loaded a Vicuna model
tokenizer = LlamaTokenizer.from_pretrained("/mnt/bn/themis/data/LLM/vicuna-7b-v1.5")

# Your input ids
input_ids = [
    1939, 29892,   278,  2305,   297,   278,
         2175,  1967,   526,   451,   278, 297,   278,  1492,  1967, 29889,     2,
]
# 1939, 29892,   278,  2305,   297,   278,  2175,
#          1967,   526,   451,   278, 
# 1021,   408,   278,  2305, 
#   297,   278, 1492,  1967, 29889,     2, 

# Decoding the tokens to text
decoded_text = tokenizer.decode(input_ids)
print(decoded_text)

# \nAre the people in the left image the same as the people in the right image? ASSISTANT: No, the people in the left image are not the same as the people in the right image.

# How do the two persons in the two images differ? ASSISTANT: One person in the second image is more hunched over at the table than the person in the first image. There is also a person not walking by the tables in the second image.
# One person in the second image is more hunched over at the table than the image. There is also a person not walking by the tables in the second image.</s>
sum([x.shape[0] for x in cur_new_input_embeds])

sum([x.shape[0] for x in cur_new_labels])