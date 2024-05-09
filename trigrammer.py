def word_pad(word: str, *, start_pad : str, end_pad : str) -> str:
   '''
   Pads the word with the specified pad character
   @param word: the word to pad
   @param start_pad: the starting pad character
   @param end_pad: the ending pad character
   @return the padded word
   '''
   return start_pad + word + end_pad

print(word_pad("hello", start_pad = "<", end_pad = ">")) # <hello>