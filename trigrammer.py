def word_cutter(word : str, start_token : str, end_token : str) -> list[str]:

   return [start_token] + word[::1] + [end_token]


print(word_cutter("hello", "<", ">"))