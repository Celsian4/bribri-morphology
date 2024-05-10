import numpy as np
import Morphemo
from math import log10

class Morphemo:
   start_token : str
   end_token : str
   text_prob : np.ndarray
   text_index : dict[str, int]
   morph_prob : np.ndarray
   morph_index : dict[str, int]
   morph_freq_data : np.ndarray

   def __init__(self, *text_files : str, morph_file : str, start_token : str = "<s>", end_token : str = "</s>"):
      # load text and morpheme probabilities
      self.text_prob, self.text_index = self.probability_loader(*text_files)
      self.morph_prob, self.morph_index = self.probability_loader(morph_file)

      # set start and end tokens
      self.start_token = start_token
      self.end_token = end_token

      # load morpheme frequency data
      self.morph_freq_data = self.morphemes_percentage(morph_file)

   def word_cutter(self, word : str, start_token : str, end_token : str) -> list[str]:
      return [start_token] + [*word.lower()] + [end_token]

   def probability_loader(self, *text_files : str, filter_token : str = None, lookahead : int = 1) -> np.ndarray:
      # read in each text file
      text : list[str] = []
      words : list[list[str]] = []
      for text_file in text_files:
         if not text_file.endswith(".txt"):
            raise ValueError("Only text files are accepted.")
         else:
            with open(text_file, 'r', encoding="utf8") as f:
               text += f.readlines()
      
         # Process text into a list of words (as lists of characters)
         text = [line.strip() for line in text]
         
         for line in text:
            for word in line.split():
               words += [self.word_cutter(word, "<s>", "</s>")]

      # enumerate all characters and assign an index to them
      set_chars = set([char for word in words for char in word])
      char_to_index = {char : i for i, char in enumerate(sorted(set_chars))}

      # create np array of probabilities (row = previous char, column = next char)
      probabilities = np.zeros((len(set_chars), len(set_chars)))
      
      for word in words:
         for i in range(1, len(word)):
            probabilities[char_to_index[word[i-1]], char_to_index[word[i]]] += 1

      # normalize the probabilities
      np.log10(np.divide(probabilities,np.sum(probabilities)), out=probabilities, where=probabilities!=0)
      probabilities[probabilities==0] = 2 * np.min(probabilities[probabilities!=0])

      return probabilities, char_to_index

   def morphemes_percentage(self, morpheme_file : str) -> np.ndarray:
      with open(morpheme_file, 'r', encoding="utf8") as f:
         text : list[str] = f.readlines()
      
      text = [line.lower().strip() for line in text]

      morpheme_freq : list[tuple[int, int]] = []
      max_morphemes = 0
      max_wordlength = 0

      for line in text:
         for word in line.split():
            word_chars : list[str] = self.word_cutter(word, "<s>", "</s>")
            morph_count : int = word_chars.count("+")
            word_length : int = len(word_chars)

            # update max values for data array dimensions
            if word_length > max_wordlength:
               max_wordlength = word_length
            if morph_count > max_morphemes:
               max_morphemes = morph_count

            morpheme_freq.append((word_length, morph_count))

      # create data array
      morph_freq_data = np.zeros((max_wordlength + 1, max_morphemes + 1))
      for word_length, morph_count in morpheme_freq:
         morph_freq_data[word_length, morph_count] += 1

      for row in morph_freq_data:
         if np.sum(row) != 0:
            np.log10(row / np.sum(row), out=row, where=row!=0)

      morph_freq_data[morph_freq_data==0] = 2 * np.min(morph_freq_data[morph_freq_data!=0])

      return morph_freq_data
   
   def morpheme_guess(self, word : str) -> float:
      # cut word into characters
      word = self.word_cutter(word, self.start_token, self.end_token)
      
      base_prob : float = 0
      [base_prob := base_prob + self.text_prob[self.text_index[word[i-1]], self.text_index[word[i]]] for i in range(1, len(word))]

      return base_prob

if __name__ == '__main__':
   morphemo : Morphemo = Morphemo("bribri-unmarked-text.txt", morph_file="bribri-conllu-20240314-tokenized-handcorrect.txt")
   
   
   print(morphemo.morpheme_guess("Sibòq"))
   print(morphemo.morpheme_guess("yéq"))

   

      

      