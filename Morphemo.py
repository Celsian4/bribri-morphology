import numpy as np
import Morphemo
from math import log10

class Morphemo:
   """
   Morphemo class is a class that predicts the likelihood of morpheme boundaries in a word given the environment.
   """
   # tokens
   start_token : str = "<s>"
   end_token : str = "</s>"
   morph_token : str = "+"
   # text probabilities
   text_forward_prob : np.ndarray
   text_backward_prob : np.ndarray
   text_forward_index : dict[str, int]
   text_backward_index : dict[str, int]
   # morph probabilities
   morph_forward_prob : np.ndarray
   morph_backward_prob : np.ndarray
   morph_forward_index : dict[str, int]
   morph_backward_index : dict[str, int]
   # morph frequency data
   morph_freq_data : np.ndarray
   # lookahead
   lookahead : int = 2
   # unseen bias
   UNSEEN_BIAS : float = 2

   def __init__(self, *text_files : str, morph_file : str, start_token : str = "<s>", end_token : str = "</s>", morph_token : str = "+", lookahead : int = 2, UNSEEN_BIAS : float = 2):
      
      self.UNSEEN_BIAS = UNSEEN_BIAS
      self.lookahead = lookahead
      # load text probabilities (same indices because it is a square matrix)
      self.text_forward_prob, self.text_forward_index, temp = self.probability_loader(*text_files, lookahead=lookahead)
      self.text_backward_prob, temp, self.text_backward_index = self.probability_loader(*text_files, reversed=True, lookahead=lookahead)

      # load morpheme probabilities
      self.morph_forward_prob, self.morph_forward_index, temp = self.probability_loader(morph_file, filter_token="+", lookahead=1)
      self.morph_backward_prob, temp, self.morph_backward_index = self.probability_loader(morph_file, filter_token="+", reversed=True, lookahead=lookahead)

      # set start, end, morphology tokens
      self.start_token = start_token
      self.end_token = end_token
      self.morph_token = morph_token

      # load morpheme frequency data
      self.morph_freq_data = self.morphemes_percentage(morph_file)

   def probability_loader(self, *text_files : str, filter_token : str = None, lookahead : int = 1, reversed : bool = False) -> np.ndarray:
      """
      Loads text files and calculates the probability of each x-gram following another character.

      Parameters:
      @param text_files: list of text files to be loaded
      @param pre_token: token to be used as a filter for the previous character
      @param post_token: token to be used as a filter for the x-gram
      @param lookahead: number of characters to look ahead (x-gram size)

      Returns:
      @return probabilities: np array of probabilities
      """
      # accounts for non-inclusive range

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
               words += [self.word_cutter(word, self.start_token, self.end_token)]

      # TODO: add lookahead functionality to grab x-grams
      # TODO: change

      # enumerate all characters and assign an index to them
      set_chars : set[str] = set()
      set_grams : set[tuple[str]] = set()
      chars_grams : dict[str, dict[tuple[str], int]] = {}
      
      # load the characters and grams into dictionaries (one-pass)
      for word in words:
         for i in range(0, len(word)):
            set_chars.add(word[i])
            if i < len(word)-lookahead:
               gram = tuple(word[i+1:i+lookahead+1])
               set_grams.add(gram)

               # new gram
               if word[i] not in chars_grams:
                  chars_grams[word[i]] = {}
               if gram not in chars_grams[word[i]]:
                  chars_grams[word[i]][gram] = 0
               
               # increment gram count
               chars_grams[word[i]][gram] += 1

      char_to_index : dict[int, str]= {char : i for i, char in enumerate(sorted(set_chars))}
      gram_to_index : dict[int, tuple[str]] = {gram : i for i, gram in enumerate(sorted(set_grams))}

      # create np array of probabilities (row = previous char, column = next char)
      probabilities : np.ndarray = np.zeros((len(set_chars), len(set_grams)))

      # update the probabilities array
      for char in chars_grams:
         for gram in chars_grams[char]:
            count = chars_grams[char][gram]
            probabilities[char_to_index[char], gram_to_index[gram]] = count

      # transform the array if looking backwards
      if reversed:
         probabilities = np.transpose(probabilities)
      
      # normalize the probabilities
      for row in probabilities:
         np.log10(np.divide(row,np.sum(row)), out=row, where=row!=0)
      probabilities[probabilities==0] = self.UNSEEN_BIAS * np.min(probabilities[probabilities!=0])

      # filter the probabilities if a token is specified
      if filter_token is not None:
         probabilities = probabilities.take([char_to_index[filter_token]], axis=1, mode='clip')

      return probabilities, char_to_index, gram_to_index

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

      # reduce the effect of length bias
      np.sqrt(morph_freq_data, out=morph_freq_data, where=morph_freq_data!=0)

      for row in morph_freq_data:
         if np.sum(row) != 0:
            np.log10(row / np.sum(row), out=row, where=row!=0)

      morph_freq_data[morph_freq_data==0] = self.UNSEEN_BIAS * np.min(morph_freq_data[morph_freq_data!=0])

      print(morph_freq_data)

      return morph_freq_data

   def predict_word(self, raw_word : str) -> str:
      word : list[str] = self.word_cutter(raw_word, self.start_token, self.end_token)

      # calculate base probabilities
      base_prob : list[float] = [0] * (len(word) - self.lookahead)
      for i in range(0, len(word) - self.lookahead):
         base_prob[i] = self.point_prob(word[i], tuple(word[i+1:i+self.lookahead+1]))
      
      # calculate morpheme probabilities
      morph_prob : list[tuple[int, float]] = [0] * (len(word) - self.lookahead)
      for i in range(0, len(word) - self.lookahead):
         morph_prob[i] = (self.morph_prob(word[i], tuple(word[i+1:i+self.lookahead+1])), i)

      morph_prob.sort(key=lambda x: x[0], reverse=True)

      # determine likelihood of morphemes that are present
      n_morphemes : int = 0
      morph_indexes : list[int] = []
      for morph, i in morph_prob:
         if base_prob[i] + self.get_morph_count_freq(len(word), n_morphemes) < morph + self.get_morph_count_freq(len(word), n_morphemes+1):
            n_morphemes += 1
            morph_indexes.append(i+1)

      morphed : list[str] = self.word_morpher(word, self.morph_token, sorted(morph_indexes))

      return "".join(morphed[1:len(morphed)-1])

   def point_prob(self, before : str, after : tuple[str]) -> float:
      forward : float
      backward : float

      # calculate forward looking probability
      if before in self.text_forward_index and after in self.text_backward_index:
         forward = self.text_forward_prob[self.text_forward_index[before], self.text_backward_index[after]]
      else:
         forward = self.text_forward_prob.min()
      # calculate backward looking probability
      if after in self.text_backward_index and before in self.text_forward_index:
         backward = self.text_backward_prob[self.text_backward_index[after], self.text_forward_index[before]]
      else:
         backward = self.text_backward_prob.min()

      return forward + backward
   
   def morph_prob(self, before : str, after : str) -> float:
      forward : float
      backward : float
      # calculate forward looking probability
      if before in self.morph_forward_index:
         forward = self.morph_forward_prob[self.morph_forward_index[before]][0]
      else:
         forward = self.morph_forward_prob.min()

      # calculate backward looking probability
      if after in self.morph_backward_index:
         backward = self.morph_backward_prob[self.morph_backward_index[after]][0]
      else:
         backward = self.morph_backward_prob.min()
      
      return forward + backward
   
   def get_morph_count_freq(self, word_length : int, morph_count : int) -> float:
      '''
      Returns the frequency of a morpheme count given a word length.
      
      Parameters:
      @param word_length: length of the word
      @param morph_count: number of morphemes in the word
      
      Returns:
      @return frequency: frequency of the morpheme count given the word length (min val if not found)
      '''
      if word_length >= self.morph_freq_data.shape[0] or morph_count >= self.morph_freq_data.shape[1]:
         return self.UNSEEN_BIAS * np.min(self.morph_freq_data)
      return self.morph_freq_data[word_length, morph_count]

   @staticmethod
   def word_cutter(word : str, start_token : str, end_token : str) -> list[str]:
      """
      Cuts a word into a list of characters with start and end tokens.
      
      Parameters:
      @param word: string representing a word
      @param start_token: token to be inserted at the beginning of the word
      @param end_token: token to be inserted at the end of the word
      
      Returns:
      @return word: list of characters with start and end tokens
      """
      return [start_token] + [*word.lower()] + [end_token]
   
   @staticmethod
   def word_morpher(word : list[str] | str, morph_token : str, morph_locations : list[int]) -> list[str]:
      """
      Inserts morpheme tokens into a word at specified locations.

      Parameters:
      @param word: list of characters representing a word
      @param morph_token: token to be inserted as morpheme boundary
      @param morph_locations: list of indices where the morpheme token should be inserted

      Returns:
      @return word: list of characters with morpheme tokens inserted
      """
      insert_count : int = 0
      
      for i in morph_locations:
         word.insert(i + insert_count, morph_token)
         insert_count += 1

      return word
         

if __name__ == '__main__':
   morphemo : Morphemo = Morphemo("bribri-unmarked-text.txt", morph_file="bribri-conllu-20240314-tokenized-handcorrect.txt", UNSEEN_BIAS=2)
   print(morphemo.predict_word("shkèxnã"))
   print(morphemo.predict_word("ujtóqkwã"))
   print(morphemo.predict_word("bua'ë"))
   print(morphemo.predict_word("kie"))
   print(morphemo.predict_word("daléqnẽ"))
   print(morphemo.predict_word("akéqkëpa"))
   print(morphemo.predict_word("stsö"))
   

      

      