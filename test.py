##############################################
# Author : Carter Anderson
# Email : carter.d.anderson.26@dartmouth.edu
# Date : 2024-05-27 (YYYY-MM-DD)
# Purpose : LING48 Final Project, 24S, Dartmouth College
# Description : This script trains BPE, WordPiece, and Morfessor tokenizers 
# on the Bribri corpus, and then tests them on a set of test word against
# the Morphemo segmented. The script outputs the results of the testing in 
# a tabular format.
##############################################

import MorphemeScorer
import math
from tqdm import tqdm
import contextlib
import random
import numpy as np
from morfessor import MorfessorIO, BaselineModel
from Morphemo import Morphemo
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

# Control the random seed for reproducibility
@contextlib.contextmanager
def rand_state(seed : int | None = None):
   state: tuple [Any, ...]= random.getstate()
   random.seed(seed)
   try:
      yield
   finally:
      random.setstate(state)


def train_models() -> None:
   """
   Train the BPE, WordPiece, and Morfessor tokenizers 
   on the Bribri corpus, and save the models to disk.
   """
   # Create a BPE tokenizer from HuggingFace
   bpe_tokenizer : Tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
   bpe_tokenizer.pre_tokenizer = Whitespace()
   bpe_trainer : BpeTrainer = BpeTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
   bpe_tokenizer.train(files = ["bribri-unmarked-corpus.txt"], trainer = bpe_trainer)
   bpe_tokenizer.save("bribri-bpe-tokenizer.json")

   # Create a BPE tokenizer from HuggingFace
   wp_tokenizer : Tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
   wp_tokenizer.pre_tokenizer = Whitespace()
   wp_trainer : WordPieceTrainer = WordPieceTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
   wp_tokenizer.train(files = ["bribri-unmarked-corpus.txt"], trainer = wp_trainer)
   wp_tokenizer.save("bribri-wordpiece-tokenizer.json")

   # Create a Morfessor tokenizer -> from https://aayushsanghavi.blogspot.com/2018/03/morphological-segmentation-of-words.html
   def log_func(x) -> int:
      return int(round(math.log(x + 1, 2)))
   
   io : MorfessorIO = MorfessorIO()
   train_data : list[tuple] = list(io.read_corpus_file("bribri-unmarked-corpus.txt"))
   mf_model : BaselineModel = BaselineModel()
   mf_model.load_data(train_data, count_modifier=log_func)
   mf_model.train_batch()
   io.write_binary_model_file("bribri-morfessor-model.bin", mf_model)

def main() -> None:
   # open the models
   bpe_tokenizer : Tokenizer = Tokenizer.from_file("models/bribri-bpe-tokenizer.json")
   wp_tokenizer : Tokenizer = Tokenizer.from_file("models/bribri-wordpiece-tokenizer.json")
   io : MorfessorIO = MorfessorIO()
   mf_model : BaselineModel = io.read_binary_model_file(file_name="models/bribri-morfessor-model.bin")

   # load the gold standard morpheme segmentations
   with open("bribri-conllu-goldstandard-corpus.txt", "r") as f:
      gold_standards_raw : list[str] = f.readlines()
      gold_standards : list[str] = [gold_standard.strip() for gold_standard in gold_standards_raw]

   # load the raw test words
   with open("bribri-conllu-20240314-corpus.txt", "r") as f:
      test_words_raw : list[str] = f.readlines()
      test_words : list[str] = [test_word.strip("\n") for test_word in test_words_raw]

   # shuffle the data and break into chunks
   q_and_a : list[tuple[str, str]] = list(zip(gold_standards, test_words))
   random.shuffle(q_and_a)

   n_chunks : int = 10
   chunks : list[list[tuple[str, str]]] = [q_and_a[i:i + len(q_and_a) // n_chunks] for i in range(0, len(q_and_a), len(q_and_a) // n_chunks)]

   # initialize the scores
   bpe_score : np.ndarray = np.array([0., 0., 0., 0.])
   wp_score : np.ndarray = np.array([0., 0., 0., 0.])
   mf_score : np.ndarray = np.array([0., 0., 0., 0.])
   morphemo_score : np.ndarray = np.array([0., 0., 0., 0.])
   num_tested : int = 0
   weight_sum : int = 0

   for i in tqdm(range(n_chunks)):
      # perform split for a test and training set, based on the n_chunks
      test : list[tuple[str, str]] = chunks[i]
      train : list[tuple[str, str]] = [pair for j, chunk in enumerate(chunks) if j != i for pair in chunk]

      test = [(gold_standard, test_word) for gold_standard, test_word in test if len(gold_standard.split("+")) >= 2]
      num_tested += len(test)

      # create training file for Morphemo and train
      with open("morphemo_training_morphs.txt", "w", encoding = "utf8") as f:
         for gold_standard, _ in train:
            f.write(gold_standard + "\n")
   
      morphemo : Morphemo = Morphemo(UNSEEN_BIAS=2, lookahead=2)
      morphemo.train("bribri-unmarked-corpus.txt", "morphemo_training_morphs.txt")

      # create the results for each model
      bpe_results : list[str] = [bpe_tokenizer.encode(test_word).tokens for _, test_word in test]
      wp_results_raw : list[str] = [wp_tokenizer.encode(test_word).tokens for _, test_word in test]
      mf_results : list[str] = [mf_model.viterbi_segment(test_word)[0] for _, test_word in test]
      morphemo_results_raw : list[str] = [morphemo.ortho_morpher(test_word) for _, test_word in test]
      gold_standards = [gold_standard.split("+") for gold_standard, _ in test]
      
      # process results into standard format of a list of morphemes for analysis
      wp_results = [[morph.replace("##", "") for morph in word] for word in wp_results_raw]
      morphemo_results = [word.split("+") for word in morphemo_results_raw]

      #WORDWISE
      wordwise : bool = False

      # score the results
      bpe_output = MorphemeScorer.MorphemeScorer.score_set(gold_standards, bpe_results, word_wise=wordwise)
      bpe_score_change = bpe_output[:4]

      # keep morpheme count for divisor
      divisor = bpe_output[4]
      weight_sum += divisor

      # add the scores to the running total
      bpe_score += np.asarray(bpe_score_change, dtype=np.float64) * divisor
      wp_score += np.asarray(MorphemeScorer.MorphemeScorer.score_set(gold_standards, wp_results, word_wise=wordwise)[:4], dtype=np.float64) * divisor
      mf_score += np.asarray(MorphemeScorer.MorphemeScorer.score_set(gold_standards, mf_results, word_wise=wordwise)[:4], dtype=np.float64) * divisor
      morphemo_score += np.asarray(MorphemeScorer.MorphemeScorer.score_set(gold_standards, morphemo_results, word_wise=wordwise)[:4], dtype=np.float64) * divisor

   # Output results to console in tabular format
   print("\n{0:10s}\t{1:4s}\t{2:4s}\t{3:4s}\t{4:4s}".format("Model", "ER", "P", "R", "F1"))
   print("{0:10s}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".format("BPE", *(bpe_score / weight_sum)))
   print("{0:10s}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".format("WordPiece", *(wp_score / weight_sum)))
   print("{0:10s}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".format("Morfessor", *(mf_score / weight_sum)))
   print("{0:10s}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".format("Morphemo", *(morphemo_score / weight_sum)))
   print("Number of words tested: ", num_tested)
   print("Number of morphemes tested: ", weight_sum)
   print(f"Wordwise Operation: {wordwise}")

if __name__ == "__main__":
   # perform the training and testing with provided seed, 4 arbitrarily selected
   with rand_state(5):
      main()
