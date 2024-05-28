class MorphemeScorer: 
   """Class to score morpheme segmentation results against gold standards."""

   @staticmethod
   def score_set(gold_standards : list[list[str]], results : list[list[str]]) -> tuple[float, float, float, float]:
      """
      Score a set of morpheme segmentation results against gold standards.

      Params:
      @param gold_standards: list of gold standard morpheme segmentations
      @param results: list of morpheme segmentation results

      Returns:
      @return: tuple of average error, precision, recall, and f1 score
      
      """
      # initialize variables
      total_error : int = 0
      total_precision : float = 0
      total_recall : float = 0
      total_f1 : float = 0

      # iterate through each pair of gold standard and result
      for gold_standard, result in zip(gold_standards, results):
         error, precision, recall, f1 = MorphemeScorer.score(gold_standard, result)
         total_error += error
         total_precision += precision
         total_recall += recall
         total_f1 += f1

      # calculate average
      avg_error : float = total_error / len(gold_standards)
      avg_precision : float = total_precision / len(gold_standards)
      avg_recall : float = total_recall / len(gold_standards)
      avg_f1 : float = total_f1 / len(gold_standards)

      return avg_error, avg_precision, avg_recall, avg_f1

   @staticmethod
   def score(gold_morphemes : list[str], result_morphemes : list[str]) -> tuple[bool, float, float, float]:
      """
      Score a single morpheme segmentation result against a gold standard.
      
      Params:
      @param gold_morphemes: list of gold standard morphemes
      @param result_morphemes: list of morphemes from the result
      
      Returns:
      @return: tuple of error, precision, recall, and f1 score
      """
      
      # boolean of error, 1 if error, 0 if no error
      error : bool = not(gold_morphemes == result_morphemes)

      # calculate f1
      tpos : list[str] = [morpheme for morpheme in result_morphemes if morpheme in gold_morphemes]

      precision : float = len(tpos) / len(result_morphemes)
      recall : float = len(tpos) / len(gold_morphemes)

      f1 : float = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

      return error, precision, recall, f1
   
if __name__ == "__main__":
   # test MorphemeScorer
   gold_standards = ["a-b-c", "ab-c-d", "a-b-c-d-e"]
   results = ["a-b-c", "a-b-cd", "a-b-cd-e"]
   avg_error, avg_precision, avg_recall, avg_f1 = MorphemeScorer.score_set(gold_standards, results, "-")
   print(avg_error, avg_precision, avg_recall, avg_f1)