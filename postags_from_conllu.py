from conllu import parse, SentenceList, TokenList, Token
from conllu.serializer import serialize

def conllu_pos_extraction(conllu_file : str, output_file : str) -> None:
   '''
   Extracts the words from the conllu file and writes them to the output file
   @param conllu_file: the conllu file to extract the words from
   '''
   with open(conllu_file, "r", encoding="utf8") as f:
      data : str = f.read()
   
   sentences : SentenceList = parse(data)

   with open(output_file, "w", encoding="utf8") as f:
      sentence : TokenList
      for sentence in sentences:
         token : Token
         tokens : list[str] = [token.get("upos") for token in sentence]
         f.write(" ".join(tokens) + "\n")

if __name__ == "__main__":
   conllu_file : str = r"bribri-conllu-20240314.txt"
   pos_file : str = r"bribri-conllu-20240314-pos.txt"

   conllu_pos_extraction(conllu_file, pos_file)