# Pre-process a Chinese text file
# For the time being, assume we are using the 2005 SIGHAN training data, which is already one clause per line
import sys

remove = [" ", "　", "，", "。", "："]
punctuation = ["（", "）", "」", "「"]

def process_line(line):
    chars = list(line)
    return " ".join([c for c in chars if c not in remove])

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print("Usage: preprocess.py input_file output_file --remove-punctuation")
    infile_path = sys.argv[1]
    outfile_path = sys.argv[2]
    
    if len(sys.argv) > 3 and sys.argv[3] == "--remove-punctuation":
        remove += punctuation
        
    with open(infile_path, "r") as infile:
        with open(outfile_path, "w") as outfile:
            for line in infile:
                outfile.write(process_line(line))