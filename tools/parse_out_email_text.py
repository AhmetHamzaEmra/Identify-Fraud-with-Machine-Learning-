#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):

    f.seek(0)
    all_text = f.read()

    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
            
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        words = text_string
            
        stemmer = SnowballStemmer("english")
            
        words = words.split()
        stemmed_words=[]
        for to_stem in words:
            stemmed_words.append(stemmer.stem(to_stem))
        words=' '.join(stemmed_words)

    return words

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text

if __name__ == '__main__':
    main()
