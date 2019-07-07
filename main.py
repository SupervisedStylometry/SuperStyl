import sys
import os
import jagen_will.preproc.tuyau as tuy
import jagen_will.preproc.features_extract as fex
import fasttext
# from importlib import reload
# tuy = reload(tuy)
#import json , json.dump, file et object, json.load sur des files, dumps et loads sur des str
#import json
import nltk.tokenize

# TODO: eliminate features that occur only n times ?
# Do the Moisl Selection ?
# Z-scores, etc. ?
# Vector-length normalisation ?
#

if __name__ == '__main__':

    # path = "meertens-song-collection-DH2019/train/34153.xml"
    myTexts = []

    #langCerts = []

    model = fasttext.load_model("preproc/models/lid.176.bin")

    for path in sys.argv[1:]:
        with open(path, 'r') as f:
            name = path.split('/')[-1]
            aut, text = tuy.XML_to_text(path)
            lang, cert = tuy.identify_lang(text, model)
            lang = lang[0].replace("__label__", "")

            myTexts.append({"name": name, "aut": aut, "text": text, "lang": lang,
                            "wordCounts": fex.count_words(text.lower(), relFreqs=True)})

            #if cert < 1:
            #langCerts.append((lang, name, cert))

            #directory = "txt/" + lang + "/" + aut + "/"

            #if not os.path.exists(directory):
            #    os.makedirs(directory)

            #with open(directory + name + ".txt", "w") as out:
            #    out.write(text)

    # with open("lang_certs.csv", 'w') as out:
    #    for line in langCerts:
    #        out.write("{}\t{}\t{}\t\n".format(line[0], line[1], float(line[2])))

    unique_words = set([k for t in myTexts for k in t["wordCounts"].keys()])

    with open("feats.csv", "w") as out:
        # First line
        out.write("\t" + "author" + "\t" + "lang")
        for word in unique_words:
            out.write("\t"+word)

        out.write("\n")

        for text in myTexts:
            out.write(text["name"] + "\t" + text["aut"] + "\t" + text["lang"])
            for word in unique_words:
                if not word in text["wordCounts"].keys():
                    out.write("\t"+"0")

                else:
                    out.write("\t"+str(text["wordCounts"][word]))

            out.write("\n")

