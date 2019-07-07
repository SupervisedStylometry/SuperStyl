import sys
import os
import jagen_will.preproc.tuyau as tuy
import jagen_will.preproc.features_extract as fex
import fasttext
import pandas
# from importlib import reload
# tuy = reload(tuy)
#import json , json.dump, file et object, json.load sur des files, dumps et loads sur des str
#import json


# TODO: eliminate features that occur only n times ?
# Do the Moisl Selection ?
# Z-scores, etc. ?
# Vector-length normalisation ?
#

if __name__ == '__main__':

    # path = "meertens-song-collection-DH2019/train/34153.xml"
    myTexts = []

    #langCerts = []

    model = fasttext.load_model("jagen_will/preproc/models/lid.176.bin")

    for path in sys.argv[1:]:
        with open(path, 'r') as f:
            name = path.split('/')[-1]
            aut, text = tuy.XML_to_text(path)
            lang, cert = tuy.identify_lang(text, model)
            lang = lang[0].replace("__label__", "")

            # Normalise text once and for all
            text = fex.normalise(text)

            myTexts.append({"name": name, "aut": aut, "text": text, "lang": lang,
                            "wordCounts": fex.count_words(text, feats="chars", n = 3, relFreqs=True)})

            #if cert < 1:
            #langCerts.append((lang, name, cert))

            #directory = "train_txt/" + lang + "/" + aut + "/"

            #if not os.path.exists(directory):
            #    os.makedirs(directory)

            #with open(directory + name + ".txt", "w") as out:
            #    out.write(text)

    #with open("lang_certs.csv", 'w') as out:
    #    for line in langCerts:
    #        out.write("{}\t{}\t{}\t\n".format(line[0], line[1], float(line[2])))

    unique_words = set([k for t in myTexts for k in t["wordCounts"].keys()])
    unique_texts = [text["name"] for text in myTexts]

    feats = pandas.DataFrame(columns=['author', 'lang'] + list(unique_words), index=unique_texts)

    for text in myTexts:

        local_freqs = []

        for word in unique_words:
            if not word in text["wordCounts"].keys():
                local_freqs.append(0)

            else:
                local_freqs.append(text["wordCounts"][word])

        feats.loc[text["name"]] = [text["aut"]] + [text["lang"]] + local_freqs

    feats.to_csv("feats.csv")
