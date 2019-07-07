import sys
import os
import jagen_will.preproc.tuyau as tuy
import jagen_will.preproc.features_extract as fex
import fasttext
import pandas
import json
# from importlib import reload
# tuy = reload(tuy)
#import json , json.dump, file et object, json.load sur des files, dumps et loads sur des str
#import json

# TODO: eliminate features that occur only n times ?
# Do the Moisl Selection ?
# Z-scores, etc. ?
# Vector-length normalisation ?


if __name__ == '__main__':

    model = fasttext.load_model("jagen_will/preproc/models/lid.176.bin")

    print(".......loading texts.......")

    myTexts = tuy.load_texts(sys.argv[1:], model)

    print(".......getting features.......")

    my_feats = fex.get_feature_list(myTexts, feats="chars", n=3, relFreqs=True)

    # and now, cut at around rank k
    k = 5000
    val = my_feats[k][1]
    my_feats = [m for m in my_feats if m[1] >= val]

    with open("feature_list.json", "w") as out:
        out.write(json.dumps(my_feats))


    print(".......getting counts.......")

    # myTexts = fex.get_counts(myTexts, feats="chars", n=3, relFreqs=True)

    unique_words = set([k for t in myTexts for k in t["wordCounts"].keys()])
    unique_texts = [text["name"] for text in myTexts]

    # print(".......feeding data frame.......")
    # feats = pandas.DataFrame(columns=list(unique_words), index=unique_texts)
    #
    # for text in myTexts:
    #
    #     local_freqs = []
    #
    #     for word in unique_words:
    #         if not word in text["wordCounts"].keys():
    #             local_freqs.append(0)
    #
    #         else:
    #             local_freqs.append(text["wordCounts"][word])
    #
    #     feats.loc[text["name"]] = local_freqs
    #
    # # And here is the place to implement selection and normalisation
    #
    # print(".......saving results.......")
    # # frequence based selection
    # # WOW, pandas is a great tool, almost as good as using R
    # # But confusing as well: boolean selection works on rows by default
    # # were elsewhere it works on columns
    # # take only rows where the number of values above 0 is superior to two
    # # (i.e. appears in at least two texts)
    # feats = feats.loc[:, feats[feats > 0].count() > 2]
    #
    # metadata = pandas.DataFrame(columns=['author', 'lang'], index=unique_texts, data =
    #                             [[t["aut"], t["lang"]] for t in myTexts])
    #
    # pandas.concat([metadata, feats], axis=1).to_csv("feats_tests.csv")
    #
    #
