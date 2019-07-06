import sys
import preproc.tuyau as tuy
import fasttext
#from importlib import reload
#tuy = reload(tuy)


if __name__ = '__main__':

    #path = "meertens-song-collection-DH2019/train/34153.xml"
    myTexts = []

    lowCerts = []

    model = fasttext.load_model("preproc/models/lid.176.bin")

    for path in sys.argv[1:]:
        with open(path, 'r') as f:
            name = path.split('/')[-1]
            aut, text = tuy.XML_to_text(path)
            lang, cert = tuy.identify_lang(text, model)

            myTexts.append({"name": name, "aut": aut, "text": text, "lang:" lang})

            if cert < 0.9:
                lowCerts.append((lang, name, cert))

            with open("txt/" + lang + "/" + aut + "/" + name + ".txt", "w") as out:
                out.write(text)






