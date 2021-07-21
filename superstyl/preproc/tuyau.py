from lxml import etree
import regex as re
import fasttext
import unidecode
import glob
import nltk.tokenize

def XML_to_text(path, correct_aut=None):
    """
    Get main text from xml file
    :param path: path to the file to transform
    :param correct_aut: optional data frame of metadata correction (authors)
    :return: a tuple with auts, and string (the text).
    """

    myxsl = etree.XML('''
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    exclude-result-prefixes="xs"
    version="1.0">
    
    <xsl:output method="text"/>
    
    <xsl:template match="/">
        <xsl:apply-templates
            select="song/text"/>
    </xsl:template>
    
</xsl:stylesheet>''')
    myxsl = etree.XSLT(myxsl)

    with open(path, 'r') as f:
        my_doc = etree.parse(f)

        auts = my_doc.findall("//author")
        auts = [a.text for a in auts]

        if not len(auts) == 1:
            print("Error: more or less than one author in" + path)

            if len(auts) == 0:
                auts = [None]

        if auts == [None]:
            aut = "unknown"

        else:
            aut = auts[0]
            if correct_aut is not None and aut in list(correct_aut.loc[:, "Original"]):
                print("correcting " + aut + " to " + correct_aut.loc[aut, "Actual"])
                aut = correct_aut.loc[aut, "Actual"]

        return aut, re.sub(r"\s+", " ", str(myxsl(my_doc)))


def TXT_to_text(path, correct_aut=None):
    """
    Get main text from xml file
    :param path: path to the file to transform
    :param correct_aut: optional data frame of metadata correction (authors)
    :return: a tuple with auts, and string (the text).
    """

    with open(path, 'r') as f:
        #txt = [line.rstrip() for line in f if line.rstrip() != '']
        txt = f.readlines()

    # get author from filename (string before first _)
    aut = path.split('/')[-1].split("_")[0]

    return aut, re.sub(r"\s+", " ", str(' '.join(txt)))


def identify_lang(string, model):
    """
    Get the language from a string
    :param string: a string, duh
    :param model, the fasttext model
    :return: the language
    """

    return model.predict(string)  # , k = 3)


def normalise(text, keep_punct = False):
    # Remove all but word chars, remove accents, and normalise space
    # and then normalise unicode

    if keep_punct:
        out = unidecode.unidecode(re.sub(r"\s+", " ", re.sub(r"[^\p{L}\p{P}]+", " ", text.strip())))

    else:
        out = unidecode.unidecode(re.sub(r"\s+", " ", re.sub(r"[\W0-9]+", " ", text.lower()).strip()))

    return out


def load_texts(paths, fasttext_model, format="txt", correct_aut=None, keep_punct=False):
    """
    Loads a collection of documents into a 'myTexts' object for further processing.
    TODO: a proper class
    :param paths: path to docs
    :param fasttext_model: model for language identification
    :param format: format of the source files (implemented values: txt [default], xml)
    :param correct_aut: optional data frame of metadata correction (authors)
    :param keep_punct: whether or not to keep punctuation and caps.
    :return: a myTexts object
    """

    myTexts = []
    # langCerts = []

    for path in paths:
        name = path.split('/')[-1]

        if format=='xml':
            aut, text = XML_to_text(path, correct_aut=correct_aut)

        else:
            aut, text = TXT_to_text(path)  # implement correct_aut

        lang, cert = identify_lang(text, fasttext_model)
        lang = lang[0].replace("__label__", "")

        # Normalise text once and for all
        text = normalise(text, keep_punct=keep_punct)

        myTexts.append({"name": name, "aut": aut, "text": text, "lang": lang})

        # if cert < 1:
        # langCerts.append((lang, name, cert))

        # directory = "train_txt/" + lang + "/" + aut + "/"

        # if not os.path.exists(directory):
        #    os.makedirs(directory)

        # with open(directory + name + ".txt", "w") as out:
        #    out.write(text)

    # with open("lang_certs.csv", 'w') as out:
    #    for line in langCerts:
    #        out.write("{}\t{}\t{}\t\n".format(line[0], line[1], float(line[2])))
    return myTexts


# Load and split in samples of length -n- a collection of files
def get_samples(path, size, step=None, units="verses", feature="tokens", format="tei", keep_punct=False):
    """
    Take samples of n words or verses from a document, and then parse it.
    ONLY IMPLEMENTED FOR NOW: XML/TEI, TXT and verses or words as units
    :param path : path to file
    :param size: sample size
    :param size: size of the step when sampling successively (determines overlap) default is the same
    as sample size (i.e. no overlap)
    :param units: the units to use, one of "words" or "verses"
    :param feature: type of tokens to extract (default is tokens, not lemmas or POS)
    :param format: type of document, one of full text, TEI or simple XML (ONLY TEI and TXT IMPLEMENTED)
    """

    if step is None:
        step = size

    if feature == "tokens" and units == "words" and format == "txt":
        my_doc = TXT_to_text(path)
        text = normalise(my_doc[1], keep_punct=keep_punct)
        units = nltk.tokenize.wordpunct_tokenize(text)

    if feature == "tokens" and units == "verses" and format == "tei":
        myxsl = etree.XML('''<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:tei="http://www.tei-c.org/ns/1.0" xmlns:txm="http://textometrie.org/1.0" 
    version="1.0">

    <xsl:output method="text"/>

    <xsl:template match="/">
        <xsl:apply-templates select="descendant::tei:l"/>
    </xsl:template>

    <xsl:template match="tei:l">
        <xsl:apply-templates select="descendant::tei:w[
            not(txm:ana[@type='#frpos'] = 'NOMpro')
            ]"/>
        <xsl:text>&#xA;</xsl:text>
    </xsl:template>

    <xsl:template match="tei:w">
        <xsl:text> </xsl:text>
        <xsl:apply-templates select="txm:form"/>
    </xsl:template>

</xsl:stylesheet>''')
        myxsl = etree.XSLT(myxsl)

        with open(path, 'r') as f:
            my_doc = etree.parse(f)

        units = str(myxsl(my_doc)).splitlines()

    # and now generating output
    samples = []
    current = 0
    while current + size <= len(units):
        samples.append({"start": current, "end": current + size, "text": list(units[current:(current + size)])})
        current = current + step

    return samples


def docs_to_samples(paths, size, step=None, units="verses", feature="tokens", format="tei", keep_punct=False):
    """
    Loads a collection of documents into a 'myTexts' object for further processing BUT with samples !
    :param paths: path to docs
    :param size: sample size
    :param size: size of the step when sampling successively (determines overlap) default is the same
    as sample size (i.e. no overlap)
    :param units: the units to use, one of "words" or "verses"
    :param feature: type of tokens to extract (default is tokens, not lemmas or POS)
    :param format: type of document, one of full text, TEI or simple XML (ONLY TEI and TXT IMPLEMENTED)
    :param keep_punct: whether or not to keep punctuation and caps.
    """
    myTexts = []
    for path in paths:
        aut = path.split('/')[-1].split('_')[0]
        lang = 'fr'  # POM POM POM
        samples = get_samples(path, size=size, step=step, units=units, feature=feature, format=format,
                              keep_punct=keep_punct)

        for sample in samples:
            name = path.split('/')[-1].split('_')[1] + '_' + str(sample["start"]) + "-" + str(sample["end"])
            text = normalise(' '.join(sample["text"]), keep_punct=keep_punct)
            myTexts.append({"name": name, "aut": aut, "text": text, "lang": lang})

    return myTexts
