from lxml import etree
import regex as re
import unidecode
import nltk.tokenize
import random
import langdetect

def XML_to_text(path):
    """
    Get main text from xml file
    :param path: path to the file to transform
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

        return aut, re.sub(r"\s+", " ", str(myxsl(my_doc)))


def TXT_to_text(path):
    """
    Get main text from xml file
    :param path: path to the file to transform
    :return: a tuple with auts, and string (the text).
    """

    with open(path, 'r') as f:
        #txt = [line.rstrip() for line in f if line.rstrip() != '']
        txt = f.readlines()

    # get author from filename (string before first _)
    aut = path.split('/')[-1].split("_")[0]

    return aut, re.sub(r"\s+", " ", str(' '.join(txt)))


def detect_lang(string):
    """
    Get the language from a string
    :param string: a string, duh
    :return: the language
    """

    return langdetect.detect(string)  # , k = 3)


def normalise(text, keep_punct=False, keep_sym=False):
    """
    Function to normalise an input string. By defaults, it removes all but word chars, remove accents,
    and normalise space, and then normalise unicode.
    :param keep_punct: if true, in addition, also keeps Punctuation and case distinction
    :param keep_sym: if true, same as keep_punct, but keeps also N?umbers, Symbols,  Marks, such as combining diacritics,
    as well as Private use characters, and no Unidecode is applied
    """
    # Remove all but word chars, remove accents, and normalise space
    # and then normalise unicode

    if keep_sym:
        out = re.sub(r"[^\p{L}\p{P}\p{N}\p{S}\p{M}\p{Co}]+", " ", text)

    else:
        if keep_punct:
            out = re.sub(r"[^\p{L}\p{P}]+", " ", text)

        else:
            #out = re.sub(r"[\W0-9]+", " ", text.lower())
            out = re.sub(r"[^\p{L}]+", " ", text.lower())

        out = unidecode.unidecode(out)

    out = re.sub(r"\s+", " ", out).strip()

    return out

def max_sampling(myTexts, max_samples=10):
    """
    Select a random number of samples, equal to max_samples, for authors or classes that have more than max_samples
    :param myTexts: the input myTexts object
    :param max_samples: the maximum number of samples for any class
    :return: a myTexts object, with the resulting selection of samples
    """
    autsCounts = dict()
    for text in myTexts:
        if text['aut'] not in autsCounts.keys():
            autsCounts[text['aut']] = 1

        else:
            autsCounts[text['aut']] += 1

    for autCount in autsCounts.items():
        if autCount[1] > max_samples:
            # get random selection
            toBeSelected = [text for text in myTexts if text['aut'] == autCount[0]]
            toBeSelected = random.sample(toBeSelected, k=max_samples)
            # Great, now remove all texts from this author from our samples
            myTexts = [text for text in myTexts if text['aut'] != autCount[0]]
            # and now concat
            myTexts = myTexts + toBeSelected

    return myTexts


def load_texts(paths, identify_lang=False, format="txt", keep_punct=False, keep_sym=False, max_samples=None):
    """
    Loads a collection of documents into a 'myTexts' object for further processing.
    TODO: a proper class
    :param paths: path to docs
    :param identify_lang: whether or not try to identify lang (default: False)
    :param format: format of the source files (implemented values: txt [default], xml)
    :param keep_punct: whether or not to keep punctuation and caps.
    :param keep_sym: whether or not to keep punctuation, caps, letter variants and numbers (no unidecode).
    :param max_samples: the maximum number of samples for any class
    :return: a myTexts object
    """

    myTexts = []

    for path in paths:
        name = path.split('/')[-1]

        if format=='xml':
            aut, text = XML_to_text(path)

        else:
            aut, text = TXT_to_text(path)

        if identify_lang:
            lang = detect_lang(text)
        else:
            lang = "NA"

        # Normalise text once and for all
        text = normalise(text, keep_punct=keep_punct, keep_sym=keep_sym)

        myTexts.append({"name": name, "aut": aut, "text": text, "lang": lang})

    if max_samples is not None:
        myTexts = max_sampling(myTexts, max_samples=max_samples)

    return myTexts


# Load and split in samples of length -n- a collection of files
def get_samples(path, size, step=None, samples_random=False, max_samples=10,
                units="words", format="txt", keep_punct=False, keep_sym=False):
    """
    Take samples of n words or verses from a document, and then parse it.
    ONLY IMPLEMENTED FOR NOW: XML/TEI, TXT and verses or words as units
    :param path : path to file
    :param size: sample size
    :param size: size of the step when sampling successively (determines overlap) default is the same
    as sample size (i.e. no overlap)
    :param samples_random: Should random sampling with replacement be performed instead of continuous sampling (default: false)
    :param max_samples: maximum number of samples per author/clas
    :param units: the units to use, one of "words" or "verses"
    :param format: type of document, one of full text, TEI or simple XML (ONLY TEI and TXT IMPLEMENTED)
    """

    if samples_random and step is not None:
        raise ValueError("random sampling is not compatible with continuous sampling (remove either the step or the samples_random argument")

    if samples_random and not max_samples:
        raise ValueError("random sampling needs a fixed number of samples (use the max_samples argument)")

    if step is None:
        step = size

    if units == "words" and format == "txt":
        my_doc = TXT_to_text(path)
        text = normalise(my_doc[1], keep_punct=keep_punct, keep_sym=keep_sym)
        units = nltk.tokenize.wordpunct_tokenize(text)

    if units == "verses" and format == "tei":
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

    if samples_random:
        for k in range(max_samples):
            samples.append({"start": str(k)+'s', "end": str(k)+'e', "text": list(random.choices(units, k=size))})

    else:
        current = 0
        while current + size <= len(units):
            samples.append({"start": current, "end": current + size, "text": list(units[current:(current + size)])})
            current = current + step

    return samples


def docs_to_samples(paths, size, step=None, units="words", samples_random=False, format="txt", keep_punct=False,
                    keep_sym=False, max_samples=None, identify_lang=False):
    """
    Loads a collection of documents into a 'myTexts' object for further processing BUT with samples !
    :param paths: path to docs
    :param size: sample size
    :param size: size of the step when sampling successively (determines overlap) default is the same
    as sample size (i.e. no overlap)
    :param units: the units to use, one of "words" or "verses"
    :param samples_random: Should random sampling with replacement be performed instead of continuous sampling (default: false)
    :param format: type of document, one of full text, TEI or simple XML (ONLY TEI and TXT IMPLEMENTED)
    :param keep_punct: whether to keep punctuation and caps.
    :param max_samples: maximum number of samples per author/class.
    :param identify_lang: whether to try to identify lang (default: False)
    :return: a myTexts object
    """
    myTexts = []
    for path in paths:
        aut = path.split('/')[-1].split('_')[0]
        if identify_lang:
            if format == 'xml':
                aut, text = XML_to_text(path)

            else:
                aut, text = TXT_to_text(path)

            lang = detect_lang(text)

        else:
            lang = 'NA'

        samples = get_samples(path, size=size, step=step, samples_random=samples_random, max_samples=max_samples,
                              units=units, format=format,
                              keep_punct=keep_punct, keep_sym=keep_sym)

        for sample in samples:
            name = path.split('/')[-1] + '_' + str(sample["start"]) + "-" + str(sample["end"])
            text = normalise(' '.join(sample["text"]), keep_punct=keep_punct, keep_sym=keep_sym)
            myTexts.append({"name": name, "aut": aut, "text": text, "lang": lang})

    if max_samples is not None:
        myTexts = max_sampling(myTexts, max_samples=max_samples)

    return myTexts
