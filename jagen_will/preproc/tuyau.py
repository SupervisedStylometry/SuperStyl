from lxml import etree
import re
import fasttext


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

def identify_lang(string, model):
    """
    Get the language from a string
    :param string: a string, duh
    :param model, the fasttext model
    :return: the language
    """

    return model.predict(string)#, k = 3)

