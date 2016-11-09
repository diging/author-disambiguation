import re, math
from collections import Counter

WORD = re.compile(r'\w+')

def cosine_similarity(vector1, vector2):
    """
    Computes the cosine similarity between 2 vectors.

    ``Example``
        >>> input1 = "CARNEGIE INST WASHINGTON,DEPT EMBRYOL"
        >>> input2 = "CARNEGIE INST WASHINGTON,DEPT"
        >>> vector1 = sentence_to_vector(input1) #Counter({'EMBRYOL': 1, 'WASHINGTON': 1, 'INST': 1, 'CARNEGIE': 1, 'DEPT': 1})
        >>> vector2 = sentence_to_vector(input2) #Counter({'WASHINGTON': 1, 'INST': 1, 'CARNEGIE': 1, 'DEPT': 1})
        >>> cosine_similarity(vector1, vector2)  #0.894427191

    :param vector1:
    :param vector2:
    :return:
    """
    common_tokens = set(vector1.keys()) & set(vector2.keys())
    dot_product = sum([vector1[x] * vector2[x] for x in common_tokens])

    vector1Magnitude = sum([vector1[x]**2 for x in vector1.keys()])
    vector2Magnitute = sum([vector2[x]**2 for x in vector2.keys()])

    productOfMagnitude = math.sqrt(vector1Magnitude) * math.sqrt(vector2Magnitute)

    if not productOfMagnitude:
        return 0.0
    else:
        return float(dot_product)/productOfMagnitude


def sentence_to_vector(sentence):
    """
    tokenizes and converts into a vector.

    ``Example``
        >>> input1 = "CARNEGIE INST WASHINGTON,DEPT EMBRYOL"
        >>> input2 = "CARNEGIE INST WASHINGTON,DEPT"
        >>> vector1 = sentence_to_vector(input1) #Counter({'EMBRYOL': 1, 'WASHINGTON': 1, 'INST': 1, 'CARNEGIE': 1, 'DEPT': 1})
        >>> vector2 = sentence_to_vector(input2) #Counter({'WASHINGTON': 1, 'INST': 1, 'CARNEGIE': 1, 'DEPT': 1})

    :param sentence : (str) Sentence to be converted into a vector.
    :return:
    """
    tokens = WORD.findall(sentence)
    return Counter(tokens)
"""
input1 = "CARNEGIE INST WASHINGTON,DEPT EMBRYOL"
input2 = "CARNEGIE INST WASHINGTON,DEPT"
vector1 = sentence_to_vector(input1)
vector2 = sentence_to_vector(input2)
print vector1
print vector2
print cosine_similarity(vector1, vector2)
"""

