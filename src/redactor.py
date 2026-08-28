import re

def layer1_regex_redactor(text):
    text = re.sub(r'\b\d{1,3}-year-old\b', '[AGE]', text)
    text = re.sub(r'\b(man|woman|male|female|boy|girl)\b', '[SEX]', text, flags=re.IGNORECASE)
    return text

def layer2_ner_redactor(text):
    replacements = {
        r'\bhis wife\b': 'their spouse',
        r'\bher husband\b': 'their spouse',
        r'\bhe\b': 'the patient',
        r'\bshe\b': 'the patient',
        r'\bhis\b': 'their',
        r'\bhim\b': 'them',
        r'\bher\b': 'them'
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text