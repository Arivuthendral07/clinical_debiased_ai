import re
import spacy


try:
    nlp=spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model")
    spacy.cli.download("en_core_web_sm")
    nlp=spacy.load("en_core_web_sm")
def layer1_regex_redactor(text):
    """Handles strict clinical patterns and basic pronouns via Regex."""
    text = re.sub(r'\b\d{1,3}-year-old\b', '[AGE]', text)
    text = re.sub(r'\b(man|woman|male|female|boy|girl)\b', '[SEX]', text, flags=re.IGNORECASE)
    
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

def layer2_ner_redactor(text):
    """
    Uses spaCy advanced Named Entity Recognition (NER) to dynamically catch 
    names, locations, cultural backgrounds, and institutions.
    """
    doc = nlp(text)
    redacted_text = text
    
    # We iterate in reverse so replacing text doesn't mess up the character indices
    for ent in reversed(doc.ents):
        # Target specific entity labels: Person, Geopolitical Entity, Location, Organization, Nationalities/Religious Groups
        if ent.label_ in ["PERSON", "GPE", "LOC", "ORG", "NORP"]:
            # Replace the identified word with its category tag
            redacted_text = redacted_text[:ent.start_char] + f"[{ent.label_}]" + redacted_text[ent.end_char:]
            
    return redacted_text
if __name__ == "__main__":
    # A test vignette loaded with demographic clues, names, and locations
    test_vignette = "Dr. Sharma referred a 45-year-old male to Apollo Hospital in Chennai because he had a severe headache."
    
    print("--- REDACTION TEST ---")
    print(f"Original: {test_vignette}\n")
    
    step_1 = layer1_regex_redactor(test_vignette)
    print(f"After Layer 1 (Regex): {step_1}\n")
    
    step_2 = layer2_ner_redactor(step_1)
    print(f"After Layer 2 (spaCy NER): {step_2}\n")