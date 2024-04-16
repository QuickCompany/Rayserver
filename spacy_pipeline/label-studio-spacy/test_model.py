import spacy

# Load the trained SpaCy NER model
nlp = spacy.load("/root/Rayserver/spacy_pipeline/label-studio-spacy/ner_model_server/output/model-last")

# Sample texts for testing
texts = [
    "In support of the complaint allegations, the Complainant has recorded his statement and presented the statement on oath of the witness Lal Bahadur Singh and as documentary evidence notice annexure-1, receipt for deposit of the bottle of sample for analysis with Public Analyst annexure-3A and application to the Public Analyst for analysis annexure-3B, report of the incident with O.S. Ghazipur annexure-4, cash memo issued by the vendor annexure-5, statement of Executive Director of Pepsi Foods Ltd. annexure-6, report of the Public Analyst annexures 7A and 7B and prescriptions of the doctor for treatment have been filed."
]

# Process each text and print the entities
for text in texts:
    doc = nlp(text)
    print("Text:", text)
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
    print()
