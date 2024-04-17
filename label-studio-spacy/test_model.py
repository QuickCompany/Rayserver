import spacy

# Load the trained SpaCy NER model
nlp = spacy.load("/home/debo/Rayserver/label-studio-spacy/ner_model_server/output/model-best")

# Sample texts for testing
texts = [
    """The complainant (second respondent) is a student. He says that he is appearing in examinations is various State and Central Services. On September 13, 1993, he went to a shop known as The Flavours Fast Food and Cool Corner and purchased 500 m1. chilled bottle of 'Lehar Pepsi' for drinking. Nitin Sachdeva is stated to have (Accused named as No.1) sold the bottle to the complainant. After he had consumed the beverage contained in the bottle, the complainant felt a strange taste. On observation, he found that the bottle contained many white particles. The complainant felt giddy and nauseated. One Divya Trivedi was present at the shop as a customer. Another shopkeeper by the name Lal Bahadur Singh who owned a shop opposite to from where the complainant purchased the 'Lehar Pepsi' bottle was also present. They were shown the bottle by the complainant. The beverage was put in two glasses to see the while particles clearly and Nitin Sachdeva accepted the presence of the particles. Suspecting adulteration, the complainant told Nitin Sachdeva that he would take sample of the beverage for analysis. He thereupon gave notice to Nitin Sachdeva, purchased three clean and dry empty new plastic jars from hereby Suri Stores and filled up the same with the beverage and which, according to the complainant, were sealed as per rules, wrapped in the paper and tied with thick yearn. Nitin Sachdeva signed the jars and put stamp of his shop thereon. The complainant obtained the stamp of the shop The Flavour Fast Food and Cool Corner on a separate paper and one jar of the sample with stamp used in the sample was deposited by the complainant in he office of the State Public Analyst, Uttar Pradesh, Lucknow on September 20, 1993 for analysis. The complainant says that the three jars were sealed in the presence of the witnesses and he also recorded their statements in writing including that of Nitin Sachdeva. The complainant also made a report to the Police on September 13, 1993 itself about the incident..
    """
]

# Process each text and print the entities
for text in texts:
    doc = nlp(text)
    # print("Text:", text)
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")
    print()
