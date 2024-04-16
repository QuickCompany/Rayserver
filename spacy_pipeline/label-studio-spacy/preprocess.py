import spacy
from spacy.tokens import DocBin

# Load a blank SpaCy model and create a DocBin
nlp = spacy.blank("en")
doc_bin = DocBin()

# JSON data
data = [
  {
    "text": "example text food",
    "id": 1092,
    "label": [
      {
        "start": 8,
        "end": 12,
        "text": "text",
        "labels": [
          "Person"
        ]
      }
    ],
    "annotator": 4,
    "annotation_id": 609,
    "created_at": "2024-04-15T09:06:33.763382Z",
    "updated_at": "2024-04-15T09:06:33.763419Z",
    "lead_time": 12.969
  },
  {
    "text": "M/S. Pepsi Foods Ltd. & Anr vs Special Judicial Magistrate & Ors on 4 November, 1997        ",
    "id": 1097,
    "label": [
      {
        "start": 5,
        "end": 10,
        "text": "Pepsi",
        "labels": [
          "Organization"
        ]
      },
      {
        "start": 0,
        "end": 28,
        "text": "M/S. Pepsi Foods Ltd. & Anr ",
        "labels": [
          "Organization"
        ]
      },
      {
        "start": 31,
        "end": 64,
        "text": "Special Judicial Magistrate & Ors",
        "labels": [
          "Organization"
        ]
      }
    ],
    "annotator": 4,
    "annotation_id": 614,
    "created_at": "2024-04-16T05:18:00.937093Z",
    "updated_at": "2024-04-16T05:18:00.937136Z",
    "lead_time": 35.979
  }
]

# Iterate over the JSON data, create SpaCy Doc objects, and add them to the DocBin
for entry in data:
    text = entry["text"]
    entities = entry.get("entities", [])
    doc = nlp.make_doc(text)
    ents = []
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["label"]
        span = doc.char_span(start, end, label=label)
        if span is None:
            print("Skipping entity:", text[start:end])
        else:
            ents.append(span)
    doc.ents = ents
    doc_bin.add(doc)

# Save the DocBin to a file
doc_bin.to_disk("test_data.spacy")
