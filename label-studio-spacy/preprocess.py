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
    "text": "M\/S. Pepsi Foods Ltd. & Anr vs Special Judicial Magistrate & Ors on 4 November, 1997        ",
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
        "text": "M\/S. Pepsi Foods Ltd. & Anr ",
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
  },
  {
    "text": "There are two appellants, second appellant is the Managing Director of first appellant, The respondents are three. First respondent is the court where the appellants alongwith others have been summoned for having committed offences under Sections 7\/16 of the Act. The second respondent is the complainant and the third respondent is the State of Uttar Pradesh.",
    "id": 1098,
    "label": [
      {
        "start": 238,
        "end": 262,
        "text": "Sections 7\/16 of the Act",
        "labels": [
          "Section"
        ]
      }
    ],
    "annotator": 4,
    "annotation_id": 613,
    "created_at": "2024-04-16T05:17:55.991896Z",
    "updated_at": "2024-04-16T05:17:55.991942Z",
    "lead_time": 38.468
  },
  {
    "text": "The allegation in the complaint is that complainant was sold a bottle of beverage under the brand Lehar Pepsi",
    "id": 1099,
    "label": [
      {
        "start": 98,
        "end": 109,
        "text": "Lehar Pepsi",
        "labels": [
          "Organization"
        ]
      }
    ],
    "annotator": 4,
    "annotation_id": 612,
    "created_at": "2024-04-16T05:17:51.002232Z",
    "updated_at": "2024-04-16T05:17:51.002276Z",
    "lead_time": 15.487
  },
  {
    "text": "Having set out the relevant provisions of law to some extent and before we consider the merits and demerits of the case and the jurisdiction of the High Court under Article 226 and 227 of the Constitution, we may refer to the complaint and the evidence which led the 1st respondent to issue summons to the appellants and others for an offence under Section 7 of the Act.",
    "id": 1100,
    "label": [
      {
        "start": 144,
        "end": 158,
        "text": "the High Court",
        "labels": [
          "Court"
        ]
      },
      {
        "start": 165,
        "end": 205,
        "text": "Article 226 and 227 of the Constitution,",
        "labels": [
          "Section"
        ]
      },
      {
        "start": 349,
        "end": 370,
        "text": "Section 7 of the Act.",
        "labels": [
          "Section"
        ]
      }
    ],
    "annotator": 4,
    "annotation_id": 611,
    "created_at": "2024-04-16T05:17:46.798209Z",
    "updated_at": "2024-04-16T05:17:46.798252Z",
    "lead_time": 38.242
  },
  {
    "text": "The complainant (second respondent) is a student. He says that he is appearing in examinations is various State and Central Services. On September 13, 1993, he went to a shop known as The Flavours Fast Food and Cool Corner and purchased 500 m1. chilled bottle of 'Lehar Pepsi' for drinking. Nitin Sachdeva is stated to have (Accused named as No.1) sold the bottle to the complainant. After he had consumed the beverage contained in the bottle, the complainant felt a strange taste. On observation, he found that the bottle contained many white particles. The complainant felt giddy and nauseated. One Divya Trivedi was present at the shop as a customer. Another shopkeeper by the name Lal Bahadur Singh who owned a shop opposite to from where the complainant purchased the 'Lehar Pepsi' bottle was also present. They were shown the bottle by the complainant. The beverage was put in two glasses to see the while particles clearly and Nitin Sachdeva accepted the presence of the particles. Suspecting adulteration, the complainant told Nitin Sachdeva that he would take sample of the beverage for analysis. He thereupon gave notice to Nitin Sachdeva, purchased three clean and dry empty new plastic jars from hereby Suri Stores and filled up the same with the beverage and which, according to the complainant, were sealed as per rules, wrapped in the paper and tied with thick yearn. Nitin Sachdeva signed the jars and put stamp of his shop thereon. The complainant obtained the stamp of the shop The Flavour Fast Food and Cool Corner on a separate paper and one jar of the sample with stamp used in the sample was deposited by the complainant in he office of the State Public Analyst, Uttar Pradesh, Lucknow on September 20, 1993 for analysis. The complainant says that the three jars were sealed in the presence of the witnesses and he also recorded their statements in writing including that of Nitin Sachdeva. The complainant also made a report to the Police on September 13, 1993 itself about the incident.",
    "id": 1101,
    "label": [
      {
        "start": 184,
        "end": 222,
        "text": "The Flavours Fast Food and Cool Corner",
        "labels": [
          "Organization"
        ]
      },
      {
        "start": 264,
        "end": 275,
        "text": "Lehar Pepsi",
        "labels": [
          "Organization"
        ]
      },
      {
        "start": 291,
        "end": 305,
        "text": "Nitin Sachdeva",
        "labels": [
          "Person"
        ]
      },
      {
        "start": 601,
        "end": 615,
        "text": "Divya Trivedi ",
        "labels": [
          "Person"
        ]
      },
      {
        "start": 684,
        "end": 702,
        "text": " Lal Bahadur Singh",
        "labels": [
          "Person"
        ]
      },
      {
        "start": 774,
        "end": 786,
        "text": "Lehar Pepsi'",
        "labels": [
          "Person"
        ]
      },
      {
        "start": 934,
        "end": 948,
        "text": "Nitin Sachdeva",
        "labels": [
          "Person"
        ]
      },
      {
        "start": 1035,
        "end": 1050,
        "text": "Nitin Sachdeva ",
        "labels": [
          "Person"
        ]
      },
      {
        "start": 1134,
        "end": 1149,
        "text": "Nitin Sachdeva,",
        "labels": [
          "Person"
        ]
      }
    ],
    "annotator": 4,
    "annotation_id": 610,
    "created_at": "2024-04-16T05:17:42.055568Z",
    "updated_at": "2024-04-16T05:17:42.055613Z",
    "lead_time": 115.325
  }
]

# Iterate over the JSON data, create SpaCy Doc objects, and add them to the DocBin
for entry in data:
    text = entry["text"]
    entities = entry.get("label", [])  # Access entities from "label" key
    doc = nlp.make_doc(text)
    ents = []
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["labels"][0]  # Assuming single label per entity
        span = doc.char_span(start, end, label=label)
        if span is None:
            print("Skipping entity:", text[start:end])
        else:
            ents.append(span)
    doc.ents = ents
    doc_bin.add(doc)

# Save the DocBin to a file
doc_bin.to_disk("trainebn_data.spacy")
