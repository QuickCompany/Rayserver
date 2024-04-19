import json
def convert_to_first_format(data):
    # Extracting relevant information from the data
    text = data['data']['text']
    spans = []
    relations = []

    # Extracting entity spans from the annotations
    for annotation in data['annotations']:
        for result in annotation['result']:
            if 'value' in result:  # Check if 'value' key exists
                value = result['value']
                if 'labels' in value:
                    label = value['labels'][0]
                    start = value['start']
                    end = value['end']
                    text_span = text[start:end]
                    spans.append({
                        "text": text_span,
                        "start": start,
                        "end": end,
                        "type": "span",
                        "label": label
                    })

    # Creating relation between entities if available
    for annotation in data['annotations']:
        for result in annotation['result']:
            if result['type'] == 'relation':
                if 'from_id' in result and 'to_id' in result:  # Check if 'from_id' and 'to_id' keys exist
                    head_id = result['from_id']
                    child_id = result['to_id']
                    head_span = next((span for span in spans if span['start'] == head_id), None)
                    child_span = next((span for span in spans if span['start'] == child_id), None)
                    if head_span and child_span:
                        relations.append({
                            "head": head_span['start'],
                            "child": child_span['start'],
                            "head_span": head_span,
                            "child_span": child_span,
                            "color": "#ffd882",
                            "label": "Pos-Reg"  # You may need to change this label based on your data
                        })

    # Constructing the final output
    output = {
        "text": text,
        "spans": spans,
        "relations": relations,
        "answer": "accept"  # You may need to modify this based on your application
    }

    return output


# Example usage:
data = json.load(open("data.json","r"))

for i in data:
    print(convert_to_first_format(i))