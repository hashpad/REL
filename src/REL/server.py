import traceback
import json
from http.server import BaseHTTPRequestHandler

from flair.models import SequenceTagger

from REL.mention_detection import MentionDetection
from REL.utils import process_results, process_ed_results

API_DOC = "API_DOC"



def make_handler(base_url, wiki_version, model, tagger_ner):
    """
    Class/function combination that is used to setup an API that can be used for e.g. GERBIL evaluation.
    """
    class GetHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.model = model
            self.tagger_ner = tagger_ner

            self.base_url = base_url
            self.wiki_version = wiki_version

            self.custom_ner = not isinstance(tagger_ner, SequenceTagger)
            self.mention_detection = MentionDetection(base_url, wiki_version)

            super().__init__(*args, **kwargs)

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                bytes(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "label": "status",
                            "message": "up",
                            "color": "green",
                        }
                    ),
                    "utf-8",
                )
            )
            return

        def do_HEAD(self):
            # send bad request response code
            self.send_response(400)
            self.end_headers()
            self.wfile.write(bytes(json.dumps([]), "utf-8"))
            return

        def do_POST(self):
            """
            Returns response.

            :return:
            """
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                self.send_response(200)
                self.end_headers()

                if self.path == '/md':
                    response = self.do_md(post_data)
                    self.wfile.write(bytes(json.dumps(response), "utf-8"))
                elif self.path == '/cg':
                    response = self.do_cg(post_data)
                    self.wfile.write(bytes(json.dumps(response), "utf-8"))
                elif self.path == '/ed':
                    response = self.do_ed(post_data)
                    self.wfile.write(bytes(json.dumps(response), "utf-8"))
                elif self.path == '/combine':
                    text, spans = self.read_json(post_data)
                    response = self.generate_response(text, spans)

                    self.wfile.write(bytes(json.dumps(response), "utf-8"))


            except Exception as e:
                print(f"Encountered exception: {repr(e)}")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(bytes(json.dumps([]), "utf-8"))
                # tb = traceback.format_exc()

            # finally:
                # print(tb)
            return

        def do_md(self, post_data):
            """
            Reads input JSON message.

            :return: document text and spans.
            """

            data = json.loads(post_data.decode("utf-8"))
            text = data["document"]["text"]
            text = text.replace("&amp;", "&")


            if len(text) == 0:
                return []

            else:
                # EL
                processed = {API_DOC: [text, []]}
                mentions_dataset, total_ment = self.mention_detection.find_mentions(
                    processed, self.tagger_ner
                )

            # Process result.
            result = mentions_dataset
            mentions = result['API_DOC']
            mentions_formated = []
            for mention in mentions:
                mentions_formated.append({ 
                                          'offset': mention['pos'],
                                          'assignment': None,
                                          'possibleAssignments': [],
                                          'originalWithoutStopwords': mention['ngram'],
                                          'detectionConfidence': mention['conf_md'],
                                          'originalMention': mention['ngram'],
                                          'mention': mention['mention']
                                          })
            
            data["document"]["mentions"] = mentions_formated

            # Singular document.
            if len(result) > 0:
                return data

            return []

        def do_cg(self, post_data):
            """
            Reads input JSON message.

            :return: document text and spans.
            """

            data = json.loads(post_data.decode("utf-8"))

            mentions = data['document']['mentions']


            for mention in mentions:
                mentions_candidates = self.mention_detection.get_candidates(
                    mention['mention']
                )
                for cand in mentions_candidates:
                    mention['possibleAssignments'].append({
                        'score': cand[1],
                        'assignment': 'https://wikipedia.org/wiki/' + cand[0]
                        })


            return data

        def do_ed(self, post_data):
            """
            Reads input JSON message.

            :return: document text and spans.
            """

            data = json.loads(post_data.decode("utf-8"))
            format_for_predictor = {'API_DOC': []}

            # TODO: FIX THE CONTEXT
            for mention in data['document']['mentions']:
                format_for_predictor['API_DOC'].append({
                    'mention': mention['mention'],
                    'context': ["", ""],
                    'candidates': [
                            [possibleAssignment['assignment'].replace("https://wikipedia.org/wiki/", ""), possibleAssignment['score']] 
                            for possibleAssignment in mention['possibleAssignments']
                        ],
                    'gold': ['NONE'],
                    'pos': mention['offset'],
                    'sent_idx': 0,
                    'ngram': mention['originalMention'],
                    'end_pos': len(mention['mention']),
                    'sentence': data['document']['text'],
                    'conf_md': mention['detectionConfidence'],
                    'tag': 'PER'
                    })

            # Disambiguation
            predictions, timing = self.model.predict(format_for_predictor)

            result = process_ed_results(
                format_for_predictor,
                predictions,
                include_offset=False,
            )

            for mention in data['document']['mentions']:
                for matched_mention in result['API_DOC']:
                    if mention['mention'] == matched_mention[2] and mention['offset'] == matched_mention[0]:
                        mention['assignment'] = {'score': matched_mention[4], 'assignment': matched_mention[3]}



            # Singular document.
            return data



        def read_json(self, post_data):
            """
            Reads input JSON message.

            :return: document text and spans.
            """

            data = json.loads(post_data.decode("utf-8"))
            text = data["text"]
            text = text.replace("&amp;", "&")

            # GERBIL sends dictionary, users send list of lists.
            if "spans" in data:
                try:
                    spans = [list(d.values()) for d in data["spans"]]
                except Exception:
                    spans = data["spans"]
                    pass
            else:
                spans = []

            return text, spans

        def generate_response(self, text, spans):
            """
            Generates response for API. Can be either ED only or EL, meaning end-to-end.

            :return: list of tuples for each entity found.
            """

            if len(text) == 0:
                return []

            if len(spans) > 0:
                # ED.
                processed = {API_DOC: [text, spans]}
                mentions_dataset, total_ment = self.mention_detection.format_spans(
                    processed
                )
            else:
                # EL
                processed = {API_DOC: [text, spans]}
                mentions_dataset, total_ment = self.mention_detection.find_mentions(
                    processed, self.tagger_ner
                )

            ## MENTION DETECTION DONE + CANDIDATES ONLY
            print(mentions_dataset)

            # Disambiguation
            predictions, timing = self.model.predict(mentions_dataset)



            # Process result.
            result = process_results(
                mentions_dataset,
                predictions,
                processed,
                include_offset=False if ((len(spans) > 0) or self.custom_ner) else True,
            )

            # Singular document.
            if len(result) > 0:
                return [*result.values()][0]

            return []

    return GetHandler


if __name__ == "__main__":
    import argparse
    from http.server import HTTPServer

    from REL.entity_disambiguation import EntityDisambiguation
    from REL.ner import load_flair_ner

    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("wiki_version")
    p.add_argument("--ed-model", default="ed-wiki-2019")
    p.add_argument("--ner-model", default="ner-fast")
    p.add_argument("--bind", "-b", metavar="ADDRESS", default="0.0.0.0")
    p.add_argument("--port", "-p", default=5555, type=int)
    args = p.parse_args()

    ner_model = load_flair_ner(args.ner_model)
    ed_model = EntityDisambiguation(
        args.base_url, args.wiki_version, {"mode": "eval", "model_path": args.ed_model}
    )
    server_address = (args.bind, args.port)
    server = HTTPServer(
        server_address,
        make_handler(args.base_url, args.wiki_version, ed_model, ner_model),
    )

    try:
        print("Ready for listening.")
        server.serve_forever()
    except KeyboardInterrupt:
        exit(0)
