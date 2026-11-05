"""Generate public, invented PDF fixtures; no user files are copied."""
import json
from pathlib import Path

TEXTS = {
    'finance/invoice.pdf': 'Invoice for office chairs. Payment of 450 dollars is due in November.',
    'travel/hotel.pdf': 'Hotel reservation in Nashville. Accommodation booked for a three night stay.',
    'study/graphs.pdf': 'Graph algorithms. Breadth first search visits neighboring vertices using a queue.',
}


def create_demo(output):
    import pymupdf
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, text in TEXTS.items():
        path = output / 'pdfs' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open() as pdf:
            pdf.new_page().insert_textbox((50, 50, 550, 750), text)
            pdf.save(path)
    queries = []
    for qid, text, answer in [
        ('q-payment', 'When must I pay for the office chairs?', 'finance/invoice.pdf'),
        ('q-stay', 'Where is my accommodation booking?', 'travel/hotel.pdf'),
        ('q-graph', 'Which graph traversal uses a queue?', 'study/graphs.pdf'),
        ('q-no-answer', 'Recipe for chocolate cake', None),
    ]:
        queries.append(dict(query_id=qid, text=text, split='test',
                            judgments={name: 3 if name == answer else 0 for name in TEXTS}))
    benchmark = dict(schema_version=1, benchmark_id='synthetic-smoke-v1', queries=queries)
    (output / 'benchmark.json').write_text(json.dumps(benchmark, indent=2) + '\n', encoding='utf-8')
