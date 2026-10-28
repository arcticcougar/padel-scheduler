"""Render generated schedule HTML and reject extra/blank/split print pages.

Run with a Python containing pdfplumber:
  python tests/check_print_pagination.py --browser PATH_TO_CHROME fixture.html ...
Fixtures must come from the real app. Use paper-size/margin overrides as well as
the default A4 fixture: a full-sheet minimum height can pass A4 yet fail Letter.
This performs local document rendering only, never live app downloads/uploads.
"""
import argparse
import json
import re
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import pdfplumber


class PageMatches(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pages = []

    def handle_starttag(self, tag, attrs):
        if tag == 'section' and dict(attrs).get('class') == 'sched-page':
            self.pages.append([])

    def handle_data(self, data):
        match = re.match(r'^Match (\d+)\b', data.strip())
        if match:
            self.pages[-1].append(int(match[1]))


def check(browser, fixture):
    fixture = fixture.resolve()
    expected = PageMatches()
    expected.feed(fixture.read_text(encoding='utf-8'))
    assert expected.pages and all(expected.pages), fixture.name
    output = fixture.with_suffix('.pdf')
    with tempfile.TemporaryDirectory(prefix='padel-print-check-') as profile:
        profile_path = Path(profile).resolve()
        assert profile_path.parent == Path(tempfile.gettempdir()).resolve()
        assert profile_path.name.startswith('padel-print-check-')
        result = subprocess.run([
            str(browser), '--headless', '--disable-gpu', '--no-first-run',
            '--disable-background-networking', '--disable-extensions',
            f'--user-data-dir={profile}', '--no-pdf-header-footer',
            f'--print-to-pdf={output}', fixture.as_uri(),
        ], capture_output=True, timeout=60,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        assert result.returncode == 0 and output.is_file(), result.stderr[-1000:]
    with pdfplumber.open(output) as pdf:
        actual = [list(map(int, re.findall(r'\bMatch (\d+)\b', page.extract_text())))
                  for page in pdf.pages]
        assert actual == expected.pages, (fixture.name, actual, expected.pages)
        for page, matches in zip(pdf.pages, expected.pages):
            assert page.extract_text().count('Resting:') == len(matches)
            assert all(10 < c['x0'] < c['x1'] < page.width-10 and
                       10 < c['top'] < c['bottom'] < page.height-10 for c in page.chars)
    return {'file': fixture.name, 'pages': len(actual), 'matches': actual}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', required=True, type=Path)
    parser.add_argument('fixtures', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([check(args.browser, path) for path in args.fixtures], indent=2))
