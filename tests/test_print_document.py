"""Portable artwork, screen-page grouping and downstairs label structure.

Actual printed pagination is checked by check_print_pagination.py, including
paper-size variants; counting HTML wrappers alone cannot detect blank sheets.
"""
import base64
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from test_downstairs import load_source


class PageImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pages = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'section' and attrs.get('class') == 'sched-page':
            self.pages.append([])
        if tag == 'img':
            self.images.append(attrs)
            self.pages[-1].append(attrs)


class PrintDocumentChecks(unittest.TestCase):
    def setUp(self):
        self.ns = load_source()
        self.ns['base64'] = base64

    def test_each_screen_page_owns_a_complete_background(self):
        marker = "<div style='page-break-after: always;'></div>"
        source_pages = ['<h1>Session</h1>Match 7', 'Focus reminder: Match 10', 'Match 13']
        result = self.ns['schedule_pages_html'](marker.join(source_pages))
        parser = PageImages()
        parser.feed(result)
        self.assertEqual([len(p) for p in parser.pages], [1, 1, 1])
        self.assertNotIn(marker, result)
        for image in parser.images:
            self.assertEqual(image['alt'], '')
            self.assertEqual(image['aria-hidden'], 'true')
            prefix, data = image['src'].split(',', 1)
            self.assertEqual(prefix, 'data:image/svg+xml;base64')
            svg = ET.fromstring(base64.b64decode(data, validate=True))
            self.assertEqual(svg.tag, '{http://www.w3.org/2000/svg}svg')
            ids = {node.get('id') for node in svg.iter() if node.get('id')}
            self.assertTrue({'doodle-racket', 'padel-doodle-tile'} <= ids)
            for node in svg.iter():
                href = node.get('href')
                if href:
                    self.assertIn(href.removeprefix('#'), ids)
        self.assertEqual([result.index(p) for p in source_pages], sorted(result.index(p) for p in source_pages))

    def test_single_group_has_one_screen_wrapper(self):
        result = self.ns['schedule_pages_html']('<h1>Session</h1>Match 1')
        self.assertEqual(result.count('class="sched-page"'), 1)
        self.assertEqual(result.count('class="sched-doodles"'), 1)

    def test_badge_and_court_name_share_the_nonwrapping_label(self):
        result = self.ns['format_match_table_html'](
            0, [(0, 1, 2, 3)], ['Court 12'], ['Aideen', 'Anna', 'Mike', 'Scott'], [],
            downstairs_idx=[0], time_label='10:30')
        self.assertIn('<span class="court-label court-label-downstairs"><span>Court 12</span>', result)
        self.assertIn('Downstairs</span></span></td>', result)
        self.assertNotIn('<br', result)


if __name__ == '__main__':
    unittest.main()
