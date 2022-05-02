from src.tweet_score.model import *
import unittest
from collections import namedtuple


class Test_Model(unittest.TestCase):
    def setUp(self):
        self.Analyzer = TweetAnalyzer()

    def test_tokenize(self):
        Test = namedtuple('Testcase', 'arg, ans')
        tests = [
            Test('#just buy', {'label': 'LABEL_1',
                 'score': 0.9707463979721069}),
            Test('just sell it', {'label': 'LABEL_0',
                 'score': 0.9894134998321533}),
            Test('entity rocket to the sky!', {
                 'label': 'LABEL_1', 'score': 0.9486345052719116}),
            Test('go down', {'label': 'LABEL_0',
                 'score': 0.9953770637512207}),
            Test('even though it is going up, I still think it will not keep this trend in the near future', {
                 'label': 'LABEL_0', 'score': 0.6002193689346313}),

        ]
        for t in tests:
            res = self.Analyzer.tokenize(t.arg, needProcessed=True)
            self.assertEqual(res[0], t.ans, 'Incorrect ans')


if __name__ == '__main__':
    unittest.main()
