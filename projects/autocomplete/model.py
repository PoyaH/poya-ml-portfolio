"""2026 AI-assisted rewrite of the course n-gram experiment; original preserved."""
from collections import Counter, defaultdict
import re

def tokenize(text):
    return re.findall(r'[a-z]+', text.lower())

class NGramCompletion:
    def __init__(self, context_size=2):
        if context_size<1:raise ValueError('context_size must be positive')
        self.context_size=context_size
        self.transitions=defaultdict(Counter)
    def fit(self, documents):
        self.transitions.clear()
        for document in documents:
            words=tokenize(document)
            for i in range(self.context_size,len(words)):
                self.transitions[tuple(words[i-self.context_size:i])][words[i]]+=1
        return self
    def suggest(self, prefix, k=5):
        if k<1:raise ValueError('k must be positive')
        words=tokenize(prefix)
        if len(words)<self.context_size:return []
        counts=self.transitions.get(tuple(words[-self.context_size:]))
        if not counts:return []
        total=sum(counts.values())
        return [(w,c/total) for w,c in sorted(counts.items(),key=lambda v:(-v[1],v[0]))[:k]]
