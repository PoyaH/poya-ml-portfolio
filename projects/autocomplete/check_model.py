import json
from pathlib import Path
from model import NGramCompletion
m=NGramCompletion(2).fit(['we like tea','we like tea','we like coffee'])
assert m.suggest('we like')==[('tea',2/3),('coffee',1/3)]
assert m.suggest('unknown context')==[]
assert m.suggest('we')==[]
assert m.suggest('WE LIKE!')==m.suggest('we like')
assert NGramCompletion(1).fit(['alpha beta','gamma delta']).suggest('beta')==[]
assert NGramCompletion(2).fit(['one two three']).suggest('one two')==[('three',1.0)]
assert NGramCompletion(2).fit(['one']).suggest('one two')==[]
r={'checks':['conditional probabilities normalized by context','unseen contexts','short input','case/punctuation handling','no cross-document transitions','last transition retained','short corpus'],'scope':'Synthetic correctness cases; not held-out prediction quality.'}
Path(__file__).with_name('checks.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
