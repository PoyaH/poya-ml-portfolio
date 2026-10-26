"""2026 reproducible extension of historical coursework, not an exact rerun."""
from pathlib import Path
import argparse, hashlib, json, os, platform
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/poya-portfolio-matplotlib')
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


def run(data_path, output, seed=42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    raw=data_path.read_bytes()
    data=json.loads(raw)
    texts=[]; labels=[]
    for intent in data['intents']:
        for phrase in intent['patterns']:
            texts.append(phrase); labels.append(intent['tag'])
    enc=LabelEncoder(); y=enc.fit_transform(labels)
    indices=np.arange(len(texts))
    dev,test=train_test_split(indices,test_size=.2,stratify=y,random_state=seed)
    train,val=train_test_split(dev,test_size=.25,stratify=y[dev],random_state=seed)
    assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    vec=CountVectorizer(binary=True)
    vec.fit([texts[i] for i in train])
    X=vec.transform(texts).toarray().astype('float32')
    baseline=LogisticRegression(max_iter=2000,random_state=seed).fit(X[train],y[train])
    model=tf.keras.Sequential([
        tf.keras.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(len(enc.classes_),activation='softmax')])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
    # Explicit batches avoid implicit tf.data thread-pool requirements on laptops.
    history=[];best=float('inf');best_weights=None;best_epoch=None
    rng=np.random.default_rng(seed)
    for epoch in range(1,51):
        order=rng.permutation(train)
        for start in range(0,len(order),32):
            batch=order[start:start+32];model.train_on_batch(X[batch],y[batch])
        prob=np.asarray(model(X[val],training=False))
        loss=float(log_loss(y[val],prob,labels=np.arange(len(enc.classes_))))
        history.append({'epoch':epoch,'validation_loss':loss,'validation_accuracy':float(accuracy_score(y[val],prob.argmax(1)))})
        if loss<best:best=loss;best_weights=model.get_weights();best_epoch=epoch
    model.set_weights(best_weights)
    results={}
    predictions={'logistic_bow':baseline.predict(X[test]),'neural_bow':np.asarray(model(X[test],training=False)).argmax(1)}
    for name,pred in predictions.items():
        results[name]={'accuracy':float(accuracy_score(y[test],pred)),
                      'macro_f1':float(f1_score(y[test],pred,average='macro')),
                      'classification_report':classification_report(y[test],pred,target_names=enc.classes_,zero_division=0,output_dict=True),
                      'confusion_matrix':confusion_matrix(y[test],pred).tolist()}
    report={'description':'New 2026 experiment, existing course dataset; not fresh external validation or exact reproduction.',
            'data_sha256':hashlib.sha256(raw).hexdigest(),'seed':seed,
            'split_counts':{'train':len(train),'validation':len(val),'test':len(test)},
            'split_indices':{'train':train.tolist(),'validation':val.tolist(),'test':test.tolist()},
            'classes':enc.classes_.tolist(),'vocabulary_size':len(vec.vocabulary_),
            'selected_neural_epoch':best_epoch,'selection':'Minimum validation loss across 50 epochs; test evaluated after selection.',
            'results':results,'history':history,
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'tensorflow':tf.__version__}}
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'split_counts':report['split_counts'],'epoch':best_epoch,'results':{k:{m:v[m] for m in ['accuracy','macro_f1']} for k,v in results.items()}},indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args();run(args.data,args.output)
