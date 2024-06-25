import pickle

def load_chunk(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def chunk_generator(filenames):
    for filename in filenames:
        yield load_chunk(filename)

def combine_chunks(generator):
    combined = {'text1': [], 'text2': [], 'score': []}
    for chunk in generator:
        combined['text1'].extend(chunk['text1'])
        combined['text2'].extend(chunk['text2'])
        combined['score'].extend(chunk['score'])
    return combined

chunk_files = [
    'trainchunk0.pkl',
    'trainchunk1.pkl',
    'trainchunk2.pkl',
    'trainchunk3.pkl',
    'trainchunk8.pkl'
]

chunk_gen = chunk_generator(chunk_files)
processed_train = combine_chunks(chunk_gen)

with open('combined.pkl', 'wb') as f:
    pickle.dump(processed_train, f)
