__all__ = ['get_label']

BO = {0:'B', 1:'O'}
BO2 = {0:'B', 1:'O1', 2:'O2'}

BIO = {0:'B', 1:'I', 2:'O1'}
BIO2 = {0:'B', 1:'I', 2:'O1', 3:'O2'}
BIO3 = {0:'B', 1:'I', 2:'O1', 3:'O2', 4:'O3'}

label_maps = {'BO':BO, 'BO2':BO2, 'BIO':BIO, 'BIO2':BIO2, 'BIO3':BIO3}

def get_label(label_id, schema):
    return label_maps[schema][label_id]

