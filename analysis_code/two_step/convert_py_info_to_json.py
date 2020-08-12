import imp
import json

def cinfo():
    info = imp.load_source('info', 'info.py')
    info_dict = {'info'      : info.info,
                 'event_IDs' : info.IDs,
                 'file_type' : info.file_type}
    with open('info.json', 'w') as f:
        f.write(json.dumps(info_dict,  indent=4))