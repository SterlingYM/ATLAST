import json
import requests
from tqdm.auto import tqdm
import os 

http_errors = {
    304: "Error 304: Not Modified: There was no new data to return.",
    400: "Error 400: Bad Request: The request was invalid. "
    "An accompanying error message will explain why.",
    403: "Error 403: Forbidden: The request is understood, but it has "
    "been refused. An accompanying error message will explain why.",
    404: "Error 404: Not Found: The URI requested is invalid or the "
    "resource requested, such as a category, does not exists.",
    500: "Error 500: Internal Server Error: Something is broken.",
    503: "Error 503: Service Unavailable.",
}
    
class QueryTNS():
    def __init__(self,url='https://www.wis-tns.org/api/get/',
                 tns_id=None,bot_name=None,api_key=None):
        self.url = url
        self.tns_id = tns_id
        self.api_key = api_key
        self.bot_name = bot_name
        self.headers = self.create_headers()

    def post(self, path, data, url=None):
        if url is None:
            url = self.url
        response = requests.post(url + path, headers = self.headers, files=data)
        if response.status_code == 200:
            return response
        else:
            print(http_errors[response.status_code])
    
    def create_headers(self,):
        user_agent = dict(tns_id=self.tns_id, 
                          type='bot', 
                          name=self.bot_name)
        headers = {"User-Agent": f'tns_marker{json.dumps(user_agent)}'}
        return headers
    
    def filter_response(self,response):
        response = response.json()
        if response['id_message'] != 'OK':
            print('Error:',response['id_message'])
            return response
        else:
            return response['data']
    
    def prep_data(self,query_data):
        data = dict(api_key=(None,self.api_key),
                    data=(None,json.dumps(query_data)))
        return data
    
    def search_objects(self, **kwargs):
        data = self.prep_data(kwargs)
        res = self.post('search', data)
        return self.filter_response(res)
    
    def get_object(self, **kwargs):
        data = self.prep_data(kwargs)
        res = self.post('object', data)
        return self.filter_response(res)
    
    def get_objects(self, object_names, jsonfile='tns_tmp.json', spectra=1,
                    save_every=100):
        ''' a wrapper to iteratively run get_object and save the results to a json file'''
        # load existing data
        if os.path.exists(jsonfile):
            self.get_objects_replies = json.loads(open(jsonfile).read())
        else:
            self.get_objects_replies = {}
        
        for i,objname in enumerate(tqdm(object_names,leave=False)):
            if objname in self.get_objects_replies and self.get_objects_replies[objname] != 'failed':
                continue
            try:
                reply = self.get_object(objname=objname,spectra=spectra)['reply']
                self.get_objects_replies[objname] = reply
            except Exception:
                self.get_objects_replies[objname] = 'failed'
            if i % save_every == 0:
                with open(jsonfile,'w') as f:
                    f.write(json.dumps(self.get_objects_replies))            
        with open(jsonfile,'w') as f:
            f.write(json.dumps(self.get_objects_replies))      
            
    def get_file(self, url, savefile='get_file.txt'):
        data = self.prep_data({})
        res = self.post('', data, url=url)
        with open(savefile,'w') as f:
            f.write(res.text)


    