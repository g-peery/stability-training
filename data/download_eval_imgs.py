import os
import sys
import urllib.request
from tqdm import tqdm

def main(lines):
    counter = 0
    i = 0
    txt_queries = []
    duplicates = {}
    while i < len(lines)//4:
        
        txt_query=lines[counter]
        query_img_url = lines[counter+1]
        pos_img_url = lines[counter+2]
        neg_img_url = lines[counter+3]
        if txt_query in txt_queries:
            #print(txt_query, "already exists")
            duplicates[txt_query] += 1
            #txt_queries.append(txt_query+str(duplicates[txt_query]+1))
            txt_query = txt_query+str(duplicates[txt_query]+1)
        else:
            duplicates[txt_query] = 0
            txt_queries.append(txt_query)
        try:
            write_imgs(txt_query, query_img_url, pos_img_url, neg_img_url)
        except(FileExistsError):
            continue
        print(i)
        counter += 4
        i += 1
    #print(txt_queries.count('snoop+dogg5'))
    #print(len(lines), counter, len(txt_queries), txt_query, query_img_url, pos_img_url, neg_img_url)
    
    return 

def write_imgs(txt_query, query_img_url, pos_img_url, neg_img_url):
    dirname = os.path.dirname(sys.argv[1]) # data folder
    query_folder = os.path.join(dirname, txt_query)
    os.makedirs(query_folder)
    urllib.request.urlretrieve(query_img_url, os.path.join(query_folder, 'query_img.jpeg'))
    urllib.request.urlretrieve(pos_img_url, os.path.join(query_folder, 'pos_img.jpeg'))
    urllib.request.urlretrieve(neg_img_url, os.path.join(query_folder, 'neg_img.jpeg'))
    ##sys.exit(0)
    return

if __name__ == "__main__":
    urls_file = sys.argv[1]
    f = open(urls_file, 'r')
    lines = [x.strip() for x in f.readlines() if x]
    f.close()
    main(lines)