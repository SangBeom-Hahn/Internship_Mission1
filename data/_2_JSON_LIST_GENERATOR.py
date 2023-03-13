import os
import shutil
from tqdm import tqdm

# ROOT 설정
# ex) ./_1_SPLITTING_JSON/
ROOT = './_1_SPLITTING_JSON/'
TYPE = 'test'
TARGET_PREFIX = './_2_JSON_LIST/'


def ListToFile(filePath, contentsList):
    print(filePath)
    with open(filePath, 'w') as lf:        
        for path in contentsList:
            lf.write(path + '\n')

def GenerateAllJsonToListFile(targetPath, TYPE, jsonFilename):
    
    path_list = []
    for (path, dir, files) in os.walk(targetPath):        
        if TYPE not in path:
            continue
            
        pbar = tqdm(files)
        for filename in pbar:
            ext = os.path.splitext(filename)[-1]
            
            if ext == '.json':
                # 수정
                path = path.replace('\\', '/')

                path_list.append(path + '/' + filename)
        pbar.close()
    ListToFile(jsonFilename , path_list)
    print(len(path_list))    

GenerateAllJsonToListFile(ROOT, TYPE, TARGET_PREFIX + TYPE + '_LIST.txt')