import json

path1 = './datasets/oodomain_train/duorc'
with open(path1, 'rb') as f:
    squad_dict = json.load(f)

# print(json.dumps(squad_dict, indent = 1))

print(squad_dict.keys())
print(squad_dict['version'])
print(type(squad_dict['version']))
print(len(squad_dict['data']))
# print(squad_dict['data'][0]['paragraphs'])

print(json.dumps(squad_dict['data'][0]['paragraphs'], indent = 1))
# version: '1.1'
# data: [
#   title: 50 char str
#   paragraphs: [

# ]
# ]



# disk_json = {'version': '1.1', data: }

# path2 = './test_dataset/squad'

# def checkContextCount(paths):
#     for path in paths:
#         with open(path, 'rb') as f:
#             squad_dict = json.load(f)
#             data = squad_dict['data']
#             for elem in data:
#                 title = elem['title']
#                 paragraphs = elem['paragraphs']
#                 if len(paragraphs != 1):
#                     print('Found a multi-context paragraph:')
#                     print(paragraphs))
#                     return False
#     print('All paragraphs are of length 1')
#     return True