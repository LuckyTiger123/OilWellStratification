import yaml
import sys
import lightgbm as lgb
import pandas as pd
import numpy as np
import las

# read the config
with open('./setting.yml', 'r') as f:
    path = yaml.load(f.read(), Loader=yaml.FullLoader)['path']

model_path = path['ModelPath']
well_las_file_path = path['WellLasFilePath']
result_save_path = path['ResultSavePath']

# get path
if model_path is None:
    print('ModelPath in setting is empty.')
    sys.exit(1)

if well_las_file_path is None:
    print('WellLasFilePath in setting is empty.')
    sys.exit(1)

if result_save_path is None:
    print('ResultSavePath in setting is empty.')
    sys.exit(1)

# get well name
well_name = well_las_file_path.split('/')[-1].replace('.las', '')

# get the model
bst = lgb.Booster(model_file=model_path)

# get the las file
item = las.LASReader(well_las_file_path)
raw_data = pd.DataFrame(item.data)

if not {'DEPTH', 'AC', 'SP', 'COND', 'ML1', 'ML2'}.issubset(raw_data.columns):
    lack_item = {'DEPTH', 'AC', 'SP', 'COND', 'ML1', 'ML2'} - set(raw_data.columns)
    print('This las file loss attribution ', lack_item)
    sys.exit(1)

col_item = raw_data[['DEPTH', 'AC', 'SP', 'COND', 'ML1', 'ML2']]
use_item = col_item[
    (col_item['AC'] != -9999) & (col_item['SP'] != -9999) & (col_item['COND'] != -9999) & (col_item['ML1'] != -9999) & (
                col_item['ML2'] != -9999)]
use_item['Well'] = well_name[:2]
pred_input = np.array(use_item)

# predict the level or not
ans = bst.predict(pred_input)
ans = np.argmax(ans, axis=1)

# generate output data
use_item['level'] = ans
output = use_item[['DEPTH', 'level']]

# save the result
output.to_excel('{}/{}.xlsx'.format(result_save_path, well_name))

print('Mission complete!')
sys.exit(0)
