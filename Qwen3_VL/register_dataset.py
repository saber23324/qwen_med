from swift.dataset import (
    ResponsePreprocessor, DatasetMeta, register_dataset, SubsetDataset, load_dataset
)
from typing import Dict, Any

class CustomPreprocessor(ResponsePreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = f"""任务：判断下面两句话语意是否相似。
        句子1: {row['text1']}
        句子2: {row['text2']}
        请输出类别[0/1]: 0代表含义不同, 1代表含义相似。
        """
        response = str(row['label'])
        row = {
            'query': query,
            'response': response
        }
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        # ms_dataset_id='qwen/kit_6',
        dataset_path = '/BDSZ6/private/user/yxd/data/qwen/data_6/',
        dataset_name = 'kit_6',
        # subsets=[SubsetDataset('train', split=['train']), SubsetDataset('test', split=['test'])],
        preprocess_func=CustomPreprocessor(),
    ))

if __name__ == '__main__':
    # load_dataset returns train_dataset and val_dataset based on `split_dataset_ratio`
    # Here, since we didn't pass `split_dataset_ratio` (defaults to 0), we take the first one (index 0)
    dataset = load_dataset('/BDSZ6/private/user/yxd/data/qwen/data_6/')[0]
    # test_dataset = load_dataset('swift/financial_classification:test')[0]
    print(f'dataset[0]: {dataset[0]}')
    # print(f'test_dataset[0]: {test_dataset[0]}')