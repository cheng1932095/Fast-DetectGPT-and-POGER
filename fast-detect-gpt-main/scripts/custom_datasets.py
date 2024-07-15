import os.path
import random
import datasets

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed']

def load_dataset(path, name=None, split=None, cache_dir=None):
    # 如果本地模型存在，则使用本地模型
    local_path = os.path.join(cache_dir, f'local.{path}_{name}_{split}')
    if os.path.exists(local_path):
        return datasets.load_from_disk(local_path)  # 从本地加载数据集
    return datasets.load_dataset(path, name, split=split, cache_dir=cache_dir)
    # 否则，从指定路径加载数据集，如果未指定数据集名称，则加载默认数据集，如果指定了split，则只加载特定的数据集分割


def load_pubmed(cache_dir):
    data = load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)

    # 将问题和长回答结合起来
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]
    # 使用列表推导式将每个问题和长回答组合成格式化的字符串

    return data  # 返回处理后的数据


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'  # 定义写作题目数据的路径

    # 从文件中读取写作题目和故事内容
    with open(f'{writing_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()  # 逐行读取题目数据
    with open(f'{writing_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()  # 逐行读取故事数据

    # 处理题目中的特定字符串，并组合题目和故事内容
    prompts = [process_prompt(prompt) for prompt in prompts]  # 处理每个题目，去除特定标记
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]  # 组合处理后的题目和故事内容，并处理空格
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]  # 过滤掉包含 'nsfw' 或 'NSFW' 的故事内容

    random.seed(0)  # 设置随机种子，确保随机性可复现
    random.shuffle(filtered)  # 将数据集中的内容随机打乱顺序

    return filtered  # 返回处理后的、随机打乱顺序的故事内容列表


def load_language(language, cache_dir):
    # 加载WMT16数据集中的英语或德语部分
    assert language in ['en', 'de']  # 断言语言参数必须是'en'或'de'

    d = load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)  # 加载WMT16数据集的指定语言部分

    docs = d['translation']  # 获取数据集中的翻译文档
    desired_language_docs = [d[language] for d in docs]  # 提取指定语言的文档内容

    lens = [len(d.split()) for d in desired_language_docs]  # 计算每个文档的单词数

    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    # 从符合长度条件（100到150个单词之间）的文档中筛选出子集

    return sub  # 返回符合条件的文档子集



def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        # 如果数据集名称在已知数据集列表中
        load_fn = globals()[f'load_{name}']  # 获取对应数据集加载函数
        return load_fn(cache_dir=cache_dir, **kwargs)  # 调用加载函数，并传递cache_dir和其他可选参数
    else:
        raise ValueError(f'Unknown dataset {name}')  # 如果数据集名称未知，则抛出值错误异常
